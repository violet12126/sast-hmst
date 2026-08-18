"""
Exportable SAST — 纯 torch 可导出包装 (ONNX/TensorRT)
=====================================================

目标: 把完整 SAST 前向 (信号 → 增强 TFR) 变成可被 torch.export + ONNX + TRT
消费的纯 torch 计算图.

与 models/sast.py 的差异 (全部为可导出性服务):
1. STFT: 复数 index_put → real/imag 分离 scatter (ONNX 的 complex index_put 无分解)
2. refine_if: 数据依赖分支 `if valid.any()` → 固定迭代次数 (torch.export 禁止数据依赖 guard)
3. 节点特征提取: 布尔掩码就地赋值 → torch.where; 动态索引 → torch.gather
4. ReassignerFunction (自定义 autograd + C++ kernel) → 纯 torch scatter_add 循环
5. 推理分支多轮挤压: `if src_mask.max()==0: break` → 固定 N_max 轮

权重: 复用训练好的 SAST 子模块 (prototype_matcher / ppm / gat / sqz_controller),
      通过 `from_pretrained` 组装, 无需改动原模型文件.

用法:
    from sast_export_model import ExportableSAST
    exportable = ExportableSAST.from_checkpoint('sast_v3_e050.pt', device)
    tfr = exportable(signal)   # [1, F, T]
"""

import math
import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


# ═══════════════════════════════════════════════════════════════
# 1. MSST — 可导出纯 torch 版
# ═══════════════════════════════════════════════════════════════

def _msst_stft_export(x: torch.Tensor, h: torch.Tensor,
                      N: int, hlength: int, Lh: int,
                      neta: int, tcol: int) -> torch.Tensor:
    """复数 STFT. real/imag 分离实现 index_put, 避免 ONNX complex index_put."""
    device = x.device
    tau = torch.arange(-Lh, Lh + 1, device=device, dtype=torch.long)
    ti = torch.arange(tcol, device=device, dtype=torch.long)
    idx = ti.unsqueeze(1) + tau.unsqueeze(0)
    valid = (idx >= 0) & (idx < N)
    idx_c = idx.clamp(0, N - 1)
    rSig = x[idx_c] * valid
    win_idx = (Lh + tau).clamp(0, hlength - 1)
    windowed = rSig * h[win_idx].unsqueeze(0)  # [tcol, 2Lh+1]

    indices = (N + tau) % N  # [2Lh+1]

    # float32: 避免 complex128 → angle 输出 float64 → ONNX Atan CPU EP 不支持
    tfr_pre_re = torch.zeros(N, tcol, dtype=torch.float64, device=device)
    # 就地 scatter: real tensor 的 index_put 可分解 (ONNX ScatterND)
    tfr_pre_re[indices, :] = windowed.t()
    tfr_pre = torch.complex(tfr_pre_re, torch.zeros_like(tfr_pre_re))
    tfr = torch.fft.fft(tfr_pre, dim=0)[:neta, :]
    return tfr


def _phase_unwrap_export(phase: torch.Tensor, dim: int = -1) -> torch.Tensor:
    dd = torch.diff(phase, dim=dim)
    ddmod = (dd + math.pi) % (2 * math.pi) - math.pi
    ph_correct = ddmod - dd
    ph_correct = torch.where(dd.abs() < math.pi, torch.zeros_like(ph_correct), ph_correct)
    up = phase.clone()
    slices_start = [slice(None)] * phase.ndim
    slices_start[dim] = slice(1, None)
    up[tuple(slices_start)] = up[tuple(slices_start)] + torch.cumsum(ph_correct, dim=dim)
    return up


def _estimate_if_export(tfr_complex: torch.Tensor, N: int) -> torch.Tensor:
    # 与原始 estimate_if 完全一致 (float64 精度, 含相位 unwrap):
    #   angle → unwrap → diff → round
    # float64 Atan 在 onnxruntime CPU EP 不支持, 但部署目标是 Jetson TRT,
    # 且本函数在 PyTorch 内验证 (见 export_sast_onnx.py 的 PyTorch 验证分支).
    angle = torch.atan2(tfr_complex.imag, tfr_complex.real)
    unwrapped = _phase_unwrap_export(angle, dim=-1)
    omega_cont = torch.diff(unwrapped, dim=-1) * N / (2.0 * math.pi)
    omega_cont = torch.cat([omega_cont, omega_cont[:, -1:]], dim=-1)
    # 保持 int64: ONNX 的 Clip/Sub 对 Python int 常量 (int32) 与 tensor 混用不做类型提升
    return torch.round(omega_cont).to(torch.int64)


def _refine_if_export(omega: torch.Tensor, neta: int, num: int):
    """固定迭代次数, 无数据依赖分支的 IF 精化."""
    F, T_col = omega.shape
    omegas = [omega.clone()]
    omega_cur = omega
    if num > 1:
        b_idx = torch.arange(T_col, device=omega.device).unsqueeze(0).expand(F, T_col)
        for _ in range(num - 1):
            valid = (omega_cur >= 1) & (omega_cur <= neta)
            k_vals = (omega_cur - 1).clamp(0, F - 1)
            # 全量 gather (等价 omega_cur[k_vals, b_idx]), 无效位置随后 masked
            gathered = omega_cur[k_vals, b_idx]
            omega_cur = torch.where(valid, gathered, torch.zeros_like(omega_cur))
            omegas.append(omega_cur.clone())
    else:
        omega_cur = omega.clone()
    return omegas, omega_cur


def msst_torch_export(x: torch.Tensor, fs: float,
                      hlength: Optional[int] = None,
                      num: int = 4) -> Dict:
    """完整 MSST (可导出): 返回 STFT + omega_final + omegas."""
    device = x.device
    N = len(x)
    if hlength is None:
        hlength = min(N, 512)
    hlength = hlength + 1 - (hlength % 2)
    ht = torch.linspace(-0.5, 0.5, hlength, device=device, dtype=torch.float64)
    h = torch.exp(-math.pi / 0.32**2 * ht**2)
    Lh = (hlength - 1) // 2
    tcol = N
    neta = int(round(N / 2))
    tfr = _msst_stft_export(x.double(), h, N, hlength, Lh, neta, tcol)
    omega = _estimate_if_export(tfr, N)
    omegas, omega_final = _refine_if_export(omega, neta, num)
    freqs = torch.arange(neta, device=device, dtype=torch.float32) / N * fs
    t_axis = torch.arange(tcol, device=device, dtype=torch.float32) / fs
    # 保持 complex128 (避免 _to_copy complex64 的 ONNX 复数分解问题);
    # 下游 .abs() / .angle() 都在 complex128 上做, 输出已转实数.
    tfr_float = tfr / (N / 2.0)
    return {
        'STFT': tfr_float,
        'freqs': freqs,
        't': t_axis,
        'omega_final': omega_final,
        'omegas': omegas,
    }


# ═══════════════════════════════════════════════════════════════
# 2. 节点特征提取 — 可导出版 (无布尔就地赋值)
# ═══════════════════════════════════════════════════════════════

def _extract_nodes_export(tfr_stft: torch.Tensor, omega_final: torch.Tensor,
                          freqs: torch.Tensor, fs: float, T_sig: int,
                          regions) -> Dict[str, torch.Tensor]:
    """单样本可导出的节点特征提取."""
    device = tfr_stft.device
    N_phys = len(regions)
    F, T = omega_final.shape

    valid_mask = omega_final >= 1  # [F, T]
    omega_f = omega_final.float()
    if_map = torch.where(valid_mask, (omega_f - 1.0) * fs / T_sig,
                         torch.zeros_like(omega_f))  # [F, T] Hz
    stft_energy = tfr_stft.abs() ** 2

    if_hz = torch.zeros(N_phys, T, device=device, dtype=torch.float32)
    energy = torch.zeros(N_phys, T, device=device, dtype=torch.float32)
    bw = torch.zeros(N_phys, T, device=device, dtype=torch.float32)

    for n_idx, region in enumerate(regions):
        freq_mask = (freqs >= region.f_min) & (freqs <= region.f_max)  # [F]
        region_if = if_map[freq_mask, :]                    # [F_r, T]
        region_valid = valid_mask[freq_mask, :]             # [F_r, T]
        region_energy_map = stft_energy[freq_mask, :]       # [F_r, T]
        F_r = region_if.shape[0]
        counts = region_valid.sum(dim=0)                    # [T]

        # median IF: 无效位置置 inf 后排序, 取中位 (gather 替代动态索引)
        region_if_sort = torch.where(region_valid, region_if,
                                     torch.full_like(region_if, float('inf')))
        region_if_sorted, _ = torch.sort(region_if_sort, dim=0)  # [F_r, T]
        med_idx = ((counts - 1) // 2).clamp(min=0).long()         # [T]
        if_hz_n = torch.gather(region_if_sorted, 0, med_idx.view(1, T))[0]  # [T]
        center = (region.f_min + region.f_max) / 2.0
        if_hz_n = torch.where(counts == 0,
                              torch.full_like(if_hz_n, center), if_hz_n)
        if_hz[n_idx] = if_hz_n

        # energy (log1p)
        region_energy = (region_energy_map * region_valid.float()).sum(dim=0)
        energy[n_idx] = torch.log1p(region_energy)

        # bandwidth / std
        if_sum = (region_if * region_valid.float()).sum(dim=0)
        mu = if_sum / counts.clamp(min=1)
        sq_dev = (region_if - mu.unsqueeze(0)) ** 2
        sq_sum = (sq_dev * region_valid.float()).sum(dim=0)
        bw_n = torch.sqrt(sq_sum / counts.clamp(min=1)).clamp(max=20.0)
        bw_n = torch.where(counts < 2, torch.ones_like(bw_n), bw_n)
        bw[n_idx] = bw_n

    # persistence
    persist = torch.zeros(N_phys, device=device, dtype=torch.float32)
    for n_idx in range(N_phys):
        persist[n_idx] = (energy[n_idx] > 0.01).float().mean()

    return {
        'node_if': if_hz,
        'node_energy': energy,
        'node_bw': bw,
        'node_persist': persist,
    }


# ═══════════════════════════════════════════════════════════════
# 3. 重排 — 纯 torch scatter_add (无自定义 autograd, 无 C++ kernel)
# ═══════════════════════════════════════════════════════════════

def _gaussian_reassign_export(tfr_mag: torch.Tensor, sigma: torch.Tensor,
                              omega_hat_int: torch.Tensor, K: int,
                              F_dim: int) -> torch.Tensor:
    """软高斯重排 (单轮). 与 ReassignerFunction Python fallback 数值一致."""
    B, F, T = tfr_mag.shape
    dtype = sigma.dtype
    eps = 1e-8
    offsets = list(range(-K, K + 1))
    Z = torch.zeros(B, F, T, device=tfr_mag.device, dtype=dtype)
    for k in offsets:
        r = k / sigma
        Z = Z + torch.exp(-0.5 * r * r)
    Z = Z + eps
    tfr_enhanced = torch.zeros(B, F, T, device=tfr_mag.device, dtype=dtype)
    for k in offsets:
        r = k / sigma
        w_k = torch.exp(-0.5 * r * r) / Z
        target = (omega_hat_int + k).clamp(0, F - 1)
        tfr_enhanced = tfr_enhanced.scatter_add(1, target, w_k * tfr_mag)
    return tfr_enhanced


def _multi_round_reassign_export(tfr_mag, sigma, omega_hat_int, K, F_dim,
                                 n_sqz_per_bin, ridge_factor, N_max):
    """推理多轮挤压 (固定 N_max 轮, 无 break)."""
    tfr_cur = tfr_mag
    for r in range(1, N_max + 1):
        src_mask = (n_sqz_per_bin >= r).to(tfr_cur.dtype)
        part = src_mask * ridge_factor
        moved = _gaussian_reassign_export(tfr_cur * part, sigma, omega_hat_int, K, F_dim)
        tfr_cur = moved + tfr_cur * (1.0 - part)
    return tfr_cur


# ═══════════════════════════════════════════════════════════════
# 4. ExportableSAST — 顶层包装
# ═══════════════════════════════════════════════════════════════

class ExportableSAST(nn.Module):
    """复用训练好的 SAST 子模块, forward 为纯 torch 可导出推理路径.

    输入: [1, T] 信号 (float32)
    输出: [1, F, T] 增强 TFR (float32)
    """

    def __init__(self, model, fs: int, n_sqz_max: int = 4):
        super().__init__()
        # 复用训练好的子模块 (权重共享, 非拷贝)
        self.prototype_matcher = model.prototype_matcher
        self.ppm = model.ppm
        self.gat = model.gat
        self.sqz_controller = model.sqz_controller
        self.register_buffer('edge_src', model.edge_src.clone())
        self.register_buffer('edge_dst', model.edge_dst.clone())

        self.fs = fs
        self.N_phys = len(model.regions)
        self.regions = model.regions
        self.n_sqz_max = n_sqz_max
        self.sigma_min = model.sigma_min
        self.sigma_max = model.sigma_max
        self.fs_half = fs / 2.0

        # 边特征常量 (w_type per edge)
        from models.sast_graph import PHYSICS_EDGES
        self.register_buffer('w_type', torch.tensor(
            [e.w_type for e in PHYSICS_EDGES], dtype=torch.float32))
        self.M_edges = len(PHYSICS_EDGES)
        # 边索引 (转 0-indexed: OP=0, phys 1..3 -> 0..2)
        self.src_idx = [e.src - 1 for e in PHYSICS_EDGES]
        self.dst_idx = [e.dst - 1 for e in PHYSICS_EDGES]
        self._e_types = [e.edge_type for e in PHYSICS_EDGES]
        self._r_nom = [e.r_nom for e in PHYSICS_EDGES]

    # ── 工厂 ──
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: torch.device) -> 'ExportableSAST':
        from sast_model import load_checkpoint
        model, _freq_enc, meta = load_checkpoint(ckpt_path, device)
        model.eval()
        cfg = meta['config']
        exportable = cls(model, fs=cfg.fs, n_sqz_max=cfg.n_sqz_max)
        exportable.eval()
        return exportable

    # ── 边特征 (可导出) ──
    def _compute_edge_feats(self, node_if, node_energy, node_persist, node_bw):
        """[B=1, M, T, 5] 边特征, 复用 compute_edge_features_torch 逻辑."""
        from models.sast_graph import compute_edge_features_torch, PHYSICS_EDGES
        return compute_edge_features_torch(
            node_if, node_energy, node_persist, node_bw,
            edges=PHYSICS_EDGES, window_size=5, fs=self.fs)

    # ── forward ──
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [1, T] 原始信号 (float32, CUDA)

        Returns:
            tfr_enhanced: [1, F, T]
        """
        B, T_sig = x.shape
        device = x.device

        # ══ Step 1+2: MSST + 节点提取 (单样本) ══
        x1 = x[0]
        msst_r = msst_torch_export(x1, self.fs, hlength=None, num=4)
        tfr_stft = msst_r['STFT']                    # [F, T] complex
        omega_final = msst_r['omega_final']          # [F, T] int32
        freqs = msst_r['freqs']                      # [F]
        omegas_traj = msst_r['omegas']               # list [F, T]

        nodes = _extract_nodes_export(tfr_stft, omega_final, freqs,
                                      self.fs, T_sig, self.regions)
        node_if = nodes['node_if'].unsqueeze(0)      # [1, N, T]
        node_energy = nodes['node_energy'].unsqueeze(0)
        node_bw = nodes['node_bw'].unsqueeze(0)
        node_persist = nodes['node_persist'].unsqueeze(0)  # [1, N]
        tfr_mag = tfr_stft.abs().unsqueeze(0)        # [1, F, T]
        omegas = torch.stack(omegas_traj).unsqueeze(0)    # [1, N_max, F, T]
        omegas = omegas.long()

        F_bins = freqs.shape[0]
        T_msst = tfr_mag.shape[-1]

        # ══ Step 3: V_obs -> cond_ctx ══
        V_obs = self.prototype_matcher.compute_V_obs(node_energy, freqs)
        cond_ctx, alpha = self.prototype_matcher(V_obs)   # [B, T, d_cond]

        # ══ Step 4: edge features ══
        edge_feats_t = self._compute_edge_feats(node_if, node_energy,
                                                node_persist, node_bw)  # [B, M, T, 5]
        r_obs_t = edge_feats_t[..., 0]                # [B, M, T]

        # ══ Step 5: PPM -> GAT -> w_i (B*T 合并) ══
        N_phys = self.N_phys
        M_edges = self.M_edges
        d_cond = cond_ctx.shape[-1]
        BT = B * T_msst

        f_norm = node_if / self.fs_half
        persist_exp = node_persist.unsqueeze(-1).expand(-1, -1, T_msst)
        raw_feats = torch.stack([f_norm, node_energy, node_bw / self.fs_half,
                                 persist_exp], dim=-1)
        raw_feats_bt = raw_feats.permute(0, 2, 1, 3).reshape(BT, N_phys, 4)
        node_if_bt = node_if.permute(0, 2, 1).reshape(BT, N_phys)
        r_obs_bt = r_obs_t.permute(0, 2, 1).reshape(BT, M_edges)
        cond_ctx_bt = cond_ctx.reshape(BT, d_cond)
        drft_bt = edge_feats_t[:, :, :, 0].permute(0, 2, 1).reshape(BT, M_edges)

        h_enhanced, C_prior_t, gate_edge_bt, gate_node_bt, cond_sim_bt = self.ppm(
            raw_feats_bt, node_if_bt, r_obs_bt, cond_ctx_bt, drft_bt, drft_bt)

        h_op_padded = F.pad(h_enhanced[:, :1, :], (0, 1))
        h_phys_cat = torch.cat([h_enhanced[:, 1:, :], C_prior_t.unsqueeze(-1)], dim=-1)
        h_cat = torch.cat([h_op_padded, h_phys_cat], dim=1)
        h_gat_in = self.ppm.gat_input_proj(h_cat)

        edge_feats_bt = edge_feats_t.permute(0, 2, 1, 3).reshape(BT, M_edges, 5)
        from models.sast_graph import CONDITION_EDGE_INDICES
        for i, m in enumerate(CONDITION_EDGE_INDICES):
            edge_feats_bt[:, m, 0] = cond_sim_bt[:, i]

        w_i_bt, A_ij_bt = self.gat(h_gat_in, edge_feats_bt,
                                   self.edge_src, self.edge_dst)
        w_i = w_i_bt.reshape(B, T_msst, N_phys).permute(0, 2, 1)   # [B, N_phys, T]

        # ══ Step 6: sigma_sq + squeeze control ══
        delta = self.sigma_max - self.sigma_min
        sigma_i = self.sigma_min + (1.0 - w_i) * delta              # [B, N, T]

        freqs_exp = freqs.view(1, 1, F_bins, 1).to(device)
        node_if_exp = node_if.unsqueeze(2)
        dist = (freqs_exp - node_if_exp).abs()
        i_star = dist.argmin(dim=1)                                 # [B, F, T]

        B_idx_f = torch.arange(B, device=device).view(B, 1, 1).expand(-1, F_bins, T_msst)
        T_idx_f = torch.arange(T_msst, device=device).view(1, 1, T_msst).expand(B, F_bins, -1)
        sigma_sq = sigma_i[B_idx_f, i_star, T_idx_f]

        bw_expected = torch.tensor([r.bw_expected for r in self.regions],
                                   device=device, dtype=torch.float32)
        lambda_sqz, ridge_factor = self.sqz_controller(
            w_i, tfr_mag, freqs, node_if, bw_expected)
        lambda_per_bin = lambda_sqz[B_idx_f, i_star, T_idx_f]
        n_sqz_per_bin = (lambda_per_bin * ridge_factor).round() \
            .clamp(1, self.n_sqz_max).long()

        # ══ Step 7: 多轮高斯重排 ══
        sigma = sigma_sq.clamp(self.sigma_min, self.sigma_max)
        omega_final_r = omegas[:, -1, :, :].float()
        # clamp 边界用 int64 tensor (ONNX Clip 不做 Python int 类型提升, 否则 int32/int64 混用)
        lo_i64 = torch.zeros((), device=device, dtype=torch.int64)
        hi_i64 = torch.tensor(F_bins - 1, device=device, dtype=torch.int64)
        omega_hat_int = (omega_final_r - 1.0).round().long().clamp(lo_i64, hi_i64)
        K = int(math.ceil(3.0 * self.sigma_max))

        # 多轮重排. 默认用 CUDA kernel 加速 (一次 launch, 避免 91×4 Python 循环);
        # 导出 ONNX 时用纯 torch 版 (循环展开, 图大但仅导出用, 部署端仍走 kernel).
        tfr_enhanced = _multi_round_reassign_export(
            tfr_mag, sigma, omega_hat_int, K, F_bins,
            n_sqz_per_bin, ridge_factor, self.n_sqz_max)

        return tfr_enhanced

    # ── 网络部分 (到 sigma_sq/ridge 为止, 不含重排) — 部署时与 CUDA kernel 组合 ──
    def forward_network(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        只算网络部分 (MSST→GAT→w_i→sigma), 输出重排所需的全部中间量.
        重排留在 CUDA kernel (deploy/reassigner.cu) 做, 避免 91 次循环展开.
        """
        B, T_sig = x.shape
        device = x.device

        x1 = x[0]
        msst_r = msst_torch_export(x1, self.fs, hlength=None, num=4)
        tfr_stft = msst_r['STFT']
        omega_final = msst_r['omega_final']
        freqs = msst_r['freqs']
        omegas_traj = msst_r['omegas']

        nodes = _extract_nodes_export(tfr_stft, omega_final, freqs,
                                      self.fs, T_sig, self.regions)
        node_if = nodes['node_if'].unsqueeze(0)
        node_energy = nodes['node_energy'].unsqueeze(0)
        node_bw = nodes['node_bw'].unsqueeze(0)
        node_persist = nodes['node_persist'].unsqueeze(0)
        tfr_mag = tfr_stft.abs().unsqueeze(0)
        omegas = torch.stack(omegas_traj).unsqueeze(0).long()

        F_bins = freqs.shape[0]
        T_msst = tfr_mag.shape[-1]

        V_obs = self.prototype_matcher.compute_V_obs(node_energy, freqs)
        cond_ctx, alpha = self.prototype_matcher(V_obs)

        edge_feats_t = self._compute_edge_feats(node_if, node_energy,
                                                node_persist, node_bw)
        r_obs_t = edge_feats_t[..., 0]

        N_phys = self.N_phys
        M_edges = self.M_edges
        d_cond = cond_ctx.shape[-1]
        BT = B * T_msst

        f_norm = node_if / self.fs_half
        persist_exp = node_persist.unsqueeze(-1).expand(-1, -1, T_msst)
        raw_feats = torch.stack([f_norm, node_energy, node_bw / self.fs_half,
                                 persist_exp], dim=-1)
        raw_feats_bt = raw_feats.permute(0, 2, 1, 3).reshape(BT, N_phys, 4)
        node_if_bt = node_if.permute(0, 2, 1).reshape(BT, N_phys)
        r_obs_bt = r_obs_t.permute(0, 2, 1).reshape(BT, M_edges)
        cond_ctx_bt = cond_ctx.reshape(BT, d_cond)
        drft_bt = edge_feats_t[:, :, :, 0].permute(0, 2, 1).reshape(BT, M_edges)

        h_enhanced, C_prior_t, gate_edge_bt, gate_node_bt, cond_sim_bt = self.ppm(
            raw_feats_bt, node_if_bt, r_obs_bt, cond_ctx_bt, drft_bt, drft_bt)

        h_op_padded = F.pad(h_enhanced[:, :1, :], (0, 1))
        h_phys_cat = torch.cat([h_enhanced[:, 1:, :], C_prior_t.unsqueeze(-1)], dim=-1)
        h_cat = torch.cat([h_op_padded, h_phys_cat], dim=1)
        h_gat_in = self.ppm.gat_input_proj(h_cat)

        edge_feats_bt = edge_feats_t.permute(0, 2, 1, 3).reshape(BT, M_edges, 5)
        from models.sast_graph import CONDITION_EDGE_INDICES
        for i, m in enumerate(CONDITION_EDGE_INDICES):
            edge_feats_bt[:, m, 0] = cond_sim_bt[:, i]

        w_i_bt, A_ij_bt = self.gat(h_gat_in, edge_feats_bt,
                                   self.edge_src, self.edge_dst)
        w_i = w_i_bt.reshape(B, T_msst, N_phys).permute(0, 2, 1)

        delta = self.sigma_max - self.sigma_min
        sigma_i = self.sigma_min + (1.0 - w_i) * delta

        freqs_exp = freqs.view(1, 1, F_bins, 1).to(device)
        node_if_exp = node_if.unsqueeze(2)
        dist = (freqs_exp - node_if_exp).abs()
        i_star = dist.argmin(dim=1)

        B_idx_f = torch.arange(B, device=device).view(B, 1, 1).expand(-1, F_bins, T_msst)
        T_idx_f = torch.arange(T_msst, device=device).view(1, 1, T_msst).expand(B, F_bins, -1)
        sigma_sq = sigma_i[B_idx_f, i_star, T_idx_f]

        bw_expected = torch.tensor([r.bw_expected for r in self.regions],
                                   device=device, dtype=torch.float32)
        lambda_sqz, ridge_factor = self.sqz_controller(
            w_i, tfr_mag, freqs, node_if, bw_expected)
        lambda_per_bin = lambda_sqz[B_idx_f, i_star, T_idx_f]
        n_sqz_per_bin = (lambda_per_bin * ridge_factor).round() \
            .clamp(1, self.n_sqz_max).long()

        sigma = sigma_sq.clamp(self.sigma_min, self.sigma_max)
        omega_final_r = omegas[:, -1, :, :].float()
        # clamp 边界用 int64 tensor (ONNX Clip 不做 Python int 类型提升, 否则 int32/int64 混用)
        lo_i64 = torch.zeros((), device=device, dtype=torch.int64)
        hi_i64 = torch.tensor(F_bins - 1, device=device, dtype=torch.int64)
        omega_hat_int = (omega_final_r - 1.0).round().long().clamp(lo_i64, hi_i64)

        return {
            'tfr_mag': tfr_mag,             # [1, F, T] 源能量
            'sigma': sigma,                 # [1, F, T] 核宽
            'omega_hat_int': omega_hat_int,  # [1, F, T] IF 目标 bin
            'ridge_factor': ridge_factor,   # [1, F, T] bin 参与因子
            'n_sqz_per_bin': n_sqz_per_bin,  # [1, F, T] 挤压轮数
            'w_i': w_i,                     # [1, N_phys, T]
            'freqs': freqs,
        }
