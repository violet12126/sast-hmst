"""SAST v4 可微图注意力 —— 最小图 (3 物理节点 + 7 条边)
=========================================================

在 v4 的双窗前端之后、分频带 β 之前, 用一个小 GAT 从**长窗谱**读出
逐节点的挤压强度 ``w_node`` 与频率落点修正 ``Δω``。这是 v3 GAT 的
"只留物理骨架"版本。

与 v3 的三处差异 (2026-11-13 定稿)
----------------------------------
1. **不用 OP 虚拟节点 / 原型上下文**. v3 的 ``StaticPrototypeMatcher`` 与
   ``PhysicsPrototypeMemory`` (``obs_proj`` 已是死代码) 一并去掉。图只剩
   3 个物理节点 (LOW_FREQ / BPF / 2xBPF) 与 7 条物理边 —— 即
   ``PHYSICS_EDGES`` 去掉 3 条以 OP 为端点的 CONDITION 边。
   动机: OP 的工况上下文在 v3 里最终塌缩成一个与类标签共线的向量,
   GAT 学到的是"工况→w", 而 v4 要的是"谱形→w", 不该再引入标签捷径。

2. **节点特征取自 v4 自己的连续 IF**. v3 的 ``node_if`` 来自 MSST 的
   ``omega_final`` 中位数; v4 没有 MSST, 直接用 ``dual_frontend`` 的
   ``omega`` (连续 IF, bin) 在带内做能量加权质心。**排除 DC bin**:
   0 Hz 行的能量是窗内局部均值 (见 ``PUMP_TURBINE_REGIONS`` 的注释),
   不排除会把 LOW_FREQ 的质心拖到 0.5 Hz。

3. **Δω 只开锁频带**. BPF / 2xBPF 的频率由转速唯一确定, 允许 ±2 Hz 的
   落点修正; LOW_FREQ 是流动不稳定性决定的水力分量, 无绝对参考, 锁死 0。
   (``Δω = 2·tanh(mlp)``, 对 LOW_FREQ 乘 0)

粗时间网格
----------
T=2000 帧逐帧过 GAT 既慢又过拟合 (边特征滑窗在 1 ms 尺度上没有物理意义)。
先 ``avg_pool1d(k=51, stride=50)`` (≈50 ms / 控制点) 降到 40 点, GAT 在这
40 点上跑, 再用 ``F.interpolate`` 线性插回 2000。40 点 ≈ 50 ms 分辨率, 远高于
长窗 (1024 点, 等效时间 std ≈ 91 ms) 的特征截止, 又能分辨转轮 180 ms 周期。

边特征
------
复用 ``sast_graph.compute_edge_features_torch``, 只传 7 条无边 OP 的边。
该函数用 ``src = e.src - 1`` 做 0-based 索引, 物理节点 1..3 → 0..2, 与
本模块的 3 节点数组一致。``window_size`` 在粗网格上从 101 改为 5 (≈250 ms)。

输出头
------
* ``w_node``: **真区间映射 (2026-11-13 定稿)**
  ``w = w_low_eff + (w_high_eff − w_low_eff)·sigmoid(mlp + logit(frac_prior))``
  其中 ``w_low_eff = w_low_min + p·(w_range_low − w_low_min)``,
  ``w_high_eff = w_high_absent + (1−a)·(w_range_high − w_high_absent)``,
  ``frac_prior`` 是 C_prior 在该区间内的归一化位置。
  这样 w **从构造上**恒落在物理可行域内 (含缺席时上界收拢到 0.20),
  初值 ≈ C_prior, 且不像 4·tanh 那样只能保证 (0.018, 0.982)。
  代价: 区间边界是输入 (presence/absence) 的函数, 故 w 的可行域随样本变化。

  **插值口径**: GAT 跑在 40 点粗网格上, 但插值到全长的量是区间内的归一化
  位置 ``s = sigmoid(mlp + bias) ∈ (0,1)``, 到全长后再乘上**该帧的**有效区间。
  若直接插值 w, 有界性只在 40 个控制点上成立 —— 50 ms 内 absence 可整体翻转
  (空转 2xBPF 的逐帧 a 就在 0/1 之间跳), 两控制点边界不同时线性插值实测可
  越上界 +0.50。插 s 后逐帧严格有界 (实测最大越界 < 1e-4)。
* ``Δω``: ``2·tanh(mlp)``, ∈ [−2, +2] Hz, LOW_FREQ 乘 0。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sast import EdgeConditionedGATLayer
from models.sast_graph import (
    NODE_OP,
    PHYSICS_EDGES,
    compute_edge_features_torch,
)
from models.sast_nodes import (
    PUMP_TURBINE_REGIONS,
    FreqRegion,
    region_weights,
)

# ── 最小图: 去掉以 OP 为端点的 3 条 CONDITION 边, 剩 7 条 ──
PHYSICS_EDGES_MIN = [e for e in PHYSICS_EDGES
                     if e.src != NODE_OP and e.dst != NODE_OP]

N_PHYS = 3
# 粗网格: 51 点窗 / 50 点步长, padding=25 -> 恰好 40 点
COARSE_KERNEL = 51
COARSE_STRIDE = 50
COARSE_PAD = COARSE_KERNEL // 2


# ═══════════════════════════════════════════════════════════════
# 节点特征 (长窗谱 -> [B, N_phys, T])
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def node_features_from_v4(S_long: torch.Tensor, omega: torch.Tensor,
                          freqs: torch.Tensor,
                          regions: Optional[List[FreqRegion]] = None,
                          conc: Optional[torch.Tensor] = None
                          ) -> Dict[str, torch.Tensor]:
    """从 v4 长窗谱与连续 IF 构造 GAT 的节点特征。

    Args:
        S_long: [B, F, T] 长窗幅值 (非负)
        omega:  [B, F, T] 长窗连续 IF (bin, 0-based)
        freqs:  [F] Hz
        regions: FreqRegion 列表

    Returns:
        dict:
          node_if      [B, N, T] Hz, 带内能量加权质心 (**排除 DC bin**)
          node_energy  [B, N, T] log(带内 Σ S_long²)
          node_bw      [B, N, T] Hz, 带内 omega 的加权标准差
          node_persist [B, N]    带内集中度 (未挤压谱上的 presence, 0..1)。
                                 旧实现是 (e_band > 自身中位数) 的帧占比 —— 对
                                 连续正实数恒等于 0.500, 是零信息死特征
    """
    if regions is None:
        regions = PUMP_TURBINE_REGIONS
    B, Fb, T = S_long.shape
    dev, dtype = S_long.device, S_long.dtype
    N = len(regions)

    w_band = region_weights(freqs, regions).to(device=dev, dtype=dtype)  # [N,F]
    energy = S_long.pow(2)                                              # [B,F,T]

    # ── 节点能量 ──
    e_band = torch.einsum('nf,bft->bnt', w_band, energy)                 # [B,N,T]
    node_energy = torch.log(e_band.clamp_min(1e-12))

    # ── 带内质心 (排除 DC bin 0) ──
    # DC 行是窗内局部均值, 不是独立分量; 排除后 LOW_FREQ 质心才落在涡带本身。
    w_nodc = w_band.clone()
    f_bins = torch.arange(Fb, device=dev).view(1, -1)
    w_nodc = w_nodc * (f_bins > 0).to(dtype)                             # [N,F]
    e_nodc = torch.einsum('nf,bft->bnt', w_nodc, energy)                 # [B,N,T]
    if_bin = torch.einsum('nf,bft->bnt', w_nodc,
                          energy * omega)                                # [B,N,T]
    df_hz = float(freqs[1] - freqs[0])
    node_if = if_bin / e_nodc.clamp_min(1e-12) * df_hz                  # [B,N,T] Hz
    # 带内无能量 -> 退化为频带中心
    r_mid = torch.tensor([(r.f_min + r.f_max) / 2.0 for r in regions],
                         device=dev, dtype=dtype).view(1, N, 1)
    node_if = torch.where(e_nodc > 1e-12, node_if,
                          r_mid.expand(B, N, T))

    # ── 带内带宽 (omega 的加权 std) ──
    # var[ω] = E[ω²] − E[ω]² (避免构造 [B,N,F,T] 中间张量)
    m2_bin = torch.einsum('nf,bft->bnt', w_nodc,
                          energy * omega.pow(2))                         # [B,N,T]
    m2_hz2 = m2_bin / e_nodc.clamp_min(1e-12) * (df_hz ** 2)
    node_bw = (m2_hz2 - node_if.pow(2)).clamp_min(0.0).sqrt()

    # ── 持续性: 带内能量高于样本中位数的帧占比 ──
    med = e_band.median(dim=-1, keepdim=True).values                    # [B,N,1]
    # 旧: (e_band > med).mean(-1) ≡ 0.5 (死特征)。改用未挤压谱上的带内集中度;
    # conc 为 None 时退回旧口径以兼容独立调用。
    node_persist = (conc[:, :, 0] if conc is not None
                    else (e_band > med).to(dtype).mean(dim=-1))          # [B,N]

    return {'node_if': node_if, 'node_energy': node_energy,
            'node_bw': node_bw, 'node_persist': node_persist}


def _coarse(x: torch.Tensor) -> torch.Tensor:
    """[B, N, T] -> [B, N, T_c]。T_c = 40 (T=2000)。"""
    return F.avg_pool1d(F.pad(x, (COARSE_PAD, COARSE_PAD), mode='replicate'),
                        kernel_size=COARSE_KERNEL, stride=COARSE_STRIDE,
                        count_include_pad=False)


# ═══════════════════════════════════════════════════════════════
# GAT
# ═══════════════════════════════════════════════════════════════
class MinimalPhysicsGAT(nn.Module):
    """3 物理节点 + 7 边的边条件 GAT -> (w_node, Δω)。

    节点特征先做逐样本逐帧的跨节点标准化 (3 个节点内部), 再叠可学习的
    节点类型嵌入 —— 三个节点物理角色不同, 同一组特征在不同节点上应当有
    不同含义, 类型嵌入把这层先验显式给出。
    """

    def __init__(self, regions: Optional[List[FreqRegion]] = None,
                 d_h: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_e: int = 5, dropout: float = 0.1,
                 edge_window: int = 5,
                 dw_hz: float = 2.0,
                 dw_nodes: Tuple[int, ...] = (1, 2),
                 w_low_min: float = 0.05,
                 w_high_absent: float = 0.20,
                 use_absence_gate: bool = True,
                 no_edges: bool = False,
                 use_delta_omega: bool = True):
        super().__init__()
        self.regions = regions if regions is not None else PUMP_TURBINE_REGIONS
        self.N = N = len(self.regions)
        self.d_h = d_h
        self.n_layers = n_layers
        self.edge_window = edge_window
        self.dw_hz = dw_hz
        self.w_low_min = w_low_min
        self.w_high_absent = w_high_absent
        # use_absence_gate=False: 关闭手工缺席门控, 硬带用固定区间 [w_low_min,
        # w_high_range] 且 bias=0 (z=0 对区间中点)。旧路径保留供消融对照。
        self.use_absence_gate = use_absence_gate
        # no_edges=True: 跳过 GAT 层, 三层节点特征各自过 MLP (GAT vs 无边 MLP 消融)
        self.no_edges = no_edges
        # use_delta_omega=False: 不建 Δω 头, delta_hz 恒 0 (实测 mean|Δω|≤0.10 Hz)
        self.use_delta_omega = use_delta_omega

        self.edges = PHYSICS_EDGES_MIN
        self.register_buffer('edge_src',
                             torch.tensor([e.src - 1 for e in self.edges],
                                          dtype=torch.long), persistent=False)
        self.register_buffer('edge_dst',
                             torch.tensor([e.dst - 1 for e in self.edges],
                                          dtype=torch.long), persistent=False)

        # 节点特征投影: [if/10, logE_z, bw/5, persist] -> d_h
        self.feat_proj = nn.Linear(4, d_h)
        self.type_embed = nn.Parameter(torch.zeros(1, N, d_h))
        self.in_norm = nn.LayerNorm(d_h)

        self.layers = nn.ModuleList([
            EdgeConditionedGATLayer(d_h, d_e, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_h)
                                          for _ in range(n_layers)])

        def _head():
            return nn.Sequential(
                nn.Linear(d_h, d_h // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_h // 2, 1))

        self.mlp_w = _head()
        self.mlp_dw = _head() if use_delta_omega else None

        # w 的先验中心 (C_prior) 与物理区间; Δω 只在锁频带开
        c_prior = torch.tensor([r.C_prior for r in self.regions])
        self.register_buffer('c_prior', c_prior.view(1, N, 1),
                             persistent=False)
        self.register_buffer(
            'w_low_range',
            torch.tensor([r.w_range_low for r in self.regions]).view(1, N, 1),
            persistent=False)
        self.register_buffer(
            'w_high_range',
            torch.tensor([r.w_range_high for r in self.regions]).view(1, N, 1),
            persistent=False)
        m = torch.zeros(N, 1)
        for i in dw_nodes:
            if 0 <= i < N:
                m[i] = 1.0
        self.register_buffer('dw_mask', m.view(1, N, 1), persistent=False)

    # ── 特征标准化 ─────────────────────────────────────────
    @staticmethod
    def _standardize(feat: Dict[str, torch.Tensor], T_c: int) -> torch.Tensor:
        ifc = feat['node_if'] / 10.0                                  # Hz / 10
        logE = feat['node_energy']
        logE = (logE - logE.mean(dim=1, keepdim=True)) \
            / (logE.std(dim=1, keepdim=True, unbiased=False) + 1.0)
        bwc = feat['node_bw'] / 5.0
        per = feat['node_persist'].unsqueeze(-1).expand(
            -1, -1, T_c)
        return torch.stack([ifc, logE, bwc, per], dim=-1)

    def forward(self, feat: Dict[str, torch.Tensor],
                presence: Optional[torch.Tensor] = None,
                absence: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """Args:
            feat:     node_features_from_v4 的输出 (全长 T)
            presence: [B, N, T] 分量存在度 (0..1); 松弛 w 的下界
            absence:  [B, N, T] 缺席度 (0..1); 收拢 w 的上界到 w_high_absent

        Returns:
            dict:
              w_node    [B, N, T]   挤压强度, 恒在 [w_low_eff, w_high_eff] 内
              delta_hz  [B, N, T]   ∈ [−dw_hz, +dw_hz] Hz, 非锁频带恒 0
              A_ij      [B, M, T_c, H] 末层注意力 (粗网格)
              h_coarse  [B, N, T_c] 末层节点表示 (诊断)
              w_bounds  [B, N, T, 2] 逐帧有效区间 (下界, 上界)
              s_node    [B, N, T]   区间内的归一化位置 ∈(0,1)
              feat_coarse, node_feats
        """
        T_full = feat['node_if'].shape[-1]
        # 粗网格
        coarse = {k: (_coarse(v) if v.dim() == 3 else v)
                  for k, v in feat.items()}
        T_c = coarse['node_if'].shape[-1]

        x = self._standardize(coarse, T_c)              # [B,N,Tc,4]
        B, N = x.shape[0], x.shape[1]
        h = self.feat_proj(x).reshape(B, N, T_c, self.d_h)
        h = self.in_norm(h + self.type_embed.unsqueeze(2))

        # 边特征: [B,M,Tc,5]
        edge_feats = compute_edge_features_torch(
            node_if=coarse['node_if'],
            node_energy=coarse['node_energy'],
            node_persist=coarse['node_persist'],
            node_bw=coarse['node_bw'],
            edges=self.edges,
            window_size=self.edge_window,
            fs=1000.0,
        )

        # 逐时间步过 GAT: 把 [B,N,Tc,d_h] 折叠成 [B*Tc,N,d_h]
        h_flat = h.permute(0, 2, 1, 3).reshape(B * T_c, N, self.d_h)
        e_flat = edge_feats.permute(0, 2, 1, 3).reshape(
            B * T_c, edge_feats.shape[1], -1)
        A_ij = None
        # no_edges=True: 不跑图层, 每个节点各自过后续 MLP (无边 3 节点 MLP 消融)
        for i, (layer, norm) in enumerate(
                [] if self.no_edges else zip(self.layers, self.layer_norms)):
            residual = h_flat
            h_new, attn = layer(h_flat, e_flat, self.edge_src, self.edge_dst)
            h_flat = norm(residual + h_new)
            if i == self.n_layers - 1:
                A_ij = attn.reshape(B, T_c, attn.shape[1], -1).permute(0, 2, 1, 3)

        h_c = h_flat.reshape(B, T_c, N, self.d_h).permute(0, 2, 1, 3)  # [B,N,Tc,dh]

        # ── w_node: 真区间映射 ─────────────────────────────────
        # 区间边界由 presence/absence 决定 (逐样本逐帧), w 从构造上恒在
        # [w_low_eff, w_high_eff] 内 —— 命题里的"有界流形"是真命题, 不再
        # 依赖 range 罚项兜底。
        logits_mlp = self.mlp_w(h_c).squeeze(-1)                       # [B,N,Tc]

        # presence/absence 全长 -> 粗网格 (avg_pool 是同一组权重的凸组合,
        # 因此 [0,1] 与边界顺序在池化后保持)
        p_c = (_coarse(presence) if presence is not None
               else torch.ones_like(coarse['node_if']))
        a_c = (_coarse(absence) if absence is not None
               else torch.zeros_like(coarse['node_if']))
        w_lo_c, w_hi_c = self._bounds(p_c, a_c)

        # C_prior 在区间内的归一化位置作为先验中心: 初值 (mlp≈0) 时
        # w ≈ C_prior, 与 v3 的 c_prior_logit 语义一致
        if self.use_absence_gate:
            # 旧路径: 令 z=0 时 w≈C_prior。代价是 s0≈0.90, sigmoid 严重饱和
            # (σ'≈0.056), Δz=0.17 只能映射成 Δw≈0.008; 且缺席时 hi<C_prior
            # 会让 frac 被 clamp 到 1, bias≈+9.2, s 焊死在 1.0。
            frac = ((self.c_prior - w_lo_c) / (w_hi_c - w_lo_c)).clamp(
                1e-4, 1 - 1e-4)
            bias = torch.log(frac) - torch.log1p(-frac)
        else:
            # 去门控: z=0 对应区间中点, 两侧都有充足线性空间
            bias = torch.zeros_like(logits_mlp)
        s_c = torch.sigmoid(logits_mlp + bias)                         # [B,N,Tc] ∈(0,1)

        # 插回全长: **插值的是区间内的归一化位置 s, 再把逐帧边界加回来**。
        # 若直接插值 w, 有界性只在 40 个控制点上成立 —— 粗网格 50 ms 内
        # absence 可以整体翻转 (空转 2xBPF 逐帧 a 就在 0/1 之间跳), 两控制点
        # 边界不同时线性插值会越界最多 0.5。插 s 则每帧都严格落在该帧边界内。
        s_full = F.interpolate(s_c, size=T_full, mode='linear',
                               align_corners=False)
        p_f = presence if presence is not None else torch.ones_like(s_full)
        a_f = absence if absence is not None else torch.zeros_like(s_full)
        w_lo, w_hi = self._bounds(p_f, a_f)
        w_node = w_lo + (w_hi - w_lo) * s_full

        # ── Δω (只开锁频带) ──
        if self.mlp_dw is not None:
            dw_c = self.dw_hz * torch.tanh(self.mlp_dw(h_c).squeeze(-1))
            dw_c = dw_c * self.dw_mask
            delta_hz = F.interpolate(dw_c, size=T_full, mode='linear',
                                     align_corners=False)
        else:
            delta_hz = torch.zeros_like(s_full)

        return {'w_node': w_node, 'delta_hz': delta_hz, 'A_ij': A_ij,
                'h_coarse': h_c, 's_node': s_full,
                'w_bounds': torch.stack([w_lo, w_hi], dim=-1),
                # 未激活 logit z_w (粗网格) —— 供 λ_z·mean(z_w²) 正则使用。
                # RE_2D 对 β 单调, 缺少制衡时 sigmoid 会贴死在区间端点并
                # 自毁梯度 (实测 mlp_w 梯度降到 1e-7)。L2 把工作点拉回
                # |z_w| ≤ 2.5 的高梯度线性区。
                'z_w': logits_mlp,
                'feat_coarse': coarse, 'node_feats': feat}

    def _bounds(self, presence: torch.Tensor, absence: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """presence/absence -> 逐帧有效区间 [w_low_eff, w_high_eff]。

        缺席 -> 上界收拢到 ``w_high_absent``; 存在度低 -> 下界松弛到
        ``w_low_min``。二者都只在对应节点启用 (其余节点 a≡0)。
        """
        if not self.use_absence_gate:
            # 固定物理区间: 下界统一 w_low_min, 上界取各节点 w_range_high;
            # 不再随 presence/absence 逐帧缩放 (absence 也不收拢上界)
            B, N, T = presence.shape
            hi = self.w_high_range.expand(B, N, T)
            return torch.full_like(hi, self.w_low_min), hi
        lo = self.w_low_min + presence * (self.w_low_range - self.w_low_min)
        hi = self.w_high_absent + (1.0 - absence) * (self.w_high_range
                                                     - self.w_high_absent)
        hi = torch.maximum(hi, lo + 1e-3)                              # 防区间翻转
        return lo, hi


def band_delta_hz(delta_hz: torch.Tensor, freqs: torch.Tensor,
                  regions: Optional[List[FreqRegion]] = None) -> torch.Tensor:
    """[B,N,T] 节点 Δω -> [B,F,T] 逐 bin 的 Δω (Hz)。

    用 ``region_weights`` 的单位分解把每个 bin 唯一归属到一个节点;
    不属于任何频带的 bin (10-36, 53-90, 105+) Δω = 0。
    """
    if regions is None:
        regions = PUMP_TURBINE_REGIONS
    w_band = region_weights(freqs, regions).to(
        device=delta_hz.device, dtype=delta_hz.dtype)                  # [N,F]
    return torch.einsum('nf,bnt->bft', w_band, delta_hz)


if __name__ == '__main__':
    torch.manual_seed(0)
    from models.dual_stft import dual_frontend

    t = torch.arange(2000, dtype=torch.float64) / 1000.0
    x = (torch.cos(2 * torch.pi * 50.0 * t)
         + 0.6 * torch.cos(2 * torch.pi * 100.0 * t)
         + 0.3 * torch.cos(2 * math.pi * 1.2 * t)
         + 0.2 * torch.randn(2000, dtype=torch.float64)).unsqueeze(0).float()

    o = dual_frontend(x)
    feat = node_features_from_v4(o['S_long'], o['omega'], o['freqs'])
    gat = MinimalPhysicsGAT()
    out = gat(feat)

    print(f'w_node   {tuple(out["w_node"].shape)}  '
          f'范围 [{out["w_node"].min():.3f}, {out["w_node"].max():.3f}]')
    print(f'delta_hz {tuple(out["delta_hz"].shape)}  '
          f'范围 [{out["delta_hz"].min():.3f}, {out["delta_hz"].max():.3f}]')
    print(f'A_ij     {tuple(out["A_ij"].shape) if out["A_ij"] is not None else None}')
    print(f'参数 {sum(p.numel() for p in gat.parameters()):,}')
    print('节点 IF (Hz) 时间均值:',
          [f'{v:.2f}' for v in out['node_feats']['node_if'][0].mean(-1)])
    print('节点 w 时间均值:',
          [f'{v:.3f}' for v in out['w_node'][0].mean(-1)])
    print('节点 Δω 时间均值:',
          [f'{v:+.3f}' for v in out['delta_hz'][0].mean(-1)])
    db = band_delta_hz(out['delta_hz'], o['freqs'])
    print(f'band Δω 形状 {tuple(db.shape)}  '
          f'LOW_FREQ 段最大 |Δω| = {db[0, :36].abs().max():.4f} Hz (应为 0)')
