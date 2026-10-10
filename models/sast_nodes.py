"""
SAST Node Feature Extraction — 从 MSST omega_final 直接提取物理节点特征
=========================================================================

设计依据: SAST_MSST_必要性分析.md §5.2.2
  "不对 TFR 做脊线追踪，而是直接从 MSST 的 omega_final 中按频率区域聚合统计量"

核心洞察:
  MSST 的 omega_final[eta, b] 已经逐 bin 计算好了 IF 估计值（解析计算，非追踪）。
  SAST 只需按频率区域聚合这些值，避免脊线追踪引入的峰值检测/跳变/交叉混淆误差。

节点定义 (基于实测数据频谱分析, 2024):
  原设计 7 节点 (fr, LOW_FREQ, BPF, 2xBPF, 3xBPF, GPF, HIGH_HARMONIC)
  → 精简为 3 节点，依据:
    1. fr (~5.5 Hz): 物理上存在但加速度传感器不可观测 (ω² 衰减 81×)
    2. 3xBPF (150 Hz): 与 50 Hz 相关性接近零, 非真正的 3 次谐波
    3. GPF (110 Hz): 频谱中无独立峰, 100 Hz 附近只有 2xBPF
    4. HIGH_HARMONIC (468 Hz): 仅 31-45% 样本出现, 能量 <0.08%, 判定为噪声
    5. LOW_FREQ (11-15 Hz): 水力来源 (涡带/压力脉动), 与 BPF 不成整数倍

  保留节点 (B4 拆节点: 42-55 Hz 原为单一 BLADE_PASS 节点, 同时装着两种机制,
            一个标量 σ_i 无法既挤窄线又留宽峰 → 拆成两个同质节点):
    LOW_FREQ:  2-18 Hz   — 流动不稳定性 (涡带/压力脉动); 无绝对锚
    BPF_DRIFT: 42-48 Hz  — 工况漂移宽峰; 无绝对锚 (频率由流动不稳定性决定)
    BPF_LINE:  48-53 Hz  — 转速锁定线 1×Z_r·f_rot; 绝对锚 49.995 Hz ±1.5
    2xBPF:     90-105 Hz — 二倍叶片通过频率, 尖峰, 极度稳定; 绝对锚 99.990 Hz ±1.5

  边: BPF_LINE → 2xBPF (r_nom=2.0, INTEGER_HARMONIC) — 唯一的确定性倍频关系
      (只有锁转速的 BPF_LINE 才是 2xBPF 的基频; BPF_DRIFT 不参与倍频约束)

IF 聚合 (A2 可微 node_if):
  if_hz 的观测值由"带内无权重中位数"改为"带内能量加权均值"——
  中位数与能量无关, 在宽带 (如 LOW_FREQ 16 Hz) 上会被大量噪声 bin 拉向频带
  几何中心 (实测偏高 6-12 Hz), 是 r_obs = 2.13 ≠ 2.00 的直接来源。

用法:
  extractor = MSSTNodeExtractor(fs=1000, freq_regions=PUMP_TURBINE_REGIONS)
  nodes = extractor(signal)  # signal: [T] numpy array
  # nodes.if_hz: [N_phys, T] 每节点每帧 IF (Hz)
  # nodes.energy: [N_phys, T] 每节点每帧能量
  # nodes.bandwidth: [N_phys, T] 每节点每帧 IF 带宽 (Hz)
  # nodes.persistence: [N_phys] 每节点持久性
  # nodes.tfr_stft: [F, T] 复数 STFT (用于后续挤压)
  # nodes.freqs: [F] 频率轴 (Hz)
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from models.tfr import msst


# ═══════════════════════════════════════════════════════════════
# 水泵水轮机频率区域定义 (基于实测数据频谱分析)
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# 机组物理参数 → 绝对频率锚
# ═══════════════════════════════════════════════════════════════
# 额定转速 333.3 rpm, 转轮叶片数 9 → f_rot = 5.5550 Hz
#   1×BPF = Z_r·f_rot = 49.995 Hz (与 50 Hz 电网频率数值重合, 见 docs 讨论)
#   2×BPF = 2·Z_r·f_rot = 99.990 Hz (实测 100.10 Hz, 逐样本标准差 0.000)
# 锚的作用: 比值约束 (r_obs = f_dst/f_src) 是尺度不变的, 无法纠正共模频率偏差,
#           绝对锚提供唯一的"绝对参考系"。
# 锚的容差 anchor_tol: 只惩罚超出容差的漂移 (flat-bottom hinge), 不惩罚真实工况漂移。
N_RATED_RPM = 333.3
Z_RUNNER = 9
F_ROT_HZ = N_RATED_RPM / 60.0        # 5.5550 Hz
F_BPF_ANCHOR = Z_RUNNER * F_ROT_HZ   # 49.995 Hz
F_2XBPF_ANCHOR = 2.0 * Z_RUNNER * F_ROT_HZ   # 99.990 Hz


@dataclass
class FreqRegion:
    """单个物理节点的频率搜索区域。"""
    name: str           # 节点名称 (如 'BPF', '2xBPF')
    f_min: float        # 区域下界 (Hz)
    f_max: float        # 区域上界 (Hz)
    f_type: str         # 分量类型 (HYDRAULIC, BLADE_DRIFT, BLADE_PASS, BLADE_HARMONIC)
    C_prior: float      # 先验可压缩性倾向 (0-1)
    bw_expected: float  # 预期带宽 (Hz) — 用于原型匹配和异常检测
    persist_expected: float  # 预期持续性 (0-1)
    w_range_low: float  # w_i 逐帧下界 (0-1, 防跳变到过低)
    w_range_high: float # w_i 逐帧上界 (0-1, 防跳变到过高)
    f_anchor: Optional[float] = None    # 绝对频率锚 (Hz); None = 无绝对参考
    anchor_tol: float = 4.0             # 锚容差 (Hz), flat-bottom 半宽


PUMP_TURBINE_REGIONS: List[FreqRegion] = [
    # LOW_FREQ: 水力来源 (涡带/压力脉动), 2-18 Hz
    #   实测评测 (probe_node_if.py / check_spectral_bands.py):
    #     - 2.0 Hz 附近在所有工况都是带内能量峰 (E峰 f = 2.0-3.0 Hz)
    #     - 15.14 Hz 在低负荷是最强线 (凸出 11.2 dB), 属同一"水力"族的漂移分量
    #     - >17 Hz 各工况均为平坦宽带 (PMR 0.8-1.5 dB), 无独立峰
    #   → 上界由 25 Hz 收到 18 Hz: 去掉纯宽带尾部, 避免 IF 中位数被抬到带中
    #   预期宽频 (12 Hz), 中低持续性 (涡带仅部分工况显著)
    #   w_range: [0.10, 0.45] — 保守, 宽核, 不锐化
    #   无绝对锚: 水力分量频率由流动不稳定性决定, 不锁定转速
    FreqRegion('LOW_FREQ',  2.0, 18.0,  'HYDRAULIC',       0.30, 12.0, 0.50, 0.10, 0.45),
    # BPF_DRIFT: 工况漂移宽峰, 42-48 Hz (B4 拆节点)
    #   实测: 带内能量峰在 46.4-48.8 Hz; 45.9 Hz 候选分量在空转 (45.65% 转速)
    #   仍位于 46.39±0.61 Hz → 频率不锁转速, 由流动不稳定性决定。
    #   故: 无绝对锚, w_range 下界低 (允许不挤压), 不参与倍频约束。
    #   w_range: [0.10, 0.50] — 不信任其"线"性, 保留展宽
    FreqRegion('BPF_DRIFT', 42.0, 48.0,  'BLADE_DRIFT',    0.30,  6.0, 0.60, 0.10, 0.50),
    # BPF_LINE: 转速锁定线 1×BPF = Z_r·f_rot = 49.995 Hz, 48-53 Hz (B4 拆节点)
    #   该线位置由转速唯一确定 (与 50 Hz 电网频率数值重合, 只能靠空转换速区分)。
    #   拆开后不再与漂移宽峰共用一个 σ_i, 锚容差因此可以收紧到 ±1.5 Hz。
    #   w_range: [0.65, 0.95] — 最信任, 极锐
    FreqRegion('BPF_LINE',  48.0, 53.0,  'BLADE_PASS',     0.80,  1.5, 0.90, 0.65, 0.95,
               f_anchor=F_BPF_ANCHOR, anchor_tol=1.5),
    # 2xBPF: 二倍叶片通过频率, 90-105 Hz
    #   预期窄带 (2 Hz), 极高持续性
    #   实测: 带内能量峰 = 100.00 Hz (全部带负荷工况), 逐样本 f=100.10 Hz 标准差 0.000,
    #        空转时该分量消失 (存在度 0.10 vs 带负荷 0.32-0.34) → 确证为 2×BPF
    #   锚容差 ±1.5 Hz (紧): 该线位置由转速唯一确定, 不应漂移
    #   w_range: [0.65, 0.90] — 最信任, 极锐
    FreqRegion('2xBPF',    90.0, 105.0, 'BLADE_HARMONIC',  0.90,  2.0, 0.95, 0.65, 0.90,
               f_anchor=F_2XBPF_ANCHOR, anchor_tol=1.5),
]

# 节点数量
N_PHYSICS_NODES = len(PUMP_TURBINE_REGIONS)

# "软"分量类型: 频率由流动不稳定性决定, 不该被鼓励集中/锐化.
# RE_2D 熵对这类频段不惩罚 (保留展宽); 硬挤压类 (BLADE_PASS/BLADE_HARMONIC) 才罚.
SOFT_F_TYPES = ('HYDRAULIC', 'BLADE_DRIFT')


# ═══════════════════════════════════════════════════════════════
# 可微节点频率 (A2) — 作用在**增强 TFR** 上, 与观测 IF 分工不同
# ═══════════════════════════════════════════════════════════════
#
# 两类节点频率必须分开:
#   IF_obs  — 由 MSST omega_final 聚合得到, 只依赖输入信号, 对网络参数不可微。
#             用于边特征 / 边门控 / bin→节点分配 (路由)。
#   IF_hat  — 由**增强 TFR** 的带内能量质心得到, 经 w_i -> σ_i -> 重排, 对参数可微。
#             用于谐波比值约束 / 绝对频率锚 (约束)。
#
# 为什么不能让约束用 IF_obs: 它的梯度恒为零 —— 模型可以在完全不满足
# "f(2xBPF) = 2·f(BPF)" 的条件下把 L_physics 降到最低, 约束等于没接上。
def band_centroid_if(tfr: torch.Tensor,
                     freqs: torch.Tensor,
                     regions: Optional[List[FreqRegion]] = None,
                     eps: float = 1e-12) -> torch.Tensor:
    """
    逐节点逐帧的带内能量质心频率 [B, N_phys, T] (对 tfr 可微).

      E(f, t) = |tfr(f, t)|²
      f_cent  = Σ_{f∈band} E·f / Σ_{f∈band} E

    与 IF_obs 的区别: 这是"能量所在处", 不是"频带几何中心"——
    即使带宽 16 Hz, 只要能量集中在 2-3 Hz, 质心就落在 2-3 Hz。

    Args:
        tfr:     [B, F, T] 复值或幅值 TFR (可微)
        freqs:   [F] 频率轴 (Hz)
        regions: FreqRegion 列表 (默认 PUMP_TURBINE_REGIONS)

    Returns:
        [B, N_phys, T] Hz; 带内无能量时退化为频带中心
    """
    if regions is None:
        regions = PUMP_TURBINE_REGIONS
    mag = tfr.abs() if torch.is_complex(tfr) else tfr
    B, F, T = mag.shape
    dtype = mag.dtype
    dev = mag.device
    out = torch.zeros(B, len(regions), T, device=dev, dtype=dtype)

    for n, r in enumerate(regions):
        m = (freqs >= r.f_min) & (freqs <= r.f_max)
        Fn = int(m.sum().item())
        if Fn == 0:
            out[:, n, :] = (r.f_min + r.f_max) / 2.0
            continue
        e = mag[:, m, :].pow(2)                                  # [B, Fn, T]
        tot = e.sum(dim=1)                                       # [B, T]
        f_ax = freqs[m].to(dtype=dtype, device=dev).view(1, -1, 1)
        cent = (e * f_ax).sum(dim=1) / tot.clamp(min=eps)        # [B, T]
        # 带内完全无能量 -> 质心无意义, 退化为频带中心
        cent = torch.where(tot > eps, cent,
                           torch.full_like(cent, (r.f_min + r.f_max) / 2.0))
        out[:, n, :] = cent

    return out


# ═══════════════════════════════════════════════════════════════
# NodeFeatures dataclass
# ═══════════════════════════════════════════════════════════════

@dataclass
class NodeFeatures:
    """从 MSST omega_final 提取的物理节点特征。"""
    if_hz: np.ndarray       # [N, T] 每节点 IF (Hz), N = N_PHYSICS_NODES
    energy: np.ndarray      # [N, T] 每节点对数能量
    bandwidth: np.ndarray   # [N, T] 每节点 IF 标准差 (Hz)
    persistence: np.ndarray # [N] 每节点时间持续性 (0-1)
    tfr_stft: np.ndarray    # [F, T] 复数 STFT (归一化后)
    freqs: np.ndarray       # [F] 频率轴 (Hz)
    t_axis: np.ndarray      # [T] 时间轴 (s)
    node_names: List[str]   # 节点名称列表
    omegas: List[np.ndarray] = None  # [N_max] list of [F, T] — MSST IF 轨迹

    @property
    def N(self) -> int:
        return self.if_hz.shape[0]

    @property
    def T(self) -> int:
        return self.if_hz.shape[1]


# ═══════════════════════════════════════════════════════════════
# MSSTNodeExtractor
# ═══════════════════════════════════════════════════════════════

class MSSTNodeExtractor:
    """
    从 MSST omega_final 提取物理节点特征。

    工作流程:
      1. 运行 msst(x, fs, num=3) → omega_final, STFT
      2. 在每个频率区域内聚合:
         IF_raw = median(omega_final 中落入该区域的 bin 的 IF 值)
         energy = sum(|STFT|² 在该区域)
         bandwidth = std(omega_final 中落入该区域的 bin 的 IF 值)
      3. 跨帧统计 persistence

    Args:
        fs:           采样率 (Hz)
        freq_regions: 频率区域定义列表
        msst_hlength: MSST 窗长 (默认 round(N/8))
        msst_num:     MSST 迭代次数 (默认 3)
    """

    def __init__(self, fs: float = 1000.0,
                 freq_regions: Optional[List[FreqRegion]] = None,
                 msst_hlength: Optional[int] = None,
                 msst_num: int = 3):
        self.fs = fs
        self.regions = freq_regions if freq_regions is not None else PUMP_TURBINE_REGIONS
        self.N_nodes = len(self.regions)
        self.msst_hlength = msst_hlength
        self.msst_num = msst_num

    def __call__(self, x: np.ndarray) -> NodeFeatures:
        """
        Args:
            x: [T] 1D 信号

        Returns:
            NodeFeatures dataclass
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        N_sig = len(x)

        # ── Step 1: MSST → omega_final, STFT, omegas ──
        result = msst(x, self.fs, hlength=self.msst_hlength, num=self.msst_num,
                      save_trajectory=True)
        omega_final = result['omega_final']  # [F_bins, T] int32, 1-indexed, 0=invalid
        tfr_stft = result['STFT']            # [F_bins, T] complex128
        freqs = result['freqs']              # [F_bins] float64
        t_axis = result['t']                 # [T] float64
        omegas = result.get('omegas', [omega_final])  # [N_max] list of [F, T]

        F_bins, T = omega_final.shape

        # omega_final → Hz: (bin - 1) * fs / N_sig  (bin 1-indexed, 0=invalid)
        # 预计算 IF 值矩阵 (Hz)
        if_map = np.zeros((F_bins, T), dtype=np.float64)
        valid_mask = omega_final >= 1
        if_map[valid_mask] = (omega_final[valid_mask] - 1) * self.fs / N_sig

        # STFT 能量
        stft_energy = np.abs(tfr_stft) ** 2  # [F, T]

        # ── Step 2: 逐区域聚合 ──
        if_hz = np.zeros((self.N_nodes, T), dtype=np.float64)
        energy = np.zeros((self.N_nodes, T), dtype=np.float64)
        bandwidth = np.zeros((self.N_nodes, T), dtype=np.float64)

        for n_idx, region in enumerate(self.regions):
            # 频率区域掩码
            freq_mask = (freqs >= region.f_min) & (freqs <= region.f_max)

            for t in range(T):
                # 该区域内且 omega_final 有效的 bin
                region_valid = freq_mask & valid_mask[:, t]
                if not region_valid.any():
                    # 无有效 IF → 使用区域中心频率作为 fallback
                    if_hz[n_idx, t] = (region.f_min + region.f_max) / 2.0
                    energy[n_idx, t] = 0.0
                    bandwidth[n_idx, t] = (region.f_max - region.f_min) / 2.0
                    continue

                if_vals = if_map[region_valid, t]
                w_vals = stft_energy[region_valid, t]
                w_tot = float(w_vals.sum())
                if w_tot > 0:
                    # A2/B1: 能量加权均值 (取代无权重中位数)
                    if_hz[n_idx, t] = float((w_vals * if_vals).sum() / w_tot)
                else:
                    if_hz[n_idx, t] = float(np.mean(if_vals))
                energy[n_idx, t] = np.log1p(stft_energy[region_valid, t].sum())
                # 带宽: IF 标准差, 钳制到合理范围
                if len(if_vals) >= 2:
                    bw = np.std(if_vals)
                    bandwidth[n_idx, t] = min(bw, 20.0)  # cap at 20 Hz
                else:
                    bandwidth[n_idx, t] = 1.0  # 默认 1 Hz

        # ── Step 3: Persistence (跨帧能量持续性) ──
        persistence = np.zeros(self.N_nodes, dtype=np.float64)
        energy_threshold = 0.01  # log1p 阈值
        for n_idx in range(self.N_nodes):
            n_valid = (energy[n_idx, :] > energy_threshold).sum()
            persistence[n_idx] = n_valid / max(1, T)

        return NodeFeatures(
            if_hz=if_hz,
            energy=energy,
            bandwidth=bandwidth,
            persistence=persistence,
            tfr_stft=tfr_stft,
            freqs=freqs,
            t_axis=t_axis,
            node_names=[r.name for r in self.regions],
            omegas=omegas,
        )

    def extract_gpu(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """批量 GPU 节点特征提取 — 全部在 GPU 上完成, 无 numpy 转换.

        Args:
            x: [B, T] torch tensor on CUDA

        Returns:
            dict of torch tensors on CUDA:
              node_if:      [B, N_phys, T]  IF (Hz)
              node_energy:  [B, N_phys, T]  log energy
              node_bw:      [B, N_phys, T]  bandwidth (Hz)
              node_persist: [B, N_phys]     persistence
              tfr_stft:     [B, F, T]       complex STFT
              freqs:        [F]             freq axis (Hz)
              omegas:       [B, N_max, F, T]  IF trajectory (int32)
        """
        from models.msst_torch import msst_torch as msst_gpu

        B, T_sig = x.shape
        device = x.device

        node_if_list = []
        node_energy_list = []
        node_bw_list = []
        node_persist_list = []
        tfr_stft_list = []
        omegas_list = []
        freqs_out = None

        for b in range(B):
            # ── GPU MSST (single sample) ──
            x_b = x[b]  # [T]
            result = msst_gpu(x_b, self.fs, hlength=self.msst_hlength,
                              num=self.msst_num, save_trajectory=True,
                              skip_squeeze=True)

            tfr_stft = result['STFT']          # [F, T] complex64
            omega_final = result['omega_final'] # [F, T] int32
            freqs = result['freqs']             # [F]
            omegas_traj = result.get('omegas', [omega_final])

            F, T = omega_final.shape

            # ── IF map (Hz), 0 for invalid bins ──
            valid_mask = omega_final >= 1  # [F, T]
            if_map = torch.zeros(F, T, device=device, dtype=torch.float32)
            if_map[valid_mask] = (omega_final[valid_mask] - 1).float() * self.fs / T_sig

            # ── STFT energy ──
            stft_energy = tfr_stft.abs() ** 2  # [F, T]

            # ── Per-region aggregation (vectorized over T) ──
            if_hz_b = torch.zeros(self.N_nodes, T, device=device, dtype=torch.float32)
            energy_b = torch.zeros(self.N_nodes, T, device=device, dtype=torch.float32)
            bw_b = torch.zeros(self.N_nodes, T, device=device, dtype=torch.float32)

            for n_idx, region in enumerate(self.regions):
                freq_mask = (freqs >= region.f_min) & (freqs <= region.f_max)  # [F]
                # ── Narrow to region bins only (F_region ≪ F, ~30 bins) ──
                region_if = if_map[freq_mask, :]         # [F_r, T]
                region_valid = valid_mask[freq_mask, :]   # [F_r, T]
                region_energy_map = stft_energy[freq_mask, :]  # [F_r, T]
                F_r, T = region_if.shape
                counts = region_valid.sum(dim=0)           # [T]

                # ── Energy-weighted mean IF (A2/B1: 取代无权重中位数) ──
                # 中位数与能量无关: 宽带里大量噪声 bin 会把 IF 拉向频带几何中心
                # (实测 LOW_FREQ 偏高 6-12 Hz), 是 r_obs = 2.13 ≠ 2.00 的直接来源。
                if_sum = (region_if * region_valid.float()).sum(dim=0)   # [T]
                arith_mean = if_sum / counts.clamp(min=1)                # [T]
                w_e = region_energy_map * region_valid.float()           # [F_r, T]
                w_sum = w_e.sum(dim=0)                                   # [T]
                if_hz_n = (region_if * w_e).sum(dim=0) / w_sum.clamp(min=1e-12)
                if_hz_n = torch.where(w_sum > 0, if_hz_n, arith_mean)    # 无能量取算术均值
                if_hz_n[counts == 0] = (region.f_min + region.f_max) / 2.0
                if_hz_b[n_idx] = if_hz_n

                # ── Energy (vectorized) ──
                region_energy = (region_energy_map * region_valid.float()).sum(dim=0)
                energy_b[n_idx] = torch.log1p(region_energy)

                # ── Bandwidth / std (vectorized) ──
                sq_dev = (region_if - arith_mean.unsqueeze(0)) ** 2      # [F_r, T]
                sq_sum = (sq_dev * region_valid.float()).sum(dim=0)     # [T]
                bw_n = torch.sqrt(sq_sum / counts.clamp(min=1)).clamp(max=20.0)
                bw_n[counts < 2] = 1.0
                bw_b[n_idx] = bw_n

            # ── Persistence ──
            persist_b = torch.zeros(self.N_nodes, device=device, dtype=torch.float32)
            for n_idx in range(self.N_nodes):
                persist_b[n_idx] = (energy_b[n_idx] > 0.01).float().mean()

            node_if_list.append(if_hz_b)
            node_energy_list.append(energy_b)
            node_bw_list.append(bw_b)
            node_persist_list.append(persist_b)
            tfr_stft_list.append(tfr_stft)
            omegas_list.append(torch.stack(omegas_traj))
            if freqs_out is None:
                freqs_out = freqs

        return {
            'node_if': torch.stack(node_if_list),
            'node_energy': torch.stack(node_energy_list),
            'node_bw': torch.stack(node_bw_list),
            'node_persist': torch.stack(node_persist_list),
            'tfr_stft': torch.stack(tfr_stft_list),
            'freqs': freqs_out,
            'omegas': torch.stack(omegas_list),
        }

    def get_region_bounds(self) -> Dict[str, Tuple[float, float]]:
        """返回各节点的频率区域边界 (用于诊断)."""
        return {r.name: (r.f_min, r.f_max) for r in self.regions}

    def get_region_center(self) -> Dict[str, float]:
        """返回各节点的频率区域中心 (用于 fallback)."""
        return {r.name: (r.f_min + r.f_max) / 2.0 for r in self.regions}
