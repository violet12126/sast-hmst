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

  保留节点:
    LOW_FREQ: 2-25 Hz   — 水力分量, 独立于机械路径
    BPF:      42-55 Hz  — 叶片通过频率, 宽峰 (工况调制)
    2xBPF:    90-105 Hz — 二倍叶片通过频率, 尖峰, 极度稳定

  边: BPF → 2xBPF (r_nom=2.0, INTEGER_HARMONIC) — 唯一的确定性倍频关系

用法:
  extractor = MSSTNodeExtractor(fs=1000, freq_regions=PUMP_TURBINE_REGIONS)
  nodes = extractor(signal)  # signal: [T] numpy array
  # nodes.if_hz: [3, T] 每节点每帧 IF (Hz)
  # nodes.energy: [3, T] 每节点每帧能量
  # nodes.bandwidth: [3, T] 每节点每帧 IF 带宽 (Hz)
  # nodes.persistence: [3] 每节点持久性
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

@dataclass
class FreqRegion:
    """单个物理节点的频率搜索区域。"""
    name: str           # 节点名称 (如 'BPF', '2xBPF')
    f_min: float        # 区域下界 (Hz)
    f_max: float        # 区域上界 (Hz)
    f_type: str         # 分量类型 (HYDRAULIC, BLADE_PASS, BLADE_HARMONIC)
    C_prior: float      # 先验可压缩性倾向 (0-1)
    bw_expected: float  # 预期带宽 (Hz) — 用于原型匹配和异常检测
    persist_expected: float  # 预期持续性 (0-1)


PUMP_TURBINE_REGIONS: List[FreqRegion] = [
    # LOW_FREQ: 水力来源 (涡带/压力脉动), 2-25 Hz
    #   预期宽频 (12 Hz), 中低持续性 (涡带仅部分工况显著)
    FreqRegion('LOW_FREQ',  2.0, 25.0,  'HYDRAULIC',       0.30, 12.0, 0.50),
    # BPF: 叶片通过频率, 42-55 Hz
    #   预期宽频 (~12 Hz), 高持续性 (有水就有)
    #   工况调制致宽: 水头/负荷/导叶开度变化 → IF 在 42-55 Hz 内漂移
    FreqRegion('BPF',      42.0, 55.0,  'BLADE_PASS',      0.60, 12.0, 0.85),
    # 2xBPF: 二倍叶片通过频率, 90-105 Hz
    #   预期窄带 (2 Hz), 极高持续性
    FreqRegion('2xBPF',    90.0, 105.0, 'BLADE_HARMONIC',  0.90,  2.0, 0.95),
]

# 节点数量
N_PHYSICS_NODES = len(PUMP_TURBINE_REGIONS)


# ═══════════════════════════════════════════════════════════════
# NodeFeatures dataclass
# ═══════════════════════════════════════════════════════════════

@dataclass
class NodeFeatures:
    """从 MSST omega_final 提取的物理节点特征。"""
    if_hz: np.ndarray       # [N, T] 每节点 IF (Hz), N=3
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
                if_hz[n_idx, t] = np.median(if_vals)
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

                # ── Median IF (sort over small F_r, vectorized over T) ──
                region_if_sort = region_if.clone()
                region_if_sort[~region_valid] = float('inf')
                region_if_sorted, _ = torch.sort(region_if_sort, dim=0)  # [F_r, T]
                med_idx = ((counts - 1) // 2).clamp(min=0).long()        # [T]
                t_arange = torch.arange(T, device=device)
                if_hz_n = region_if_sorted[med_idx, t_arange]             # [T]
                if_hz_n[counts == 0] = (region.f_min + region.f_max) / 2.0
                if_hz_b[n_idx] = if_hz_n

                # ── Energy (vectorized) ──
                region_energy = (region_energy_map * region_valid.float()).sum(dim=0)
                energy_b[n_idx] = torch.log1p(region_energy)

                # ── Bandwidth / std (vectorized) ──
                if_sum = (region_if * region_valid.float()).sum(dim=0)  # [T]
                mu = if_sum / counts.clamp(min=1)                       # [T]
                sq_dev = (region_if - mu.unsqueeze(0)) ** 2             # [F_r, T]
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
