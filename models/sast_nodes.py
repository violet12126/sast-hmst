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

  保留节点 (3 物理节点; B4 的 4 节点拆分已于本轮撤回, 理由见下):
    LOW_FREQ:  2-18 Hz   — 流动不稳定性 (涡带/压力脉动); 唯一软带; 无绝对锚
    BPF:      40-53 Hz   — 叶片通过族: 41 Hz 流动峰 + 49.995 Hz 转速锁定线;
                           无绝对锚 (带内有非锁频分量, 见 FreqRegion 注释)
    2xBPF:    90-105 Hz  — 二倍叶片通过频率, 尖峰; 绝对锚 99.990 Hz ±1.5

  为什么撤回 B4 拆分:
    B4 把 42-55 Hz 拆成 BPF_DRIFT / BPF_LINE, 理由是"一个标量 σ_i 不能既挤
    窄线又留宽峰"。实测推翻了这个前提 —— 两段各自单帧本征展宽只有 0.686 /
    0.584 Hz, 都是窄带脊线, 拆分并不解决 σ 的分配问题。真正的后果是:
      (a) 41 Hz 主峰掉出 BPF_DRIFT 左界 [42, 48.5), 该带内 IF 描述的是
          46-47 Hz 的次峰, 与 FFT 主峰不符;
      (b) 两段 w_i / σ_i 不同 (0.85 vs 0.98), 在 48.5 Hz 接缝两侧形成
          不同的背景抽取比例 → 垂直梳状毛刺;
      (c) 41-50 Hz 整段被一个偏宽的核摊平, 峰值低于 100 Hz, 造成色阶反转。
    合并成 [40, 53) 后整段只有一个 w_i/σ_i, 41 Hz 与 50 Hz 落在同一带内,
    各自按自身 IF 重排 —— 既没有接缝, 也不再有"用 46 Hz 的 IF 代表 41 Hz"。

  频带划分 (统一由 region_weights 实现, 全仓库单一真源):
    区间语义为左闭右开 [f_min, f_max)。3 节点后三带**互不相邻**
    ([2,18) [40,53) [90,105)), 不再有共享内部边, 故 SOFT_SPLIT_HZ 的
    单位分解过渡带在此配置下自动失效 (canon_split 为空), 全部退化为硬边界。

  边: BPF → 2xBPF (r_nom=2.0, INTEGER_HARMONIC) — 唯一的确定性倍频关系

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

import math
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
# ⚠ 本轮 3 节点合并后 **没有节点使用 F_BPF_ANCHOR**: BPF 带 [40,53) 同时装着
#   非锁频的 41 Hz 流动峰与锁频的 50 Hz 线, 能量加权 node_if 必然落在两者之间,
#   绝对锚会恒被违反。保留该常量是因为它是物理基准 (转速公式), 供脚本/测试引用。
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
    # ── 缺席判定门限 (presence 的两个拐点) ──
    # 缺席度 a = clamp((p_hi - e)/(p_hi - p_lo), 0, 1):
    #   e >= p_hi → a = 0 (分量存在, 上界保持 w_range_high)
    #   e <= p_lo → a = 1 (分量缺席, 上界收拢到 w_high_absent=0.20)
    # p_hi <= 0 表示**该节点不启用缺席门控** (a≡0, 行为与修改前完全一致)。
    # ⚠ 门限必须逐节点给, 不能用全局阈值: presence 的尺度是**压缩且节点相关**的,
    #   实测 (3 节点 e15 ckpt, 5 类) presence 均值
    #     LOW_FREQ 0.195/0.148/0.176/0.161/0.147
    #     BPF      0.211/0.156/0.149/0.129/0.126   ← 空转**最高**, 与负荷反相
    #     2xBPF    0.084/0.275/0.332/0.345/0.311   ← 空转最低, 与负荷同相
    #   BPF 的 presence 与"是否加载"反相 (空转带内只有一条平坦的 48 Hz 漂移峰,
    #   集中度反而高; 加载后 41 Hz 流动峰把能量摊开), 若对它门控会把**加载**工况
    #   的上界压低、打掉硬挤压。故 BPF / LOW_FREQ 一律 p_hi=0 (不启用)。
    p_absent_lo: float = 0.0            # 缺席端 (e <= lo → a = 1)
    p_absent_hi: float = 0.0            # 存在端 (e >= hi → a = 0); 0 = 不启用


PUMP_TURBINE_REGIONS: List[FreqRegion] = [
    # LOW_FREQ: 水力来源 (涡带/压力脉动), 2-18 Hz
    #   实测评测 (probe_node_if.py / check_spectral_bands.py):
    #     - 2.0 Hz 附近在所有工况都是带内能量峰 (E峰 f = 2.0-3.0 Hz)
    #     - 15.14 Hz 在低负荷是最强线 (凸出 11.2 dB), 属同一"水力"族的漂移分量
    #     - >17 Hz 各工况均为平坦宽带 (PMR 0.8-1.5 dB), 无独立峰
    #   → 上界由 25 Hz 收到 18 Hz: 去掉纯宽带尾部, 避免 IF 中位数被抬到带中
    #   预期宽频, 中低持续性 (涡带仅部分工况显著)
    #   w_range: [0.05, 0.35] — 保守, 宽核, 不锐化
    #   无绝对锚: 水力分量频率由流动不稳定性决定, 不锁定转速
    #   bw_expected 旧值 12.0 是错的: ridge_decay = exp(-(|f-if|/(2·bw))²/8),
    #   bw=12 使 ±24 Hz 内的 decay 恒 ≈ 0.90, 挤压参与因子在该节点内失效。
    FreqRegion('LOW_FREQ',  2.0, 18.0,  'HYDRAULIC',       0.20,  4.5, 0.50, 0.05, 0.35),
    # BPF: 叶片通过族, 40-53 Hz —— 41 Hz 流动峰 + 49.995 Hz 转速锁定线
    #   下界 40 Hz 而非 42 Hz: FFT 实测 40-50 Hz 段的主峰在 41.0 Hz (class 2),
    #   旧边界 42 Hz 把它整个排除在带外 —— 于是 BPF_DRIFT 的 node_if 描述的是
    #   46-48 Hz 的次峰, 与 FFT 主峰不符, 且主峰所在的 40-42 Hz 得不到挤压。
    #   上界 53 Hz: 覆盖 49.995 Hz 线 (与 50 Hz 电网频率差 0.005 Hz, 远低于
    #   0.5 Hz 分辨率, 数值上不可区分)。
    #   ⚠ 无绝对锚: 该带现在同时装着非锁频的 41 Hz 流动分量与锁频的 50 Hz 线,
    #     能量加权的 node_if 落在两者之间 (≈43-46 Hz), 单个绝对锚不再有意义。
    #     倍频关系仍由 BPF→2xBPF 的 r_nom=2.0 边表达 (诊断口径, 见 sast_losses)。
    #   w_range [0.80, 0.98]: 硬带, 与 M=2 软门控配合 ⇒ r=2w≥1.6 ⇒ g_1=1.0,
    #     第 1 轮 100% 清掉原始 STFT 底座, 把脊线压成与 Raw MSST 同等锐度。
    #   bw_expected 3.0: 带内两条脊线 (41 / 50 Hz) 相距 9 Hz, 该值只用于
    #     ridge_factor 的参与因子 (train_ridge 默认关), 取可比的两脊折中。
    FreqRegion('BPF',       40.0, 53.0,  'BLADE_PASS',     0.90,  3.0, 0.90, 0.80, 0.98),
    # 2xBPF: 二倍叶片通过频率, 90-105 Hz
    #   预期窄带, 极高持续性
    #   实测: 带内能量峰 = 100.00 Hz (全部带负荷工况), 逐样本 f=100.10 Hz 标准差 0.000,
    #        空转时该分量消失 (过渡带口径 presence_e 0.074 vs 带负荷 0.238-0.256;
    #        旧硬矩形口径为 0.097 vs 0.319-0.343) → 确证为 2×BPF
    #   锚容差 ±1.5 Hz (紧): 该线位置由转速唯一确定, 不应漂移
    #   w_range: [0.80, 0.98] — 最信任, 极锐
    #   ⚠ bw_expected 4.5 直接取自 probe 的带内 IF 标准差 (五类 5.26/4.81/4.17/
    #     4.04/4.33 的均值 4.52): 该值被空转样本抬高 (空转此带无分量, 带内 IF 是
    #     纯噪声游走)。若只按"分量本身有多宽"标定, 应取更小的 ~2.0。取实测均值的
    #     理由是标定方法可复现 (论文口径统一); 代价是 53-90 Hz 空带内的参与因子
    #     由 ridge_floor(0.15) 抬到 ~0.39, 属可接受的松弛。
    #   缺席门限 p_lo=0.13 / p_hi=0.17: 按**帧级**分布标定 (probe_absence_frames.py,
    #     50 样本 × 2000 帧)。空转 2xBPF 逐帧 presence: 均值 0.092, p90 0.144;
    #     加载 2xBPF: 均值 0.285-0.342, p10 0.199-0.232。候选门限网格的实测代价:
    #       (lo,hi)      空转 a 均值 / 加载误判帧占比 (类1-4)
    #       (0.10,0.20)   0.88 / 9% 6% 5% 7%
    #       (0.13,0.17)   0.91 / 5% 2% 0% 2%     <- 取此
    #       (0.14,0.19)   0.95 / 8% 5% 4% 6%
    #     取 (0.13,0.17): 空转 91% 的帧判为缺席, 同时加载工况只有 0-5% 的帧
    #     误判 (且那些帧的 a 很小, 上界仍 ≥0.9), 不会打掉加载工况的硬挤压。
    FreqRegion('2xBPF',    90.0, 105.0, 'BLADE_HARMONIC',  0.90,  4.5, 0.95, 0.80, 0.98,
               f_anchor=F_2XBPF_ANCHOR, anchor_tol=1.5,
               p_absent_lo=0.13, p_absent_hi=0.17),
]

# 节点数量
N_PHYSICS_NODES = len(PUMP_TURBINE_REGIONS)

# "软"分量类型: **单帧内**真正弥散的宽带随机分量, 不该被鼓励集中/锐化.
# RE_2D 熵对这类频段不惩罚 (保留展宽); 其余类型才罚.
# 只保留 HYDRAULIC: LOW_FREQ 的涡带/压力脉动单帧本征展宽 2.789 Hz, 是锁定带
# (BPF 0.584 / 2xBPF 0.650) 的 4-6 倍 —— 硬挤压会把它撕裂成断续的虚假细丝。
SOFT_F_TYPES = ('HYDRAULIC',)


# ── 节点 IF 估计的鲁棒化参数 (见 MSSTNodeExtractor 的聚合逻辑) ──
# WHY: 区域 bin 的**频率位置**在带内, 不代表它的**IF 估计值**也在带内。
#   MSST 的 IF 落在哪由相位导数决定, 弱分量/噪声 bin 的 IF 可游走到很远的频率;
#   不筛的话 node_if 会在单样本内跑出 32-60 Hz (带本身只有十几 Hz 宽), 并经
#   corr(w, IF) 直接驱动 w_i 摆动 → 时频图抖。故按 IF **落点**再过滤一次。
# 落点容差取 0: IF 必须落在**本带**内才算本带分量。3 节点后三带为
# [2,18) [40,53) [90,105), 带与带之间是空白 (不在任何带内的 bin 不参与挤压),
# 故"IF 落在哪个带"无歧义, 不存在双计。
IF_LANDING_TOL_HZ = 0.0
IF_MIN_LANDED_BINS = 2      # 落点 bin 少于此数 → 视为该帧无分量, 走回退
IF_SNR_FRAC = 0.05          # 带内加权能量 < 本样本中位数×该比例 → 低信噪帧, 走回退


# ═══════════════════════════════════════════════════════════════
# 区域频带权重 (单一真源) — 半开区间 + 共享内部边的单位分解过渡带
# ═══════════════════════════════════════════════════════════════
#
# 全仓库曾有 7 份各自独立的 `(freqs >= f_min) & (freqs <= f_max)` 副本
# (sast_nodes 的提取器/质心, sast_losses 的 Rényi/存在度, ONNX 导出侧),
# 任何边界改动都要手工同步 7 处。现收敛到本函数。
SOFT_SPLIT_HZ = 0.5   # 共享内部边的过渡带半宽 (Hz); 0 = 退化为纯硬边界


def _clamp01(x, is_torch: bool):
    return x.clamp(0.0, 1.0) if is_torch else np.clip(x, 0.0, 1.0)


def _raised_cosine(x):
    """升余弦过渡核; 入参需已 clamp 到 [0,1]. 0→0, 1→1, C1 连续。"""
    if isinstance(x, torch.Tensor):
        return 0.5 * (1.0 - torch.cos(x * math.pi))
    return 0.5 * (1.0 - np.cos(x * math.pi))


def _w_ge(freqs, v):
    """左闭阶跃 (numpy / torch 通用)。"""
    if isinstance(freqs, torch.Tensor):
        return (freqs >= v).to(freqs.dtype)
    return (freqs >= v).astype(freqs.dtype)


def _w_lt(freqs, v):
    """右开阶跃 (numpy / torch 通用)。"""
    if isinstance(freqs, torch.Tensor):
        return (freqs < v).to(freqs.dtype)
    return (freqs < v).astype(freqs.dtype)


def region_weights(freqs, regions: Optional[List[FreqRegion]] = None,
                   soft_split_hz: float = SOFT_SPLIT_HZ):
    """
    区域频带权重 w_n(f) ∈ [0, 1], 形状 [N_phys, F]。支持 numpy 与 torch,
    返回类型与 freqs 一致。

    规则:
      1. 区间语义统一为左闭右开 [f_min, f_max)。
      2. 若 regions[i].f_max == regions[i+1].f_min (共享内部边 b), 则两侧在
         [b-h, b+h] (h = soft_split_hz) 上用互补升余弦过渡:
             w_upper(f) = raised_cosine((f - (b-h)) / 2h)
             w_lower(f) = 1 - w_upper(f)
         故 w_lower + w_upper ≡ 1 于整个过渡带 —— **单位分解**, 带能量既不
         重复也不丢失, Σ_n E_n = E_total 成立 (compute_V_obs 的比例运算依赖
         此性质; 若两侧各自独立软窗则需额外重归一化)。
      3. 间隙侧边界 (18/40、53/90) 无相邻区域, 保持硬 0/1。
      4. **本机制在当前 3 节点定义下是空转的**: [2,18) [40,53) [90,105) 两两
         不相邻, canon_split 为空, 所有边界都退化为硬 0/1。保留实现是为兼容
         调用方的参数, 并避免将来若重新引入相邻带时又要重写一遍。

    Args:
        freqs: [F] 频率轴 (Hz)
        regions: FreqRegion 列表 (默认 PUMP_TURBINE_REGIONS)
        soft_split_hz: 共享内部边的过渡带半宽; 0 → 全部硬边界

    Returns:
        [N_phys, F] 权重
    """
    if regions is None:
        regions = PUMP_TURBINE_REGIONS
    is_t = isinstance(freqs, torch.Tensor)
    f = freqs if is_t else np.asarray(freqs)

    # 共享内部边的位置取自**规范表** PUMP_TURBINE_REGIONS, 而非本次传入的
    # regions —— 否则只传单个区域的调用方 (如 Rényi 熵逐区域遍历) 会因
    # range(N-1) 为空而丢掉过渡带, 行为随调用方式漂移。
    canon_split = set()
    if soft_split_hz > 0:
        for i in range(len(PUMP_TURBINE_REGIONS) - 1):
            a, b = PUMP_TURBINE_REGIONS[i], PUMP_TURBINE_REGIONS[i + 1]
            if abs(a.f_max - b.f_min) < 1e-9:
                canon_split.add(round(b.f_min, 6))

    soft_hi = [soft_split_hz if round(r.f_max, 6) in canon_split else 0.0
               for r in regions]
    soft_lo = [soft_split_hz if round(r.f_min, 6) in canon_split else 0.0
               for r in regions]

    ws = []
    for n, r in enumerate(regions):
        # ── 下界 ──
        if soft_lo[n] > 0:
            h = soft_lo[n]
            t = (f - (r.f_min - h)) / (2.0 * h)          # 在 [b-h, b+h] 上 0→1
            u_lo = _raised_cosine(_clamp01(t, is_t))
        else:
            u_lo = _w_ge(f, r.f_min)
        # ── 上界 ──
        if soft_hi[n] > 0:
            h = soft_hi[n]
            t = (f - (r.f_max - h)) / (2.0 * h)          # 在 [b-h, b+h] 上 0→1
            u_hi = 1.0 - _raised_cosine(_clamp01(t, is_t))
        else:
            u_hi = _w_lt(f, r.f_max)

        ws.append(u_lo * u_hi)

    if is_t:
        return torch.stack(ws, dim=0)
    return np.stack(ws, axis=0)


def region_masks(freqs, regions: Optional[List[FreqRegion]] = None,
                 soft_split_hz: float = SOFT_SPLIT_HZ):
    """布尔掩码 [N_phys, F] = region_weights(...) > 0。

    仅供**必须**用布尔索引的调用方 (如 Rényi 熵的频轴切片)。需要"权重"
    语义 (求和/质心/存在度) 时应直接用 region_weights, 否则会丢掉过渡带。
    """
    return region_weights(freqs, regions, soft_split_hz) > 0


def physics_band_mask(freqs, regions: Optional[List[FreqRegion]] = None):
    """[F] bool — 4 个物理频带的**并集**支撑 (带外为 False)。

    用途: 把 FreqEncoder 的输入限制在物理带内。带外 (F=1000 时占 92%) 仍携带
    可观的工况信息 —— 实测带外 916 个 bin 的 1-NN 工况分类精度 **0.800**
    (chance 0.200), 而带内 84 个 bin 是 0.950。编码器据此可以把 SupCon 刷到 0
    而不依赖 GAT 的挤压, 即"旁路": 实测训练后 sc 塌到 0.000、Val acc 饱和在 1.000。
    见 docs §16.4.3。
    """
    if regions is None:
        regions = PUMP_TURBINE_REGIONS
    return region_weights(freqs, regions).sum(dim=0) > 0.0


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

      E(f, t) = w_n(f) · |tfr(f, t)|²          (w_n 见 region_weights)
      f_cent  = Σ_f E·f / Σ_f E

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

    # 区域权重 [N_phys, F]: 半开区间 + 共享内部边的单位分解过渡带。
    # 旧实现用两端皆闭的硬矩形, 导致 (a) 48.0 Hz 被 BPF_DRIFT/BPF_LINE 双计,
    # (b) 漂移峰右翼整体计入 BPF_LINE —— 空转因此出现 E峰f = 48.000 Hz 的
    # 劈裂伪峰。加权后 48.0 Hz 完全归 BPF_DRIFT, 过渡带 [48.0, 49.0] 内按
    # 升余弦互补分配。
    w_all = region_weights(freqs, regions).to(dtype=dtype, device=dev)  # [N, F]
    e_full = mag.pow(2)                                          # [B, F, T]
    f_ax_full = freqs.to(dtype=dtype, device=dev).view(1, F, 1)   # [1, F, 1]

    for n, r in enumerate(regions):
        w = w_all[n].view(1, F, 1)                               # [1, F, 1]
        e = e_full * w                                           # [B, F, T]
        tot = e.sum(dim=1)                                       # [B, T]
        cent = (e * f_ax_full).sum(dim=1) / tot.clamp(min=eps)   # [B, T]
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
        msst_hlength: MSST 窗长 (默认 1024, 与 SastConfig.msst_hlength 一致)
        msst_num:     MSST 迭代次数 (默认 1, 与 SastConfig.msst_num 一致)

    ⚠ 默认值必须与 SastConfig 一致: 旧默认是 num=3 / hlength=None(=N/8=250),
      而模型走的是 config 的 1 / 1024 —— 任何"只实例化 extractor"的脚本
      (如 build_static_prototypes) 都会静默用另一套口径算节点特征。
    """

    def __init__(self, fs: float = 1000.0,
                 freq_regions: Optional[List[FreqRegion]] = None,
                 msst_hlength: Optional[int] = 1024,
                 msst_num: int = 1):
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

        # 区域权重: 统一真源 (半开区间 + 共享内部边单位分解过渡带)。
        # 旧实现用两端皆闭的硬矩形 → 48.0 Hz 双计 + 漂移峰右翼计入 BPF_LINE。
        w_all = region_weights(freqs, self.regions)                  # [N, F] float64

        for n_idx, region in enumerate(self.regions):
            w_f = w_all[n_idx]                                       # [F]
            # 落点容差 + 带中心 (回退基准)
            land_lo = region.f_min - IF_LANDING_TOL_HZ
            land_hi = region.f_max + IF_LANDING_TOL_HZ
            center = 0.5 * (region.f_min + region.f_max)
            est = np.full(T, center, dtype=np.float64)   # 逐帧能量加权 IF
            bw_tmp = np.ones(T, dtype=np.float64)        # 逐帧落点 IF 标准差
            kept = np.zeros(T, dtype=np.int64)           # 逐帧落点 bin 数
            band_e = np.zeros(T, dtype=np.float64)       # 逐帧带内能量 (原始, 供 SNR 门控)

            for t in range(T):
                # 该区域内 (权重 > 0) 且 omega_final 有效的 bin
                region_valid = (w_f > 0.0) & valid_mask[:, t]
                if not region_valid.any():
                    continue
                idx_t = np.flatnonzero(region_valid)
                if_vals_all = if_map[idx_t, t]
                # 加权能量: 过渡带内的 bin 按 w<1 打折 (与 band_centroid_if 一致)
                w_vals_all = stft_energy[idx_t, t] * w_f[idx_t]
                band_e[t] = float(w_vals_all.sum())
                # ── IF 落点过滤: bin 的频率位置在带内 ≠ 它的 IF 估计值也在带内 ──
                # 实测 BPF_LINE 的 bin 会被 MSST 迭代拖到 40-46 Hz, 若不筛,
                # node_if 可在单样本内跑出 32-60 Hz (带本身仅 4.5 Hz 宽)。
                land = (if_vals_all >= land_lo) & (if_vals_all <= land_hi)
                kept[t] = int(land.sum())
                if kept[t] == 0:
                    continue
                if_vals = if_vals_all[land]
                w_vals = w_vals_all[land]
                w_tot = float(w_vals.sum())
                if w_tot > 0:
                    # A2/B1: 能量加权均值 (取代无权重中位数)
                    est[t] = float((w_vals * if_vals).sum() / w_tot)
                    if kept[t] >= 2:
                        var = float((w_vals * (if_vals - est[t]) ** 2).sum() / w_tot)
                        bw_tmp[t] = min(math.sqrt(max(var, 0.0)), 20.0)
                else:
                    est[t] = float(if_vals.mean())
                    if kept[t] >= 2:
                        bw_tmp[t] = min(float(np.std(if_vals)), 20.0)

            # ── 有效帧判据 + 低信噪回退 ──
            good = kept >= IF_MIN_LANDED_BINS
            if good.any():
                e_ref = float(np.median(band_e[good]))
                good &= band_e >= IF_SNR_FRAC * e_ref
            ref = float(np.median(est[good])) if good.any() else center
            if_hz[n_idx] = np.where(good, est, ref)
            bandwidth[n_idx] = np.where(good, bw_tmp, 1.0)
            # 能量特征口径不变 (带内全部支撑 bin, 不按落点裁剪)
            energy[n_idx] = np.log1p(band_e)

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

            # 区域权重 [N_phys, F] — 统一真源 (半开区间 + 共享内部边单位分解过渡带)
            w_all = region_weights(freqs, self.regions)              # [N, F]

            for n_idx, region in enumerate(self.regions):
                w_f = w_all[n_idx]                                       # [F]
                freq_mask = w_f > 0.0                                    # [F] 区域支撑
                # ── Narrow to region bins only (F_region ≪ F, ~30 bins) ──
                region_if = if_map[freq_mask, :]         # [F_r, T]
                region_valid = valid_mask[freq_mask, :]   # [F_r, T]
                # 加权能量: 过渡带内的 bin 按 w<1 打折 (取代两端皆闭的硬矩形)
                region_energy_map = (stft_energy[freq_mask, :] *
                                     w_f[freq_mask].unsqueeze(1))  # [F_r, T]
                F_r, T = region_if.shape

                # ── IF 落点过滤 (本轮新增) ──
                # 区域 bin 的**频率位置**在带内, 不代表其 **IF 估计值**也在带内:
                # MSST 固定点迭代会把 BPF_LINE 的 bin 拖到 40-46 Hz, 未经筛除时
                # node_if 在单样本内可跑出 32.6-60.0 Hz (带本身仅 4.5 Hz 宽),
                # 并经 corr(w, IF)=+0.72 直接驱动 w_i 摆动 → 时频图抖。
                land_lo = region.f_min - IF_LANDING_TOL_HZ
                land_hi = region.f_max + IF_LANDING_TOL_HZ
                center = torch.tensor(0.5 * (region.f_min + region.f_max),
                                      device=device, dtype=torch.float32)
                landed = region_valid & (region_if >= land_lo) & (region_if <= land_hi)
                kept = landed.sum(dim=0)                                 # [T]
                w_e = region_energy_map * landed.float()                 # [F_r, T]
                w_sum = w_e.sum(dim=0)                                   # [T]

                # ── Energy-weighted mean IF (A2/B1: 取代无权重中位数) ──
                # 中位数与能量无关: 宽带里大量噪声 bin 会把 IF 拉向频带几何中心
                # (实测 LOW_FREQ 偏高 6-12 Hz), 是 r_obs = 2.13 ≠ 2.00 的直接来源。
                est = (region_if * w_e).sum(dim=0) / w_sum.clamp(min=1e-12)

                # ── Energy (vectorized) — 口径不变: 带内全部支撑 bin ──
                region_energy = (region_energy_map * region_valid.float()).sum(dim=0)
                energy_b[n_idx] = torch.log1p(region_energy)

                # ── 有效帧判据 + 低信噪回退 ──
                good = (kept >= IF_MIN_LANDED_BINS) & (w_sum > 0)
                if bool(good.any()):
                    e_ref = region_energy[good].median()
                    good = good & (region_energy >= IF_SNR_FRAC * e_ref)
                if bool(good.any()):
                    # 回退值 = 有效帧的中位 IF (落点已保证 est ∈ [lo, hi])
                    ref = est[good].median()
                else:
                    ref = center
                if_hz_b[n_idx] = torch.where(good, est, ref)

                # ── Bandwidth: 落点 bin 的能量加权标准差 ──
                sq_dev = (region_if - est.unsqueeze(0)) ** 2             # [F_r, T]
                var = (sq_dev * w_e).sum(dim=0) / w_sum.clamp(min=1e-12)
                bw_n = torch.sqrt(var.clamp(min=0.0)).clamp(max=20.0)
                bw_b[n_idx] = torch.where(good, bw_n,
                                          torch.ones_like(bw_n))

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
