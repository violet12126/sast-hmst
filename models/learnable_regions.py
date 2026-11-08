"""可学习的参数化频带软掩码 (方案一)
=====================================

把 `models.sast_nodes.PUMP_TURBINE_REGIONS` 的硬矩形边界换成**可微平顶窗**
(flat-top soft window):

    B_n(f) = sigmoid( (f - (mu_n - k*sigma_n)) / tau )
           * sigmoid( ((mu_n + k*sigma_n) - f) / tau )

    w_n(f) = B_n(f)  ,  f in fence_n 且 |f - mu_n| <= k*sigma_n + 2*tau

  mu_n       可学中心 (nn.Parameter)
  sigma_n    可学带宽 = sigma_floor + softplus(raw)  (nn.Parameter)
  k          支撑截断倍数 (不可学), 逐节点可配: BandPreset.k
  tau        通带边缘的过渡宽度 (Hz, 不可学), 默认 0.3
  fence_n    不可学的硬外框, 防止节点跑到别的物理带 (§见下)

逐节点 k 与 sigma 上限 (2026-11-06)
------------------------------------
k 现在可以按节点给 (`BandPreset.k`), 并且 sigma 有软上限

    sigma_max,n = support_frac_n * (fence_hi - fence_lo) / (2 * k_n)

即通带半宽 `k*sigma` 最多占 fence 半宽的 `support_frac` 倍。默认
support_frac=0.85 —— 通带停在 fence 内部, mu/sigma 仍然决定掩码形状。

为什么主轴 preset 的"谐波簇"节点要把 support_frac 放到 2.0 (允许越墙):
该节点 fence 宽 17.5 Hz, 内含 2.5 Hz 涡带单音 (占涡带类 90% 能量) 与
8.5 / 16.5 Hz 两条谐波。`band_envelope_loss` 的 mu 目标是 |S|^0.5 加权质心,
被 2.5 Hz 单音拽到 4.6-5.4 Hz; 而 sigma 上限只有 0.85*8.75=7.44 Hz,
通带随之缩到 [-2.6, 12.2], **16.5 Hz (2xf_n, 实测 +12~20 dB) 掉出通带,
w_bin 归零、完全不被挤压**。把该节点的半宽上限放到 2.0 (k*sigma 可达
17.5 Hz) 后, 无论 mu 漂到哪里, 通带都盖满整条 fence, 簇内每条线拿到完全
相同的挤压权重 —— 这正是"整带同权"想要的。代价是 mu/sigma 对该节点不再
改变形状, 只剩诊断意义。

为什么不再用高斯钟形
--------------------
`exp(-(f-mu)^2 / 2 sigma^2)` 只在 mu 一点取 1。对主轴数据, 三个谐波簇
(8.5 / 11.5 / 15 / 17.5 Hz) 共用一个节点时, 高斯在 11.5 / 15 / 17.5 Hz 处已经
衰减到 0.75 / 0.23 / 0.05 —— 这些谱线拿到的挤压权重随之暴跌, 重排核宽
sigma_sqz 反而膨胀, 于是保留了 3.65 Hz 宽的 STFT 裙边, 谱线粘连。平顶窗在
整个通带内恒为 1, 让簇内每条线拿到**同一个** w_bin。

代价 (必须知道): 当 k*sigma 覆盖到 fence 之外时 (主轴 preset 的三个节点都是
这样), 通带退化为"整条 fence + 软边", mu/sigma 不再改变掩码形状, 只保留在
诊断与 `as_regions()` 快照里。这是有意的 —— 谐波簇本来就该整带同权。

设计上刻意保留的三个不变量
--------------------------
1. **支撑仍是有限集**。sigmoid 的尾巴是指数衰减但永不为零, 直接替换会让
   `region_weights(...) > 0` (physics_band_mask / IF 落点过滤) 退化成全频带。
   这里用 `|f-mu| <= k*sigma + 2*tau` 与 fence 双重截断, 支撑始终是有限个 bin。
2. **峰值归一**。weights 除以带内最大值, 峰值恒为 1, 与旧硬掩码 (0/1) 同一
   量纲; 不求和归一, 因为三个带互不相邻, 归一化只会把带外噪声放大。
3. **fence 不可学**。`mu/sigma` 只能在外框内漂移; 跨数据集时外框由 preset
   重新给 (见 `BAND_PRESETS`), 而不是让梯度自由探索整个频谱 —— 否则三个
   节点可能互相吞并, 结果不可解释。

梯度来源
--------
本模块自身不产生梯度路径; 训练侧由 `sast_losses.band_envelope_loss` 提供:
用当前 batch 原始谱在 fence 内算经验均值/标准差 (target 端 detach, 且用
**亚线性**权重 |S|^0.5 而非 |S|^2, 防止单条强线垄断方差), 惩罚
`(mu - mu*)^2` 与 `(log sigma - log sigma*)^2`。

**为什么不只用 Rényi 熵**: 熵对带内"更集中"是单调奖励的, 把 sigma 接到熵上
会把它一路压到下限, 与"保留宽带水力分量展宽"的目标相反。经验矩目标不会塌,
因为它由数据谱宽决定。

用法
----
    mask = LearnableBandMask.from_preset('shaft', k_support=3.0)
    freqs = torch.fft.rfftfreq(2000, d=1/1000)
    w = mask.weights(freqs)        # [N, F], 峰值 1
    s = mask.support(freqs)        # [N, F] bool
    regions = mask.as_regions()    # 快照成 FreqRegion, 供 numpy/绘图/导出
"""
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sast_nodes import FreqRegion, PUMP_TURBINE_REGIONS


@dataclass(frozen=True)
class BandPreset:
    """单个节点的频带预设。

    fence_lo/hi 是**不可学**的硬外框; mu_init/sigma_init 是可学参数的初值。
    """
    name: str
    fence_lo: float
    fence_hi: float
    mu_init: float
    sigma_init: float
    # 通带半宽倍数; None → 用构造参数 k_support 的全局默认
    k: Optional[float] = None
    # 通带半宽上限 = support_frac * fence 半宽 (再除以 k 得到 sigma 上限)。
    # 0.85 = 停在 fence 内部; >=1.0 = 允许越墙 (整条 fence 同权)。
    support_frac: float = 0.85


# ═══════════════════════════════════════════════════════════════
# 数据集预设
# ═══════════════════════════════════════════════════════════════
# rsi: 沿用现有 3 节点定义, 初值取物理主频 (BPF 50 / 2xBPF 100)
#
# shaft: 实测谱线 (见 scripts/analysis/probe_shaft_lines.py) 为
#   f_n=8.33 Hz 的整数倍: 8.3 / 16.5(2x) / 25.0(3x) / 33.3(4x) / 41.5(5x) /
#   50.0(6x) / 58.3(7x) / 116.5(14x), 加上涡带单音 2.5 Hz。
#   注意 11.75 / 19.45 Hz **不是谱线** (局部 SNR≈0, 逐样本检出率 3-35%),
#   14.0 Hz 只在两个涡带类里弱显著 —— 不要再按"谐波簇 = 11.5/15/17.5"建模。
#
#   3 个节点槽位 (1 软 2 硬) 重新映射:
#     节点 0 LOW_FREQ [19,45]  = **宽带湍流判别带** (软带 HYDRAULIC, 不罚展宽)
#     节点 1 BPF      [1.5,19] = 2.5 Hz 涡带单音 + 8.3/16.5 谐波, 整带同权硬挤压
#     节点 2 2xBPF    [105,130]= 116.5 Hz (=14*f_n, 全谱最强线)
#
#   为什么把 2.5 Hz 挪进硬带: 原先它独占 LOW_FREQ 软带 (w_i≈0.13, sigma_sqz 不降),
#   于是 2.5 Hz 保持 3.1 Hz 宽的深红粗带; 而 8.5 / 116.5 在硬带被锐化 4 倍,
#   vmax=0.5*max 归一化下发生幅值色阶反转 (极弱的 116.5 变成刺眼亮红)。
#   同一硬带内等权挤压后三者按原始幅度比例一起变锐, 反转消失。
#
#   为什么去掉 5-6 Hz 的切口: 旧配置 [1,5] / [6,25] 之间有一条不属于任何节点的
#   缝隙, 8.5 Hz 左裙边落在缝里不被挤压, 在 7.5 Hz 形成边界假肩峰。新配置
#   [1.5,19] 连续覆盖, 肩峰消失。此时软带 (19-45) 频率高于硬带 1, 是有意的:
#   DRIFT 边只把 r_obs 当特征喂 GAT, HARMONIC 边的 r_nom=2.0 项由
#   freq_constraints_active=False 挡在反向图外 (见 templates_for_preset)。
#
#   k / support_frac 逐节点:
#     节点 0, 2: k=1.6, support_frac=0.85 -> 通带停在 fence 内, mu/sigma 有效
#     节点 1   : k=3.0, support_frac=2.0  -> 通带盖满整条 fence (整带同权)
#   节点 1 必须允许越墙的原因见模块 docstring: mu 被 2.5 Hz 单音拽到 ~5 Hz,
#   若半宽只到 0.85*8.75=7.44 Hz, 16.5 Hz 会掉出通带、失去挤压。
BAND_PRESETS = {
    # rsi 保持 support_frac=2.0 (= 通带盖满整条 fence), 与本次改动前的行为
    # 逐位一致 —— 已有的 5 类实验都跑在这套掩码上, 不能顺手改掉。
    'rsi': [
        BandPreset('LOW_FREQ', 2.0, 18.0, 10.0, 4.0, k=3.0, support_frac=2.0),
        BandPreset('BPF',      40.0, 53.0, 50.0, 3.25, k=3.0, support_frac=2.0),
        BandPreset('2xBPF',    90.0, 105.0, 100.0, 3.0, k=3.0, support_frac=2.0),
    ],
    'shaft': [
        BandPreset('LOW_FREQ', 19.0, 45.0, 26.0, 6.0, k=1.6, support_frac=0.85),
        BandPreset('BPF',       1.5, 19.0,  8.5, 4.5, k=3.0, support_frac=2.00),
        BandPreset('2xBPF',   105.0, 130.0, 116.5, 2.5, k=1.6, support_frac=0.85),
    ],
    # 对照实验 (2026-11-07): 硬/软交界 19.0 -> 10.5 Hz。
    #   硬带 BPF [1.5,10.5] 只压 2.5 Hz 涡带线与 8.5 Hz 转轮基频线;
    #   软带 LOW_FREQ [10.5,42] 接管 8.5 Hz 右侧整片 10.5-35 Hz 湍流裙边,
    #   软带 w_i 低 (HYDRAULIC 的 w∈[0.05,0.35]) => 该区不挤压, 保持云团。
    # 目的: 湍流类 (c2/c5) 上让 10-35 Hz 宽带肉眼可见地"变粗", 与 MSST 的
    # 细丝形成反差; 代价是 16.5 Hz (2*f_n) 落进软带, 不再被单独压细。
    # BPF 的 mu_init 取 2.5 与 8.5 的中点 5.5, sigma 上限
    # (fence 半宽 4.5)*2.0/3.0 = 3.0 => 通带在 mu=5.5 时为 [1.5,10.5],
    # 即使 band_envelope_loss 把 mu 拽到 4 或 9 也仍盖满整条 fence。
    'shaft_split105': [
        BandPreset('LOW_FREQ', 10.5, 42.0, 26.0, 6.0, k=1.6, support_frac=0.85),
        BandPreset('BPF',       1.5, 10.5,  5.5, 2.5, k=3.0, support_frac=2.00),
        BandPreset('2xBPF',   105.0, 130.0, 116.5, 2.5, k=1.6, support_frac=0.85),
    ],
}


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """softplus 的反函数 (数值稳定版): log(exp(y) - 1)。"""
    return y + torch.log(-torch.expm1(-y).clamp(max=-1e-12))


class LearnableBandMask(nn.Module):
    """三个物理节点的可学习平顶软掩码。

    Args:
        presets:      BandPreset 列表 (节点顺序即其他模块的节点索引)
        k_support:    通带半宽倍数 (mu ± k*sigma); 默认 3
        sigma_floor:  sigma 下限 (Hz), 防止塌成 delta
        templates:    取 f_type / C_prior / w_range / p_absent 等非频率字段的
                      模板 (默认 PUMP_TURBINE_REGIONS); 长度必须与 presets 一致
        tau:          通带边缘的过渡宽度 (Hz); 默认 0.3。越小越接近硬矩形。
    """

    def __init__(self, presets: Sequence[BandPreset],
                 k_support: float = 3.0,
                 sigma_floor: float = 1.0,
                 templates: Optional[Sequence[FreqRegion]] = None,
                 tau: float = 0.3):
        super().__init__()
        if len(presets) == 0:
            raise ValueError('LearnableBandMask 至少需要一个 BandPreset')
        if tau <= 0:
            raise ValueError(f'tau 必须为正 (Hz), 收到 {tau}')
        self.presets = list(presets)
        self.names = [p.name for p in self.presets]
        # 全局兜底值 (functools 兼容旧接口); 逐节点覆盖见 BandPreset.k
        self.k_support = float(k_support)
        self.sigma_floor = float(sigma_floor)
        self.tau = float(tau)
        self.templates = list(templates) if templates is not None else list(PUMP_TURBINE_REGIONS)
        if len(self.templates) != len(self.presets):
            raise ValueError(
                f'templates 长度 {len(self.templates)} != presets 长度 {len(self.presets)}')

        fence_lo = torch.tensor([p.fence_lo for p in self.presets], dtype=torch.float32)
        fence_hi = torch.tensor([p.fence_hi for p in self.presets], dtype=torch.float32)
        mu = torch.tensor([p.mu_init for p in self.presets], dtype=torch.float32)
        # 初值必须落在 fence 内, 否则有效中心会被 clamp 到边界
        mu = torch.minimum(torch.maximum(mu, fence_lo + 1e-3), fence_hi - 1e-3)

        s0 = torch.tensor([p.sigma_init for p in self.presets], dtype=torch.float32)
        s0 = s0.clamp(min=self.sigma_floor + 1e-3)
        # sigma = sigma_floor + softplus(raw) -> raw = inv_softplus(sigma - floor)
        raw0 = _inv_softplus(s0 - self.sigma_floor)

        self.register_buffer('fence_lo', fence_lo)
        self.register_buffer('fence_hi', fence_hi)
        # 逐节点通带半宽倍数与"半宽/fence 半宽"上限比例
        k_node = torch.tensor(
            [p.k if p.k is not None else float(k_support) for p in self.presets],
            dtype=torch.float32)
        frac_node = torch.tensor([p.support_frac for p in self.presets],
                                 dtype=torch.float32)
        if not torch.all(k_node > 0):
            raise ValueError(f'k 必须为正, 收到 {k_node.tolist()}')
        if not torch.all(frac_node > 0):
            raise ValueError(f'support_frac 必须为正, 收到 {frac_node.tolist()}')
        self.register_buffer('k_node', k_node)
        self.register_buffer('support_frac', frac_node)
        # 记录初值, 供 band_envelope_loss 的 sigma 先验与日志使用 (不是参数)
        self.register_buffer('sigma_init', s0)
        self.mu = nn.Parameter(mu)
        self.raw_sigma = nn.Parameter(raw0)

    # ── 基本量 ──
    @property
    def n_nodes(self) -> int:
        return len(self.presets)

    def _k_vec(self, k: Optional[float] = None) -> torch.Tensor:
        """[N] 逐节点通带半宽倍数; 显式给 k 时整体覆盖 (兼容旧的标量调用)。"""
        if k is None:
            return self.k_node
        return torch.full_like(self.k_node, float(k))

    def mu_effective(self) -> torch.Tensor:
        """clamp 到 fence 内的中心 [N]。fence 边界处梯度为 0 —— 这是有意的:
        节点不该越过物理外框, 到了边界就停止。"""
        return torch.minimum(torch.maximum(self.mu, self.fence_lo + 1e-3),
                             self.fence_hi - 1e-3)

    def sigma(self) -> torch.Tensor:
        """[N] 正带宽, 下限 sigma_floor, 上限使 k*sigma = support_frac * fence 半宽。

        support_frac=0.85 (默认): 通带停在 fence 内部, mu/sigma 决定形状。
        support_frac>=1.0: 通带可以盖满/越过 fence, 由 fence 裁剪 —— 用于
        "整条带内所有谱线必须同权"的节点。
        旧实现的上限是 fence 半宽本身 (与 k 无关), 在 k=1.6 时会允许
        k*sigma 达到 1.6 倍半宽, 通带越墙 60% —— 与"不饱和"的目标相反。
        """
        s = self.sigma_floor + F.softplus(self.raw_sigma)
        half_width = (self.fence_hi - self.fence_lo).clamp(min=1.0) * 0.5
        cap = half_width * self.support_frac / self.k_node
        return torch.minimum(s, cap)

    def effective_bounds(self) -> torch.Tensor:
        """[N, 2] 有效支撑的左右界 (mu ± k*sigma, 再 clamp 到 fence)。"""
        half = self.k_node * self.sigma()
        lo = torch.maximum(self.mu_effective() - half, self.fence_lo)
        hi = torch.minimum(self.mu_effective() + half, self.fence_hi)
        return torch.stack([lo, hi], dim=-1)

    # ── 掩码 ──
    def weights(self, freqs: torch.Tensor, k: Optional[float] = None,
                normalize: bool = True) -> torch.Tensor:
        """[N, F] 平顶窗权重, 通带内恒为 1, 边缘为 tau 宽的 sigmoid 过渡,
        支撑外严格为 0。

        Args:
            freqs: [F] 频率轴 (Hz)
            k:     通带半宽倍数; None -> 逐节点 self.k_node
            normalize: True -> 每个节点除以自身峰值 (通带内峰值=1)
        """
        k_vec = self._k_vec(k)
        tau = self.tau
        f = freqs.to(device=self.mu.device, dtype=self.mu.dtype).view(1, -1)  # [1,F]
        mu = self.mu_effective().view(-1, 1)                                  # [N,1]
        sig = self.sigma().view(-1, 1)                                        # [N,1]
        half = k_vec.view(-1, 1) * sig
        lo_edge = mu - half
        hi_edge = mu + half
        # 平顶窗: 通带 [lo_edge, hi_edge] 内 ≈ 1, 两侧各 ~2*tau 内衰减到 0
        w = (torch.sigmoid((f - lo_edge) / tau)
             * torch.sigmoid((hi_edge - f) / tau))
        in_fence = (f >= self.fence_lo.view(-1, 1)) & (f <= self.fence_hi.view(-1, 1))
        # 过渡段完整包在支撑里 (否则硬截断会切掉 sigmoid 的下降沿)
        in_reach = (f >= lo_edge - 2.0 * tau) & (f <= hi_edge + 2.0 * tau)
        w = w * (in_fence & in_reach).to(w.dtype)
        if normalize:
            peak = w.max(dim=1, keepdim=True).values
            # 通带整个落在 fence 外时 peak≈0; 不能去除以 1e-12 把噪声放大成 1
            w = torch.where(peak > 1e-6, w / peak.clamp(min=1e-12),
                            torch.zeros_like(w))
        return w

    def support(self, freqs: torch.Tensor, k: Optional[float] = None) -> torch.Tensor:
        """[N, F] bool —— 支撑 (|f-mu| <= k*sigma + 2*tau 且 f 在 fence 内)。

        与 `weights(...) > 0` 的范围一致: 多留 2*tau 是为了不切掉 sigmoid 的
        下降沿。
        """
        k_vec = self._k_vec(k)
        f = freqs.to(device=self.mu.device, dtype=self.mu.dtype).view(1, -1)
        d = (f - self.mu_effective().view(-1, 1)).abs()
        in_fence = (f >= self.fence_lo.view(-1, 1)) & (f <= self.fence_hi.view(-1, 1))
        half = k_vec.view(-1, 1) * self.sigma().view(-1, 1)
        return in_fence & (d <= half + 2.0 * self.tau)

    def union_support(self, freqs: torch.Tensor, k: Optional[float] = None) -> torch.Tensor:
        """[F] bool —— 三个节点支撑的并集, 供 physics_band_mask 的替代路径。"""
        return self.support(freqs, k).any(dim=0)

    # ── 与旧接口的桥 ──
    def as_regions(self) -> List[FreqRegion]:
        """快照成 FreqRegion 列表 (detach)。

        供 numpy 提取器 / IF 落点过滤 / 绘图 / 导出等**不需要梯度**的路径使用。
        若模板的绝对锚已落在可学带外 (例如 shaft preset 的 2xBPF 从 99.99
        移到 116.5), 锚会被丢弃 —— 否则诊断量会拿一个不可能满足的锚去比。
        """
        lo_hi = self.effective_bounds().detach().cpu()
        out: List[FreqRegion] = []
        for i, tpl in enumerate(self.templates):
            lo = float(lo_hi[i, 0])
            hi = float(lo_hi[i, 1])
            if hi <= lo:
                hi = lo + 1e-2
            f_anchor = tpl.f_anchor
            if f_anchor is not None and not (lo - 1e-6 <= f_anchor <= hi + 1e-6):
                f_anchor = None
            out.append(replace(tpl, name=self.names[i], f_min=lo, f_max=hi,
                               f_anchor=f_anchor))
        return out

    def describe(self) -> str:
        """一行可读摘要, 训练开始时打印。"""
        mu = self.mu_effective().detach().cpu()
        sg = self.sigma().detach().cpu()
        lo_hi = self.effective_bounds().detach().cpu()
        parts = []
        for i, name in enumerate(self.names):
            parts.append(
                f'{name}: mu={mu[i]:.2f} sigma={sg[i]:.2f} k={float(self.k_node[i]):.1f} '
                f'support=[{lo_hi[i, 0]:.2f},{lo_hi[i, 1]:.2f}] '
                f'fence=[{float(self.fence_lo[i]):.1f},{float(self.fence_hi[i]):.1f}]')
        return ' | '.join(parts)

    def coverage_report(self, freqs: torch.Tensor,
                        lines: Sequence[float]) -> str:
        """诊断: 给定谱线是否落在本节点通带内 (权重 > 0.5), 训练日志用。"""
        w = self.weights(freqs).detach().cpu()
        f = freqs.detach().cpu()
        out = []
        for i, name in enumerate(self.names):
            hits = []
            for ln in lines:
                j = int(torch.argmin((f - ln).abs()))
                if float(w[i, j]) > 0.5:
                    hits.append(f'{ln:g}')
            hit_str = '/'.join(hits) or '-'
            out.append(f'{name}:[{hit_str}]')
        return ' '.join(out)

    @classmethod
    def from_preset(cls, preset: str = 'rsi', k_support: float = 3.0,
                    sigma_floor: float = 1.0,
                    templates: Optional[Sequence[FreqRegion]] = None,
                    tau: float = 0.3) -> 'LearnableBandMask':
        if preset not in BAND_PRESETS:
            raise KeyError(f'未知 preset {preset!r}; 可选: {sorted(BAND_PRESETS)}')
        if templates is None:
            templates = templates_for_preset(preset)
        return cls(BAND_PRESETS[preset], k_support=k_support,
                   sigma_floor=sigma_floor, templates=templates, tau=tau)


def templates_for_preset(preset: str,
                         base: Optional[Sequence[FreqRegion]] = None
                         ) -> List[FreqRegion]:
    """按 preset 改写非频率字段 (f_anchor / p_absent), 供 `as_regions()` 使用。

    `BAND_PRESETS` 只给频率外框; f_type / w_range / p_absent / f_anchor 仍来自
    `PUMP_TURBINE_REGIONS` 模板。但模板上的绝对锚 (2xBPF = 99.99 Hz) 与缺席
    门限 (p_absent 0.13/0.17) 是**在 RSI 数据上标定的**, 搬到主轴数据上不成立:

      · 主轴的 2xBPF 落在 116.5 Hz, 锚若仍在 99.99 Hz, 而学习到的频带在
        [107,123], `as_regions()` 会把锚丢掉 —— 结果正确但靠巧合;
      · 主轴各类未必都有这条高频线, 用 RSI 的缺席门限会把"本来就在"的样本
        误判为缺席 (或反之), 进而错误地收拢 w_i 上界、截断 RE_2D 推力。

    所以主轴 preset 直接把这两项关掉 (p_absent=0 -> a≡0, 行为与未启用一致)。
    另外图上的 BPF→2xBPF 边 r_nom=2.0 在主轴数据上不成立 (116.5/8.5≈13.7),
    该项由 `SASTConfig.freq_constraints_active=False` (默认) 挡在反向图外。
    """
    tpl = list(PUMP_TURBINE_REGIONS if base is None else base)
    if preset == 'shaft':
        return [replace(r, f_anchor=None, p_absent_lo=0.0, p_absent_hi=0.0)
                for r in tpl]
    return tpl