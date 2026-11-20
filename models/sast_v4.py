"""SAST v4 前向组装
==================

把 v4 的独立部件接起来, 产出一张最终时频图 T(f,t):

    1. 双窗前端     dual_frontend      -> S_short / S_long / S_base / ω / q / W_main
    2. 节点门控     GAT(长窗谱)        -> w_node [B,N,T], Δω [B,N,T]
    3. 分频带门控   β(f,t)             (锁频带 β=w_node; 低频带 β≡0, 见 low_squeeze)
    4. 连续落点重排 conserved_reassign -> R_long  (落点 = ω + Δω, σ 固定 0.30 bin)
    5. 合成         T = (1-β)·S_base + β·R_long

``use_gat=False`` 时退回固定 C_prior 与 Δω≡0 (诊断/消融口径)。

定稿依据 (外部会诊, 2026-11-12):
  - 短窗 512 + 长窗 1024; α(f) 在 3-9 Hz 升余弦切换底图 (见 dual_stft.alpha_short)
  - 「换窗(α)」与「挤压(β)」正交: β 只控制挤压强度, 不决定基准窗
  - 锁频带 (BPF/2xBPF) β = w_node (不用 q, 因为 46/50 干扰会拖拽 IF, 让 q 失真)
  - 低频带 (<10 Hz) **β≡0, 不挤压** (2026-11-14)。旧 β=w_node·q 实测只占该带
    能量 1.5%, 且 σ 被映射到最宽核, 效果是低通而非挤压 (FWHM +17%)。
  - β 斜坡 10→12 Hz (升余弦), 12 Hz 起 β = w_BPF (2026-11-14 定稿; 上表)
  - 底噪连续性: >35 Hz 的带外背景统一除以 W_main, 避免频带边界亮暗断崖

GAT 由**未挤压**的 ``S_base`` 算出 presence / absence, 作为 ``w_node`` 有效
区间的上下界门控 (只作门控, 不参与反传); 区间映射在 ``MinimalPhysicsGAT``
内部完成, 因此 ``w_node`` 逐帧恒落在物理可行域内。

训练入口: ``train_sast_v4.py`` (纯物理损失, 见 ``models/sast_v4_losses``)。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.dual_stft import dual_frontend
from models.reassign_v4 import (
    SIGMA_ANTIALIAS_BIN,
    apply_delta_omega,
    conserved_reassign,
    sigma_anti_alias,
)
from models.sast_losses import absence_gate, node_presence
from models.sast_gat_v4 import (
    MinimalPhysicsGAT,
    band_delta_hz,
    node_features_from_v4,
)
from models.sast_nodes import (
    PUMP_TURBINE_REGIONS,
    FreqRegion,
)


# β 斜坡的 (起点, 终点) [Hz]。**已在 2026-11-14 定稿为 (10, 12)**。
#
# 起点 10 Hz: β≡0 区 (低频不挤压区) 的上界。依据是 0-18 Hz 里住着两族不同的
#   东西: 0-10 Hz 是尾水管涡带 (0.5-3 Hz) + 转轮旋转基频 8.33 Hz, 单帧本征
#   展宽宽; 10-18 Hz 是 c1/c2 的离散水力尖峰 (实测 15.5 / 14.5 Hz), 时均谱上
#   是尖峰。切在 10 Hz 把后者剥离出 LOW_FREQ, 让该节点只描述涡带与转频;
#   10 Hz 同时紧接 alpha_short 的 9 Hz 终点 (9 Hz 以上已 100% 长窗)。
#   ⚠ 必须与 models/sast_nodes.py 的 LOW_FREQ.f_max 保持一致: 前者决定 β 的
#   分段, 后者决定 node_if/node_energy/presence 与损失覆盖的频段。
#
# 终点 12 Hz: **与起点解耦**（两者不必是同一个数）。10-18 Hz 的实测是
#   R_long/S_base ≈ 1/W_main = 0.16 —— 该带没有相干脊线, 重排不聚能, 只把
#   背景除以 W_main, 因此 β 在那里是**对比度旋钮**而非锐化旋钮。要让这一段
#   在图上呈现 (b) R_long 的观感, β 必须在 12 Hz 前就饱和。实测 (五类平均):
#
#     斜坡      Δβ/bin   12-18 Hz 对比度(dB)   15 Hz 电平     8.33 Hz 电平
#     (10,35)   0.029    1.2-5.0             -0.5 dB       0.0 dB
#     (10,20)   0.073    1.8-5.3             -3.0 dB       0.0 dB
#     (10,12)   0.328    3.4-16.9           -10.3 dB       0.0 dB
#     ( 4,12)   0.091    3.4-16.9           -10.3 dB      -6.0 dB
#
#   (4,12) 与 (10,12) 在 ≥12 Hz **逐位相同**, 它多做的只是把 4-9 Hz 一起压暗:
#   该段 R_long/S_base 仅约 0.1 (低于 1/W_main, 重排在往外搬能量), 压下去
#   换不到锐化, 却让 8.33 Hz 转轮旋转线掉 6 dB。故取 (10,12)。
#   代价是 10 Hz 处一条约 -12 dB / 2 Hz 的横向暗边; 它落在 10.5 Hz 尖峰的
#   低侧, 尖峰保留 -3.3 dB 而其上方背景掉 -10 dB, 判别特征反而更亮。
LOW_BETA_MAX_HZ = 10.0
BETA_RAMP_HI_HZ = 12.0


def prior_w_node(regions: List[FreqRegion], B: int, T: int,
                 device=None, dtype=None) -> torch.Tensor:
    """由 FreqRegion.C_prior 构造的固定节点先验 [B, N_phys, T].

    诊断阶段在没有训练好的 GAT 时的 w_node 来源; 训练时由 GAT 替换。
    """
    N = len(regions)
    c = torch.tensor([r.C_prior for r in regions], device=device, dtype=dtype)
    return c.view(1, N, 1).expand(B, N, T).contiguous()


def band_beta_map(w_node: torch.Tensor, q: torch.Tensor, freqs: torch.Tensor,
                  alpha: torch.Tensor,
                  regions: Optional[List[FreqRegion]] = None,
                  ramp: Optional[torch.Tensor] = None,
                  low_squeeze: bool = False,
                  low_beta_max_hz: Optional[float] = None) -> torch.Tensor:
    """全频段连续的 β(f,t) ∈ [0,1], 形状 [B, F, T].

    单公式 T = (1-β)·S_base + β·R_long 统一覆盖全频段。β 在频率轴上分段连续:

        [0, 10):   β ≡ 0  (默认)  |  β = w_LOW(t)·q(f,t)  (low_squeeze=True)
        [10, 36):  β = ramp(f) · w_BPF(t)  升余弦 0→1 over (10, 12); 12 Hz 后 ramp ≡ 1
        [36, 90):  β = w_BPF(t)                       BPF 带 + 53-90 空带
        [90, F):   β = w_2xBPF(t)                     2xBPF 带 + >105 空带

    斜坡起止点由 BETA_RAMP_HI_HZ / LOW_BETA_MAX_HZ 给出 (可在实例上覆盖),
    与 ``alpha`` 已解耦 —— 见下方段落。

    **2026-11-14 关闭低频软挤压 (low_squeeze 默认 False)**。旧的
    ``β = w_LOW·q`` 在实测中不是"轻轻挤", 而是一个**衰减过的低通**:

      · 五类各 2 条实测 β_low = 0.075, 而 σ(0.075) = 1.87 bin ≈ 0.93 Hz
        —— 恰好贴着 sigma_max, 是**最宽**的核。行归一化散射下, 已落在
        自己 IF 上的 bin 被摊成 σ 宽一团、幅度再除以 Z ≈ 4.7。
      · R_long 只占低频能量的 **1.5%**; 低频 FWHM 却从 3.90 Hz 涨到
        **4.55 Hz (+17%)** —— 软挤压把带抹宽了。
      · 5-18 Hz 可分辨峰数: T = 去掉β_low = S_base = 20, S_short 单独 = 15。
        也就是说"把 6.5/10.5/15 Hz 分开"完全是 α 换窗 (0.51 s → 1.02 s)
        做的, β_low 一个峰都没帮上。
      · 没有任何损失项覆盖它: RE_2D 跳过 SOFT_F_TYPE, L_soft_ent 单边
        hinge 实测恒为 0, 只剩 L_w_center 与 L_z 双双把 w_LOW 钉在
        C_prior=0.20 (z_LOW ≈ 0.01-0.02, mlp_w 梯度死)。
      · 语义上还与 α 冲突: f<3 Hz 用短窗是为了保时间定位, 而 R_long 是
        1.02 s 长窗的分布, 混进来正好抵消这一点。

    置 0 后斜坡起点 (10 Hz) 处的 β 恰为 0, 与左侧 β≡0 连续, 不会切出硬边。
    保留开关仅为复现旧 checkpoint 与论文消融。

    ``ramp`` 即 ``dual_stft.beta_ramp``, 2026-11-14 起与 ``alpha`` **解耦**:
    窗口切换已从 18-35 Hz 下移到 3-9 Hz, 若继续用 ``1-alpha`` 作斜坡, 9 Hz
    以上它恒为 1, β 会在斜坡起点一步跳到 w_BPF, 切出硬边。
    ``ramp=None`` 时退回 ``1-alpha`` (旧行为, 仅供消融复现)。
    """
    if regions is None:
        regions = PUMP_TURBINE_REGIONS
    B, F, T = q.shape
    dev = q.device
    dtype = q.dtype

    name_to_idx = {r.name: i for i, r in enumerate(regions)}
    i_low = name_to_idx['LOW_FREQ']
    i_bpf = name_to_idx['BPF']
    i_2x = name_to_idx['2xBPF']

    w_low = w_node[:, i_low, :].view(B, 1, T)          # [B, 1, T]
    w_bpf = w_node[:, i_bpf, :].view(B, 1, T)
    w_2x = w_node[:, i_2x, :].view(B, 1, T)

    f = freqs.view(1, F, 1)
    beta = torch.zeros(B, F, T, device=dev, dtype=dtype)

    # [0, LOW_BETA_MAX_HZ): 默认恒 0 (软挤压关闭, 见 docstring 的实测依据)
    f_low_max = LOW_BETA_MAX_HZ if low_beta_max_hz is None else low_beta_max_hz
    if low_squeeze:
        beta = beta + (f < f_low_max).to(dtype) * w_low * q
    # [LOW_BETA_MAX_HZ, 90): 升余弦斜坡 × w_BPF。斜坡一旦饱和 (f >= ramp_hi)
    # 就有 ramp_f ≡ 1, 整段退化为 β = w_BPF。**不再在 36 Hz 切分支** —— 那样
    # 一旦 ramp_hi > 36 就会在 36 Hz 切出一个台阶。定稿 ramp_hi=12 时两种写法
    # 逐位相同。
    ramp_f = (ramp if ramp is not None else (1.0 - alpha)).view(1, F, 1)
    # [LOW_BETA_MAX_HZ, 88): 斜坡 × w_BPF (斜坡在 ramp_hi 饱和)
    beta = beta + ((f >= f_low_max) & (f < 88.0)).to(dtype) * ramp_f * w_bpf

    # [88, 90) 与 [103, 105): 2 Hz 升余弦过渡; [90, 103): w_2xBPF;
    # [105, ∞): **回到 w_BPF** —— 105 Hz 以上是空化/水力空带, 不是 2xBPF
    # 分量。旧实现把 f≥90 全部绑在 w_2xBPF 上, 空转 (w_2x≈0.22) 时整段
    # 105-250 Hz 背景不再被 1/W_main 压暗, 在 90 Hz 切出 3.7 倍的亮暗台阶
    # (实测 53-90 保留 0.219 vs 105-250 保留 0.816)。剥离后台阶收窄到
    # 90-105 这一条带本身, 不再波及整段高频。
    def _cos_blend(x, lo, hi):
        t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
        return 0.5 * (1.0 - torch.cos(torch.pi * t))

    w_bg = w_bpf
    m_up = (f >= 88.0) & (f < 90.0)          # w_BPF -> w_2x
    m_2x = (f >= 90.0) & (f < 103.0)         # 纯 w_2x
    m_dn = (f >= 103.0) & (f < 105.0)        # w_2x -> w_bg
    m_bg = (f >= 105.0)                      # 空带
    k_up = _cos_blend(f, 88.0, 90.0)
    k_dn = _cos_blend(f, 103.0, 105.0)
    beta = beta + m_up.to(dtype) * ((1 - k_up) * w_bpf + k_up * w_2x)
    beta = beta + m_2x.to(dtype) * w_2x
    beta = beta + m_dn.to(dtype) * ((1 - k_dn) * w_2x + k_dn * w_bg)
    beta = beta + m_bg.to(dtype) * w_bg

    return beta.clamp(0.0, 1.0)


class SASTv4(nn.Module):
    """v4 前向 (诊断口径): 双窗 + 分频带 β + 连续重排 + 底噪连续合成。"""

    def __init__(self, regions: Optional[List[FreqRegion]] = None,
                 sigma_min: Optional[float] = None,
                 sigma_max: Optional[float] = None,
                 use_gat: bool = True, gat_kwargs: Optional[dict] = None,
                 low_squeeze: bool = False,
                 low_beta_max_hz: Optional[float] = None,
                 beta_ramp_hi_hz: Optional[float] = None):
        super().__init__()
        self.regions = regions if regions is not None else PUMP_TURBINE_REGIONS
        # β 斜坡的 (起点, 终点)。**与 LOW_FREQ 节点区域解耦**: 节点区域决定
        # node_if/node_energy/presence 的积分范围 (0-10 Hz), 斜坡只决定 TFR
        # 哪一段开始混入 R_long。两者不必是同一个数 (2026-11-14)。
        self.low_beta_max_hz = (LOW_BETA_MAX_HZ if low_beta_max_hz is None
                                else low_beta_max_hz)
        self.beta_ramp_hi_hz = (BETA_RAMP_HI_HZ if beta_ramp_hi_hz is None
                                else beta_ramp_hi_hz)
        # sigma_min / sigma_max 已废弃 (2026-11-14): σ 现在是固定的亚 bin
        # 抗锯齿核 SIGMA_ANTIALIAS_BIN, 不再是挤压宽度旋钮。保留形参只是为
        # 了让旧调用不报错; 传值会被忽略。
        self.sigma_min = SIGMA_ANTIALIAS_BIN
        self.sigma_max = SIGMA_ANTIALIAS_BIN
        # 低频软挤压默认关闭 (理由与实测见 band_beta_map 的 docstring)
        self.low_squeeze = low_squeeze
        self.gat = (MinimalPhysicsGAT(self.regions, **(gat_kwargs or {}))
                    if use_gat else None)

    def forward(self, x: torch.Tensor,
                w_node: Optional[torch.Tensor] = None,
                delta_hz: Optional[torch.Tensor] = None,
                return_detail: bool = False,
                detach_frontend: bool = False) -> torch.Tensor:
        """前向。

        Args:
            x:       [B, N] 实数信号
            w_node:  [B, N_phys, T] 节点挤压强度; None 则由 GAT (或无 GAT 时
                     用 C_prior 常量) 给出
            delta_hz:[B, N_phys, T] 落点修正 (Hz); None 则沿用 GAT 输出 / 0
            return_detail: 为 True 时返回 (T, detail_dict)
            detach_frontend: 前端 (双窗 STFT + IF) **无任何可训练参数**, 截断
                     其反向图不影响任何参数梯度, 但能省掉 float64 STFT 的
                     反向显存与时间 (实测 B=4 峰值 5.0→约 2 GB, 吞吐翻倍)。
                     训练时置 True; 需要对 x 求导时保持 False。

        Returns:
            T [B, F, T] 最终时频图; 或 (T, detail) 当 return_detail
        """
        # β 斜坡的起止点由实例属性给出 (默认取模块常量)
        if detach_frontend:
            with torch.no_grad():
                o = dual_frontend(x, beta_ramp_lo=self.low_beta_max_hz,
                                  beta_ramp_hi=self.beta_ramp_hi_hz)
        else:
            o = dual_frontend(x, beta_ramp_lo=self.low_beta_max_hz,
                              beta_ramp_hi=self.beta_ramp_hi_hz)
        S_base = o['S_base']
        S_long = o['S_long']
        omega = o['omega']
        q = o['q']
        freqs = o['freqs']
        w_main = o['w_main']
        B, F, T = S_long.shape
        df_hz = float(freqs[1] - freqs[0])

        gat_out = None
        presence = absence = None
        if (w_node is None or delta_hz is None) and self.gat is not None:
            # presence/absence 由**未挤压**的 S_base 决定 (否则模型可制造尖峰
            # 自证存在), 且只作门控, 不参与反传
            with torch.no_grad():
                presence = node_presence(S_base, freqs, self.regions)
                absence = absence_gate(presence, self.regions)
            # 带内集中度替代原来的死特征 node_persist (恒 0.5), 作为第 4 维
            feat = node_features_from_v4(S_long, omega, freqs, self.regions,
                                         conc=presence)
            gat_out = self.gat(feat, presence, absence)
        if w_node is None:
            w_node = (gat_out['w_node'] if gat_out is not None
                      else prior_w_node(self.regions, B, T, device=x.device,
                                        dtype=S_base.dtype))
        if delta_hz is None:
            delta_hz = (gat_out['delta_hz'] if gat_out is not None
                        else torch.zeros_like(w_node))

        # 节点 Δω -> 逐 bin Δω -> 落点 (只修落点, 不动 STFT 幅度)
        omega_c = apply_delta_omega(omega, band_delta_hz(delta_hz, freqs,
                                                         self.regions), df_hz)

        beta = band_beta_map(w_node, q, freqs, o['alpha'], self.regions,
                             ramp=o.get('beta_ramp'),
                             low_squeeze=self.low_squeeze,
                             low_beta_max_hz=self.low_beta_max_hz)  # [B,F,T]
        # σ 是固定的亚 bin 抗锯齿核 (0.30 bin), 与 β 无关 —— 挤压强度只由 β 决定。
        # 旧映射 σ = σ_min + (1−β)(σ_max−σ_min) 已删除, 见 reassign_v4 模块 docstring。
        sigma = sigma_anti_alias(beta)
        R_long = conserved_reassign(S_long, omega_c, sigma, w_main=w_main)

        T = (1.0 - beta) * S_base + beta * R_long

        if not return_detail:
            return T
        detail = {
            'S_base': S_base, 'S_long': S_long, 'S_short': o['S_short'],
            'R_long': R_long, 'beta': beta, 'sigma': sigma, 'q': q,
            'alpha': o['alpha'], 'freqs': freqs, 'w_main': w_main,
            'w_node': w_node, 'delta_hz': delta_hz, 'omega': omega_c,
        }
        if gat_out is not None:
            detail.update({'A_ij': gat_out['A_ij'],
                           'h_coarse': gat_out['h_coarse'],
                           'w_bounds': gat_out['w_bounds'],
                           'presence': presence, 'absence': absence,
                           'z_w': gat_out['z_w'],
                           'node_feats': gat_out['node_feats']})
        return T, detail


if __name__ == '__main__':
    torch.manual_seed(0)
    # 合成自检: 两条线 (50 / 100 Hz) + 宽带底噪, 验证形状与基本量纲
    t = torch.arange(2000, dtype=torch.float64) / 1000.0
    x = (torch.cos(2 * torch.pi * 50.0 * t)
         + 0.6 * torch.cos(2 * torch.pi * 100.0 * t)
         + 0.2 * torch.randn(2000, dtype=torch.float64)).unsqueeze(0)
    m = SASTv4()
    T, d = m(x, return_detail=True)
    print(f'T shape {tuple(T.shape)}  S_base {tuple(d["S_base"].shape)}')
    lo, hi = m.low_beta_max_hz, m.beta_ramp_hi_hz
    print(f'β 斜坡 [{lo:g}, {hi:g}) Hz | β 带内均值: '
          f'LOW_FREQ(0-{lo:g}) {d["beta"][0, :int(lo*2)].mean():.3f}  '
          f'斜坡({lo:g}-{hi:g}) '
          f'{d["beta"][0, int(lo*2):int(hi*2)].mean():.3f}  '
          f'BPF(36-53) {d["beta"][0, 36*2:53*2].mean():.3f}  '
          f'空带(53-90) {d["beta"][0, 53*2:90*2].mean():.3f}  '
          f'2xBPF(90-105) {d["beta"][0, 90*2:105*2].mean():.3f}')
    print(f'W_main = {d["w_main"]:.3f}')
    # 50 Hz 峰应接近 0.5 (保峰口径)
    seg = T[0, 99:102]
    print(f'50 Hz 峰 (合成) = {seg.max().item():.3f} (期望 ≈0.5)')
