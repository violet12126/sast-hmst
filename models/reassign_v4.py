"""
SAST v4 重排算子
================

相对 v3 的三点变化：

1. **落点连续**：v3 把 ω̂ round 成整数 bin 后当常量用，梯度只能经 σ 走。
   v4 保留连续落点 ω*，高斯权重按**整数目标 bin 到连续中心的距离**计算：
       r_k = (j − ω*) / σ ,  j = round(ω*) + k
   于是 ∂输出/∂ω* 存在，物理约束（倍频比、绝对锚）第一次有真实梯度。

2. **幅度内生守恒（行归一 + 窗常数）**：
   v3 把主瓣内约 6 个 bin 的幅值全部累加到同一目标，峰值被动放大 W_main 倍，
   只能事后除以 γ 标定。**柱平均**（除以落点处累计权重）看似能避免 γ，
   但它是平滑而非输运：分母 = 有多少个 bin 报告同一 IF，随信噪比变化，
   干净的强线反而被罚得最重（实测峰因子 0.4–0.95，总能量按合并数被吃掉）。

   v4 改为**行归一**——每个源像素先把自己的核质量归一到 1，再散射：
       out_j = (1/W_main) · Σ_f [ w_{f→j} / Z_f ] · A_f
       Z_f   = Σ_k w_{f→k}                       （逐像素实际核质量）
       W_main = Σ_η |H(η)| / |H(0)|              （由实际窗算出的等效 bin 数）
   对一条干净谱线，Z_f 与 W_main 相消，输出峰 = Σ_f A_f / W_main ≈ 原始峰
   （参与 bin 少时 0.85，主瓣全参与时 1.00），且与 σ 无关；
   底噪按 1/W_main 压低，信噪比反而提升。总能量严格守恒到常数 1/W_main。

3. **σ 是固定的亚 bin 抗锯齿核，不是挤压宽度旋钮（2026-11-14 定稿）**：
   落点 ω* 连续，而幅度只能落在离散频率网格上；σ 的职责只有一件——把这个
   连续落点平滑分配到相邻 bin（σ=0.30 bin 时中心拿 99.2%，两侧各 0.39%），
   避免取整跳变。**挤压强度统一由 β_n(t) 控制。**

   旧映射 σ = σ_min + (1−β)(σ_max−σ_min) 把"软/硬"编码进了核宽：β 小 → σ 大
   → 核摊得更开，于是"软挤压"实际是"低通滤波"而不是"少搬"。实测（5 类各 2 条）：
   低频 β=0.075 时 σ=1.87 bin，R_long 只占低频能量 1.5%，却把该带 FWHM 从
   3.90 Hz 抹宽到 4.55 Hz（+17%），5–18 Hz 可分辨峰数一个没加。低频软挤压
   已关闭（见 ``sast_v4.band_beta_map`` 的 ``low_squeeze=False``），本条耦合
   一并取消。改成常数对输出影响很小：0–10 Hz 完全不变（那里 β≡0），36 Hz 以上
   1–2%，只有 10–12 Hz 斜坡 4.5–6.1%（该处 β 从 0 爬到 0.93，旧 σ 跨 2.0→0.33）。

注意：**重排不能同时守恒峰和总能量**。一条线是「6 个 bin 各一份」变成
「1 个 bin 六份」，二者只能选一个。这里选峰（时频显示、CNN、L_marginal
的归一化都只看形状，常数 1/W_main 会被消掉）。

用法：
    W_main = window_mainlobe_mass(1024, 2000)
    R = conserved_reassign(|S_long|, omega, sigma, w_main=W_main)
"""
from __future__ import annotations

import math
from typing import Optional

import torch

EPS = 1e-8


SIGMA_ANTIALIAS_BIN = 0.30
"""重排核宽（bin）—— 固定的亚 bin 抗锯齿核。

σ=0.30 bin（=0.15 Hz @ df=0.5 Hz）时，一个恰好落在 bin 中心上的源像素有
99.2% 的质量落在该 bin、两侧各 0.39%；落点在两 bin 之间时按距离线性分配。
这就是它的全部作用：让连续落点 ω* 能被离散网格平滑表示。

**不要再把它当"挤压宽度"来读**——挤压强度由 β_n(t) 唯一决定。理由见模块
docstring 第 3 条。
"""


def sigma_anti_alias(beta: torch.Tensor) -> torch.Tensor:
    """β [B,F,T] → 常数核宽 [B,F,T]。

    保留 β 参数与输出形状，只为与 ``conserved_reassign`` 的接口（以及旧权重
    的形状约定）一致；数值上不依赖 β。
    """
    return torch.full_like(beta, SIGMA_ANTIALIAS_BIN)


def kernel_mass(sigma: torch.Tensor, K: int) -> torch.Tensor:
    """Z(σ) = Σ_{k=-K..K} exp(-k²/(2σ²)) —— 单源核的解析总质量。

    仅用于标定/自检；算子内部用的是逐像素实际质量 Z_f（落点可能不在
    整数上，Z_f 会随小数偏移变小）。
    """
    k = torch.arange(-K, K + 1, device=sigma.device, dtype=sigma.dtype)
    return torch.exp(-0.5 * (k / sigma.clamp_min(EPS)) ** 2).sum()


def _weights(omega: torch.Tensor, idx0: torch.Tensor, sg: torch.Tensor,
             k: int) -> torch.Tensor:
    """第 k 个整数目标的核权重（含连续落点的小数偏移）。"""
    r = ((idx0 + k).to(omega.dtype) - omega) / sg
    return torch.exp(-0.5 * r * r)


def conserved_reassign(amp: torch.Tensor,
                       omega: torch.Tensor,
                       sigma: torch.Tensor,
                       K: Optional[int] = None,
                       w_main: Optional[float] = None,
                       normalize: str = 'row') -> torch.Tensor:
    """连续落点 + 幅度守恒的高斯重排。

    Args:
        amp:    [B, F, T] 源幅值 |S|
        omega:  [B, F, T] 连续落点 (bin, 0-based, 可含 Δω 修正)
        sigma:  [B, F, T] 核宽 (bin)
        K:      核半径 (bin)；None 时按 3·sigma 的最大值取
        w_main: 窗主瓣等效 bin 数；给定则输出除以它（峰还原到 STFT 标尺），
                None 则只做质量守恒散射（峰被放大 W_main 倍）
        normalize:
            'row'     行归一 + (可选)w_main —— **默认，v4 语义**
            'column'  柱平均（v4 早期版本，仅用于消融对照）
            'none'    纯散射（v3 行为，峰被放大）

    Returns:
        [B, F, T] 重排后幅值
    """
    B, F, T = amp.shape
    if K is None:
        K = int(math.ceil(3.0 * sigma.detach().amax().item()))
        K = max(1, min(K, F - 1))

    idx0 = omega.detach().round().long()
    # amp 可以是复数 (模态分离的反演需要: 对复 STFT 施加同一重排算子)。
    # 核质量 z 必须用**实数** dtype, 不能 zeros_like(amp) —— 复数没有 clamp。
    rdtype = amp.real.dtype if torch.is_complex(amp) else amp.dtype
    omega = omega.to(rdtype)
    sg = sigma.clamp_min(EPS).to(rdtype)

    # 行归一：先累加每个源像素的核质量 Z_f（与散射循环分开，避免存 2K+1 个 [B,F,T]）
    if normalize == 'row':
        z = torch.zeros(amp.shape, device=amp.device, dtype=rdtype)
        for k in range(-K, K + 1):
            z = z + _weights(omega, idx0, sg, k)
        z = z.clamp_min(EPS)
    else:
        z = None

    out = torch.zeros_like(amp)
    wsum = torch.zeros_like(amp) if normalize == 'column' else None
    zero = torch.zeros_like(amp)

    for k in range(-K, K + 1):
        target = (idx0 + k).clamp(0, F - 1)
        wk = _weights(omega, idx0, sg, k)
        contrib = wk * amp
        if normalize == 'row':
            contrib = contrib / z
        out = out + torch.scatter_add(zero, 1, target, contrib)
        if normalize == 'column':
            wsum = wsum + torch.scatter_add(zero, 1, target, wk)

    if normalize == 'column':
        out = out / wsum.clamp_min(EPS)
    if w_main is not None:
        out = out / w_main
    return out


def apply_delta_omega(omega: torch.Tensor, delta_hz: torch.Tensor,
                      df_hz: float) -> torch.Tensor:
    """节点级频率校准残差 (Hz) → bin 偏移，广播到 [B,F,T]。

    delta_hz: [B, N_phys, T] 或 [B, 1, T]（已按节点广播）
    """
    if delta_hz.dim() == 2:
        delta_hz = delta_hz.unsqueeze(1)
    return omega + delta_hz / df_hz


if __name__ == '__main__':
    torch.manual_seed(0)

    # ── 1) 真实窗常数与单根高斯线 ──────────────────────────────
    from models.dual_stft import window_mainlobe_mass

    N, hl = 2000, 1024
    W_main = window_mainlobe_mass(hl, N)
    F, T = 128, 4
    sigma_line = 1.22 / 0.5          # 1.22 Hz → bin

    def build_line(f0):
        """一条谱线 + 真实 IF 剖面：主瓣内报 f0，之外报自己的 bin。"""
        bins = torch.arange(F, dtype=torch.float64).view(1, F, 1)
        amp = torch.exp(-0.5 * ((bins - f0) / sigma_line) ** 2)
        amp = amp.expand(1, F, T).clone()
        omega = bins.expand(1, F, T).clone()
        mask = ((bins - f0).abs() <= 3).expand_as(omega)
        omega = torch.where(mask, torch.full_like(omega, f0), omega)
        return amp, omega

    print(f'[1] 单根高斯线 (σ_f=1.22Hz)  W_main={W_main:.3f}')
    for f0 in (64.0, 64.3):
        amp, omega = build_line(f0)
        sigma = torch.full((1, F, T), 0.35, dtype=torch.float64)
        r_row = conserved_reassign(amp, omega, sigma, K=3, w_main=W_main)
        r_col = conserved_reassign(amp, omega, sigma, K=3, normalize='column')
        r_raw = conserved_reassign(amp, omega, sigma, K=3, normalize='none')
        pk_in = amp[0, :, 0].max().item()
        tag = '整数频点' if f0 == int(f0) else '非整数频点'
        print(f'    {tag} f0={f0:.1f}: 输入峰 {pk_in:.3f} | '
              f'行+W_main {r_row[0, :, 0].max() / pk_in:.3f} | '
              f'柱平均 {r_col[0, :, 0].max() / pk_in:.3f} | '
              f'纯散射 {r_raw[0, :, 0].max() / pk_in:.2f}x')

    # 整数频点是锁频线 (50/100 Hz) 的情形，峰应尽量还原；非整数频点受
    # 整 bin 栅格限制，峰会分到相邻两格，天然低一截。
    amp, omega = build_line(64.0)
    sigma = torch.full((1, F, T), 0.35, dtype=torch.float64)
    pk_on = conserved_reassign(amp, omega, sigma, K=3,
                               w_main=W_main)[0, :, 0].max().item()
    assert 0.80 <= pk_on / amp[0, :, 0].max().item() <= 0.87

    amp, omega = build_line(64.3)
    pk_off = conserved_reassign(amp, omega, sigma, K=3,
                                w_main=W_main)[0, :, 0].max().item()
    assert 0.60 <= pk_off / amp[0, :, 0].max().item() <= 0.72

    # ── 2) 总能量：行归一应严格守恒到 1/W_main ─────────────────
    r_row = conserved_reassign(amp, omega, sigma, K=3, w_main=W_main)
    e_ratio = (r_row.sum() / amp.sum()).item()
    print(f'[2] 能量 Σout/Σin = {e_ratio:.4f}   '
          f'期望 1/W_main = {1 / W_main:.4f}   '
          f'相对误差 {abs(e_ratio * W_main - 1):.2e}')
    assert abs(e_ratio * W_main - 1) < 2e-3

    # ── 3) 对 ω 与 σ 的可微性 ─────────────────────────────────
    # 注意：行归一让 Σout 与 ω/σ 无关（质量守恒），所以损失必须取局部值。
    # 扰动点取 bin 64（ω=64.3，落在两 bin 之间，idx0 稳定）。
    for name in ('omega', 'sigma'):
        base = omega if name == 'omega' else sigma
        p = base.clone().requires_grad_(True)
        o = p if name == 'omega' else omega
        s = p if name == 'sigma' else sigma
        L = conserved_reassign(amp, o, s, K=3, w_main=W_main)[0, 64, 0]
        g_auto = torch.autograd.grad(L, p)[0][0, 64, 0].item()

        h = 1e-5
        vals = []
        for dh in (h, -h):
            q = base.clone()
            q[0, 64, 0] += dh
            oo = q if name == 'omega' else omega
            ss = q if name == 'sigma' else sigma
            vals.append(conserved_reassign(amp, oo, ss, K=3,
                                           w_main=W_main)[0, 64, 0].item())
        g_fd = (vals[0] - vals[1]) / (2 * h)
        rel = abs(g_auto - g_fd) / (abs(g_fd) + 1e-12)
        print(f'[3] d/d{name}: autograd {g_auto:+.6e}  有限差分 {g_fd:+.6e}  '
              f'相对误差 {rel:.2e}')
        assert abs(g_fd) > 1e-6, f'{name} 的有限差分≈0，测试点退化'
        assert rel < 1e-4, f'{name} 梯度与有限差分不符'

    # ── 4) σ 增大只应展宽、不应增峰 ───────────────────────────
    peaks = []
    for sg_val in (0.20, 0.35, 0.70, 1.20, 2.00):
        r = conserved_reassign(amp, omega, torch.full_like(sigma, sg_val),
                               K=int(math.ceil(3 * sg_val)), w_main=W_main)
        peaks.append(r[0, :, 0].max().item() / amp[0, :, 0].max().item())
    print('[4] 峰随 σ: ' + '  '.join(f'σ={s:.2f}→{p:.3f}'
                                     for s, p in zip((0.20, 0.35, 0.70, 1.20, 2.00), peaks)))
    assert all(peaks[i] >= peaks[i + 1] - 1e-6 for i in range(len(peaks) - 1))

    print('\n全部自检通过。')
