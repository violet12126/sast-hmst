"""
SAST v4 双窗前端
================

一次前向同时给出两套时频图：

    短窗 (hlength=512, ≈0.51 s)  |S_short|   中时间分辨率 —— 低频脉动 / 瞬态
    长窗 (hlength=1024, ≈1.02 s) |S_long|, ω_cont, q   高频率分辨率 —— 46.9 vs 50 Hz

两套窗共用同一个 FFT 长度 N=2000，因此频率轴完全一致 (0.5 Hz/档)，
差别只在时间分辨率与相位差分 IF 的稳定性上。

由此得到的三个逐像素量：

    ω_cont(f,t)  连续瞬时频率 (bin, 0-based)，不做 round —— 供 v4 的可微落点使用
    q(f,t)       局部相干度 = 垂直(±1 bin) × 时间(±20 帧) 的相位驻点共识
    g(f,t)       能量显著性 = E / (E + τ·median_f E)，用于压掉底噪

设计依据见 docs/SAST_v4_架构改造方案.md §二。
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from models.msst_torch import msst_stft_torch, phase_unwrap

# 与 v3 一致的窗形；只有长度是可变量
WINDOW_BETA = 0.32
# 256 -> 512 (2026-11-12, 外部会诊定稿): 256 窗 σ_f=4.87 Hz, 46/50 Hz 抑制系数
#   exp(-16/(2·4.87²))=0.71, 无法把 46 Hz 流动分量与 50 Hz 锁频线分开。512 窗
#   σ_f=2.43 Hz, 抑制系数 0.26; 而 σ_t 从 32.7 ms 升到 65.4 ms, 仍远小于转轮
#   旋转周期 (180 ms) 与涡带脉动周期 (500-2000 ms), 低频水力瞬态不受影响。
SHORT_HLENGTH = 512
LONG_HLENGTH = 1024

# q(f,t) 的平滑尺度：与 sast_inference.Q_IF_* 同源
Q_SIGMA_V = 1.0      # bin
Q_SIGMA_T = 1.5      # bin
Q_DT = 20            # 帧


def gaussian_window(hlength: int, device=None, dtype=torch.float64) -> torch.Tensor:
    ht = torch.linspace(-0.5, 0.5, hlength, device=device, dtype=dtype)
    return torch.exp(-math.pi / WINDOW_BETA ** 2 * ht ** 2)


def analytic_signal(x: torch.Tensor) -> torch.Tensor:
    """单边解析信号 (负频置零, 正频 ×2, 再 ÷2 回原标尺).

    WHY (2026-11-14, 0 Hz 拍频与跨带增益失配的根因):
      实信号 x = A·cos(2πf₀t) 含 +f₀ 与镜像 −f₀ 两个分量。短窗 (hl=512,
      σ_f=2.43 Hz) 下二者相距仅 2f₀, 在 f≈0 处强烈重叠:

          |S(0,t)| = A·H(f₀)·|cos(2πf₀t)|

      表现为 (a) 0 Hz 行出现 max/mean = 1.6~2.9 的"红青交替"拍频;
      (b) 低频峰值由 f₀ 搬到 0 Hz 并抬高 (实测 5/5 样本全图 vmax 都掉在
      0.0 Hz); (c) 短窗的 f₀ 行峰也因此抬到 (A/2)(1+H(2f₀)/Σh), 在
      f₀≈1.5 Hz 时是长窗的 1.4~1.5 倍 —— 这就是 T 的低频带系统性比
      高频带亮的原因。

      解析信号只有正频, H(f+f₀) 项消失, 上述三条同时消失。

    ÷2 的作用: 解析信号把单边幅度翻倍 (单音 A → A 而非 A/2), 除以 2 后
      恢复"单音幅度 A → 谱峰 A/2"的标尺。白噪声底噪也恰好回到原值
      (正频内容 ×4 能量, ÷2 幅度后 ×1), 因此本变换只去掉交叉项,
      不改变信噪比。

    Args:
        x: [B, N] 实数信号 (或 [N])
    Returns:
        [B, N] complex128
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    N = x.shape[-1]
    x64 = x.to(torch.float64)
    x64 = x64 - x64.mean(dim=-1, keepdim=True)          # 去静压偏置
    Xf = torch.fft.fft(x64, dim=-1)
    m = torch.zeros(N, dtype=torch.float64, device=x.device)
    m[1:N // 2] = 2.0                                    # 严格正频 ×2, 负频与 Nyquist 置零
    return torch.fft.ifft(Xf * m, dim=-1) * 0.5


def stft_complex(x: torch.Tensor, hlength: int,
                 fs: float = 1000.0,
                 normalize: bool = True) -> torch.Tensor:
    """复 STFT, 边界零填充 (与 msst_torch.msst_stft_torch 同口径).

    **按窗和归一**：msst_stft_torch 返回未归一化的谱，v3 只除以 N/2，
    导致幅值随窗长线性变化（hl=1024 是 hl=256 的 4.0 倍）。双窗合成
    `(1-β)|S_short| + β·R_long` 需要两套窗同标尺，故这里除以 Σh：
    单音幅度 A 在两套窗上都得到 A/2。相位不受标量除法影响。

    ``x`` 允许是解析信号那样的 complex128 —— msst_stft_torch 的 scatter
    缓冲本身就是 complex128, 实输入只是虚部为 0 的特例。

    Args:
        x: [B, N] 实数或复数信号
    Returns:
        [B, F, T] complex128, F = N//2
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    B, N = x.shape
    hl = hlength + 1 - (hlength % 2)
    h = gaussian_window(hl, device=x.device, dtype=torch.float64)
    Lh = (hl - 1) // 2
    neta = int(round(N / 2))
    xc = x if torch.is_complex(x) else x.double()
    outs = []
    for b in range(B):
        tfr = msst_stft_torch(xc[b], h, N, hl, Lh, neta, N)
        if normalize:
            tfr = tfr / h.sum()
        outs.append(tfr)
    return torch.stack(outs)


def window_mainlobe_mass(hlength: int, N: int = 2000,
                         device=None) -> float:
    """W_main = Σ_η |H(η)| / |H(0)| —— 窗主瓣吸引盆的等效 bin 数.

    与 v3 `SAST._window_mainlobe_mass` 逐位一致（同一高斯窗、同一
    `indices = (N+tau) % N` 放置、全谱求和含负频镜像）。它是重排的
    幅度标定常数：行归一后的输出除以它，干净谱线的峰就回到原始标尺。

    N=2000 / hlength=1024 时给出 6.12；hlength=256 时 24.5。
    """
    hl = hlength + 1 - (hlength % 2)
    h = gaussian_window(hl, device=device, dtype=torch.float64)
    Lh = (hl - 1) // 2
    buf = torch.zeros(N, dtype=h.dtype, device=h.device)
    tau = torch.arange(-Lh, Lh + 1, device=h.device)
    buf[(N + tau) % N] = h[Lh + tau]
    mag = torch.fft.fft(buf).abs()
    return float(mag.sum() / mag[0])


def continuous_if(tfr_complex: torch.Tensor, N: int) -> torch.Tensor:
    """连续瞬时频率 (bin, 0-based, 不做 round).

    ω(f,t) = (φ(f,t+1) − φ(f,t)) · N / 2π

    与 msst_torch.estimate_if 唯一的区别：那里随后 round 成 1-based 整数索引，
    这里保留连续值，让 v4 的重排落点对 Δω 可微。
    """
    angle = torch.angle(tfr_complex)
    unwrapped = phase_unwrap(angle, dim=-1)
    w = torch.diff(unwrapped, dim=-1) * N / (2.0 * math.pi)
    return torch.cat([w, w[..., -1:]], dim=-1)


def local_coherence(omega_bins: torch.Tensor,
                    sigma_v: float = Q_SIGMA_V,
                    sigma_t: float = Q_SIGMA_T,
                    dt: int = Q_DT) -> torch.Tensor:
    """逐像素相位驻点共识度 q(f,t) = q_vert · q_temp ∈ [0,1].

    q_vert: 相邻 bin 的 IF 差应很小 (脊柱是连贯的，不是随机游走)
    q_temp: 相隔 dt 帧的 IF 差应很小 (脊线是持续的，不是一闪而过)

    边界 (首末 bin / 首末 dt 帧) 置 0 —— 那里无法判断，按"不参与挤压"处理。
    """
    B, F, T = omega_bins.shape
    q = torch.ones_like(omega_bins)

    if F > 2:
        dv = omega_bins[:, 2:] - omega_bins[:, :-2]
        q[:, 1:F - 1] = torch.exp(-dv ** 2 / (2.0 * sigma_v ** 2))
    q[:, 0] = 0.0
    q[:, -1] = 0.0

    if T > 2 * dt:
        dw = omega_bins[:, :, 2 * dt:] - omega_bins[:, :, :-2 * dt]
        qt = torch.exp(-dw ** 2 / (2.0 * sigma_t ** 2))
        # 两端各 dt 帧没有跨距可比
        full = torch.ones_like(omega_bins)
        full[:, :, dt:T - dt] = qt
        full[:, :, :dt] = 0.0
        full[:, :, T - dt:] = 0.0
        q = q * full
    return q


def noise_floor_gate(mag: torch.Tensor, tau: float = 3.0,
                     eps: float = 1e-12) -> torch.Tensor:
    """能量显著性门控 E / (E + τ·median_f E) ∈ [0,1).

    底噪 (E 远低于该帧频率中位数) → 0；真实谱线 → 接近 1。
    在能量域做，与"不相关噪声能量可加"一致。
    """
    E = mag ** 2
    med = E.median(dim=1, keepdim=True).values
    return E / (E + tau * med + eps)


def alpha_short(freqs: torch.Tensor, f_lo: float = 3.0,
                f_hi: float = 9.0) -> torch.Tensor:
    """短窗基准权重 α(f) ∈ [0,1] —— 分频带基准窗.

      S_base(f,t) = α(f)·|S_short| + (1-α(f))·|S_long|

      α = 1 于 f <= f_lo (涡带区, 用短窗保展宽与时间分辨率),
      α = 0 于 f >= f_hi (转轮旋转线/独立水力峰区, 用长窗保频率分辨率),
      中间用升余弦过渡。

    过渡带从 (18, 35) 下移到 (3, 9) 的依据 (2026-11-14):
      短窗 σ_f = 2.43 Hz, 主瓣宽约 6 Hz。6.5 / 8.33 / 10 / 15 Hz 这些独立峰
      在 α=1 下被涂抹成一团 (实测 5-18 Hz 剖面"半高以上局部峰数"= 1)。
      下移后 6.5 Hz 处 α=0.37、8.33 Hz 处 α=0.03, 实测半高峰数升到 2~3、
      峰/中位提升 30~50%; 而 0-3 Hz 涡带区完全不动, 展宽与脉动保住。
      两套窗对单音的幅度响应在解析信号下处处严格相等 (都是 A/2),
      因此任何频率上切换底图都无接缝。
    """
    f = freqs.to(dtype=torch.float64)
    t = ((f - f_lo) / (f_hi - f_lo)).clamp(0.0, 1.0)
    # 升余弦 1 -> 0: t=0 -> 1, t=1 -> 0
    a = 0.5 * (1.0 + torch.cos(math.pi * t))
    return a.to(dtype=torch.float32)


def beta_ramp(freqs: torch.Tensor, f_lo: float = 10.0,
              f_hi: float = 12.0) -> torch.Tensor:
    """β 爬坡权重 (0 → 1), 与 ``alpha_short`` **解耦**.

    WHY 解耦 (2026-11-14): 原实现让 β 的斜坡复用 ``1-α(f)``。α 已下移到
    3-9 Hz, 9 Hz 以上 1-α 恒为 1, 若继续复用, β 会在斜坡起点一步跳到
    w_BPF, 切出一条硬边。故使用独立的升余弦: f_lo 处 0 (与左侧 β≡0 连续),
    f_hi 处 1 (完全交给 w_BPF), 两端之外由 ``clamp`` 保持不变。

    默认区间 (10, 12) Hz 是与 ``models/sast_v4.py`` 的 LOW_BETA_MAX_HZ /
    BETA_RAMP_HI_HZ 同源的定稿值 —— 取舍依据见那边的常量注释 (10-18 Hz 的
    R_long/S_base ≈ 0.16, β 在该带是对比度旋钮而非锐化旋钮, 必须在 12 Hz
    前饱和才能呈现 R_long 的观感)。
    """
    f = freqs.to(dtype=torch.float64)
    t = ((f - f_lo) / (f_hi - f_lo)).clamp(0.0, 1.0)
    r = 0.5 * (1.0 - torch.cos(math.pi * t))
    return r.to(dtype=torch.float32)


def dual_frontend(x: torch.Tensor, fs: float = 1000.0,
                  short_hlength: int = SHORT_HLENGTH,
                  long_hlength: int = LONG_HLENGTH,
                  need_q: bool = True,
                  analytic: bool = True,
                  beta_ramp_lo: float = 10.0,
                  beta_ramp_hi: float = 12.0) -> Dict[str, torch.Tensor]:
    """双窗前端.

    Args:
        x: [B, N] 实数信号
        analytic: 是否先转解析信号 (见 ``analytic_signal``)。默认 True;
                  置 False 可复现 2026-11-14 之前的"实信号"口径 (消融对照)。
    Returns:
        S_short  [B,F,T] 短窗幅值
        S_long   [B,F,T] 长窗幅值
        S_base   [B,F,T] 分频带底图 = α(f)·S_short + (1-α(f))·S_long
        alpha    [F]     短窗基准权重 (窗口切换)
        beta_ramp[F]     β 爬坡权重 (0→1), 区间由 beta_ramp_lo/hi 给出,
                         默认 10→12, 与 alpha 解耦
        omega    [B,F,T] 长窗连续 IF (bin, 0-based)
        q        [B,F,T] 局部相干度 (need_q=False 时不返回)
        w_main   float   长窗主瓣等效 bin 数 (重排幅度标定)
        freqs    [F] Hz
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    B, N = x.shape
    if analytic:
        x = analytic_signal(x)

    S_long_c = stft_complex(x, long_hlength, fs)
    S_short_c = stft_complex(x, short_hlength, fs)

    omega = continuous_if(S_long_c, N)
    S_long = S_long_c.abs().float()
    S_short = S_short_c.abs().float()
    freqs = torch.arange(S_long_c.shape[1], device=x.device,
                         dtype=torch.float32) / N * fs
    alpha = alpha_short(freqs)                                    # [F]
    ramp = beta_ramp(freqs, beta_ramp_lo, beta_ramp_hi)           # [F]
    w_main = window_mainlobe_mass(long_hlength, N, device=x.device)
    # S_base 的长窗分量**不**除 W_main —— 两套窗在同一标尺上 (单音都峰到 A/2),
    # 底图必须保持原始 STFT 标尺。
    #
    # WHY 撤销 2026-11-14 上午的 ÷W_main: 那次改动的目标是消除 c0 的"长方形
    # 垂直缺口", 但缺口的真因是 node_presence 用了 0.2 s 滑动窗导致 β 逐帧
    # 翻转 (已由整段均值修复, β 在 2 s 内恒定)。而 ÷W_main 的副作用是:
    #   (a) 在 18-35 Hz 过渡带切出 6.57 倍的底噪台阶 (旧口径 1.08 倍);
    #   (b) 一旦某带被门控判成缺席 (β 低), 该带 S_base 只剩 S_long/6.109,
    #       真实谱线被压低 6.1 倍 —— 实测 c0 的 95-105 Hz 带亮度从 0.18
    #       掉到 0.04, 与 FFT 完全不符。
    # β 对背景明暗的调制本身就是设计语义 (挤压把底噪按 1/W_main 压掉),
    # 不该用缩放底图去抹平。
    S_base = (alpha.view(1, -1, 1) * S_short
              + (1.0 - alpha).view(1, -1, 1) * S_long)

    out = {
        'S_long': S_long,
        'S_short': S_short,
        'S_base': S_base,
        'alpha': alpha,
        'beta_ramp': ramp,
        'omega': omega.float(),
        'w_main': w_main,
        'freqs': freqs,
    }
    if need_q:
        out['q'] = local_coherence(omega).float()
    return out


if __name__ == '__main__':
    # 自检：两种窗的时间/频率分辨率与 q 在噪声上的地板
    import numpy as np
    torch.manual_seed(0)
    for hl in (SHORT_HLENGTH, LONG_HLENGTH):
        st = WINDOW_BETA / math.sqrt(2 * math.pi) * hl / 1000.0
        print(f'hlength={hl:4d}  等效时间 std={st * 1000:6.1f} ms  '
              f'频率 std={1 / (2 * math.pi * st):5.2f} Hz')

    # 两套窗必须同标尺：单音 A=1 的峰都应 ≈ A/2
    t = torch.arange(2000, dtype=torch.float64) / 1000.0
    tone = torch.cos(2 * math.pi * 50.0 * t).unsqueeze(0)
    o = dual_frontend(tone)
    p_short = o['S_short'].max().item()
    p_long = o['S_long'].max().item()
    print(f'单音峰: S_short {p_short:.4f}  S_long {p_long:.4f}  '
          f'比值 {p_long / p_short:.4f} (应为 1.000)')
    print(f'W_main(long) = {o["w_main"]:.3f}')
    assert abs(p_long / p_short - 1.0) < 1e-3

    x = torch.randn(2, 2000)
    out = dual_frontend(x)
    print(f"S_short {tuple(out['S_short'].shape)}  S_long {tuple(out['S_long'].shape)}")
    print(f"q 均值 {out['q'].mean().item():.3f}  "
          f"q 中位 {out['q'].median().item():.3f}  (纯噪声应偏低)")
    g = noise_floor_gate(out['S_long'])
    print(f"噪声显著性门控 均值 {g.mean().item():.3f}")