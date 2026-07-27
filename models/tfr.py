"""
TFR: Time-Frequency Representation 计算模块
============================================
包含 STFT, SST, WSST, WSST2, HMST 及其 IF 估计子程序。

参考实现:
  - Wsst2_new.m  (Pham & Meignen, 2015): CWT 域二阶 SST
  - sst2_new.m   (Pham & Meignen):       STFT 域二阶 SST
  - sstn.m       (Oberlin & Meignen):    STFT 域 1-4 阶 SST
  - MSST_Y_new.m (Yu et al.):            MSST 多重挤压

所有 TFR 函数接受 numpy 或 torch 输入，返回 numpy 数组，
便于直接用于绘图和诊断。
"""
import numpy as np
import torch
import math
from scipy.interpolate import interp1d


# ============================================================
# 0. 工具函数
# ============================================================

def to_numpy(x):
    """torch tensor → numpy array。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def renyi_entropy(tfr, alpha=3):
    """Rényi entropy (α=3), lower = more concentrated."""
    tfr = np.asarray(tfr).ravel()
    tfr_n = tfr / (tfr.sum() + 1e-12)
    tfr_n = tfr_n[tfr_n > 1e-12]
    if len(tfr_n) == 0:
        return float('inf')
    return 1 / (1 - alpha) * np.log2((tfr_n ** alpha).sum())


# ============================================================
# 1. WSST2: CWT 域二阶同步压缩 (Wsst2_new.m 直译)
# ============================================================

def _cmor(Fb, Fc, xi):
    """
    Complex Morlet wavelet in frequency domain.
    MATLAB cmor.m 直译.

    Args:
        Fb: bandwidth parameter
        Fc: center frequency
        xi: frequency grid (normalized, as in Wsst2_new.m: xi = arange(N)/2)

    Returns:
        psih:   wavelet in freq domain
        dpsih:  d(psih)/dxi
        ddpsih: d²(psih)/dxi²
    """
    psih = np.sqrt(Fb) * np.exp(-Fb**2 * np.pi * (xi - Fc)**2)
    dpsih = -2 * Fb**2 * np.pi * (xi - Fc) * psih
    ddpsih = 2 * np.pi * Fb**2 * (
        2 * np.pi * Fb**2 * Fc**2
        - 4 * np.pi * Fb**2 * Fc * xi
        + 2 * np.pi * Fb**2 * xi**2
        - 1
    ) * psih
    return psih, dpsih, ddpsih


def _morlet(xi, omega0=6.0):
    """
    Morlet wavelet in frequency domain.

    ψ̂(ξ) = π^{-1/4} · exp(-(ξ - ω₀)² / 2)

    Args:
        xi:     frequency grid
        omega0: center frequency (default 6.0)

    Returns:
        psih:   wavelet in freq domain
        dpsih:  d(psih)/dxi
    """
    pi14 = np.pi ** (-0.25)
    psih = pi14 * np.exp(-(xi - omega0)**2 / 2.0)
    dpsih = -(xi - omega0) * psih
    return psih, dpsih


def _morse(xi, beta=20.0, gamma=3.0):
    """
    Analytic Morse wavelet in frequency domain (MATLAB 'amor' default).

    ψ̂_{β,γ}(ξ) = U(ξ) · a_{β,γ} · ξ^β · exp(-ξ^γ)

    where:
      - U(ξ) is the Heaviside step (ξ > 0 only, analytic)
      - a_{β,γ} = 2 · (e·γ/β)^{β/γ}  (peak normalization)
      - β: asymmetry parameter (default 20)
      - γ: compactness parameter (default 3)

    Ref: Lilly & Olhede (2012), "Generalized Morse Wavelets"

    Args:
        xi:    frequency grid (positive half, same convention as _cmor)
        beta:  asymmetry parameter (default 20)
        gamma: compactness parameter (default 3)

    Returns:
        psih:   wavelet in freq domain (zero for xi <= 0)
        dpsih:  d(psih)/dxi
    """
    psih = np.zeros_like(xi, dtype=complex)
    dpsih = np.zeros_like(xi, dtype=complex)

    pos = xi > 0
    x = xi[pos]
    # Normalization constant
    a_bg = 2.0 * (np.exp(1.0) * gamma / beta) ** (beta / gamma)
    psih[pos] = a_bg * x**beta * np.exp(-x**gamma)
    # Derivative: dψ̂/dξ = ψ̂ · (β/ξ - γ·ξ^{γ-1})
    dpsih[pos] = psih[pos] * (beta / x - gamma * x**(gamma - 1.0))
    return psih, dpsih


def _mypad(s):
    """
    Symmetric padding to next power of 2.
    MATLAB mypad.m 直译: N = 2^(1+round(log2(n+eps)))

    Returns:
        N:   padded length (power of 2)
        sx:  padded signal
        n1:  number of samples padded at left
    """
    n = len(s)
    N = 2 ** (1 + int(np.round(np.log2(n + np.finfo(float).eps))))
    n1 = (N - n) // 2
    # symmetric padding
    sx = np.pad(s, (n1, N - n - n1), mode='symmetric')
    return N, sx.astype(np.float64), n1


def wsst2(s, fs, gamma=0.01, mywav='cmor2-1', nv=32):
    """
    Wavelet-based Second-Order Synchrosqueezing Transform.

    MATLAB Wsst2_new.m (Pham & Meignen, 2015) 的 Python 直译。
    参考文献:
      Pham, D-H., Meignen, S. "Second-Order Synchrosqueezing Transform:
      The Wavelet Case and Comparisons." IEEE TSP, 2015.

    算法:
      1. 复 Morlet 小波 CWT (频域快速卷积)
      2. 一阶 IF: ω̂₁ = Re(V^{ξψ} / (a·V^ψ))
      3. 群延迟:   τ = Re(a·V^{ψ'} / (2iπ·V^ψ))
      4. 二阶 chirp 率: q̂ = 2iπ/a²·(V^{ξ²ψ}V^ψ - (V^{ξψ})²)/(V^ψ²+V^{ξψ'}V^ψ-V^{ψ'}V^{ξψ})
      5. 二阶 IF: ω̂₂ = Re(ω̂₁ - q̂·τ)
      6. 挤压: k = round(1 + nv·log₂(ω̂))

    Args:
        s:       [T] 1D 信号 (numpy array)
        fs:      采样率 (Hz)
        gamma:   小波系数阈值 (默认 0.01)
        mywav:   小波类型, 如 'cmor2-1' (Fb=2, Fc=1)
        nv:      每倍频程尺度数 (默认 32)

    Returns:
        dict:
          'WT':     [na, T] 复数 CWT
          'WSST':   [na, T] 一阶 SST (复数)
          'WSST2':  [na, T] 二阶 SST (复数)
          'freqs':  [na] 频率轴 (Hz), 对数间隔
          'omega':  [na, T] 一阶 IF (归一化频率)
          'omega2': [na, T] 二阶 IF (归一化频率)
          'scales': [na] 尺度数组
    """
    s = np.asarray(s, dtype=np.float64).ravel()
    n = len(s)

    # ── 解析小波参数 ──
    if mywav.startswith('cmor'):
        parts = mywav[4:].split('-')
        Fb = float(parts[0])
        Fc = float(parts[1])
        wav_type = 'cmor'; omega0 = None; beta = gamma_m = None
    elif mywav.startswith('morl'):
        if len(mywav) > 4:
            omega0 = float(mywav[4:])
        else:
            omega0 = 6.0
        wav_type = 'morl'; Fb = Fc = None; beta = gamma_m = None
    elif mywav.startswith('morse'):
        # 'morse' or 'morse20-3' → beta=20, gamma=3 (MATLAB default)
        if len(mywav) > 5:
            parts = mywav[5:].split('-')
            beta = float(parts[0])
            gamma_m = float(parts[1])
        else:
            beta = 20.0; gamma_m = 3.0
        wav_type = 'morse'; Fb = Fc = None; omega0 = None
    else:
        raise ValueError(f"Unsupported wavelet: {mywav}")

    # ── 尺度网格 (MATLAB L31: noct = log2(n)-1, +1 octave 以覆盖 500 Hz) ──
    noct = np.log2(n)                    # n=1000 → noct≈9.97
    na = int(np.floor(noct * nv + 1))    # floor(9.97*32+1)=319
    as_ = (2 ** (-1.0 / nv)) ** np.arange(na)

    # ── 对称填充到 2 的幂 ──
    N, sx, n1 = _mypad(s)

    # ── FFT ──
    xh = np.fft.fft(sx)
    xh[N//2+1:] = 0    # 切断负频率，构造解析信号频谱（避免高频混叠）
    xi = np.arange(N) / 2.0   # 归一化频率轴 (匹配 MATLAB: xi = (0:N-1)/2)

    # ── 初始化 (omega/tau 存复数, 循环后取 real — 匹配 MATLAB L99-100) ──
    WT = np.zeros((na, N), dtype=complex)
    omega = np.zeros((na, N), dtype=complex)
    tau = np.zeros((na, N), dtype=complex)
    phipp = np.zeros((na, N), dtype=complex)
    omega2 = np.zeros((na, N))

    # ── 逐尺度 CWT + IF 估计 (MATLAB L76-95) ──
    norm2psi = np.zeros(na)
    for ai in range(na):
        a = as_[ai]

        # 小波频域响应
        if wav_type == 'cmor':
            psih, dpsih, _ = _cmor(Fb, Fc, a * xi)
        elif wav_type == 'morl':
            psih, dpsih = _morlet(a * xi, omega0)
        elif wav_type == 'morse':
            psih, dpsih = _morse(a * xi, beta, gamma_m)

        norm2psi[ai] = np.linalg.norm(psih)  # MATLAB L81

        # CWT 矩变换 (频域快速卷积) — MATLAB L83-87
        Wtmp   = np.fft.ifft(np.conj(psih) * xh)               # V^ψ
        Wnu    = np.fft.ifft(np.conj(a * xi * psih) * xh)      # V^{ξ·ψ}
        Wnunu  = np.fft.ifft(np.conj((a * xi)**2 * psih) * xh) # V^{ξ²·ψ}
        Wp     = np.fft.ifft(np.conj(dpsih) * xh)               # V^{ψ'}
        Wnup   = np.fft.ifft(np.conj(a * xi * dpsih) * xh)     # V^{ξ·ψ'}

        WT[ai, :] = Wtmp

        # 一阶 IF: ω̂₁ = 1/a · V^{ξψ}/V^ψ  — MATLAB L91 (complex)
        omega[ai, :] = 1.0 / a * Wnu / Wtmp

        # 群延迟: τ = a·V^{ψ'} / (2iπ·V^ψ)  — MATLAB L92 (complex)
        tau[ai, :] = a * Wp / Wtmp / 2.0 / 1j / np.pi

        # 二阶 chirp 率: q̂ = 2iπ/a² · (V^{ξ²ψ}V^ψ - (V^{ξψ})²) / (V^ψ²+V^{ξψ'}V^ψ-V^{ψ'}V^{ξψ})
        # MATLAB L93 — 不做正则化 (MATLAB L97 是被注释掉的)
        num = Wnunu * Wtmp - Wnu**2
        den = Wtmp**2 + Wnup * Wtmp - Wp * Wnu
        phipp[ai, :] = 2j * np.pi / a**2 * num / den

    # ── 取实部 (消除虚部交叉项, 必须在 omega2 之前) ──
    omega = np.real(omega)    # MATLAB L99
    tau = np.real(tau)        # MATLAB L100

    # tau 正则化: 群延迟不可靠时零化 chirp 率, 抑制竖条纹
    # MATLAB L97: phipp(abs(real(tau)*n)<1)=0 (原始被注释, 论文图启用此正则化)
    mask_tau = np.abs(tau) * n < 1.0
    phipp[mask_tau] = 0

    # 二阶 IF: ω̂₂ = Re(ω̂₁) - Re(q̂·τ)  — 匹配 Wsst2_new.m L98
    # CWT 域 τ = b - t_inst (与 STFT 域 sstn.m 符号相反):
    #   ω₂ = ω₁ - q·(b - t_inst) = ω₁ + q·(t_inst - b) = Taylor 展开结果
    # τ 已是实数, Re(q·τ) = Re(q)·τ (无虚部交叉项)
    # MATLAB L97: %phipp(abs(real(tau)*n)<1)=0;  (被注释, 不执行)
    omega2 = omega - np.real(phipp * tau)
    # ── 去除填充 (MATLAB L103-108) ──
    WT     = WT[:, n1:n1 + n]
    omega  = omega[:, n1:n1 + n]
    tau    = tau[:, n1:n1 + n]
    phipp  = phipp[:, n1:n1 + n]
    omega2 = omega2[:, n1:n1 + n]

    # ── 阈值处理 (MATLAB L116-121) ──
    omega[omega < 0] = np.nan
    omega[np.abs(WT) < gamma] = np.nan
    tau[np.abs(WT) < gamma] = np.nan
    omega2[omega2 < 0] = np.nan
    omega2[np.abs(WT) < gamma] = np.nan
    phipp[np.abs(WT) < gamma] = np.nan

    # ── 挤压 (MATLAB L124-148) ──
    WSST  = np.zeros((na, n), dtype=complex)
    WSST2 = np.zeros((na, n), dtype=complex)

    for b in range(n):
        for ai in range(na):
            if np.isnan(omega[ai, b]):
                continue
            # MATLAB L130: 自适应阈值 (per-scale norm2psi)
            thresh = 2 * gamma * np.sqrt(2) * norm2psi[ai] / np.sqrt(2 * n)
            if np.abs(WT[ai, b]) <= thresh:
                continue

            # WSST2: squeeze to 2nd-order IF  — MATLAB L132-135
            if not np.isnan(omega2[ai, b]):
                k2 = int(np.round(1 + nv * np.log2(omega2[ai, b])))
                if 1 <= k2 <= na:
                    WSST2[k2 - 1, b] += WT[ai, b]

            # WSST: squeeze to 1st-order IF  — MATLAB L142-145
            k1 = int(np.round(1 + nv * np.log2(omega[ai, b])))
            if 1 <= k1 <= na:
                WSST[k1 - 1, b] += WT[ai, b]

    # 归一化 (MATLAB L150-151)
    WSST  = WSST / nv * np.log(2)
    WSST2 = WSST2 / nv * np.log(2)

    # ── 频率轴 (Hz) ──
    # MATLAB: fs_out = 1./as, 即 f_k = 2^{(k-1)/nv}
    # 物理频率: 2·fs/N · 2^{(k-1)/nv}  (k=1..na, MATLAB 1-indexed)
    freqs_hz = 2.0 * fs / N * 2.0 ** (np.arange(na) / nv)

    return {
        'WT': WT,
        'WSST': WSST,
        'WSST2': WSST2,
        'freqs': freqs_hz.astype(np.float64),
        'omega': omega,
        'omega2': omega2,
        'scales': as_,
    }


# ============================================================
# 2. STFT-based tools
# ============================================================
# 2a. STFT 域二阶 SST (sstn.m 直译, Oberlin & Meignen)
# ============================================================

def sst2_stft(s, fs, gamma=0.001, sigma=0.04):
    """
    STFT 域二阶同步压缩变换 (vertical second-order SST).

    MATLAB sstn.m (Oberlin & Meignen) 的 Python 直译 — 仅 2 阶 (SST2)。
    使用高斯窗及其时间矩。

    算法:
      1. 高斯窗 g(t) = 1/σ·exp(-π·t²/σ²), gp = g'
      2. 6 个 STFT: V^{t^i·g}, V^{t^i·gp} (i=0,1,2)
      3. 一阶 IF: ω̂ = ω - Im(V^{gp}/(2π·V^g))
      4. 群延迟:  τ̂ = V^{t·g}/V^g
      5. 二阶 chirp 率: q̂ = W2/Y22
      6. 二阶 IF: ω̂₂ = ω̂ + q̂·τ̂
      7. 挤压: k = round(1 + ω̂)

    Args:
        s:      [T] 1D 信号 (numpy array)
        fs:     采样率 (Hz)
        gamma:  幅值阈值 (默认 0.001)
        sigma:  高斯窗参数 (默认 0.04)

    Returns:
        dict:
          'STFT':   [F, T] 复数 STFT
          'SST1':   [F, T] 一阶 SST (复数)
          'SST2':   [F, T] 二阶 SST (复数)
          'freqs':  [F] 频率轴 (Hz)
          't':      [T] 时间轴 (s)
          'omega':  [F, T] 一阶 IF (频率 bin 索引)
          'omega2': [F, T] 二阶 IF (频率 bin 索引)
    """
    s = np.asarray(s, dtype=np.float64).ravel()
    n = len(s)

    # ── 填充 (MATLAB sstn.m L42-47: 零填充 n/2 两侧) ──
    n_pad = n // 2
    x_pad = np.concatenate([np.zeros(n_pad), s, np.zeros(n - n_pad)])
    # 确保 x_pad 长度至少为 2n (匹配 MATLAB 行为)
    if len(x_pad) < 2 * n:
        x_pad = np.concatenate([x_pad, np.zeros(2 * n - len(x_pad))])

    # ── 时间轴与窗 (MATLAB L50-53) ──
    t_win = np.arange(n) / n - 0.5  # t = -0.5 : 1/n : 0.5-1/n
    g = 1.0 / sigma * np.exp(-np.pi / sigma**2 * t_win**2)        # g
    gp = -2.0 * np.pi / sigma**2 * t_win * g                       # g'
    tg = t_win * g                                                  # t·g
    t2g = t_win**2 * g                                              # t²·g
    t2gp = t_win**2 * gp                                            # t²·gp

    # ── STFT (hop_length=1, n_fft=n) ──
    neta = n // 2  # 单侧频率 bin 数
    nb = n         # 时间帧数

    STFT = np.zeros((neta, nb), dtype=complex)
    SST1 = np.zeros((neta, nb), dtype=complex)
    SST2 = np.zeros((neta, nb), dtype=complex)
    omega = np.zeros((neta, nb))
    omega2 = np.zeros((neta, nb))

    # 预计算 FFT 频率索引 (MATLAB: ft = 1:n/2)
    ft = np.arange(1, neta + 1)  # 1-indexed 频率 bin

    for b in range(nb):
        # 取 n 点片段 (MATLAB L83: x(bt(b):bt(b)+n-1))
        seg = x_pad[b:b + n]

        # STFT with windows t^i * g  (MATLAB L85-88, i=0,1,2)
        vg0 = np.fft.fft(seg * g) / n
        vg1 = np.fft.fft(seg * tg) / n
        vg2 = np.fft.fft(seg * t2g) / n

        # STFT with windows t^i * gp (MATLAB L91-94, i=0,2 needed for W2)
        vgp0 = np.fft.fft(seg * gp) / n
        vgp2 = np.fft.fft(seg * t2gp) / n

        # 取正频率部分 (ft indices, 1-indexed → 0-indexed python)
        Vg0 = vg0[ft]
        Vg1 = vg1[ft]
        Vg2 = vg2[ft]
        Vgp0 = vgp0[ft]
        Vgp2 = vgp2[ft]

        # STFT 存储 (补偿 1/2 平移相位, MATLAB L140)
        STFT[:, b] = Vg0 * np.exp(1j * np.pi * (ft - 1))

        # 一阶 IF (MATLAB L119): omega = (ft-1) - real(Vgp0/(2iπ)/Vg0)
        omega[:, b] = (ft - 1) - np.real(Vgp0 / (2j * np.pi) / Vg0)

        # 二阶群延迟 (MATLAB L97): tau2 = Vg1/Vg0
        tau2 = Vg1 / Vg0

        # 二阶 chirp 率 (MATLAB L113-114, L123)
        # W2 = 1/(2iπ)*(Vg0² + Vg0*Vgp2 - Vg2*Vgp0)
        W2 = 1.0 / (2j * np.pi) * (Vg0**2 + Vg0 * Vgp2 - Vg2 * Vgp0)
        # Y22 = Vg0*Vg3 - Vg2²  ... but we don't have Vg3.
        # Actually sstn.m uses Y(:,2,2) = vg1*vg3 - vg2*vg2 where vg(:,i) has i starting from 1.
        # vg(:,1)=Vg0, vg(:,2)=Vg1(with i=1: t^1*g), vg(:,3)=Vg2(with i=2: t^2*g)
        # Wait — Y(:,i,j) = vg(:,1)*vg(:,i+1) - vg(:,j)*vg(:,i-j+2)
        # Y(:,2,2) = vg(:,1)*vg(:,3) - vg(:,2)*vg(:,2) = Vg0*Vg2 - Vg1^2
        Y22 = Vg0 * Vg2 - Vg1**2
        phi22p = W2 / Y22

        # 二阶 IF (MATLAB L124): omega2 = omega + real(phi22p * tau2)
        # NOTE: + sign — matches Taylor expansion ω₂ = ω₁ + q·τ
        omega2[:, b] = omega[:, b] + np.real(phi22p * tau2)

    # ── 归一化 (MATLAB L144) ──
    STFT = STFT * sigma * 2.0

    # ── 挤压 (MATLAB L146-176) ──
    for b in range(nb):
        for eta in range(neta):
            if np.abs(STFT[eta, b]) <= 0.001 * gamma:
                continue

            # SST1: k = 1 + round(omega)
            k1 = int(np.round(1 + omega[eta, b]))
            if 1 <= k1 <= neta:
                SST1[k1 - 1, b] += STFT[eta, b]

            # SST2: k = 1 + round(omega2)
            k2 = int(np.round(1 + omega2[eta, b]))
            if 1 <= k2 <= neta:
                SST2[k2 - 1, b] += STFT[eta, b]

    # ── 频率轴 (Hz) ──
    freqs_hz = (ft - 1) / n * fs  # MATLAB: ft-1 gives frequency in bins

    # ── 时间轴 (s) ──
    t_axis = np.arange(nb) / fs

    return {
        'STFT': STFT,
        'SST1': SST1,
        'SST2': SST2,
        'freqs': freqs_hz.astype(np.float64),
        't': t_axis,
        'omega': omega,
        'omega2': omega2,
    }


# ============================================================
# 2c. MSST — Multi-Synchrosqueezing Transform (STFT domain)
# ============================================================

def msst(x, fs, hlength=None, num=3, save_trajectory=False):
    """
    Multi-Synchrosqueezing Transform (MSST) — STFT 域迭代 IF 精化 + 单次挤压.

    MATLAB MSST_Y_new.m (Yu et al., IEEE TIE 2019, eq 31) 的 Python 直译。

    算法:
      1. Gaussian-windowed STFT (hop=1, N_fft=N, 仅在正频率)
      2. 一阶 IF 估计: omega = round(diff(unwrap(angle(STFT))) · N/(2π))
      3. IF 迭代精化 (num-1 次):
         omega2(η, b) = omega(omega(η, b), b)
         本质: 把每个 bin 的 IF 估计"挤"到它自己指向的位置, 再读取那里的 IF
      4. 最终单次硬挤压到精化后的 IF 位置

    与 sst2_stft 的区别:
      - sst2_stft: 二阶 chirp 率修正, 单次挤压
      - msst:      一阶 IF + 迭代精化, 最终单次挤压
      - msst 不修正 chirp 率, 而是通过迭代逐步逼近真值

    Args:
        x:        [T] 1D 信号
        fs:       采样率 (Hz)
        hlength:  窗长 (样本数)。默认 round(T/8), 自动调整为奇数。
        num:      迭代次数 (≥1)。num=1 → 等价于标准一阶 SST。
        save_trajectory: 是否保存中间迭代的 omega (默认 False).
                        若 True, 返回值包含 'omegas': list of [F, T] 长度=num.

    Returns:
        dict:
          'MSST':        [F, T] MSST 幅度谱
          'STFT':        [F, T] 复数 STFT (归一化后)
          'freqs':       [F] 频率轴 (Hz)
          't':           [T] 时间轴 (s)
          'omega_final': [F, T] 最终 IF 估计 (1-indexed bin, 0=无效)
          'omegas':      (仅 save_trajectory=True) list of [F, T],
                         长度=num, omegas[k]=第 k 次迭代后的 IF.
                         omegas[0]=一阶 IF, omegas[-1]=omega_final.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = len(x)

    # ── 窗长默认值 ──
    # 与 ssqueezepy 对齐: win_len = min(N, 512)
    # N/8 太短, 对低频 (如 12Hz) 分辨率不足, 导致脊线呈波浪形
    if hlength is None:
        hlength = min(N, 512)

    # 确保奇数 (MATLAB: hlength+1-rem(hlength,2))
    hlength = hlength + 1 - (hlength % 2)

    # Gaussian window: h = exp(-π/0.32² · t²), t ∈ [-0.5, 0.5]
    ht = np.linspace(-0.5, 0.5, hlength)
    h = np.exp(-np.pi / 0.32**2 * ht**2)
    Lh = (hlength - 1) // 2

    tcol = N                       # hop=1 — 每样本一帧
    neta = int(round(N / 2))       # 正频率 bin 数

    # ═══════════════════════════════════════════════════════════
    # Step 1: STFT (MATLAB 的直接窗式 FFT)
    # ═══════════════════════════════════════════════════════════
    # 预分配 N×tcol 缓冲区 (MATLAB 在赋值时 auto-expand)
    tfr_pre = np.zeros((N, tcol), dtype=np.complex128)

    for icol in range(tcol):
        ti = icol  # 0-indexed, 对应 MATLAB t(icol)

        # tau 范围: 对称窗, 边界处截断
        tau_min = -min(neta - 1, Lh, ti)
        tau_max = min(neta - 1, Lh, N - 1 - ti)

        if tau_min > tau_max:
            continue

        tau = np.arange(tau_min, tau_max + 1)

        # MATLAB: indices = rem(N+tau, N) + 1  → Python: (N+tau) % N
        indices = (N + tau) % N

        # 信号片段 + 窗共轭
        rSig = x[ti + tau]
        win_idx = Lh + tau  # MATLAB: h(Lh+1+tau), h 为实值
        tfr_pre[indices, icol] = rSig * np.conj(h[win_idx])

    # FFT 沿频率轴 (axis=0), 取正频率
    tfr = np.fft.fft(tfr_pre, axis=0)
    tfr = tfr[:neta, :]  # 复数 STFT

    # ═══════════════════════════════════════════════════════════
    # Step 2: 一阶 IF 估计 (相位差分)
    # ═══════════════════════════════════════════════════════════
    omega = np.zeros((neta, tcol - 1))
    for i in range(neta):
        phase = np.unwrap(np.angle(tfr[i, :]))
        omega[i, :] = np.diff(phase) * N / (2.0 * np.pi)

    # 末列填充 + 四舍五入到整数 bin (MATLAB: omega=round(omega))
    omega = np.column_stack([omega, omega[:, -1]])
    omega = np.round(omega).astype(np.int32)

    # ═══════════════════════════════════════════════════════════
    # Step 3: IF 迭代精化 (MSST 核心)
    # ═══════════════════════════════════════════════════════════
    omegas = [omega.copy()]  # omega_0: 一阶 IF

    if num > 1:
        omega2 = np.zeros((neta, tcol), dtype=np.int32)
        for _ in range(num - 1):
            # omega2(η, b) = omega(k, b)  where k = omega(η, b)
            valid = (omega >= 1) & (omega <= neta)
            eta_idx, b_idx = np.nonzero(valid)
            k_vals = omega[eta_idx, b_idx] - 1  # → 0-indexed
            omega2[eta_idx, b_idx] = omega[k_vals, b_idx]
            omega = omega2.copy()
            omegas.append(omega.copy())  # save trajectory
    else:
        omega2 = omega.copy()

    # ═══════════════════════════════════════════════════════════
    # Step 4: 最终硬挤压
    # ═══════════════════════════════════════════════════════════
    Ts = np.zeros((neta, tcol), dtype=np.complex128)
    threshold = 0.0001

    for b in range(tcol):
        for eta in range(neta):
            if np.abs(tfr[eta, b]) > threshold:
                k = omega2[eta, b]
                if 1 <= k <= neta:
                    Ts[k - 1, b] += tfr[eta, b]

    # ── 归一化 (MATLAB: / (xrow/2)) ──
    tfr = tfr / (N / 2.0)
    Ts = Ts / (N / 2.0)

    # ── 频率/时间轴 ──
    freqs_hz = np.arange(neta) / N * fs
    t_axis = np.arange(tcol) / fs

    result = {
        'MSST': np.abs(Ts),
        'STFT': tfr,
        'freqs': freqs_hz.astype(np.float64),
        't': t_axis,
        'omega_final': omega2,
    }
    if save_trajectory:
        result['omegas'] = omegas  # list of [F, T], length=num
    return result


# ============================================================
# 2b. STFT utilities (PyTorch)

def compute_stft(x, fs, n_fft=512, hop_length=128, window='hann',
                  win_length=None, sigma=None):
    """
    计算 STFT 短时傅里叶变换。

    Args:
        x:          [B, T] 或 [T] 信号 (torch tensor)
        fs:         采样率 (Hz)
        n_fft:      FFT 点数
        hop_length: 帧移
        window:     窗类型 'hann' / 'gaussian'
        win_length: 窗长 (默认 n_fft)
        sigma:      Gaussian 窗参数 (仅 window='gaussian')

    Returns:
        X:         [B, F, T_frames] 复数 STFT
        freqs:     [F] 频率轴 (Hz)
        t_frames:  [T_frames] 时间轴 (s)
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    B, T = x.shape
    device = x.device
    wl = win_length if win_length is not None else n_fft

    if window == 'hann':
        win = torch.hann_window(wl, device=device)
    elif window == 'gaussian' and sigma is not None:
        t_win = torch.arange(wl, device=device, dtype=torch.float32) - wl / 2.0 + 0.5
        win = (math.pi * sigma**2)**(-0.25) * torch.exp(-t_win**2 / (2.0 * sigma**2))
    else:
        win = torch.ones(wl, device=device)

    X = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length,
        win_length=wl, window=win,
        center=True, pad_mode='reflect',
        normalized=False, onesided=True, return_complex=True,
    )

    F_bins = X.shape[1]
    T_frames = X.shape[2]
    freqs = torch.linspace(0, fs / 2, F_bins, device=device)
    t_frames = torch.arange(T_frames, device=device).float() * hop_length / fs

    return X, freqs, t_frames


# ============================================================
# 3. SST (一阶同步压缩, PyTorch)
# ============================================================

def sst_stft(x, fs, n_fft=512, hop_length=128, sigma=None):
    """
    STFT 域一阶同步压缩变换 (SST)。

    一阶 IF: ω̂(t,η) = Im(V^{w'}/V^w) / (2π) + η

    Args:
        x:          [B, T] 或 [T] 信号
        fs:         采样率
        n_fft:      FFT 点数
        hop_length: 帧移
        sigma:      Gaussian 窗 σ (样本数), None→n_fft/8

    Returns:
        TFR:   [B, F, T_if] 挤压后幅度
        freqs: [F] 频率轴 (Hz)
        t_axis:[T_if] 时间轴 (s)
    """
    if sigma is None:
        sigma = n_fft / 8.0

    if x.dim() == 1:
        x = x.unsqueeze(0)

    device = x.device
    eps = 1e-8

    # Gaussian 窗 + 导数窗 (FFT 频域微分)
    w_gauss = _gaussian_window(n_fft, sigma, device=device)
    w_deriv = _gaussian_deriv_window(n_fft, sigma, fs, device=device)

    V_w = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=w_gauss,
        center=True, pad_mode='reflect',
        normalized=False, onesided=True, return_complex=True,
    )
    V_wp = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=w_deriv,
        center=True, pad_mode='reflect',
        normalized=False, onesided=True, return_complex=True,
    )

    F_bins = V_w.shape[1]
    T_frames = V_w.shape[2]
    T_if = T_frames - 2

    V_w_mid = V_w[:, :, 1:T_frames - 1]
    V_wp_mid = V_wp[:, :, 1:T_frames - 1]

    freqs_hz = torch.linspace(0, fs / 2, F_bins, device=device)
    omega = 2.0 * math.pi * freqs_hz.view(1, F_bins, 1)

    # N=1 IF: Im(b₁)/(2π), b₁ = -V^{w'}/V^w + jω
    b1 = -V_wp_mid / (V_w_mid + eps) + 1j * omega
    IF = torch.imag(b1) / (2.0 * math.pi)
    IF = IF.clamp(0, fs / 2)

    mag = V_w.abs()
    gamma_mask = mag[:, :, 1:-1].max().item() * 1e-4
    IF = torch.where(mag[:, :, 1:-1] > gamma_mask, IF,
                     freqs_hz.view(1, F_bins, 1).expand(-1, -1, T_if))

    # 挤压
    TFR = _squeeze_torch(mag[:, :, 1:-1], IF, freqs_hz, gamma_mask)

    return TFR, freqs_hz, torch.arange(T_if, device=device) * hop_length / fs


# ============================================================
# 4. 窗函数 (PyTorch)
# ============================================================

def _gaussian_window(n_fft, sigma, device='cpu', dtype=torch.float32):
    """L2 归一化高斯窗。"""
    t = torch.arange(n_fft, device=device, dtype=dtype) - n_fft / 2.0 + 0.5
    p = (math.pi * sigma ** 2) ** (-0.25)
    return p * torch.exp(-t ** 2 / (2.0 * sigma ** 2))


def _gaussian_deriv_window(n_fft, sigma, fs, device='cpu', dtype=torch.float32):
    """高斯窗的物理时间导数 (FFT 频域微分 + fs 缩放)。"""
    w = _gaussian_window(n_fft, sigma, device=device, dtype=dtype)
    wf = torch.fft.fft(w)
    xi = _fft_freq_axis(n_fft, device=device, dtype=dtype)
    return torch.fft.ifft(wf * 1j * xi).real * fs


def _fft_freq_axis(N, device='cpu', dtype=torch.float32):
    """频域微分用频率轴 (radians)。"""
    xi = torch.zeros(N, device=device, dtype=dtype)
    h = 2.0 * math.pi / N
    half = N // 2
    for i in range(half + 1):
        xi[i] = i * h
    for i in range(half + 1, N):
        xi[i] = (i - N) * h
    return xi


# ============================================================
# 4b. HMST IF 估计 (N=1, PyTorch) — 供 SAST 使用
# ============================================================

def compute_hmst_if(x, fs, n_fft=512, hop_length=128, order=1, sigma=None,
                    return_stft=False):
    """
    HMST 一阶 IF 估计 (Bao et al. 2023, eq 3)。

    IF = Im(b₁)/(2π),  b₁ = -V^{w'}/V^w + jω

    Args:
        x:          [B, T] 或 [T] 信号
        fs:         采样率 (Hz)
        n_fft:      FFT 点数
        hop_length: 帧移
        order:      仅支持 N=1
        sigma:      高斯窗 σ (样本数), 默认 n_fft/8
        return_stft:是否返回复数 STFT

    Returns:
        IF:   [B, F_bins, T_if] IF (Hz)
        mag:  [B, F_bins, T_frames] STFT 幅度
    """
    if order != 1:
        raise ValueError(f"compute_hmst_if (tfr.py) 仅支持 order=1, 收到 {order}")

    if x.dim() == 1:
        x = x.unsqueeze(0)

    B = x.shape[0]
    device = x.device
    if sigma is None:
        sigma = n_fft / 8.0
    eps = 1e-8

    w_gauss = _gaussian_window(n_fft, sigma, device=device)
    w_deriv = _gaussian_deriv_window(n_fft, sigma, fs, device=device)

    V_w = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=w_gauss,
        center=True, pad_mode='reflect',
        normalized=False, onesided=True, return_complex=True,
    )
    V_wp = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=w_deriv,
        center=True, pad_mode='reflect',
        normalized=False, onesided=True, return_complex=True,
    )

    F_bins = V_w.shape[1]
    T_frames = V_w.shape[2]
    T_if = T_frames - 2

    freqs_hz = torch.linspace(0, fs / 2, F_bins, device=device)
    omega = 2.0 * math.pi * freqs_hz

    V_w_mid = V_w[:, :, 1:T_frames - 1]
    V_wp_mid = V_wp[:, :, 1:T_frames - 1]
    omega_grid = omega.view(1, F_bins, 1)

    V_w_safe = V_w_mid + eps * torch.sgn(V_w_mid)
    b1 = -V_wp_mid / V_w_safe + 1j * omega_grid
    IF = torch.imag(b1) / (2.0 * math.pi)
    IF = IF.clamp(0, fs / 2)

    mag = V_w.abs()
    gamma_mask = mag[:, :, 1:-1].max().item() * 1e-4
    freqs_exp = freqs_hz.view(1, F_bins, 1).expand(-1, -1, T_if)
    IF = torch.where(mag[:, :, 1:-1] > gamma_mask, IF, freqs_exp)

    if return_stft:
        return IF, mag, V_w
    return IF, mag


# ============================================================
# 5. 挤压 (Squeeze)
# ============================================================

def _squeeze_torch(mag, IF, freqs_hz, gamma=1e-6):
    """
    单次挤压: 幅值重分配到 IF 指示的频率 bin (最近邻)。

    Args:
        mag:      [B, F, T] 幅值
        IF:       [B, F, T] IF 估计 (Hz)
        freqs_hz: [F] 频率网格
        gamma:    幅值阈值

    Returns:
        Tx: [B, F, T] 挤压后幅值
    """
    B, F_bins, T = mag.shape
    device = mag.device
    f0 = freqs_hz[0].item()
    df = (freqs_hz[1] - freqs_hz[0]).item()

    Tx = torch.zeros(B, F_bins, T, device=device, dtype=mag.dtype)

    for b in range(B):
        for j in range(T):
            for i in range(F_bins):
                val = mag[b, i, j].item()
                if val < gamma:
                    continue
                k = int(round((IF[b, i, j].item() - f0) / df))
                if 0 <= k < F_bins:
                    Tx[b, k, j] += val

    return Tx


# ============================================================
# 6. Rényi Entropy 浓度度量
# ============================================================

def compute_renyi(tfr, alpha=3):
    """计算 Rényi entropy (越低越集中)。"""
    tfr = np.asarray(tfr).ravel()
    tfr_n = tfr / (tfr.sum() + 1e-12)
    tfr_n = tfr_n[tfr_n > 1e-12]
    if len(tfr_n) == 0:
        return float('inf')
    return 1 / (1 - alpha) * np.log2((tfr_n ** alpha).sum())
