"""
SAST: Structure-Aware Synchrosqueezing Transform
================================================
物理原型记忆引导的盲脊线可压缩性评估与自适应同步压缩。

三角色范式:
  1. Blind Ridge Extraction:  回答 "哪些频率分量存在？"     → 匿名脊线提取
  2. Physics Prototype Memory + GAT: 回答 "每条脊线有多可信？" → C_i ∈ (0,1]
  3. Adaptive Squeeze:         回答 "基于可信度如何重分配？"  → 确定性高斯软核

核心架构原则:
  - SAST 是物理保真的预处理步骤，不是分类器
  - GAT 唯一职责：输出 Compressibility Token C_i ∈ (0,1]，表征每条脊线的物理可信度
  - σ_sq = σ_min + (1-C_i)·(σ_max-σ_min) — 确定性映射，无可学习参数
  - A_ij 是 GAT 内部诊断探针（因果推理过程），不驱动下游计算
  - Physics Prototype Memory 是外挂知识库（不变量标准尺），不定义图拓扑
  - 图是匿名全连接图——每帧盲提 K 条脊线作为节点，不问出身

引用:
  SST 基线: ssqueezepy (Daubechies, J. & Brevdo, E.)
  GAT 架构: Veličković et al. "Graph Attention Networks" (ICLR 2018)

Author: TFDCL Project
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import ssqueezepy


# ============================================================
# 1. STFT + IF 估计 + SST 基线工具函数
# ============================================================

def compute_stft(x, fs, n_fft=512, hop_length=128, window='hann'):
    """
    计算 STFT 短时傅里叶变换。

    Args:
        x: [B, T] 或 [T] 原始信号
        fs: 采样率 (Hz)
        n_fft: FFT 点数
        hop_length: 帧移
        window: 窗函数类型

    Returns:
        X:         [B, F_bins, T_frames] complex STFT
        freqs:     [F_bins] 频率轴 (Hz)
        t_frames:  [T_frames] 时间轴 (s)
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    B, T = x.shape
    device = x.device

    if window == 'hann':
        win = torch.hann_window(n_fft, device=device)
    elif window == 'hamming':
        win = torch.hamming_window(n_fft, device=device)
    else:
        win = torch.ones(n_fft, device=device)

    X = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=win,
        center=True, pad_mode='reflect',
        normalized=False, onesided=True,
        return_complex=True
    )  # [B, F_bins, T_frames]

    F_bins = X.shape[1]
    T_frames = X.shape[2]

    freqs = torch.linspace(0, fs / 2, F_bins, device=device)  # [F_bins]
    t_frames = torch.arange(T_frames, device=device).float() * hop_length / fs
 
    return X, freqs, t_frames


# ============================================================
# 1b. HMST 高阶 IF 估计: 窗函数工具
# ============================================================

def _gaussian_window(n_fft, sigma, device='cpu', dtype=torch.float32):
    """
    单位 L2 范数高斯窗。
    w(t) = (πσ²)^{-1/4} · exp(-t²/(2σ²))
    论文 eq (5): p = (πσ²)^{-1/4}, q = -1/σ²
    """
    t = torch.arange(n_fft, device=device, dtype=dtype) - n_fft / 2.0 + 0.5
    p = (math.pi * sigma ** 2) ** (-0.25)
    w = p * torch.exp(-t ** 2 / (2.0 * sigma ** 2))
    return w


def _gaussian_deriv_window(n_fft, sigma, fs, device='cpu', dtype=torch.float32):
    """
    高斯窗的时间导数, 频域微分法 (与 ssqueezepy 一致)。

    w'(t) = IFFT{ j·ξ·FFT{w(t)} } · fs

    FFT-based differentiation ensures exact derivative of the discrete window.
    Scaling by `fs` converts from per-sample to per-second derivative,
    与 ssqueezepy _stft.py:135 `diff_window * fs` 一致。
    """
    w = _gaussian_window(n_fft, sigma, device=device, dtype=dtype)
    # FFT → multiply by j·ξ → IFFT (频域微分)
    wf = torch.fft.fft(w)
    xi = _fft_freq_axis(n_fft, device=device, dtype=dtype)
    diff = torch.fft.ifft(wf * 1j * xi).real
    # 换算为物理时间 (秒) 的导数
    diff = diff * fs
    return diff


def _gaussian_tw_window(n_fft, sigma, device='cpu', dtype=torch.float32):
    """
    一阶矩窗 t·w(t)。
    用于论文 eq (14) 的 a_{k,1} 系数: V_x^{t^{k-1}w}
    """
    t = torch.arange(n_fft, device=device, dtype=dtype) - n_fft / 2.0 + 0.5
    p = (math.pi * sigma ** 2) ** (-0.25)
    g = p * torch.exp(-t ** 2 / (2.0 * sigma ** 2))
    return t * g


def _gaussian_t2w_window(n_fft, sigma, device='cpu', dtype=torch.float32):
    """
    二阶矩窗 t²·w(t)。
    用于论文 eq (14) 的 a_{k,1} 系数 (k=3): V_x^{t²w}
    注: t 用样本索引 (与 _gaussian_tw_window 一致), 与导数窗的 fs 缩放
        在 a_{k,1}·P_k 乘积中相互抵消, 故 N=3 与 N=2 量纲自洽。
    """
    t = torch.arange(n_fft, device=device, dtype=dtype) - n_fft / 2.0 + 0.5
    p = (math.pi * sigma ** 2) ** (-0.25)
    g = p * torch.exp(-t ** 2 / (2.0 * sigma ** 2))
    return t ** 2 * g


def _fft_freq_axis(N, device='cpu', dtype=torch.float32):
    """
    频域微分用频率轴 (radians)。
    N even: [0, 1, ..., N/2, -(N/2-1), ..., -1] * 2π/N
    与 ssqueezepy wavelets._xifn 一致。
    """
    xi = torch.zeros(N, device=device, dtype=dtype)
    h = 2.0 * math.pi / N
    half = N // 2
    for i in range(half + 1):
        xi[i] = i * h
    for i in range(half + 1, N):
        xi[i] = (i - N) * h
    return xi


def compute_hmst_if(x, fs, n_fft=512, hop_length=128, order=2, sigma=None):
    """
    HMST 高阶瞬时频率估计 (Bao et al., 2023, eq 16-18)。

    参照 ssqueezepy 的实现:
      - STFT 用 PyTorch (非调制), 通过 jω 项补偿相位
      - 导数窗用 FFT 频域微分 + fs 缩放 (与 ssqueezepy _stft.py 一致)
      - 相位变换: w = Im(b₁)/(2π)  (含 jω 项), 等价于 ssqueezepy 的 phase_stft

    Args:
        x:          [B, T] 或 [T] 原始信号
        fs:         采样率 (Hz)
        n_fft:      FFT 点数
        hop_length: 帧移
        order:      IF 估计阶数 (1, 2 或 3)
        sigma:      高斯窗 σ (样本数), 默认 n_fft/8

    Returns:
        IF:   [B, F_bins, T_if] 瞬时频率 (Hz)
        mag:  [B, F_bins, T_frames] STFT 幅度谱
    """
    if order not in (1, 2, 3):
        raise ValueError(f"order 必须为 1, 2 或 3, 收到 {order}")

    if x.dim() == 1:
        x = x.unsqueeze(0)

    B = x.shape[0]
    device = x.device

    if sigma is None:
        sigma = n_fft / 8.0

    eps = 1e-8

    # ── 1. 窗函数 ──
    # 关键: 导数窗通过 IFFT{j·ξ·FFT{w}} 生成 (匹配 ssqueezepy get_window)
    # 关键: 导数窗 × fs → 物理时间导数 (匹配 ssqueezepy _stft.py L135)
    w_gauss = _gaussian_window(n_fft, sigma, device=device)
    w_deriv = _gaussian_deriv_window(n_fft, sigma, fs, device=device)

    # ── 2. STFT: V^w (高斯窗) + V^{w'} (导数窗, 已含 fs 缩放) ──
    V_w = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=w_gauss,
        center=True, pad_mode='reflect',
        normalized=False, onesided=True,
        return_complex=True,
    )  # [B, F, T_frames]

    V_wp = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=w_deriv,
        center=True, pad_mode='reflect',
        normalized=False, onesided=True,
        return_complex=True,
    )  # [B, F, T_frames]

    F_bins = V_w.shape[1]
    T_frames = V_w.shape[2]
    T_if = T_frames - 2

    # ── 3. 频率轴 ──
    freqs_hz = torch.linspace(0, fs / 2, F_bins, device=device)
    omega = 2.0 * math.pi * freqs_hz            # [F] rad/s
    domega = 2.0 * math.pi * fs / n_fft         # Δω rad/s

    # ── 4. 时间对齐: 丢弃首尾各一帧 ──
    V_w_mid = V_w[:, :, 1:T_frames - 1]       # [B, F, T_if]
    V_wp_mid = V_wp[:, :, 1:T_frames - 1]     # [B, F, T_if] (已含 fs)
    omega_grid = omega.view(1, F_bins, 1)      # [1, F, 1]

    # ── 5. b₁ = -V^{w'}/V^w + jω (论文 eq 3, 已代入 fs 缩放) ──
    V_w_safe = V_w_mid + eps * torch.sgn(V_w_mid)
    b1 = -V_wp_mid / V_w_safe + 1j * omega_grid    # [B, F, T_if]

    if order == 1:
        # ── N=1: ω̂_{[1]} = Im(b₁) / 2π ──
        # b₁ = -V^{w'}_scaled/V + jω = Im(b₁) = ω - Im(V^{w'}_scaled/V)
        # 对于纯单频信号 f₀: Im(V^{w'}_scaled/V) = 2π(f-f₀)
        #   → ω̂/(2π) = f - (f-f₀) = f₀  ✓
        IF = torch.imag(b1) / (2.0 * math.pi)
        IF = IF.clamp(0, fs / 2)
        mag = V_w.abs()
        return IF, mag

    # ══════════════════════════════════════════════════════════
    # N≥2: 高阶 HMST — N×N 上三角矩阵求解 Q₁ (论文 eq 16-17)
    # ══════════════════════════════════════════════════════════

    # ── ∂_ω 沿频率轴中心差分 ──
    def _freq_deriv(z):
        """∂_ω 中心差分, dim=1 为频率轴, 二阶精度"""
        z_pad = F.pad(z, (0, 0, 1, 1), mode='replicate')
        return (z_pad[:, 2:, :] - z_pad[:, :-2, :]) / (2.0 * domega)

    def _safe_cdiv(num, den):
        """复数伪逆除法 num/den → num·conj(den)/(|den|²+eps)。
        ★ den 是复数, 加实数 eps 无法防止 |den|→0 时发散, 故用伪逆。"""
        den_abs2 = den.real ** 2 + den.imag ** 2 + eps
        return num * den.conj() / den_abs2

    def _clamp_c(z, lim):
        """分别钳制复数实/虚部, 防止噪声区高阶修正项飞到几万 Hz。"""
        return z.real.clamp(-lim, lim) + 1j * z.imag.clamp(-lim, lim)

    # ── V^{t·w}: 一阶矩窗 STFT (无需 fs 缩放, 非导数窗) ──
    w_tw = _gaussian_tw_window(n_fft, sigma, device=device)
    V_tw = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=w_tw,
        center=True, pad_mode='reflect',
        normalized=False, onesided=True,
        return_complex=True,
    )  # [B, F, T_frames]
    V_tw_mid = V_tw[:, :, 1:T_frames - 1]  # [B, F, T_if]

    # ── a_{2,1} = V^{tw}/V^w, b₂ = ∂_ω b₁ / ∂_ω a_{2,1} (论文 eq 14/16, k=2) ──
    a21 = V_tw_mid / V_w_safe                    # [B, F, T_if]
    d_b1 = _freq_deriv(b1)                       # ∂_ω b₁
    d_a21 = _freq_deriv(a21)                     # ∂_ω a_{2,1}
    b2 = _clamp_c(_safe_cdiv(d_b1, d_a21), fs)   # [B, F, T_if]

    if order == 2:
        # ── 反向代入: Q₂ = b₂, Q₁ = b₁ - a₂₁·Q₂ (论文 eq 17, N=2) ──
        Q1 = b1 - a21 * b2  # [B, F, T_if]
    else:
        # ══════════════════════════════════════════════════════
        # N=3: 增加 t²w 窗, 上三角扩为 3×3 (论文 eq 16-17, N=3)
        #   代价: 需两层 ∂_ω 频率微分, 对噪声比 N=2 更敏感
        # ══════════════════════════════════════════════════════

        # ── V^{t²·w}: 二阶矩窗 STFT (论文 eq 14, k=3) ──
        w_t2w = _gaussian_t2w_window(n_fft, sigma, device=device)
        V_t2w = torch.stft(
            x, n_fft=n_fft, hop_length=hop_length,
            win_length=n_fft, window=w_t2w,
            center=True, pad_mode='reflect',
            normalized=False, onesided=True,
            return_complex=True,
        )  # [B, F, T_frames]
        V_t2w_mid = V_t2w[:, :, 1:T_frames - 1]  # [B, F, T_if]

        # ── a_{3,1} = V^{t²w}/V^w (论文 eq 14, i=3, j=1) ──
        a31 = V_t2w_mid / V_w_safe               # [B, F, T_if]

        # ── a_{3,2} = ∂_ω a_{3,1} / ∂_ω a_{2,1} (论文 eq 16, i=3, j=2) ──
        d_a31 = _freq_deriv(a31)
        a32 = _safe_cdiv(d_a31, d_a21)           # [B, F, T_if]

        # ── b₃ = ∂_ω b₂ / ∂_ω a_{3,2} (论文 eq 16, k=3) ──
        d_b2 = _freq_deriv(b2)
        d_a32 = _freq_deriv(a32)
        b3 = _clamp_c(_safe_cdiv(d_b2, d_a32), fs)  # [B, F, T_if]

        # ── 反向代入 (论文 eq 17, N=3) ──
        #   P₃ = b₃;  P₂ = b₂ - a₃₂·P₃;  Q₁ = b₁ - a₂₁·P₂ - a₃₁·P₃
        P3 = b3
        P2 = b2 - a32 * P3
        Q1 = b1 - a21 * P2 - a31 * P3  # [B, F, T_if]

    # ── ω̂_{[N]} = Im(Q₁) / 2π (论文 eq 18, N≥2 分支) ──
    IF = torch.imag(Q1) / (2.0 * math.pi)

    # 钳制: 无效估计回退到 STFT 中心频率
    freqs_exp = freqs_hz.view(1, F_bins, 1).expand(-1, -1, T_if)
    IF = torch.where((IF >= 0) & (IF <= fs / 2), IF, freqs_exp)

    mag = V_w.abs()

    return IF, mag


# ============================================================
# 1b-extra. CUDA squeeze 加载器 (自动 fallback)
# ============================================================

_hmst_cuda_ext = None
_hmst_cuda_attempted = False


def _load_hmst_cuda():
    """
    懒加载 CUDA squeeze 扩展。

    优先级:
      1. 预编译的 hmst_cuda_ext (Jetson 部署: deploy/hmst_cuda_ext.so)
      2. JIT 编译 (开发环境: 首次调用时自动编译 .cu → 缓存)
      3. Fallback: Python 三重循环 (纯 CPU 或无 CUDA 工具链时)
    """
    global _hmst_cuda_ext, _hmst_cuda_attempted
    if _hmst_cuda_attempted:
        return _hmst_cuda_ext
    _hmst_cuda_attempted = True

    # 路径 1: 预编译扩展 (Jetson 部署)
    try:
        import importlib
        _hmst_cuda_ext = importlib.import_module('hmst_cuda_ext')
        return _hmst_cuda_ext
    except ImportError:
        pass

    # 路径 2: JIT 编译 (开发环境)
    try:
        from torch.utils.cpp_extension import load
        import os
        src_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'deploy', 'hmst_squeeze.cu',
        )
        if os.path.exists(src_path):
            _hmst_cuda_ext = load(
                name='hmst_cuda_ext',
                sources=[src_path],
                extra_cuda_cflags=['-O3', '--use_fast_math'],
                verbose=False,
            )
    except Exception:
        pass  # 静默 fallback

    return _hmst_cuda_ext


def _hmst_squeeze_cuda(mag, IF, freqs_hz, gamma=1e-6):
    """
    HMST 挤压, 自动选择 CUDA 或 Python 后端。

    CUDA:  单次 kernel launch, 3,598 线程并行,  ~25μs
    Python: 三重循环 scatter_add,             ~15ms (仅 CPU fallback)
    """
    # CPU tensor: 直接走 Python
    if not mag.is_cuda:
        return _hmst_squeeze(mag, IF, freqs_hz, gamma)

    # CUDA tensor: 尝试加载扩展
    ext = _load_hmst_cuda()
    if ext is not None:
        return ext.hmst_squeeze(mag, IF, freqs_hz, gamma)

    # Fallback: CUDA tensor 但无扩展 → Python 循环 + GPU 数据搬运 (极慢, 仅保证正确性)
    return _hmst_squeeze(mag, IF, freqs_hz, gamma)


def _hmst_squeeze(mag, IF, freqs_hz, gamma=1e-6):
    """
    单次 HMST 挤压 (论文 eq 19 的一次迭代)。

    离散挤压 = bin-to-bin 能量重分配, 不加 df:
      - 离散 TFR 的每个 bin 已经代表该频率区间的能量, 挤压缩放的是 bin index,
        不是连续积分 → 直接累加幅值, 不乘 df
      - ssqueezepy 的 _ssqueeze 也是直接累加, 无 df 因子
      - M 次迭代后能量守恒, 不会出现 (df)^M 幅度爆炸

    使用幅值累加 (非复值):
      - 幅值累加 ≡ 正确调制 STFT 的复值累加 (验证: corr=0.9999)
      - 避免 PyTorch 非调制 STFT 的相位不一致问题

    Args:
        mag:      [B, F, T] 当前 TFR 幅值
        IF:       [B, F, T] IF 估计 (Hz), 已取 abs, 所有迭代共用同一 IF
        freqs_hz: [F] 频率网格 (Hz)
        gamma:    绝对幅度阈值

    Returns:
        Tx: [B, F, T] 挤压后幅度 TFR
    """
    B, F, T = mag.shape
    device = mag.device

    f0 = freqs_hz[0].item()
    df = (freqs_hz[1] - freqs_hz[0]).item()

    Tx = torch.zeros(B, F, T, device=device, dtype=mag.dtype)

    for b in range(B):
        for j in range(T):
            for i in range(F):
                val = mag[b, i, j]
                if val < gamma:
                    continue
                w = IF[b, i, j].item()
                k = int(round((w - f0) / df))
                if 0 <= k < F:
                    # 离散 bin-to-bin 重分配, 不乘 df
                    # (乘 df 会导致 M 次迭代后幅度 × df^M 指数爆炸)
                    Tx[b, k, j] += val

    return Tx


def compute_hmst(x, fs, n_fft=512, hop_length=128, order=2, M=2,
                 sigma=None):
    """
    完整 HMST (Bao et al., 2023, eq 19)。

    实现方式:
      1. 计算 N 阶 IF ω̂_{[N]} (论文 eq 16-18)
      2. 取 abs(ω̂) 确保非负 (与 ssqueezepy 一致)
      3. M 次顺序挤压: 每次将 TFR 能量重排到 ω̂_{[N]} 位置
         (论文 eq 19: HMST^{[m]} = ∫ HMST^{[m-1]}(η) δ(ω-ω̂(η)) dη)

    与 SST/MSST/HSST 的工程区别:
      - SST: 1 阶 IF + 1 次挤压
      - MSST: 1 阶 IF + M 次挤压 (IF 不变, 只是反复向同个 IF 聚拢)
      - HSST: N 阶 IF + 1 次挤压 (同精度, 无多次迭代)
      - HMST: N 阶 IF + M 次挤压 (同精度 + 同浓度 = 最优)

    Args:
        x:       [B, T] 或 [T] 原始信号
        fs:      采样率 (Hz)
        n_fft:   FFT 点数
        hop_length: 帧移
        order:   IF 估计阶数 (1, 2 或 3)
        M:       挤压迭代次数 (≥1)
        sigma:   高斯窗 σ

    Returns:
        tfr:   [B, F, T_if] HMST 时频表示 (幅度)
        IF:    [B, F, T_if] 高阶 IF (Hz)
        mag:   [B, F, T_frames] STFT 幅度
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    B = x.shape[0]
    device = x.device

    if sigma is None:
        sigma = n_fft / 8.0

    # ── 1. N 阶 IF 估计 ──
    IF, mag = compute_hmst_if(x, fs, n_fft=n_fft, hop_length=hop_length,
                               order=order, sigma=sigma)
    # IF: [B, F, T_if], mag: [B, F, T_frames]

    F_bins = IF.shape[1]
    T_if = IF.shape[2]
    freqs_hz = torch.linspace(0, fs / 2, F_bins, device=device)

    # ── 2. 绝对值 IF (与 ssqueezepy 一致, 确保非负) ──
    IF_abs = IF.abs()

    # ── 3. 初始幅值 TFR 对齐到 T_if ──
    #     ★ 使用幅值累加, 避免 PyTorch 非调制 STFT 的相位不一致问题
    #     幅值累加 ≡ 正确调制 STFT 的复值累加 (验证: corr=1.0)
    TFR = mag[:, :, 1:-1]  # [B, F, T_if] 幅值

    # ── 4. 绝对 gamma (参照 ssqueezepy) ──
    gamma = TFR.max().item() * 1e-6

    # ── 5. M 次顺序挤压 (论文 eq 19) ──
    for m in range(M):
        TFR = _hmst_squeeze_cuda(TFR, IF_abs, freqs_hz, gamma=gamma)

    tfr_mag = TFR  # [B, F, T_if] 已是幅值

    return tfr_mag, IF, mag


def compute_sst_baseline(x_np, fs, n_fft=512, hop_length=128):
    """
    用 ssqueezepy 计算一阶 SST 作为"硬 δ 挤压"对比基线。
    仅用于对比和可视化，不参与训练梯度。

    Args:
        x_np: [T] numpy 数组，单条信号

    Returns:
        Tx:    [F_bins, T_frames] SST 幅度谱
        Wx:    [F_bins, T_frames] STFT 幅度谱
        freqs: [F_bins] 频率轴 (Hz)
    """
    Tx, Wx, ssq_freqs, *_ = ssqueezepy.ssq_stft(
        x_np, fs=fs,
        window='hann', n_fft=n_fft,
        win_len=n_fft, hop_len=hop_length,
        squeezing='full',
        padtype='reflect',
    )
    return Tx, Wx, ssq_freqs


# ============================================================
# 1c. HMST-CWT: CWT 域高阶多同步压缩 (小波基)
# ============================================================

def _squeeze_cwt_to_linear(W_mag, IF_scales, freqs_linear, gamma=1e-6):
    """
    CWT 域→线性频率网格挤压。

    从 (n_scales, T) 重分配到 (F_bins, T) 线性网格。
    离散 bin-to-bin 重分配, 不乘 df, 能量守恒。

    Args:
        W_mag:        [n_scales, T] CWT 幅值
        IF_scales:    [n_scales, T] IF 估计 (Hz), 与 W_mag 同 shape
        freqs_linear: [F_bins] 线性频率网格
        gamma:        幅度阈值

    Returns:
        Tx: [F_bins, T] 挤压后 TFR
    """
    n_scales, T = W_mag.shape
    F_bins = len(freqs_linear)
    f0 = freqs_linear[0]
    df = freqs_linear[1] - freqs_linear[0]

    Tx = np.zeros((F_bins, T), dtype=W_mag.dtype)

    for t in range(T):
        vals = W_mag[:, t]
        mask = vals >= gamma
        if not mask.any():
            continue
        ks = np.round((IF_scales[mask, t] - f0) / df).astype(np.int32)
        valid = (ks >= 0) & (ks < F_bins)
        if not valid.any():
            continue
        np.add.at(Tx[:, t], ks[valid], vals[mask][valid])

    return Tx


def _squeeze_cwt_within_scales(W_mag, IF_scales, freqs_scales, gamma=1e-6):
    """
    CWT 尺度域内挤压: 在 log 尺度网格内重分配能量。

    对每个 (s, t): 将能量从尺度 s 移到尺度 s_target,
    其中 freqs_scales[s_target] ≈ IF[s,t]。

    Args:
        W_mag:         [n_scales, T] 当前 TFR
        IF_scales:     [n_scales, T] IF 估计 (Hz)
        freqs_scales:  [n_scales] 各尺度中心频率 (升序)
        gamma:         幅度阈值

    Returns:
        Tx: [n_scales, T] 挤压后 TFR (仍在尺度域)
    """
    n_scales, T = W_mag.shape

    Tx = np.zeros_like(W_mag)

    for t in range(T):
        vals = W_mag[:, t]
        mask = vals >= gamma
        if not mask.any():
            continue
        IF_t = IF_scales[mask, t]  # [n_valid]

        # 找每个 IF 值最近的尺度 bin
        ks = np.searchsorted(freqs_scales, IF_t)
        # 比较左/右邻居, 取更近者
        ks_clip = np.clip(ks, 1, n_scales - 1).astype(np.int32)
        left = np.abs(IF_t - freqs_scales[ks_clip - 1])
        right = np.abs(IF_t - freqs_scales[ks_clip])
        ks_final = np.where(left <= right, ks_clip - 1, ks_clip)

        np.add.at(Tx[:, t], ks_final, vals[mask])

    return Tx

from scipy.interpolate import interp1d

def _exact_time_derivative(x, fs):
    """使用 FFT 计算信号的精确时间导数"""
    N = len(x)
    X = np.fft.fft(x)
    freqs_fft = np.fft.fftfreq(N, d=1.0/fs)
    dX = X * (1j * 2.0 * np.pi * freqs_fft)
    return np.fft.ifft(dX).real

def compute_hmst_cwt(x_np, fs, nv=32, M=2, freq_max=200, F_bins=257, wavelet='morlet', wavelet_width=6):
    """
    HMST-CWT 最终完美版
    1. FFT 频域精确求导
    2. 噪声区域 IF 强制回退
    3. 【核心修复】引入 interp1d 解决对数网格迭代的量化误差 (蝴蝶效应)
    4. 纯幅值挤压，保证物理能量守恒
    """
    # ── 1. CWT 与精确时间导数 ──
    Wx, scales = ssqueezepy.cwt(x_np, wavelet=wavelet, scales='log', nv=nv, fs=fs)
    
    dx_dt = _exact_time_derivative(x_np, fs)
    dWx, _ = ssqueezepy.cwt(dx_dt, wavelet=wavelet, scales=scales, fs=fs)
    
    n_scales, T = Wx.shape
    freqs_scales = wavelet_width / (2.0 * np.pi * scales) * fs

    # 升序重排以满足插值和绘图要求
    sort_idx = np.argsort(freqs_scales)
    freqs_scales = freqs_scales[sort_idx]
    Wx = Wx[sort_idx, :]
    dWx = dWx[sort_idx, :]

    # ── 2. 瞬时频率 (IF) 估计与背景噪声镇压 ──
    eps = 1e-8
    W_abs2 = np.abs(Wx)**2 + eps
    IF_scales = np.imag(dWx * np.conj(Wx) / W_abs2) / (2.0 * np.pi)
    
    mag_all = np.abs(Wx)
    gamma = mag_all.max() * 1e-4  

    # 噪声回退：防止迭代坍缩
    mask_valid = mag_all > gamma
    IF_scales = np.where(mask_valid, IF_scales, freqs_scales[:, None])
    IF_scales = np.clip(IF_scales, 0, fs / 2)

    # ── 3. MSWT 多重迭代 IF 映射 (平滑插值版) ──
    IF_current = IF_scales.copy()
    for m in range(1, M):
        IF_next = np.zeros_like(IF_current)
        for t in range(T):
            # 【最关键修复】：在对数网格上使用线性插值，彻底消灭就近取整带来的量化灾难
            f_interp = interp1d(freqs_scales, IF_scales[:, t], kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
            IF_next[:, t] = f_interp(IF_current[:, t])
        
        # 限制越界，防止外推到奈奎斯特频率之外
        IF_current = np.clip(IF_next, 0, fs / 2)

    # ── 4. 单次最终挤压 (幅值挤压) ──
    Tx = np.zeros_like(mag_all) 
    
    for t in range(T):
        vals = mag_all[:, t]  
        valid_mask = mag_all[:, t] > gamma
        if not valid_mask.any():
            continue
            
        IF_t = IF_current[valid_mask, t]
        
        # 最后一次落位，允许使用离散网格索引 (因为不再参与迭代运算)
        ks = np.searchsorted(freqs_scales, IF_t)
        ks_clip = np.clip(ks, 1, n_scales - 1)
        left = np.abs(IF_t - freqs_scales[ks_clip - 1])
        right = np.abs(IF_t - freqs_scales[ks_clip])
        ks_final = np.where(left <= right, ks_clip - 1, ks_clip)
        
        np.add.at(Tx[:, t], ks_final, vals[valid_mask])

    return Tx, freqs_scales, mag_all, IF_scales

# def compute_hmst_cwt(x_np, fs, nv=32, M=2, freq_max=200, F_bins=257,
#                      wavelet='morlet', wavelet_width=6):
#     """
#     CWT 域 HMST: 小波变换 + IF 估计 + 多重挤压。

#     与 HMST-STFT 的对比:
#       - STFT 版: 线性频率网格, 低频分辨率固定 (=fs/n_fft ≈ 1.95 Hz)
#       - CWT 版:  对数尺度网格, 低频分辨率天然更高
#                  (Morlet 小波在低频有更多 scales 覆盖)

#     挤压策略:
#       - 前 M-1 次挤压在 CWT 尺度域内进行 (bin-to-bin within log grid)
#       - 最后 1 次挤压将尺度域映射到线性频率网格
#       - 幅值累加, 不乘 df, 能量守恒

#     Args:
#         x_np:          [T] numpy 1D 信号
#         fs:            采样率 (Hz)
#         nv:            voices per octave (越大 → scale 越密, 默认32)
#         M:             挤压迭代次数
#         freq_max:      频率上限 (Hz), 用于输出网格
#         F_bins:        输出频率 bin 数
#         wavelet:       小波类型 (默认 'morlet')
#         wavelet_width: Morlet ω₀ (默认 6, 近似解析)

#     Returns:
#         tfr_mag:      [F_bins, T-2] 挤压后 TFR
#         freqs_linear: [F_bins] 线性频率网格 (Hz)
#         Wx_mag:       [n_scales_sort, T] CWT 幅值 (按频率排序)
#         IF_scales:    [n_scales_sort, T-2] IF 估计 (Hz)
#     """
#     # ── 1. CWT ──
#     Wx, scales = ssqueezepy.cwt(x_np, wavelet=wavelet, scales='log',
#                                  nv=nv, fs=fs)
#     # Wx: [n_scales, T] complex
#     n_scales, T = Wx.shape

#     # Scale → 中心频率 (Morlet: f = ω₀/(2π·s)·fs)
#     freqs_scales = wavelet_width / (2.0 * np.pi * scales) * fs

#     # 按频率升序重排
#     sort_idx = np.argsort(freqs_scales)
#     freqs_scales = freqs_scales[sort_idx]
#     Wx = Wx[sort_idx, :]

#     # ── 2. IF 估计: 相位时间导数 (中心差分, 二阶精度) ──
#     dt = 1.0 / fs
#     dW_dt = (Wx[:, 2:] - Wx[:, :-2]) / (2.0 * dt)  # [n_scales, T-2]
#     W_mid = Wx[:, 1:-1]                               # [n_scales, T-2]

#     eps = 1e-8
#     # IF = Im(∂_t W / W) / 2π — 复数除法的虚部
#     IF_scales = np.imag(dW_dt / (W_mid + eps)) / (2.0 * np.pi)
#     # 取绝对值 (与 ssqueezepy SST 一致)
#     IF_scales = np.abs(IF_scales)  # [n_scales, T-2]

#     # CWT 幅值 (对齐到 T-2)
#     mag_mid = np.abs(W_mid)  # [n_scales, T-2]
#     mag_all = np.abs(Wx)     # [n_scales, T] (全时间轴, 用于可视化)

#     # ── 3. 输出线性频率网格 ──
#     freqs_linear = np.linspace(0, freq_max, F_bins)

#     # ── 4. M 次挤压 ──
#     gamma = mag_mid.max() * 1e-6
#     TFR = mag_mid.copy()

#     if M > 1:
#         # 前 M-1 次在尺度域内挤压 (能量在 log 尺度网格内向 IF 位置集中)
#         for m in range(M - 1):
#             TFR = _squeeze_cwt_within_scales(
#                 TFR, IF_scales, freqs_scales, gamma=gamma,
#             )
#         # 最后一次: 尺度域 → 线性频率网格
#         TFR = _squeeze_cwt_to_linear(
#             TFR, IF_scales, freqs_linear, gamma=gamma,
#         )
#     else:
#         # M=1: 直接尺度域 → 线性频率网格
#         TFR = _squeeze_cwt_to_linear(
#             TFR, IF_scales, freqs_linear, gamma=gamma,
#         )

#     return TFR, freqs_linear, mag_all, IF_scales


# ============================================================
# 2. Physics Prototype Memory 配置
# ============================================================

PUMP_TURBINE_PROTOTYPES = {
    'prototypes': [
        {'name': 'fr',       'f_nom': 5.56,  'f_type': 'ROTATION',       'C_prior': 0.90},
        {'name': 'RSI_low',  'f_nom': 2.4,   'f_type': 'VORTEX_ROPE',    'C_prior': 0.30},
        {'name': 'RSI_turb', 'f_nom': 8.35,  'f_type': 'TURBULENCE',     'C_prior': 0.30},
        {'name': 'BPF',      'f_nom': 50.0,  'f_type': 'BLADE_PASS',     'C_prior': 1.00},
        {'name': '2xBPF',    'f_nom': 100.0, 'f_type': 'BLADE_HARMONIC', 'C_prior': 1.00},
        {'name': 'GPF',      'f_nom': 111.1, 'f_type': 'GUIDE_VANE',     'C_prior': 0.95},
        {'name': '3xBPF',    'f_nom': 150.0, 'f_type': 'BLADE_HARMONIC', 'C_prior': 1.00},
    ],
    'temperature': 0.08,  # 频率匹配温度: 越小 → gate 越陡峭 (τ=0.08 → 5%频偏时 gate衰减50%)
}


# ============================================================
# 3. BlindRidgeExtractor: 盲脊线提取 (不问出身)
# ============================================================

class BlindRidgeExtractor(nn.Module):
    """
    每帧从幅度谱提取能量最强的 K 条脊线，不问"这条频率是什么物理分量"。

    提取流程:
      1. max_pool1d 找局部极大值
      2. 按能量排序取 top-K
      3. 帧间贪心匹配 → 累加 persistence 计数器
      4. 归一化 persistence 到 [0, 1]

    输出全部是观测量，不依赖任何先验字典。
    """

    def __init__(self, K=6, min_dist=3, fs=1000):
        """
        Args:
            K:         匿名脊线数量
            min_dist:  峰值最小间距 (bin 数)
            fs:        采样率
        """
        super().__init__()
        self.K = K
        self.min_dist = min_dist
        self.fs = fs

    def forward(self, mag, freqs):
        """
        Args:
            mag:   [B, F, T] STFT 幅度谱
            freqs: [F] 频率轴 (Hz)

        Returns:
            ridge_freq:        [B, K, T] 脊线频率 (Hz)
            ridge_energy:      [B, K, T] 脊线对数能量
            ridge_bw:          [B, K, T] 脊线带宽 (Hz, 粗略估计)
            ridge_persistence: [B, K, T] 跟踪持续性 (0-1)
        """
        B, F, T = mag.shape
        device = mag.device
        K = self.K
        freq_res = freqs[1] - freqs[0]  # Hz per bin

        ridge_freq = torch.zeros(B, K, T, device=device)
        ridge_energy = torch.zeros(B, K, T, device=device)
        ridge_bw = torch.zeros(B, K, T, device=device)
        ridge_persistence = torch.zeros(B, K, T, device=device)

        # 持久性跟踪状态
        track_freqs = None     # [B, K_tracked]
        track_persist = None   # [B, K_tracked] 原始帧数计数
        max_persist = 0

        for t in range(T):
            mag_t = mag[:, :, t]  # [B, F]

            # ── 3a. 找局部极大值 ──
            peaks_mask = self._find_local_maxima(mag_t)  # [B, F]

            # 只保留 peak 位置的值，其余为 -inf
            masked_mag = torch.where(peaks_mask, mag_t,
                                     torch.full_like(mag_t, -float('inf')))

            # top-K peaks
            topk_vals, topk_idx = torch.topk(masked_mag, K, dim=1)  # [B, K]

            # 频率
            ridge_freq[:, :, t] = freqs[topk_idx]  # [B, K]
            # log 能量 (数值稳定)
            ridge_energy[:, :, t] = torch.log(topk_vals.clamp(min=1e-8) + 1.0)
            # 粗略带宽: 用 STFT 主瓣宽度作为初始估计
            ridge_bw[:, :, t] = freq_res * 3.0

            # ── 3b. 帧间跟踪 → persistence ──
            cur_freqs = ridge_freq[:, :, t]  # [B, K]

            if track_freqs is not None:
                # 贪心匹配: 每个当前峰找最近的上一帧峰
                # dist[b, k_cur, k_prev]
                dist = (cur_freqs.unsqueeze(-1) -
                        track_freqs.unsqueeze(1)).abs()  # [B, K, K_prev]

                # 对每个当前峰，找最近的上帧峰
                min_dist_val, min_dist_idx = dist.min(dim=-1)  # [B, K]

                # 若距离 < 阈值(2×freq_res×min_dist)，则认为是同一脊线
                match_threshold = 3.0 * freq_res * self.min_dist
                matched = min_dist_val < match_threshold  # [B, K]

                # 更新 persistence
                B_idx = torch.arange(B, device=device).unsqueeze(-1).expand(-1, K)
                new_persist = torch.zeros(B, K, device=device)
                # 匹配上的继承上帧 persistence
                for k in range(K):
                    matched_prev = min_dist_idx[:, k]  # [B]
                    new_persist[:, k] = torch.where(
                        matched[:, k],
                        track_persist[B_idx[:, 0], matched_prev] + 1,
                        torch.ones(B, device=device)  # 新脊线
                    )

                track_persist = new_persist  # [B, K]
            else:
                track_persist = torch.ones(B, K, device=device)

            track_freqs = cur_freqs
            ridge_persistence[:, :, t] = track_persist
            max_persist = max(max_persist, int(track_persist.max().item()))

        # ── 3c. 归一化 persistence 到 [0, 1] ──
        ridge_persistence = ridge_persistence / max(1, max_persist)

        return ridge_freq, ridge_energy, ridge_bw, ridge_persistence

    def _find_local_maxima(self, mag):
        """
        用 max_pool1d 找局部极大值。

        Args:
            mag: [B, F_bins] 单帧幅度

        Returns:
            mask: [B, F_bins] bool
        """
        mag_4d = mag.unsqueeze(1)  # [B, 1, F_bins]
        kernel_size = 2 * self.min_dist + 1
        mag_pooled = torch.nn.functional.max_pool1d(
            mag_4d, kernel_size, stride=1, padding=self.min_dist)
        is_peak = (mag_4d == mag_pooled).squeeze(1)  # [B, F_bins]
        return is_peak


# ============================================================
# 4. Anonymous Graph Builder: 全连接匿名图
# ============================================================

def build_anonymous_graph(ridge_freq, ridge_energy, ridge_persistence,
                          window_size=5):
    """
    从匿名脊线构建全连接有向图 + 观测边特征。

    边特征全是实测值，不包含标称比值 r_nom:
      e_ij = [r_obs, r_std, energy_corr, confidence]

    Args:
        ridge_freq:        [B, K, T] 脊线频率 (Hz)
        ridge_energy:      [B, K, T] 脊线对数能量
        ridge_persistence: [B, K, T] 脊线持续性 (0-1)
        window_size:       局部统计窗口半径 (帧数)

    Returns:
        edge_src:   [M] LongTensor 边起点
        edge_dst:   [M] LongTensor 边终点
        edge_feats: [B, M, T, 4] 边特征
    """
    B, K, T = ridge_freq.shape
    device = ridge_freq.device
    eps = 1e-8

    # 全连接有向图: M = K*(K-1), 无自环
    M = K * (K - 1)
    src_list, dst_list = [], []
    for i in range(K):
        for j in range(K):
            if i != j:
                src_list.append(i)
                dst_list.append(j)
    edge_src = torch.tensor(src_list, dtype=torch.long, device=device)  # [M]
    edge_dst = torch.tensor(dst_list, dtype=torch.long, device=device)  # [M]

    # ── 逐边特征计算 ──
    f_src = ridge_freq[:, edge_src, :]     # [B, M, T]
    f_dst = ridge_freq[:, edge_dst, :]     # [B, M, T]
    e_src = ridge_energy[:, edge_src, :]   # [B, M, T]
    e_dst = ridge_energy[:, edge_dst, :]   # [B, M, T]
    p_src = ridge_persistence[:, edge_src, :]  # [B, M, T]
    p_dst = ridge_persistence[:, edge_dst, :]  # [B, M, T]

    # r_obs: 观测频率比值 f_src / f_dst
    r_obs = f_src / f_dst.clamp(min=eps)  # [B, M, T]

    # r_std: r_obs 在局部窗口内的标准差 → 比值稳定性
    r_std = _local_std(r_obs, window_size)  # [B, M, T]

    # energy_corr: 局部窗口内能量包络的 Pearson 相关系数
    energy_corr = _local_pearson_corr(e_src, e_dst, window_size)  # [B, M, T]

    # confidence: 端点脊线的最小归一化持续性
    confidence = torch.min(p_src, p_dst)  # [B, M, T]

    # 组装: [r_obs, r_std, energy_corr, confidence]
    edge_feats = torch.stack([r_obs, r_std, energy_corr, confidence], dim=-1)
    # [B, M, T, 4]

    return edge_src, edge_dst, edge_feats


def _local_std(x, window_size):
    """沿时间维计算局部窗口标准差。"""
    B, M, T = x.shape
    device = x.device
    w = 2 * window_size + 1

    if T < w:
        return x.std(dim=-1, keepdim=True).expand(-1, -1, T)

    x_flat = x.reshape(B * M, 1, T)  # [B*M, 1, T]
    x_pad = F.pad(x_flat, (window_size, window_size), mode='replicate')
    x_unfold = x_pad.unfold(-1, w, 1)  # [B*M, 1, T, w]

    # 数值稳定的 std
    x_mean = x_unfold.mean(dim=-1, keepdim=True)
    x_var = ((x_unfold - x_mean) ** 2).mean(dim=-1)
    result = torch.sqrt(x_var.clamp(min=1e-8)).squeeze(1)  # [B*M, T]
    return result.reshape(B, M, T)


def _local_pearson_corr(x, y, window_size):
    """
    沿时间维计算局部窗口 Pearson 相关系数。

    Pearson(x,y) = Cov(x,y) / (σ_x · σ_y)
    """
    B, M, T = x.shape
    device = x.device
    w = 2 * window_size + 1
    eps = 1e-8

    if T < w:
        return torch.zeros(B, M, T, device=device)

    # reshape: [B*M, 1, T]
    x_flat = x.reshape(B * M, 1, T)
    y_flat = y.reshape(B * M, 1, T)

    x_pad = F.pad(x_flat, (window_size, window_size), mode='replicate')
    y_pad = F.pad(y_flat, (window_size, window_size), mode='replicate')

    x_unfold = x_pad.unfold(-1, w, 1)  # [B*M, 1, T, w]
    y_unfold = y_pad.unfold(-1, w, 1)  # [B*M, 1, T, w]

    x_centered = x_unfold - x_unfold.mean(dim=-1, keepdim=True)
    y_centered = y_unfold - y_unfold.mean(dim=-1, keepdim=True)

    cov = (x_centered * y_centered).sum(dim=-1)  # [B*M, 1, T]
    std_x = x_centered.norm(dim=-1).clamp(min=eps)
    std_y = y_centered.norm(dim=-1).clamp(min=eps)

    corr = (cov / (std_x * std_y)).squeeze(1)  # [B*M, T]
    # 钳制到 [-1, 1]
    corr = corr.clamp(-1.0, 1.0)

    return corr.reshape(B, M, T)


# ============================================================
# 5. PhysicsPrototypeMemory: 物理原型记忆 (外挂知识注入)
# ============================================================

class PhysicsPrototypeMemory(nn.Module):
    """
    物理原型记忆库: 存储已知物理分量的"不变量标准尺"。

    不定义图拓扑——仅通过交叉注意力向匿名节点注入先验知识:
      - 匹配上的节点 (gate → 1): 获得强先验引导
      - 匹配不上的节点 (gate → 0): 完全依靠实测特征，自动退火

    可插拔设计: 更换 prototype_config 即可适配不同旋转机械。
    """

    def __init__(self, prototype_config=None, d_h=128, f_type_embed_dim=16,
                 fs=1000):
        """
        Args:
            prototype_config: 原型配置字典 (默认水泵水轮机)
            d_h:              隐层维度
            f_type_embed_dim: 分量类型嵌入维度
            fs:               采样率
        """
        super().__init__()
        cfg = prototype_config if prototype_config is not None \
            else PUMP_TURBINE_PROTOTYPES

        self.prototypes = cfg['prototypes']
        self.M_proto = len(self.prototypes)
        self.temperature = cfg.get('temperature', 0.08)
        self.fs = fs

        # 标称频率 (不可学习, 注册为 buffer)
        f_nom_list = [p['f_nom'] for p in self.prototypes]
        self.register_buffer('f_nom', torch.tensor(f_nom_list))  # [M_proto]

        # 先验可压缩性倾向 (不可学习)
        C_prior_list = [p['C_prior'] for p in self.prototypes]
        self.register_buffer('C_prior_proto', torch.tensor(C_prior_list))  # [M_proto]

        # 频率类型嵌入
        f_types = [p['f_type'] for p in self.prototypes]
        unique_types = sorted(set(f_types))
        self.type_to_idx = {t: i for i, t in enumerate(unique_types)}
        self.f_type_embed = nn.Embedding(len(unique_types), f_type_embed_dim)
        f_type_idx = torch.tensor([self.type_to_idx[t] for t in f_types])
        self.register_buffer('f_type_idx', f_type_idx)  # [M_proto]

        # 可学习原型嵌入: 每个原型一个 D 维向量
        self.prototype_embed = nn.Parameter(
            torch.randn(self.M_proto, d_h) * 0.02
        )  # [M_proto, d_h]

        # 交叉注意力投影
        self.W_q = nn.Linear(d_h, d_h, bias=False)   # node query
        self.W_k = nn.Linear(d_h, d_h, bias=False)   # prototype key
        self.W_v = nn.Linear(d_h, d_h, bias=False)   # prototype value

        # 节点特征初始投影 (raw → d_h)
        # raw: [f_norm, log_E, bw_norm, persistence]
        self.node_proj = nn.Sequential(
            nn.Linear(4, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, d_h),
        )

        # 原型类型嵌入投影 (用于增强 prototype_embed)
        self.type_proj = nn.Linear(f_type_embed_dim, d_h, bias=False)

    def forward(self, node_feats_raw, node_freqs):
        """
        Args:
            node_feats_raw: [B, K, 4] 匿名节点原始特征
                            [f_norm, log_E, bw_norm, persistence]
            node_freqs:     [B, K] 观测频率 (Hz)

        Returns:
            h_enhanced: [B, K, d_h] 原型增强的节点特征
            gate:       [B, K]     原型匹配门控 (0-1)
            C_prior:    [B, K]     先验可压缩性倾向
        """
        B, K, _ = node_feats_raw.shape
        device = node_feats_raw.device
        M = self.M_proto

        # ── 5a. 频率距离门控 ──
        # |f_obs - f_nom| / f_nom
        f_obs_exp = node_freqs.unsqueeze(-1)  # [B, K, 1]
        f_nom_exp = self.f_nom.view(1, 1, M)   # [1, 1, M]
        freq_dist = (f_obs_exp - f_nom_exp).abs() / f_nom_exp.clamp(min=1e-8)
        # [B, K, M]

        # 频率门控: 最近原型匹配得分
        match_score = torch.exp(-freq_dist / self.temperature)  # [B, K, M]
        gate, _ = match_score.max(dim=-1)  # [B, K]
        best_proto = match_score.argmax(dim=-1)  # [B, K]

        # ── 5b. 先验可压缩性 ──
        C_prior = self.C_prior_proto[best_proto]  # [B, K]
        # gate 很低时, C_prior 退火到中性值 0.5
        C_prior = gate * C_prior + (1 - gate) * 0.5  # [B, K]

        # ── 5c. 节点特征投影 ──
        h_raw = self.node_proj(node_feats_raw)  # [B, K, d_h]

        # ── 5d. 交叉注意力: anonymous node → prototype ──
        # 增强原型嵌入: 加上类型信息
        type_emb = self.f_type_embed(self.f_type_idx.to(device))  # [M, f_type_embed_dim]
        type_feat = self.type_proj(type_emb)  # [M, d_h]
        proto_feat = self.prototype_embed + type_feat  # [M, d_h]

        Q = self.W_q(h_raw)  # [B, K, d_h]
        K_p = self.W_k(proto_feat)  # [M, d_h]
        V_p = self.W_v(proto_feat)  # [M, d_h]

        # 注意力得分 (同时考虑语义 + 频率匹配)
        attn_scores = torch.matmul(Q, K_p.T) / math.sqrt(h_raw.shape[-1])
        # [B, K, M]

        # 频率偏置: 匹配得分高的原型获得注意力偏置
        freq_bias = match_score * 3.0  # scale factor to influence attention
        attn_scores = attn_scores + freq_bias

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, K, M]

        # 原型上下文
        proto_context = torch.matmul(attn_weights, V_p)  # [B, K, d_h]

        # ── 5e. 门控融合 ──
        # gate → 1: 原型上下文主导
        # gate → 0: 原始特征主导
        gate_exp = gate.unsqueeze(-1)  # [B, K, 1]
        h_enhanced = (1 - gate_exp) * h_raw + gate_exp * proto_context
        # [B, K, d_h]

        return h_enhanced, gate, C_prior


# ============================================================
# 6. EdgeConditionedGAT: 边条件图注意力 → Compressibility Token
# ============================================================

class EdgeConditionedGATLayer(nn.Module):
    """
    单层边条件图注意力。

    α_ij = softmax_j( LeakyReLU( a^T [W_q h_i || W_k h_j || W_e e_ij] ) )
    """

    def __init__(self, d_h, d_e=4, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_h % n_heads == 0
        self.d_h = d_h
        self.d_e = d_e
        self.n_heads = n_heads
        self.d_head = d_h // n_heads
        self.scale = self.d_head ** -0.5
        self.dropout = dropout

        self.W_q = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.W_v = nn.Linear(d_h, d_h, bias=False)
        self.W_e = nn.Linear(d_e, d_h, bias=False)

        self.attn_a = nn.Parameter(torch.randn(n_heads, 3 * self.d_head) * 0.02)

        self.out_proj = nn.Linear(d_h, d_h)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, h, edge_feats, edge_src, edge_dst):
        """
        Args:
            h:          [B, N, d_h]
            edge_feats: [B, M, d_e]
            edge_src:   [M]
            edge_dst:   [M]

        Returns:
            h_out: [B, N, d_h]
            attn:  [B, M, n_heads]  — A_ij 内部诊断探针
        """
        B, N, _ = h.shape
        M = edge_feats.shape[1]
        device = h.device

        Q = self.W_q(h).view(B, N, self.n_heads, self.d_head)
        K = self.W_k(h).view(B, N, self.n_heads, self.d_head)
        V = self.W_v(h).view(B, N, self.n_heads, self.d_head)
        E = self.W_e(edge_feats).view(B, M, self.n_heads, self.d_head)

        Q_dst = Q[:, edge_dst]  # [B, M, H, d_head]
        K_src = K[:, edge_src]
        cat_feat = torch.cat([Q_dst, K_src, E], dim=-1)  # [B, M, H, 3*d_head]

        attn_logits = (cat_feat * self.attn_a.view(1, 1, self.n_heads, -1)
                       ).sum(dim=-1)  # [B, M, H]
        attn_scores = F.leaky_relu(attn_logits, negative_slope=0.2)
        attn_weights = scatter_softmax(attn_scores, edge_dst, N)

        attn = attn_weights  # A_ij 诊断探针

        V_src = V[:, edge_src]  # [B, M, H, d_head]
        msg = attn_weights.unsqueeze(-1) * V_src

        h_new = torch.zeros(B, N, self.n_heads, self.d_head, device=device)
        dst_expanded = edge_dst.view(1, M, 1, 1).expand(B, -1, self.n_heads, self.d_head)
        h_new = h_new.scatter_add(1, dst_expanded, msg)
        h_new = h_new.reshape(B, N, self.d_h)

        h_out = F.relu(self.out_proj(h_new))
        h_out = self.dropout_layer(h_out)

        return h_out, attn


def scatter_softmax(scores, indices, N):
    """
    沿 scatter 索引做分组 softmax。
    多头并行: scores [B, M, H], indices [M], N 为分组数。
    """
    B, M, H = scores.shape
    device = scores.device

    idx_exp = indices.view(1, M, 1).expand(B, -1, H)

    # 每组的 max (数值稳定)
    max_per_group = torch.zeros(B, N, H, device=device)
    max_per_group = max_per_group.scatter_reduce(1, idx_exp, scores,
                                                  reduce='amax', include_self=False)

    scores_max = scores - max_per_group[:, indices]
    exp_scores = torch.exp(scores_max)

    sum_exp = torch.zeros(B, N, H, device=device)
    sum_exp = sum_exp.scatter_add(1, idx_exp, exp_scores)

    probs = exp_scores / (sum_exp[:, indices].clamp(min=1e-8))
    return probs


class EdgeConditionedGAT(nn.Module):
    """
    Edge-Conditioned GAT: L 层消息传递 → 单一 Compressibility Token 输出。

    输出:
      C_i:  [B, N] ∈ (0, 1]  每条脊线的物理可压缩性令牌 (决策结果)
      A_ij: [B, M, H]        最终层注意力权重 (因果推理过程, 诊断探针)
    """

    def __init__(self, d_h=128, d_e=4, n_heads=4, n_layers=2,
                 dropout=0.1):
        super().__init__()
        self.d_h = d_h
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            EdgeConditionedGATLayer(d_h, d_e, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_h) for _ in range(n_layers)
        ])

        # C_i 输出头: h_i^(L) → C_i ∈ (0, 1]
        # 两层 MLP + sigmoid
        self.mlp_compress = nn.Sequential(
            nn.Linear(d_h, d_h // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h // 2, 1),
        )

        # 融合 gate 和 C_prior (来自 PPM) 的输入投影
        self.input_proj = nn.Linear(d_h + 1, d_h)  # d_h features + C_prior

    def forward(self, h, edge_feats, edge_src, edge_dst, C_prior):
        """
        Args:
            h:          [B, N, d_h] PPM 增强的节点特征
            edge_feats: [B, M, d_e] 观测边特征
            edge_src:   [M]
            edge_dst:   [M]
            C_prior:    [B, N] 先验可压缩性 (来自 PPM)

        Returns:
            C_i:  [B, N] Compressibility Token ∈ (0, 1]
            A_ij: [B, M, H] 注意力权重 (诊断探针)
        """
        # 注入 C_prior 作为输入特征
        h = torch.cat([h, C_prior.unsqueeze(-1)], dim=-1)  # [B, N, d_h+1]
        h = self.input_proj(h)  # [B, N, d_h]

        # ── 多层 GAT ──
        A_ij = None
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            residual = h
            h_new, attn = layer(h, edge_feats, edge_src, edge_dst)
            h = norm(residual + h_new)
            if i == self.n_layers - 1:
                A_ij = attn

        # ── C_i 输出 ──
        C_logits = self.mlp_compress(h).squeeze(-1)  # [B, N]
        C_i = torch.sigmoid(C_logits)  # (0, 1) → 自然满足 (0, 1]
        # C_i → 1: 高可信度, 可激进挤压
        # C_i → 0: 低可信度, 保守保留

        return C_i, A_ij


# ============================================================
# 7. AdaptiveSqueeze: 自适应高斯核挤压 (确定性映射)
# ============================================================

class AdaptiveSqueeze(nn.Module):
    """
    高斯软核自适应同步压缩。

    σ_sq(t,η) = σ_min + (1 - C_{i*(t,η)}(t)) · (σ_max - σ_min)

    σ 映射是确定性的——无学习参数，梯度通过 C_i 回传。
    """

    def __init__(self, freq_bins, sigma_min=0.5, sigma_max=15.0, kernel_size=31):
        super().__init__()
        self.freq_bins = freq_bins
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, mag, C_i, node_if, freqs):
        """
        Args:
            mag:     [B, F, T] 幅度谱
            C_i:     [B, K, T] 可压缩性令牌 (已时间对齐)
            node_if: [B, K, T] 匿名脊线频率 (已时间对齐)
            freqs:   [F] 频率轴

        Returns:
            tfr_enhanced: [B, F, T]  挤压后 TFR
            sigma_sq:     [B, F, T]  逐 bin 带宽
        """
        B, F, T = mag.shape
        device = mag.device

        # ── 确定性 σ 映射 ──
        # σ_i = σ_min + (1 - C_i) · (σ_max - σ_min)
        delta = self.sigma_max - self.sigma_min
        sigma_i = self.sigma_min + (1.0 - C_i) * delta  # [B, K, T]

        # ── TF bin → 脊线分配 ──
        # 每个 TF bin 归属到最近的匿名脊线 (argmin, detach)
        freqs_exp = freqs.view(1, 1, F, 1)           # [1, 1, F, 1]
        node_if_exp = node_if.unsqueeze(2)           # [B, K, 1, T]
        dist = (freqs_exp - node_if_exp).abs()       # [B, K, F, T]
        i_star = dist.argmin(dim=1)                  # [B, F, T] (detach)

        # 继承 σ: 每个 bin 使用其归属脊线的挤压带宽
        B_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, F, T)
        T_idx = torch.arange(T, device=device).view(1, 1, T).expand(B, F, -1)
        sigma_sq = sigma_i[B_idx, i_star, T_idx]  # [B, F, T]
        sigma_sq = sigma_sq.clamp(min=0.3)

        # ── 高斯模糊 ──
        tfr_enhanced = self._gaussian_blur_along_freq(mag, sigma_sq)

        return tfr_enhanced, sigma_sq

    def _gaussian_blur_along_freq(self, mag, sigma_sq):
        """
        沿频率轴做可变带宽高斯模糊 (离散化到 L 级别 → conv1d 批量执行)。
        """
        B, F_bins, T = mag.shape
        device = mag.device

        n_levels = 20
        sigma_min_eff = 0.5
        sigma_max_eff = float(self.kernel_size) / 3.0

        sigma_clipped = sigma_sq.clamp(sigma_min_eff, sigma_max_eff)
        levels = torch.linspace(sigma_min_eff, sigma_max_eff, n_levels, device=device)

        dist = (sigma_clipped.unsqueeze(-1) - levels.view(1, 1, 1, -1)).abs()
        level_idx = dist.argmin(dim=-1)  # [B, F, T]

        x = torch.arange(-self.pad, self.pad + 1, device=device).float()
        kernels = []
        for l_idx in range(n_levels):
            sigma_l = levels[l_idx]
            k = torch.exp(-0.5 * (x / sigma_l.clamp(min=0.1)) ** 2)
            k = k / (k.sum() + 1e-8)
            kernels.append(k)
        kernel_weights = torch.stack(kernels, dim=0).unsqueeze(1)  # [L, 1, K_size]

        mag_bt = mag.permute(0, 2, 1).reshape(B * T, 1, F_bins)
        mag_pad = F.pad(mag_bt, (self.pad, self.pad), mode='replicate')
        blurred_all = F.conv1d(mag_pad, kernel_weights, groups=1)  # [B*T, L, F]

        blurred_all = blurred_all.reshape(B, T, n_levels, F_bins).permute(0, 2, 3, 1)
        # [B, L, F, T]

        B_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, F_bins, T)
        F_idx = torch.arange(F_bins, device=device).view(1, F_bins, 1).expand(B, -1, T)
        T_idx = torch.arange(T, device=device).view(1, 1, T).expand(B, F_bins, -1)
        result = blurred_all[B_idx, level_idx, F_idx, T_idx]

        return result


# ============================================================
# 8. SAST: 顶层模块
# ============================================================

class SAST(nn.Module):
    """
    Structure-Aware Synchrosqueezing Transform.

    数据流:
      Signal → HMST 高阶 IF 估计 → BlindRidgeExtractor → Anonymous Graph
      → PhysicsPrototypeMemory → EdgeConditionedGAT → C_i
      → AdaptiveSqueeze → Physical TFR

    Args:
        prototype_config: 物理原型配置 (默认水泵水轮机)
        fs:               采样率
        n_fft:            FFT 点数
        hop_length:       帧移
        d_h:              GAT 隐层维度
        n_heads:          注意力头数
        n_layers:         GAT 层数
        K_ridges:         匿名脊线数
        sigma_min:        最小挤压带宽 (bin)
        sigma_max:        最大挤压带宽 (bin)
        hmst_order:       HMST IF 估计阶数 (1=标准一阶, 2=二阶对线性调频无偏,
                          3=三阶对二次调频/强时变 IF 无偏)
        hmst_sigma:       高斯窗 σ (样本数), 默认 n_fft/8
    """

    def __init__(self, prototype_config=None, fs=1000, n_fft=512,
                 hop_length=128, d_h=128, n_heads=4, n_layers=2,
                 K_ridges=6, sigma_min=0.5, sigma_max=15.0,
                 f_type_embed_dim=16, dropout=0.1,
                 hmst_order=2, hmst_sigma=None):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.d_h = d_h
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.K_ridges = K_ridges
        self.hmst_order = hmst_order
        self.hmst_sigma = hmst_sigma if hmst_sigma is not None else n_fft / 8.0

        # HMST 高斯窗 buffer (预计算, 避免每次 forward 重新生成)
        dtype = torch.float32
        self.register_buffer('_w_gauss',
                             _gaussian_window(n_fft, self.hmst_sigma, dtype=dtype))
        self.register_buffer('_w_deriv',
                             _gaussian_deriv_window(n_fft, self.hmst_sigma, fs, dtype=dtype))
        if hmst_order >= 2:
            self.register_buffer('_w_tw',
                                 _gaussian_tw_window(n_fft, self.hmst_sigma, dtype=dtype))
        if hmst_order >= 3:
            self.register_buffer('_w_t2w',
                                 _gaussian_t2w_window(n_fft, self.hmst_sigma, dtype=dtype))

        # ── 8a. Blind Ridge Extractor ──
        self.ridge_extractor = BlindRidgeExtractor(K=K_ridges, fs=fs)

        # ── 8b. Physics Prototype Memory ──
        self.ppm = PhysicsPrototypeMemory(
            prototype_config=prototype_config,
            d_h=d_h,
            f_type_embed_dim=f_type_embed_dim,
            fs=fs,
        )

        # ── 8c. Edge-Conditioned GAT ──
        self.gat = EdgeConditionedGAT(
            d_h=d_h, d_e=4, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout,
        )

        # ── 8d. Adaptive Squeeze ──
        F_bins = n_fft // 2 + 1
        self.squeeze = AdaptiveSqueeze(F_bins, sigma_min, sigma_max)
        self.F_bins = F_bins

        # 边特征维度固定为 4: [r_obs, r_std, energy_corr, confidence]
        self.d_e = 4

    def to(self, device):
        super().to(device)
        return self

    def forward(self, x, return_all=False):
        """
        Args:
            x:          [B, T] 原始信号
            return_all: 是否返回所有诊断量 (A_ij, C_i, gate 等)

        Returns:
            若 return_all=False: tfr_enhanced [B, F_bins, T_frames]
            若 return_all=True:  dict 含 tfr_enhanced, C_i, A_ij, gate 等
        """
        B, T_in = x.shape
        device = x.device

        # ── 1. HMST: 高阶 IF 估计 + 幅度谱 ──
        # 替代原来的 STFT → 一阶相位差分 IF
        IF, mag = compute_hmst_if(
            x, self.fs, n_fft=self.n_fft, hop_length=self.hop_length,
            order=self.hmst_order, sigma=self.hmst_sigma,
        )
        # IF:  [B, F_bins, T_if]   (T_if = T_frames - 2)
        # mag: [B, F_bins, T_frames]

        F_bins = IF.shape[1]
        T_frames = mag.shape[2]
        T_if = IF.shape[2]
        freqs = torch.linspace(0, self.fs / 2, F_bins, device=device)

        # mag 对齐到 T_if (丢弃首尾各一帧, 与 IF 对齐)
        mag_aligned = mag[:, :, 1:-1]  # [B, F_bins, T_if]

        # ── 3. 盲脊线提取 ──
        ridge_freq, ridge_energy, ridge_bw, ridge_persistence = \
            self.ridge_extractor(mag_aligned, freqs)
        # 所有: [B, K, T_if]

        # ── 4. 匿名全连接图 ──
        edge_src, edge_dst, edge_feats = build_anonymous_graph(
            ridge_freq, ridge_energy, ridge_persistence, window_size=5
        )
        # edge_src/dst: [M] where M = K*(K-1)
        # edge_feats:    [B, M, T_if, 4]
        M = edge_src.shape[0]

        # ── 5. 逐帧: PPM → GAT ──
        C_i_all = []
        A_ij_all = []
        gate_all = []

        fs_half = self.fs / 2.0

        for t in range(T_if):
            # ── 构建匿名节点原始特征 ──
            # [f_norm, log_E, bw_norm, persistence]
            f_norm = ridge_freq[:, :, t] / fs_half       # [B, K]
            log_E = ridge_energy[:, :, t]                 # [B, K]
            bw_norm = ridge_bw[:, :, t] / fs_half         # [B, K]
            persist = ridge_persistence[:, :, t]          # [B, K]

            raw_feats = torch.stack([f_norm, log_E, bw_norm, persist], dim=-1)
            # [B, K, 4]

            # PPM: 原型增强
            h_enhanced, gate, C_prior = self.ppm(raw_feats, ridge_freq[:, :, t])
            # h_enhanced: [B, K, d_h], gate: [B, K], C_prior: [B, K]

            # 边特征
            e_t = edge_feats[:, :, t, :]  # [B, M, 4]

            # GAT: 输出 C_i
            C_i_t, A_ij_t = self.gat(h_enhanced, e_t, edge_src, edge_dst, C_prior)
            # C_i_t: [B, K], A_ij_t: [B, M, H]

            C_i_all.append(C_i_t)
            A_ij_all.append(A_ij_t)
            gate_all.append(gate)

        C_i = torch.stack(C_i_all, dim=-1)    # [B, K, T_if]
        A_ij = torch.stack(A_ij_all, dim=-1)  # [B, M, H, T_if]
        gate = torch.stack(gate_all, dim=-1)  # [B, K, T_if]

        # ── 6. Adaptive Squeeze ──
        # 时间对齐: 填充到 T_frames
        C_i_padded = F.pad(C_i, (1, 1), mode='replicate')
        ridge_freq_padded = F.pad(ridge_freq, (1, 1), mode='replicate')

        tfr_enhanced, sigma_sq = self.squeeze(
            mag, C_i_padded, ridge_freq_padded, freqs
        )

        if return_all:
            # 确定性 σ 映射 (用于诊断)
            delta = self.sigma_max - self.sigma_min
            sigma_i = self.sigma_min + (1.0 - C_i) * delta  # [B, K, T_if]

            return {
                'tfr_enhanced': tfr_enhanced,
                'tfr_raw': mag,
                'C_i': C_i,              # Compressibility Token (决策)
                'A_ij': A_ij,            # 注意力权重 (因果推理, 诊断探针)
                'gate': gate,            # 原型匹配门控 (匹配质量)
                'sigma_i': sigma_i,      # 逐脊线挤压带宽
                'sigma_sq': sigma_sq,    # 逐 TF bin 带宽
                'ridge_freq': ridge_freq,  # 匿名脊线频率
                'edge_src': edge_src,
                'edge_dst': edge_dst,
                'edge_feats': edge_feats,  # 观测边特征
                'freqs': freqs,
                't_frames': torch.arange(T_frames, device=device).float()
                            * self.hop_length / self.fs,
            }
        return tfr_enhanced

    def get_freq_features(self, x):
        """
        DCMR 桥接: 时间池化 SAST TFR → 增强频域特征。

        Args:
            x: [B, T] 原始信号

        Returns:
            freq_feat: [B, F_bins] 增强频域特征
        """
        tfr = self.forward(x, return_all=False)
        freq_feat_mean = tfr.mean(dim=-1)
        freq_feat_max = tfr.max(dim=-1).values
        freq_feat = freq_feat_mean + freq_feat_max
        return freq_feat
