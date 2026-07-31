"""
纯 PyTorch MSST (无 C++ kernel 依赖)
=====================================
用 torch.fft 向量化重写 modulated-form STFT, 匹配 models/tfr.py:msst() Step1
和 C++ msst_stft_cuda. IF 估计/精化复用纯 torch 实现. squeeze 用硬挤压
(匹配 numpy msst Step4; C++ 版是线性插值, 略有不同).

用途:
  - 验证 torch STFT 重写是否匹配 C++ (见 scripts/plot/plot_torch_msst_compare.py)
  - 作为 C++ 版的免编译备选 (无需 nvcc/MSVC, 跨平台)

数据流:
  1. msst_stft_torch  -> torch.fft modulated-form STFT (向量化, cuFFT 后端)
  2. estimate_if      -> torch phase unwrap + diff -> omega
  3. refine_if        -> torch IF refinement (num-1 次)
  4. msst_squeeze_hard -> torch scatter_add 硬挤压

用法:
  from models.msst_torch import msst_torch
  x = torch.randn(2000, device='cuda')
  result = msst_torch(x, fs=1000, num=4)
"""

import torch
import math
from typing import Optional, List, Dict


# ═══════════════════════════════════════════════════════════════
# Phase Unwrap (纯 torch, 匹配 numpy np.unwrap)
# ═══════════════════════════════════════════════════════════════

def phase_unwrap(phase: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """NumPy np.unwrap 的 PyTorch 实现, 支持任意形状, GPU 可用."""
    dd = torch.diff(phase, dim=dim)
    ddmod = (dd + math.pi) % (2 * math.pi) - math.pi
    ph_correct = ddmod - dd
    ph_correct[dd.abs() < math.pi] = 0.0
    up = phase.clone()
    slices_start = [slice(None)] * phase.ndim
    slices_start[dim] = slice(1, None)
    up[tuple(slices_start)] = up[tuple(slices_start)] + torch.cumsum(ph_correct, dim=dim)
    return up


# ═══════════════════════════════════════════════════════════════
# 1. STFT (向量化 torch, 匹配 numpy msst Step1 / C++ msst_stft_cuda)
# ═══════════════════════════════════════════════════════════════

def msst_stft_torch(x: torch.Tensor, h: torch.Tensor,
                    N: int, hlength: int, Lh: int,
                    neta: int, tcol: int) -> torch.Tensor:
    """
    向量化 modulated-form STFT (匹配 numpy msst() Step1).

    numpy 原始 (per-frame 循环):
      for icol in range(tcol):
          tau = [tau_min, tau_max] (边界截断)
          tfr_pre[(N+tau)%N, icol] = x[ti+tau] * conj(h[Lh+tau])
      tfr = fft(tfr_pre, axis=0)[:neta]

    torch 向量化: 一次性构造 [tcol, 2Lh+1] 窗×信号矩阵, scatter 到 tfr_pre 行.

    Args:
        x:        [N] float64 on GPU
        h:        [hlength] float64 Gaussian window
        N, hlength, Lh, neta, tcol: STFT 参数

    Returns:
        tfr: [neta, tcol] complex128 (未归一化, 后续 / (N/2))
    """
    device = x.device
    tau = torch.arange(-Lh, Lh + 1, device=device, dtype=torch.long)   # [2Lh+1]
    ti = torch.arange(tcol, device=device, dtype=torch.long)           # [tcol]
    idx = ti.unsqueeze(1) + tau.unsqueeze(0)                           # [tcol, 2Lh+1]
    valid = (idx >= 0) & (idx < N)                                     # 边界截断
    idx_c = idx.clamp(0, N - 1)
    rSig = x[idx_c] * valid                                            # [tcol, 2Lh+1], 边界 0
    win_idx = (Lh + tau).clamp(0, hlength - 1)
    windowed = rSig * h[win_idx].unsqueeze(0)                          # [tcol, 2Lh+1] (h 实窗)
    indices = (N + tau) % N                                            # [2Lh+1] 目标频率行
    tfr_pre = torch.zeros(N, tcol, dtype=torch.complex128, device=device)
    tfr_pre[indices, :] = windowed.t().to(torch.complex128)            # scatter 到行 (indices 唯一)
    tfr = torch.fft.fft(tfr_pre, dim=0)[:neta, :]                      # [neta, tcol] complex128
    return tfr


# ═══════════════════════════════════════════════════════════════
# 2. IF Estimation (纯 torch, 复用 msst_gpu 逻辑)
# ═══════════════════════════════════════════════════════════════

def estimate_if(tfr_complex: torch.Tensor, N: int) -> torch.Tensor:
    """一阶 IF (相位差分法). -> [F, T] int32 (1-indexed bin, 0=invalid)."""
    angle = torch.angle(tfr_complex)
    unwrapped = phase_unwrap(angle, dim=-1)
    omega_cont = torch.diff(unwrapped, dim=-1) * N / (2.0 * math.pi)
    omega_cont = torch.cat([omega_cont, omega_cont[:, -1:]], dim=-1)
    return torch.round(omega_cont).to(torch.int32)


# ═══════════════════════════════════════════════════════════════
# 3. IF Refinement (纯 torch)
# ═══════════════════════════════════════════════════════════════

def refine_if(omega: torch.Tensor, neta: int, num: int) -> tuple:
    """MSST IF 迭代精化: omega2(eta,b) = omega(omega(eta,b), b)."""
    F, T_col = omega.shape
    omegas: List[torch.Tensor] = [omega.clone()]
    omega_cur = omega
    if num > 1:
        omega2 = torch.zeros(F, T_col, dtype=torch.int32, device=omega.device)
        for _ in range(num - 1):
            valid = (omega_cur >= 1) & (omega_cur <= neta)
            omega2.zero_()
            if valid.any():
                b_idx = torch.arange(T_col, device=omega.device).unsqueeze(0).expand(F, T_col)
                k_vals = (omega_cur - 1).clamp(0, F - 1)
                omega2[valid] = omega_cur[k_vals[valid], b_idx[valid]]
            omega_cur = omega2.clone()
            omegas.append(omega_cur.clone())
    else:
        omega2 = omega.clone()
    return omegas, omega2


# ═══════════════════════════════════════════════════════════════
# 4. Squeeze (硬挤压, 匹配 numpy msst Step4)
# ═══════════════════════════════════════════════════════════════

def msst_squeeze_hard(tfr_mag: torch.Tensor, omega_final: torch.Tensor,
                      neta: int, gamma: float = 1e-6) -> torch.Tensor:
    """
    硬挤压: Ts[k-1, b] += tfr_mag[eta, b] where k = omega_final[eta, b].

    注意: C++ msst_squeeze_linear 是线性插值 (双 bin), 此处用硬挤压
    (匹配 numpy msst). 两者 MSST 略有不同, 但 STFT 对比不受影响.

    Args:
        tfr_mag:     [F, T] STFT 幅值
        omega_final: [F, T] int32 IF (1-indexed, 0=invalid)
        neta:        正频率 bin 数
        gamma:       幅值阈值

    Returns:
        Ts: [F, T] 挤压后幅值
    """
    F, T = tfr_mag.shape
    Ts = torch.zeros(F, T, device=tfr_mag.device, dtype=tfr_mag.dtype)
    valid = (tfr_mag.abs() > gamma) & (omega_final >= 1) & (omega_final <= neta)
    idx = (omega_final - 1).clamp(0, F - 1).long()                     # [F, T] 0-indexed (int64 for scatter)
    Ts.scatter_add_(0, idx, tfr_mag * valid)
    return Ts


# ═══════════════════════════════════════════════════════════════
# Full torch MSST Pipeline
# ═══════════════════════════════════════════════════════════════

def msst_torch(x: torch.Tensor, fs: float,
               hlength: Optional[int] = None,
               num: int = 4,
               save_trajectory: bool = False,
               gamma: float = 1e-6,
               skip_squeeze: bool = False) -> Dict:
    """
    纯 torch MSST (无 C++ 依赖, 接口匹配 msst_gpu).

    Args:
        x:       [T] 1D signal on CUDA (float32/float64)
        fs:      采样率 (Hz)
        hlength: 窗长 (默认 min(N, 512), 自动奇数)
        num:     MSST 迭代次数
        save_trajectory: 保存中间 omega
        gamma:   幅值阈值

    Returns:
        dict: 'MSST' [F,T], 'STFT' [F,T] complex, 'freqs' [F], 't' [T],
              'omega_final' [F,T] int32, ['omegas']
    """
    if x.dim() != 1:
        raise ValueError(f"x must be 1D, got {x.shape}")
    device = x.device
    N = len(x)
    x_f64 = x.double()

    if hlength is None:
        hlength = min(N, 512)
    hlength = hlength + 1 - (hlength % 2)

    ht = torch.linspace(-0.5, 0.5, hlength, device=device, dtype=torch.float64)
    h = torch.exp(-math.pi / 0.32**2 * ht**2)
    Lh = (hlength - 1) // 2

    tcol = N
    neta = int(round(N / 2))

    # Step 1: torch STFT (向量化)
    tfr = msst_stft_torch(x_f64, h, N, hlength, Lh, neta, tcol)  # [neta, tcol] complex128

    # Step 2: IF
    omega = estimate_if(tfr, N)

    # Step 3: refine
    omegas, omega_final = refine_if(omega, neta, num)

    # Step 4: normalization + (optional) squeeze
    freqs_hz = torch.arange(neta, device=device, dtype=torch.float32) / N * fs
    t_axis = torch.arange(tcol, device=device, dtype=torch.float32) / fs
    tfr_float = tfr.to(torch.complex64) / (N / 2.0)

    result = {
        'STFT':        tfr_float,
        'freqs':       freqs_hz,
        't':           t_axis,
        'omega_final': omega_final,
    }
    if save_trajectory:
        result['omegas'] = omegas

    if not skip_squeeze:
        mag_msst = msst_squeeze_hard(tfr_float.abs(), omega_final, neta, gamma)
        result['MSST'] = mag_msst

    return result
