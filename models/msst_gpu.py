"""
GPU-Accelerated MSST Pipeline
=============================
使用 CUDA 内核 + PyTorch GPU 计算, 完整替代 numpy msst().

数据流:
  1. msst_stft_cuda    → cuFFT modulated-form STFT
  2. estimate_if_gpu   → PyTorch GPU phase unwrap + diff → omega [F, T]
  3. refine_if_gpu     → PyTorch GPU IF refinement (num-1 次迭代)
  4. msst_squeeze_linear → CUDA 线性插值挤压 (连续 IF Hz, 双 bin)

用法:
  from models.msst_gpu import msst_gpu

  x = torch.randn(2000, device='cuda')  # 单条信号
  result = msst_gpu(x, fs=1000, num=3)
  # result['MSST']: [F, T] magnitude
  # result['STFT']: [F, T] complex
  # result['omega_final']: [F, T] int32 bin indices
"""

import torch
import math
import sys
from pathlib import Path
from typing import Optional, List, Dict

# ── torch must be imported before CUDA kernels (DLL dependency) ──
_ = torch  # ensure torch DLLs are loaded before .pyd import

# ── Import CUDA kernels (from deploy/*.pyd) ──
import deploy.msst_stft as _msst_stft_mod
import deploy.msst_squeeze_linear as _msst_sqz_lin_mod

msst_stft_cuda = _msst_stft_mod.msst_stft_cuda
msst_squeeze_linear_cuda = _msst_sqz_lin_mod.msst_squeeze_linear


# ═══════════════════════════════════════════════════════════════════
# Phase Unwrap (PyTorch GPU, numpy-compatible)
# ═══════════════════════════════════════════════════════════════════

def phase_unwrap(phase: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """NumPy np.unwrap 的 PyTorch 实现, 支持任意形状, GPU 可用.

    Args:
        phase: [..., T] phase angles in radians
        dim:   unwrap dimension

    Returns:
        unwrapped: same shape as phase
    """
    dd = torch.diff(phase, dim=dim)                          # [..., T-1]
    ddmod = (dd + math.pi) % (2 * math.pi) - math.pi         # wrap to [-pi, pi]
    ph_correct = ddmod - dd
    ph_correct[dd.abs() < math.pi] = 0.0                     # no correction where |dd| < pi

    up = phase.clone()
    slices_start = [slice(None)] * phase.ndim
    slices_start[dim] = slice(1, None)
    up[tuple(slices_start)] = up[tuple(slices_start)] + torch.cumsum(ph_correct, dim=dim)
    return up


# ═══════════════════════════════════════════════════════════════════
# GPU STFT (wrapper around CUDA kernel)
# ═══════════════════════════════════════════════════════════════════

def msst_stft_gpu(x: torch.Tensor, h_window: torch.Tensor,
                  N: int, hlength: int, Lh: int,
                  neta: int, tcol: int) -> torch.Tensor:
    """GPU modulated-form STFT, exact match to numpy msst() Step 1.

    Args:
        x:        [N] float64 on CUDA
        h_window: [hlength] float64 on CUDA (Gaussian window)
        N, hlength, Lh, neta, tcol: STFT parameters

    Returns:
        tfr: [neta, tcol] complex128 on CUDA
    """
    return msst_stft_cuda(x, h_window, N, hlength, Lh, neta, tcol)


# ═══════════════════════════════════════════════════════════════════
# IF Estimation (PyTorch GPU)
# ═══════════════════════════════════════════════════════════════════

def estimate_if_gpu(tfr_complex: torch.Tensor, N: int) -> torch.Tensor:
    """从复数 STFT 估计一阶 IF (相位差分法).

    Args:
        tfr_complex: [F, T] complex tensor on GPU
        N:           signal length (for normalization)

    Returns:
        omega: [F, T] int32 (1-indexed bin, 0=invalid), on GPU
    """
    F, T_col = tfr_complex.shape
    angle = torch.angle(tfr_complex)                              # [F, T]

    # Phase unwrap per frequency row
    unwrapped = phase_unwrap(angle, dim=-1)                       # [F, T]

    # Phase difference → instantaneous frequency
    omega_cont = torch.diff(unwrapped, dim=-1) * N / (2.0 * math.pi)  # [F, T-1]

    # Pad last column (replicate)
    omega_cont = torch.cat([omega_cont, omega_cont[:, -1:]], dim=-1)  # [F, T]

    # Round to integer bin index
    omega = torch.round(omega_cont).to(torch.int32)
    return omega


# ═══════════════════════════════════════════════════════════════════
# IF Refinement (MSST core, PyTorch GPU)
# ═══════════════════════════════════════════════════════════════════

def refine_if_gpu(omega: torch.Tensor, neta: int, num: int
                  ) -> tuple:
    """MSST IF 迭代精化: omega2(η, b) = omega(omega(η, b), b).

    Args:
        omega: [F, T] int32, 1-indexed IF estimates
        neta:  number of positive frequency bins (=F)
        num:   total iterations (≥1). num=1 → 不精化.

    Returns:
        omegas:    list of [F, T] int32, length=num (trajectory)
        omega_final: [F, T] int32 (最后一步的 omega)
    """
    F, T_col = omega.shape
    omegas: List[torch.Tensor] = [omega.clone()]
    omega_cur = omega

    if num > 1:
        omega2 = torch.zeros(F, T_col, dtype=torch.int32, device=omega.device)
        for _ in range(num - 1):
            valid = (omega_cur >= 1) & (omega_cur <= neta)
            if not valid.any():
                omega2.zero_()
            else:
                # Reset omega2 for this iteration
                omega2.zero_()
                # Gather: omega2[eta, b] = omega_cur[k, b] where k = omega_cur[eta, b] - 1
                b_idx = torch.arange(T_col, device=omega.device).unsqueeze(0).expand(F, T_col)
                k_vals = (omega_cur - 1).clamp(0, F - 1)  # 0-indexed
                omega2[valid] = omega_cur[k_vals[valid], b_idx[valid]]
            omega_cur = omega2.clone()
            omegas.append(omega_cur.clone())
    else:
        omega2 = omega.clone()

    return omegas, omega2


# ═══════════════════════════════════════════════════════════════════
# Full GPU MSST Pipeline
# ═══════════════════════════════════════════════════════════════════

def msst_gpu(x: torch.Tensor, fs: float,
             hlength: Optional[int] = None,
             num: int = 3,
             save_trajectory: bool = False,
             gamma: float = 1e-6) -> Dict:
    """GPU-accelerated Multi-Synchrosqueezing Transform.

    API 兼容 models/tfr.py:msst(), 但全部在 GPU 上运行。

    算法:
      1. GPU STFT (cuFFT) → complex [F, T]
      2. PyTorch phase unwrap + diff → 1st-order IF int bins
      3. PyTorch IF refinement (num-1 次迭代)
      4. CUDA linear interpolation squeeze → MSST magnitude

    Args:
        x:       [T] 1D signal on CUDA (float32 or float64)
        fs:      采样率 (Hz)
        hlength: 窗长 (样本数), 默认 min(N, 512), 自动调整为奇数
        num:     迭代次数 (≥1), num=1 → 等价于标准一阶 SST
        save_trajectory: 是否保存中间迭代的 omega
        gamma:   幅值阈值 (默认 1e-6)

    Returns:
        dict:
          'MSST':        [F, T] MSST 幅度谱 (float32)
          'STFT':        [F, T] 复数 STFT (complex64, 归一化后)
          'freqs':       [F] 频率轴 (Hz, float32)
          't':           [T] 时间轴 (s, float32)
          'omega_final': [F, T] 最终 IF bin 索引 (int32, 1-indexed, 0=invalid)
          'omegas':      (仅 save_trajectory=True) list of [F, T] int32
    """
    if x.dim() != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")
    if not x.is_cuda:
        raise ValueError("x must be on CUDA")

    device = x.device
    N = len(x)
    x_f64 = x.double()  # CUDA kernel requires float64

    # ── Window parameters (match numpy msst) ──
    if hlength is None:
        hlength = min(N, 512)
    hlength = hlength + 1 - (hlength % 2)  # ensure odd

    # Gaussian window: h = exp(-π/0.32² · t²), t ∈ [-0.5, 0.5]
    ht = torch.linspace(-0.5, 0.5, hlength, device=device, dtype=torch.float64)
    h = torch.exp(-math.pi / 0.32**2 * ht**2)
    Lh = (hlength - 1) // 2

    tcol = N                          # hop=1 — one frame per sample
    neta = int(round(N / 2))          # positive frequency bins

    # ═══════════════════════════════════════════════════════════
    # Step 1: GPU STFT (cuFFT)
    # ═══════════════════════════════════════════════════════════
    tfr = msst_stft_cuda(x_f64, h, N, hlength, Lh, neta, tcol)  # [neta, tcol] complex128

    # ═══════════════════════════════════════════════════════════
    # Step 2: IF estimation (PyTorch GPU)
    # ═══════════════════════════════════════════════════════════
    omega = estimate_if_gpu(tfr, N)  # [neta, tcol] int32

    # ═══════════════════════════════════════════════════════════
    # Step 3: IF refinement (PyTorch GPU)
    # ═══════════════════════════════════════════════════════════
    omegas, omega_final = refine_if_gpu(omega, neta, num)

    # ═══════════════════════════════════════════════════════════
    # Step 4: Linear interpolation squeeze (CUDA)
    # ═══════════════════════════════════════════════════════════
    freqs_hz = torch.arange(neta, device=device, dtype=torch.float32) / N * fs  # [F]
    t_axis = torch.arange(tcol, device=device, dtype=torch.float32) / fs         # [T]

    # STFT magnitude (normalized)
    tfr_float = tfr.to(torch.complex64) / (N / 2.0)
    mag_stft = tfr_float.abs().unsqueeze(0)  # [1, F, T]

    # IF in Hz (continuous, for linear squeeze)
    if_map = torch.zeros(neta, tcol, device=device, dtype=torch.float32)
    valid_mask = omega_final >= 1
    if_map[valid_mask] = ((omega_final[valid_mask] - 1).float() * fs / N)

    # CUDA linear squeeze
    mag_msst = msst_squeeze_linear_cuda(mag_stft, if_map.unsqueeze(0),
                                         freqs_hz, gamma)  # [1, F, T]
    mag_msst = mag_msst.squeeze(0)  # [F, T]

    result = {
        'MSST':        mag_msst,
        'STFT':        tfr_float,
        'freqs':       freqs_hz,
        't':           t_axis,
        'omega_final': omega_final,
    }
    if save_trajectory:
        result['omegas'] = omegas

    return result
