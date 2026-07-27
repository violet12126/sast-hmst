"""
对比三种挤压方式的输出差异
==========================
1. 硬最近邻 (numpy MSST — 基准)
2. 硬最近邻 (CUDA kernel — 应与 1 完全一致)
3. 线性插值 (CUDA kernel — 平滑版)

用法: python test_squeeze_compare.py
"""
import numpy as np
import torch
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models.tfr import msst

FS = 1000


def msst_hard_numpy(x, fs, num=3):
    """基准: 当前 numpy MSST (硬最近邻)."""
    return msst(x, fs, num=num, save_trajectory=True)


def msst_hard_cuda(mag, omega, gamma=0.0001):
    """CUDA 硬最近邻挤压 (调用 msst_squeeze_hard kernel).

    Args:
        mag:   [B, F, T] torch tensor on CUDA
        omega: [B, F, T] torch int32 tensor on CUDA (1-indexed bin)
        gamma: threshold

    Returns:
        Tx: [B, F, T] torch tensor on CUDA
    """
    try:
        import msst_squeeze_hard
        return msst_squeeze_hard.msst_squeeze_hard(mag, omega, gamma)
    except ImportError:
        print("[WARN] msst_squeeze_hard not built — falling back to torch")
        return _msst_hard_torch(mag, omega, gamma)


def msst_linear_cuda(mag, if_hz, freqs, gamma=1e-6):
    """CUDA 线性插值挤压 (调用 msst_squeeze_linear kernel).

    Args:
        mag:      [B, F, T] torch tensor on CUDA
        if_hz:    [B, F, T] torch float32 tensor on CUDA (IF in Hz)
        freqs:    [F] frequency axis
        gamma:    threshold

    Returns:
        Tx: [B, F, T] torch tensor on CUDA
    """
    try:
        import msst_squeeze_linear
        return msst_squeeze_linear.msst_squeeze_linear(mag, if_hz, freqs, gamma)
    except ImportError:
        print("[WARN] msst_squeeze_linear not built — falling back to torch")
        return _msst_linear_torch(mag, if_hz, freqs, gamma)


def _msst_hard_torch(mag, omega, gamma=0.0001):
    """PyTorch fallback for hard squeeze."""
    B, F, T = mag.shape
    device = mag.device
    Tx = torch.zeros(B, F, T, device=device, dtype=mag.dtype)

    valid = (omega >= 1) & (omega <= F) & (mag > gamma)
    b_idx, f_idx, t_idx = torch.where(valid)
    k_idx = omega[b_idx, f_idx, t_idx] - 1
    Tx[b_idx, k_idx, t_idx] += mag[b_idx, f_idx, t_idx]
    return Tx


def _msst_linear_torch(mag, if_hz, freqs, gamma=1e-6):
    """PyTorch fallback for linear squeeze."""
    B, F, T = mag.shape
    device = mag.device
    f0 = freqs[0].item()
    df = (freqs[1] - freqs[0]).item()
    Tx = torch.zeros(B, F, T, device=device, dtype=mag.dtype)

    valid = mag > gamma
    b_idx, f_idx, t_idx = torch.where(valid)

    k_float = (if_hz[b_idx, f_idx, t_idx] - f0) / df
    k_floor = k_float.long().clamp(0, F - 1)
    alpha = (k_float - k_floor.float()).clamp(0, 1)

    Tx[b_idx, k_floor, t_idx] += (1 - alpha) * mag[b_idx, f_idx, t_idx]
    k_hi = (k_floor + 1).clamp(0, F - 1)
    Tx[b_idx, k_hi, t_idx] += alpha * mag[b_idx, f_idx, t_idx]
    return Tx


def compute_renyi(tfr, alpha=3):
    """Rényi entropy (lower = more concentrated)."""
    tfr = np.asarray(tfr).ravel()
    tfr_n = tfr / (tfr.sum() + 1e-12)
    tfr_n = tfr_n[tfr_n > 1e-12]
    if len(tfr_n) == 0:
        return float('inf')
    return 1 / (1 - alpha) * np.log2((tfr_n ** alpha).sum())


def main():
    print("=" * 70)
    print("MSST Squeeze Comparison: Hard NN vs Linear Interpolation")
    print("=" * 70)

    # ── 合成测试信号 ──
    t = np.arange(0, 1.0, 1 / FS)
    sig = (np.sin(2 * np.pi * 48 * t + 0.15 * np.sin(2 * np.pi * 3 * t)) +
           0.6 * np.sin(2 * np.pi * 96 * t) +
           0.25 * np.sin(2 * np.pi * 12 * t))
    sig = sig.astype(np.float64)
    print(f"\nSignal: T={len(sig)}, fs={FS} Hz")
    print("Components: BPF(~48 Hz FM) + 2xBPF(96 Hz) + LOW_FREQ(12 Hz)")

    # ── 1. numpy 基准 ──
    print("\n[1] Numpy MSST (hard NN, baseline)")
    t0 = time.perf_counter()
    r_np = msst_hard_numpy(sig, FS, num=5)
    t_np = time.perf_counter() - t0
    print(f"  Time:   {t_np*1000:.1f} ms")
    print(f"  Shape:  {r_np['MSST'].shape}")
    print(f"  Rényi:  {compute_renyi(r_np['MSST']):.2f}")
    print(f"  omegas: {len(r_np['omegas'])} trajectories")

    # ── Prepare CUDA data ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mag_cuda = torch.from_numpy(np.abs(r_np['STFT'])).float().unsqueeze(0).to(device)
    omega_cuda = torch.from_numpy(r_np['omega_final'].copy()).int().unsqueeze(0).to(device)
    freqs_t = torch.from_numpy(r_np['freqs']).float().to(device)

    # IF in Hz for linear squeeze
    N_sig = len(sig)
    if_hz = np.zeros_like(r_np['omega_final'], dtype=np.float32)
    valid_mask = r_np['omega_final'] >= 1
    if_hz[valid_mask] = (r_np['omega_final'][valid_mask] - 1) * FS / N_sig
    if_hz_cuda = torch.from_numpy(if_hz).float().unsqueeze(0).to(device)

    # ── 2. CUDA hard squeeze (should match numpy exactly) ──
    print("\n[2] CUDA Hard NN Squeeze")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    r_hard = msst_hard_cuda(mag_cuda, omega_cuda)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_hard = time.perf_counter() - t0
    r_hard_np = r_hard.cpu().numpy().squeeze(0)
    diff_hard = np.abs(r_np['MSST'] - r_hard_np)
    print(f"  Time:       {t_hard*1000:.2f} ms (speedup: {t_np/t_hard:.0f}x)")
    print(f"  Rényi:      {compute_renyi(r_hard_np):.2f}")
    print(f"  Max diff:   {diff_hard.max():.6f}")
    print(f"  Mean diff:  {diff_hard.mean():.8f}")
    print(f"  Match?      {'YES' if diff_hard.max() < 1e-5 else 'NO — investigate'}")

    # ── 3. CUDA linear squeeze ──
    print("\n[3] CUDA Linear Squeeze")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    r_linear = msst_linear_cuda(mag_cuda, if_hz_cuda, freqs_t)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_linear = time.perf_counter() - t0
    r_linear_np = r_linear.cpu().numpy().squeeze(0)
    diff_linear = np.abs(r_np['MSST'] - r_linear_np)
    print(f"  Time:       {t_linear*1000:.2f} ms (speedup: {t_np/t_linear:.0f}x)")
    print(f"  Rényi:      {compute_renyi(r_linear_np):.2f} (lower=better)")
    print(f"  Max diff:   {diff_linear.max():.4f}")
    print(f"  Mean diff:  {diff_linear.mean():.6f}")
    print(f"  vs Hard NN? Rényi delta: {compute_renyi(r_linear_np) - compute_renyi(r_np['MSST']):+.2f}")

    # ── 4. Summary ──
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    re_np = compute_renyi(r_np['MSST'])
    re_hard = compute_renyi(r_hard_np)
    re_linear = compute_renyi(r_linear_np)
    print(f"  {'Method':<25s} {'Rényi':>8s} {'Time':>10s} {'Match numpy?':>15s}")
    print(f"  {'-'*60}")
    print(f"  {'Numpy MSST (hard NN)':<25s} {re_np:>8.2f} {t_np*1000:>8.1f} ms  {'(baseline)':>15s}")
    print(f"  {'CUDA Hard NN':<25s} {re_hard:>8.2f} {t_hard*1000:>8.2f} ms  {'YES' if diff_hard.max() < 1e-5 else 'NO':>15s}")
    print(f"  {'CUDA Linear':<25s} {re_linear:>8.2f} {t_linear*1000:>8.2f} ms  {'N/A (improved)':>15s}")

    if re_linear < re_np:
        print(f"\n  => Linear interpolation is {re_np - re_linear:.2f} Rényi units more concentrated")
    elif re_linear > re_np:
        print(f"\n  => Linear interpolation is {re_linear - re_np:.2f} Rényi units less concentrated (unexpected)")
    else:
        print(f"\n  => Same concentration")


if __name__ == '__main__':
    main()
