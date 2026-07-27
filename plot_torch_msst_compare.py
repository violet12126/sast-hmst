"""
CUDA MSST vs Numpy MSST: 时频图对比
=====================================
CUDA 全流程 MSST (cuFFT STFT + scatter_add squeeze) vs 原始 numpy MSST。
STFT 精确匹配 numpy (diff < 3e-14)。

用法: python plot_torch_msst_compare.py
"""

import numpy as np
import torch
import time
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from models.tfr import msst, compute_renyi

# Import CUDA kernels
import msst_stft

FS = 1000
SAVE_DIR = Path('hmst_figures/cuda_msst_compare')
SAVE_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.family'] = 'sans-serif'


# ═══════════════════════════════════════════════════════════════
# CUDA MSST (STFT + IF estimation + squeeze, all GPU)
# ═══════════════════════════════════════════════════════════════

def msst_cuda(x_np, fs=FS, hlength=None, device='cuda'):
    """
    CUDA 全流程 MSST.
    STFT 精确匹配 numpy modulated-form STFT (cuFFT).
    挤压使用 torch scatter_add (与 numpy 硬挤压一致).

    Returns:
        tfr_sqz:  [F, T] float64 — 挤压 TFR 幅度
        tfr_stft: [F, T] complex128 — 归一化 STFT
        freqs:    [F] float64 — 频率轴
        timings:  dict — 各步骤耗时
    """
    x_np = np.asarray(x_np, dtype=np.float64).ravel()
    N = len(x_np)

    # Window parameters
    if hlength is None:
        hlength = min(N, 512)
    hlength = hlength + 1 - (hlength % 2)
    Lh = (hlength - 1) // 2
    neta = int(round(N / 2))
    tcol = N

    # Gaussian window
    ht = np.linspace(-0.5, 0.5, hlength)
    h_np = np.exp(-np.pi / 0.32**2 * ht**2)

    # ── STFT (CUDA cuFFT) ──
    x_t = torch.from_numpy(x_np).cuda()
    h_t = torch.from_numpy(h_np).cuda()

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    tfr = msst_stft.msst_stft_cuda(x_t, h_t, N, hlength, Lh, neta, tcol)

    if device == 'cuda':
        torch.cuda.synchronize()
    t_stft = time.perf_counter() - t0

    # ── IF estimation (torch GPU, matching numpy Step 2) ──
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Phase diff: omega = round(diff(unwrap(angle(tfr))) * N/(2pi))
    phase = torch.angle(tfr)
    # Unwrap along time
    d = torch.diff(phase, dim=-1)
    dd = torch.where(d > np.pi, d - 2*np.pi, d)
    dd = torch.where(dd < -np.pi, dd + 2*np.pi, dd)
    phase_uw = torch.cat([phase[:, :1],
                          phase[:, :1] + torch.cumsum(dd, dim=-1)], dim=-1)
    d_phase = torch.diff(phase_uw, dim=-1)
    omega = d_phase * N / (2.0 * np.pi)
    omega = torch.round(omega).to(torch.int32)
    omega = torch.cat([omega, omega[:, -1:]], dim=-1)

    if device == 'cuda':
        torch.cuda.synchronize()
    t_if = time.perf_counter() - t0

    # ── Squeeze (torch scatter_add, matching numpy Step 4) ──
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    mag = tfr.abs()
    valid = (omega >= 1) & (omega <= neta) & (mag > 1e-4)
    f_idx, t_idx = torch.where(valid)
    k_idx = omega[f_idx, t_idx] - 1

    Tx = torch.zeros(neta, tcol, dtype=torch.float64, device=device)
    Tx.index_put_((k_idx, t_idx), mag[f_idx, t_idx].double(), accumulate=True)

    # Normalize: /(N/2)
    Tx = Tx / (N / 2.0)
    tfr_norm = tfr / (N / 2.0)

    if device == 'cuda':
        torch.cuda.synchronize()
    t_sqz = time.perf_counter() - t0

    freqs = np.arange(neta, dtype=np.float64) / N * fs

    return (Tx.cpu().numpy(),
            tfr_norm.cpu().numpy(),
            freqs,
            dict(stft=t_stft, if_est=t_if, squeeze=t_sqz))


# ═══════════════════════════════════════════════════════════════
# Main comparison
# ═══════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("[ERROR] CUDA required")
        return

    T_SIG = 2000
    print(f"CUDA MSST vs Numpy MSST — TFR Comparison")
    print(f"T={T_SIG}, fs={FS} Hz")
    print("=" * 60)

    # Test signal
    t_axis = np.arange(T_SIG) / FS
    x = (np.sin(2*np.pi*48*t_axis + 0.15*np.sin(2*np.pi*3*t_axis)) +
         0.6*np.sin(2*np.pi*96*t_axis) +
         0.25*np.sin(2*np.pi*12*t_axis)).astype(np.float64)
    print(f"Signal: {T_SIG} samples, {T_SIG/FS:.1f}s")

    # ── Numpy MSST ──
    print("\n[1] Numpy MSST (original)...")
    t0 = time.perf_counter()
    r_np = msst(x, FS, num=1, save_trajectory=True)
    t_np = time.perf_counter() - t0
    tfr_np = r_np['MSST']
    re_np = compute_renyi(tfr_np)
    print(f"  Time:  {t_np*1000:.0f} ms, shape: {tfr_np.shape}, Renyi={re_np:.2f}")

    # ── CUDA MSST (warmup first) ──
    print("\n[2] CUDA MSST (GPU)...")
    _ = msst_cuda(x, FS)  # warmup (CUDA init)
    tfr_cuda, stft_cuda, freqs_cuda, timings = msst_cuda(x, FS)
    t_total = sum(timings.values())
    re_cuda = compute_renyi(tfr_cuda)
    print(f"  STFT:    {timings['stft']*1000:.1f} ms")
    print(f"  IF est:  {timings['if_est']*1000:.1f} ms")
    print(f"  Squeeze: {timings['squeeze']*1000:.1f} ms")
    print(f"  Total:   {t_total*1000:.1f} ms, shape: {tfr_cuda.shape}, Renyi={re_cuda:.2f}")

    # ── Compare ──
    F_min = min(tfr_np.shape[0], tfr_cuda.shape[0])
    T_min = min(tfr_np.shape[1], tfr_cuda.shape[1])
    tfr_np_crop = tfr_np[:F_min, :T_min]
    tfr_cuda_crop = tfr_cuda[:F_min, :T_min]

    diff = np.abs(tfr_np_crop - tfr_cuda_crop)
    tfr_corr = np.corrcoef(tfr_np_crop.ravel(), tfr_cuda_crop.ravel())[0, 1]

    # Ridge check
    f48 = np.argmin(np.abs(r_np['freqs'][:F_min] - 48))
    ridge_np = np.argmax(tfr_np_crop[f48-3:f48+4, :], axis=0)
    ridge_cuda = np.argmax(tfr_cuda_crop[f48-3:f48+4, :], axis=0)
    np_jumps = np.sum(np.abs(np.diff(ridge_np.astype(float))) != 0)
    cuda_jumps = np.sum(np.abs(np.diff(ridge_cuda.astype(float))) != 0)

    print(f"\n  TFR max diff:      {diff.max():.6f}")
    print(f"  TFR mean diff:     {diff.mean():.8f}")
    print(f"  TFR correlation:   {tfr_corr:.8f}")
    print(f"  Speedup:           {t_np/t_total:.0f}x")
    print(f"  48Hz ridge jumps:  numpy={np_jumps}, cuda={cuda_jumps}")
    print(f"  Renyi:             numpy={re_np:.2f}, cuda={re_cuda:.2f}")

    if diff.max() < 1e-10:
        print("\n  >>> EXACT MATCH! <<<")
    elif tfr_corr > 0.9999:
        print("\n  >>> Near-perfect match <<<")
    else:
        print(f"\n  >>> TFR differs (corr={tfr_corr:.6f}) <<<")

    # ── Plot ──
    print("\n[3] Plotting...")
    plot_comparison(tfr_np, tfr_cuda, r_np['freqs'], freqs_cuda,
                    re_np, re_cuda, T_SIG, FS, F_min, T_min)
    print(f"  Saved to {SAVE_DIR}/")


def plot_comparison(tfr_np, tfr_cuda, freqs_np, freqs_cuda,
                    re_np, re_cuda, T_sig, FS, F_min, T_min):
    """Side-by-side TFR comparison."""
    FMAX = 200
    t_axis = np.arange(T_min) / FS
    mask_np = freqs_np[:F_min] <= FMAX
    mask_t = freqs_cuda[:F_min] <= FMAX

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    vmax = max(tfr_np[:F_min, :T_min].max(),
               tfr_cuda[:F_min, :T_min].max()) * 0.5

    for ax, data, freqs, mask, title, renyi in [
        (axes[0], tfr_np[:F_min, :T_min], freqs_np[:F_min], mask_np,
         'Numpy MSST\n(original)', re_np),
        (axes[1], tfr_cuda[:F_min, :T_min], freqs_cuda[:F_min], mask_t,
         'CUDA MSST\n(GPU)', re_cuda),
    ]:
        im = ax.pcolormesh(t_axis, freqs[mask], data[mask, :],
                          shading='gouraud', cmap='jet', vmax=vmax)
        ax.set_ylim(0, FMAX)
        ax.set_title(f'{title}\nRenyi={renyi:.2f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')

    # Panel 3: |diff|
    ax = axes[2]
    diff = np.abs(tfr_np[:F_min, :T_min] - tfr_cuda[:F_min, :T_min])
    im = ax.pcolormesh(t_axis, freqs_np[:F_min][mask_np], diff[mask_np, :],
                      shading='gouraud', cmap='hot', vmax=vmax * 0.01)
    ax.set_ylim(0, FMAX)
    corr = np.corrcoef(tfr_np[:F_min, :T_min].ravel(),
                       tfr_cuda[:F_min, :T_min].ravel())[0, 1]
    ax.set_title(f'|Numpy - CUDA|\nCorr={corr:.6f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'CUDA MSST — T={T_sig} ({T_sig/FS:.1f}s)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'cuda_vs_numpy_msst.png', dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
