"""
CUDA MSST on 5_dataset.npz — each class 1 sample TFR
=====================================================
用法: python plot_dataset_tfr_cuda.py
"""

import numpy as np
import torch
import time
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import CUDA MSST
import msst_stft
from models.tfr import compute_renyi

FS = 1000
N_CLASSES = 5
CLASS_NAMES = ['No-load (0)', 'Low load (1)', 'Mid load (2)',
               'High load (3)', 'Pumping (4)']

SAVE_DIR = Path('hmst_figures/dataset_cuda_msst')
SAVE_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.family'] = 'sans-serif'


def msst_cuda(x_np, hlength=None, device='cuda'):
    """CUDA 全流程 MSST."""
    x_np = np.asarray(x_np, dtype=np.float64).ravel()
    N = len(x_np)

    if hlength is None:
        hlength = min(N, 512)
    hlength = hlength + 1 - (hlength % 2)
    Lh = (hlength - 1) // 2
    neta = int(round(N / 2))
    tcol = N

    ht = np.linspace(-0.5, 0.5, hlength)
    h_np = np.exp(-np.pi / 0.32**2 * ht**2)

    # STFT
    x_t = torch.from_numpy(x_np).cuda()
    h_t = torch.from_numpy(h_np).cuda()
    tfr = msst_stft.msst_stft_cuda(x_t, h_t, N, hlength, Lh, neta, tcol)

    # IF estimation
    phase = torch.angle(tfr)
    d = torch.diff(phase, dim=-1)
    dd = torch.where(d > np.pi, d - 2*np.pi, d)
    dd = torch.where(dd < -np.pi, dd + 2*np.pi, dd)
    phase_uw = torch.cat([phase[:, :1],
                          phase[:, :1] + torch.cumsum(dd, dim=-1)], dim=-1)
    d_phase = torch.diff(phase_uw, dim=-1)
    omega = torch.round(d_phase * N / (2.0 * np.pi)).int()
    omega = torch.cat([omega, omega[:, -1:]], dim=-1)

    # Squeeze
    mag = tfr.abs()
    valid = (omega >= 1) & (omega <= neta) & (mag > 1e-4)
    f_idx, t_idx = torch.where(valid)
    k_idx = omega[f_idx, t_idx] - 1
    Tx = torch.zeros(neta, tcol, dtype=torch.float64, device=device)
    Tx.index_put_((k_idx, t_idx), mag[f_idx, t_idx].double(), accumulate=True)
    Tx = Tx / (N / 2.0)

    freqs = np.arange(neta, dtype=np.float64) / N * FS
    return Tx.cpu().numpy(), freqs


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("[ERROR] CUDA required")
        return

    # ── Load data ──
    print("Loading 5_dataset.npz...")
    data = np.load('5_dataset.npz')
    X_all = np.concatenate([data['train_X'], data['test_X']], axis=0)
    y_all = np.concatenate([data['train_y'], data['test_y']], axis=0)
    T = X_all.shape[1]
    print(f"  {X_all.shape[0]} samples, T={T}, classes={np.unique(y_all)}")

    # ── Pick 1 sample per class ──
    samples = {}
    for cls in range(N_CLASSES):
        idx = np.where(y_all == cls)[0][0]
        samples[cls] = X_all[idx].astype(np.float64)
        print(f"  Class {cls}: sample {idx}")

    # ── Run CUDA MSST on all 5 ──
    print("\nRunning CUDA MSST...")
    # Warmup
    _ = msst_cuda(samples[0])

    results = {}
    t_total = 0
    for cls in range(N_CLASSES):
        t0 = time.perf_counter()
        tfr, freqs = msst_cuda(samples[cls])
        torch.cuda.synchronize()
        t_elapsed = time.perf_counter() - t0
        t_total += t_elapsed
        results[cls] = dict(tfr=tfr, freqs=freqs, time=t_elapsed,
                           renyi=compute_renyi(tfr))
        print(f"  Class {cls}: {t_elapsed*1000:.1f} ms, R={results[cls]['renyi']:.2f}")

    print(f"  Total: {t_total*1000:.0f} ms, Avg: {t_total/N_CLASSES*1000:.0f} ms/sample")

    # ── Plot ──
    print("\nPlotting...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    FMAX = 200
    t_axis = np.arange(T) / FS

    for cls in range(N_CLASSES):
        ax = axes[cls // 3, cls % 3]
        tfr = results[cls]['tfr']
        freqs = results[cls]['freqs']
        mask = freqs <= FMAX

        vmax = tfr.max() * 0.4
        ax.pcolormesh(t_axis, freqs[mask], tfr[mask, :],
                      shading='gouraud', cmap='jet', vmax=vmax)
        ax.set_xlim(0, T/FS)
        ax.set_ylim(0, FMAX)
        ax.grid(True, alpha=0.2)
        re_str = results[cls]['renyi']
        t_str = results[cls]['time'] * 1000
        ax.set_title(f'{CLASS_NAMES[cls]}\nR={re_str:.2f}, {t_str:.0f}ms',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)')

    # Turn off empty 6th panel
    axes[1, 2].axis('off')

    plt.suptitle(f'CUDA MSST — 5_dataset.npz (T={T}, fs={FS}Hz, one sample per class)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'dataset_cuda_msst_5class.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Also save individual full-size plots
    for cls in range(N_CLASSES):
        fig, ax = plt.subplots(figsize=(12, 8))
        tfr = results[cls]['tfr']
        freqs = results[cls]['freqs']
        mask = freqs <= FMAX
        vmax = tfr.max() * 0.4
        ax.pcolormesh(t_axis, freqs[mask], tfr[mask, :],
                      shading='gouraud', cmap='jet', vmax=vmax)
        ax.set_xlim(0, T/FS)
        ax.set_ylim(0, FMAX)
        ax.grid(True, alpha=0.2)
        re_str = results[cls]['renyi']
        t_str = results[cls]['time'] * 1000
        ax.set_title(f'{CLASS_NAMES[cls]} — CUDA MSST\n'
                     f'R={re_str:.2f}, {t_str:.0f}ms',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f'dataset_cuda_msst_class{cls}.png', dpi=200,
                    bbox_inches='tight')
        plt.close()

    print(f"  Saved to {SAVE_DIR}/")


if __name__ == '__main__':
    main()
