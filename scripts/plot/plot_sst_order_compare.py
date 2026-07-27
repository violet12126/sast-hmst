"""
不同 SST 阶数的完整时频图对比 (0-2s, 0-200Hz)
===============================================
N=1~5 阶 SST 在同一信号上的全貌 TFR。

用法: python plot_sst_order_compare.py
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
from models.tfr import msst, compute_renyi
import msst_stft

FS = 1000
SAVE_DIR = Path('hmst_figures/sst_order_compare')
SAVE_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.family'] = 'sans-serif'


def msst_cuda_with_orders(x_np, hlength=None, device='cuda'):
    """CUDA MSST: 返回 STFT + 5 层 omega (N=1..5)."""
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

    x_t = torch.from_numpy(x_np).cuda()
    h_t = torch.from_numpy(h_np).cuda()
    tfr = msst_stft.msst_stft_cuda(x_t, h_t, N, hlength, Lh, neta, tcol)

    # omega_0 (N=1)
    phase = torch.angle(tfr)
    d = torch.diff(phase, dim=-1)
    dd = torch.where(d > np.pi, d - 2*np.pi, d)
    dd = torch.where(dd < -np.pi, dd + 2*np.pi, dd)
    pu = torch.cat([phase[:, :1], phase[:, :1] + torch.cumsum(dd, dim=-1)], dim=-1)
    omega_0 = torch.round(torch.diff(pu, dim=-1) * N / (2.0 * np.pi)).int()
    omega_0 = torch.cat([omega_0, omega_0[:, -1:]], dim=-1)

    omegas = [omega_0]
    omega_cur = omega_0
    for _ in range(4):
        omega_next = torch.zeros_like(omega_cur)
        valid = (omega_cur >= 1) & (omega_cur <= neta)
        fi, ti = torch.where(valid)
        kv = omega_cur[fi, ti] - 1
        omega_next[fi, ti] = omega_cur[kv, ti]
        omega_cur = omega_next
        omegas.append(omega_cur.clone())

    freqs = np.arange(neta, dtype=np.float64) / N * FS
    return tfr.cpu().numpy(), [o.cpu().numpy() for o in omegas], freqs


def squeeze_with_order(tfr_np, omega_np, N_sig):
    """硬挤压."""
    neta, tcol = omega_np.shape
    Ts = np.zeros((neta, tcol), dtype=np.complex128)
    for b in range(tcol):
        for eta in range(neta):
            if np.abs(tfr_np[eta, b]) > 1e-4:
                k = omega_np[eta, b]
                if 1 <= k <= neta:
                    Ts[k - 1, b] += tfr_np[eta, b]
    return np.abs(Ts / (N_sig / 2.0))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("[ERROR] CUDA required"); return

    T = 2000
    t = np.arange(T) / FS

    # ── Test signal ──
    x_test = (
        np.sin(2 * np.pi * 48 * t + 0.15 * np.sin(2 * np.pi * 3 * t))
        + 0.6 * np.sin(2 * np.pi * 96 * t)
        + 0.25 * np.sin(2 * np.pi * 12 * t)
    ).astype(np.float64)

    # ── Real signal (Class 2: mid-load) ──
    data = np.load('5_dataset.npz')
    X = np.concatenate([data['train_X'], data['test_X']], axis=0)
    y = np.concatenate([data['train_y'], data['test_y']], axis=0)
    idx_real = np.where(y == 2)[0][0]
    x_real = X[idx_real].astype(np.float64)

    for label, x in [("Test Signal (synthetic)", x_test),
                      ("Real Signal (Class 2, mid-load)", x_real)]:
        print(f"\n{'='*60}")
        print(f"{label}  (T={T}, 0-{T/FS:.1f}s)")
        print(f"{'='*60}")

        # Compute all orders
        print("Computing MSST with N=1..5...")
        tfr, omegas, freqs = msst_cuda_with_orders(x)
        N_sig = len(x)
        neta = tfr.shape[0]

        orders = ['N=1', 'N=2', 'N=3', 'N=4', 'N=5']
        tfrs = {}
        renyi_vals = {}
        for oi, oname in enumerate(orders):
            tfrs[oname] = squeeze_with_order(tfr, omegas[oi], N_sig)
            renyi_vals[oname] = compute_renyi(tfrs[oname])
            print(f"  {oname}: R_all={renyi_vals[oname]:.2f}")

        # ── Plot: 1 row × 5 cols, full 0-200Hz TFR ──
        print("Plotting...")
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        FMAX = 200
        t_axis = np.arange(T) / FS
        mask = freqs[:neta] <= FMAX

        vmax = max(tfrs[o][mask, :].max() for o in orders) * 0.4

        for cidx, oname in enumerate(orders):
            ax = axes[cidx]
            data = tfrs[oname][mask, :]
            f_ax = freqs[:neta][mask]

            ax.pcolormesh(t_axis, f_ax, data,
                         shading='gouraud', cmap='jet', vmax=vmax)
            ax.set_ylim(0, FMAX)
            ax.set_xlim(0, T / FS)
            ax.set_xlabel('Time (s)', fontsize=9)
            if cidx == 0:
                ax.set_ylabel('Frequency (Hz)', fontsize=9)

            # Annotate component regions
            for f_lo, f_hi, c, lbl in [
                (8, 20, 'cyan', 'LOW_FREQ'), (42, 55, 'yellow', 'BPF'),
                (90, 105, 'lime', '2xBPF')]:
                ax.axhspan(f_lo, f_hi, alpha=0.08, color=c)
                ax.text(0.02, f_lo + 2, lbl, fontsize=6, color=c, va='bottom',
                       fontweight='bold')

            ax.set_title(f'{oname}\nR={renyi_vals[oname]:.2f}',
                        fontsize=11, fontweight='bold')

        plt.suptitle(f'{label}\nFull TFR 0-200 Hz | N=1..5 SST',
                     fontsize=14, fontweight='bold', y=1.03)
        plt.tight_layout()
        fname = label.split('(')[0].strip().replace(' ', '_').lower()
        plt.savefig(SAVE_DIR / f'sst_order_{fname}.png', dpi=200, bbox_inches='tight')
        plt.close()

    print(f"\nDone. Saved to {SAVE_DIR}/")


if __name__ == '__main__':
    main()
