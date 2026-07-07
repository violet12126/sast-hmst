"""
绘制 HMST 时频图对比: WSST vs HMST N=1 vs HMST N=2

用法: python plot_hmst_tfr.py
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from models.sast import compute_hmst, compute_hmst_if

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9

CMAP = 'jet'


def compute_wsst_baseline(x_np, fs, nv=32):
    """WSST 基线: ssqueezepy.ssq_cwt, morlet 小波."""
    import ssqueezepy as ssq
    Tx, Wx, ssq_freqs, *_ = ssq.ssq_cwt(
        x_np, wavelet='morlet', scales='log', nv=nv, fs=fs,
        padtype='reflect', squeezing='sum',
    )
    return np.abs(Tx), ssq_freqs


def main():
    save_dir = Path('hmst_figures')
    save_dir.mkdir(exist_ok=True)

    data = np.load('5_dataset.npz', allow_pickle=True)
    X, y = data['train_X'], data['train_y']
    if X.ndim == 3:
        X = X[:, :, 0]

    fs = 1000
    n_fft = 512
    hop_length = 128
    sigma = n_fft / 8.0
    freq_max = 200
    vmin, vmax = -30, 10

    # STFT 幅值归一化: PyTorch STFT (normalized=False) 补偿 2/n_fft
    stft_norm = 2.0 / n_fft

    samples = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        energies = np.sum(X[idx_c] ** 2, axis=1)
        best_idx = idx_c[np.argmax(energies)]
        samples.append((c, best_idx, X[best_idx]))

    # ── 预计算 ──
    print(f"Computing HMST for {len(samples)} samples...")
    hmst_data = {}
    wsst_data = {}
    for c, idx, signal in samples:
        x = torch.from_numpy(signal).float().unsqueeze(0)
        x_np = signal
        T_in = len(signal)

        tfr1, _, _ = compute_hmst(x, fs, n_fft=n_fft, hop_length=hop_length,
                                   order=1, M=2, sigma=sigma)
        tfr2, _, _ = compute_hmst(x, fs, n_fft=n_fft, hop_length=hop_length,
                                   order=2, M=2, sigma=sigma)
        hmst_data[c] = (tfr1, tfr2)

        try:
            w_mag, w_freqs = compute_wsst_baseline(x_np, fs, nv=32)
            wsst_data[c] = (w_mag, w_freqs, T_in)
        except Exception:
            wsst_data[c] = None

    # ── 公共坐标 ──
    T_if = hmst_data[0][0].shape[2]
    freqs_hz = np.linspace(0, fs / 2, hmst_data[0][0].shape[1])
    t_hmst = np.arange(T_if) * hop_length / fs + n_fft / (2 * fs)

    # ── 逐类 1×3 图 ──
    print("Plotting per-class figures...")
    for c, idx, _ in samples:
        tfr1, tfr2 = hmst_data[c]
        db1 = 10 * np.log10(tfr1[0].numpy() * stft_norm + 1e-12)
        db2 = 10 * np.log10(tfr2[0].numpy() * stft_norm + 1e-12)

        if wsst_data[c] is not None:
            w_mag, w_freqs, T_in = wsst_data[c]
            t_wsst = np.linspace(0, T_in / fs, w_mag.shape[1])
            db_w = 10 * np.log10(w_mag + 1e-12)
        else:
            w_freqs = freqs_hz
            t_wsst = t_hmst
            db_w = np.full((len(freqs_hz), len(t_hmst)), -120.0)

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        configs = [
            (t_wsst, w_freqs, db_w,
             f'WSST (Wavelet SST)\nClass {c} (#{idx})'),
            (t_hmst, freqs_hz, db1,
             f'HMST N=1 M=2\nClass {c} (#{idx})'),
            (t_hmst, freqs_hz, db2,
             f'HMST N=2 M=2\nClass {c} (#{idx})'),
        ]
        for ax, (t_ax, f_ax, db, title) in zip(axes, configs):
            ax.pcolormesh(t_ax, f_ax, db, shading='gouraud',
                          cmap=CMAP, vmin=vmin, vmax=vmax)
            ax.set_ylim(0, freq_max)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title(title)

        # 标注频率轴类型
        axes[0].text(0.98, 0.02, 'CWT log scale', transform=axes[0].transAxes,
                     ha='right', va='bottom', fontsize=7, color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.4))
        axes[1].text(0.98, 0.02, 'STFT linear\n(1st-order IF)', transform=axes[1].transAxes,
                     ha='right', va='bottom', fontsize=7, color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.4))
        axes[2].text(0.98, 0.02, 'STFT linear\n(2nd-order IF)', transform=axes[2].transAxes,
                     ha='right', va='bottom', fontsize=7, color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.4))

        plt.suptitle(f'WSST vs HMST N=1 vs HMST N=2 — Class {c} (Sample #{idx})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fn = save_dir / f'hmst_class{c}_sample{idx}.png'
        plt.savefig(fn, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {fn}')

    # ── IF 叠加图 ──
    print("\nPlotting IF overlay...")
    c1_idx = np.where(y == 1)[0]
    energies = np.sum(X[c1_idx] ** 2, axis=1)
    best_c1 = c1_idx[np.argmax(energies)]
    signal = X[best_c1]
    x = torch.from_numpy(signal).float().unsqueeze(0)

    IF1, mag = compute_hmst_if(x, fs, n_fft=n_fft, hop_length=hop_length,
                                order=1, sigma=sigma)
    IF2, _ = compute_hmst_if(x, fs, n_fft=n_fft, hop_length=hop_length,
                              order=2, sigma=sigma)

    mag_np = mag[0, :, 1:-1].numpy()
    mag_db = 10 * np.log10(mag_np + 1e-12)
    freqs_stft = torch.linspace(0, fs / 2, mag.shape[1])
    t_if = np.arange(IF1.shape[2]) * hop_length / fs + n_fft / (2 * fs)
    threshold = np.percentile(mag_np, 75)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, IF, title in zip(
        axes,
        [IF1, IF2],
        ['HMST N=1 — IF Estimates', 'HMST N=2 — IF Estimates']
    ):
        ax.pcolormesh(t_if, freqs_stft, mag_db, shading='gouraud',
                       cmap=CMAP, vmin=-30, vmax=10)

        for f_idx in range(8, min(120, IF.shape[1]), 4):
            if_val = IF[0, f_idx].numpy()
            mask = mag_np[f_idx] > threshold
            if mask.sum() > 3:
                ax.scatter(t_if[mask], if_val[mask], s=0.6, c='white',
                           alpha=0.6, rasterized=True, linewidths=0)

        ax.set_ylim(0, 250)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(title)

    plt.suptitle(f'HMST IF Estimation — Class 1 (Sample #{best_c1})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fn = save_dir / 'hmst_if_overlay_class1.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fn}')

    # ── 全类汇总 5×3 ──
    print("\nPlotting summary...")
    fig, axes = plt.subplots(5, 3, figsize=(21, 28))

    for ci, (c, idx, _) in enumerate(samples):
        tfr1, tfr2 = hmst_data[c]
        db1 = 10 * np.log10(tfr1[0].numpy() * stft_norm + 1e-12)
        db2 = 10 * np.log10(tfr2[0].numpy() * stft_norm + 1e-12)

        if wsst_data[c] is not None:
            w_mag, w_freqs, T_in = wsst_data[c]
            t_wsst = np.linspace(0, T_in / fs, w_mag.shape[1])
            db_w = 10 * np.log10(w_mag + 1e-12)
        else:
            w_freqs = freqs_hz
            t_wsst = t_hmst
            db_w = np.full((len(freqs_hz), len(t_hmst)), -120.0)

        for col, (t_ax, f_ax, db, title) in enumerate([
            (t_wsst, w_freqs, db_w, 'WSST\n(Wavelet SST)'),
            (t_hmst, freqs_hz, db1, 'HMST N=1 M=2\n(1st-order IF)'),
            (t_hmst, freqs_hz, db2, 'HMST N=2 M=2\n(2nd-order IF)'),
        ]):
            ax = axes[ci, col]
            ax.pcolormesh(t_ax, f_ax, db, shading='gouraud',
                          cmap=CMAP, vmin=vmin, vmax=vmax)
            ax.set_ylim(0, freq_max)
            ax.set_ylabel('Freq [Hz]')
            if ci == 0:
                ax.set_title(title)
            if ci == 4:
                ax.set_xlabel('Time [s]')
            if col == 0:
                ax.text(0.02, 0.95, f'Class {c}', transform=ax.transAxes,
                        va='top', fontweight='bold', fontsize=11,
                        color='white',
                        bbox=dict(boxstyle='round', facecolor='black',
                                  alpha=0.6))

    plt.suptitle('WSST vs HMST N=1 vs HMST N=2 (5 classes × 3 methods)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fn = save_dir / 'hmst_summary_5x3.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fn}')

    print(f"\nDone! All figures saved to {save_dir}/")


if __name__ == '__main__':
    main()
