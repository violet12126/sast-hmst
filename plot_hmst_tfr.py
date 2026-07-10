"""
绘制时频图对比: WSST vs STFT-SST vs HMST N=1 vs HMST N=2

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
M_SQUEEZE = 8          # 挤压迭代次数 (MSST 路线, 越大越聚集)


def compute_wsst_baseline(x_np, fs, nv=32):
    """WSST 基线: ssqueezepy.ssq_cwt, morlet 小波 (对数频率轴)."""
    import ssqueezepy as ssq
    Tx, Wx, ssq_freqs, *_ = ssq.ssq_cwt(
        x_np, wavelet='morlet', scales='log', nv=nv, fs=fs,
        padtype='reflect', squeezing='sum',
    )
    return np.abs(Tx), ssq_freqs


def compute_sst_stft_baseline(x_np, fs, n_fft, hop_length):
    """STFT 基线 (ssqueezepy, hann 窗, 线性频率轴)。

    返回 STFT 幅值 (与 PyTorch STFT 同尺度, 无挤压稀疏问题)。
    SST(Tx) 天然 30% 零值 → dB 转换后产生斑点; 改用 STFT(Wx) 保证
    背景连续均匀, 同时仍作为"STFT-based"对照。
    """
    import ssqueezepy as ssq
    Tx, Wx, ssq_freqs, *_ = ssq.ssq_stft(
        x_np, fs=fs, window='hann', n_fft=n_fft,
        win_len=n_fft, hop_len=hop_length,
        squeezing='sum', padtype='reflect',
    )
    return np.abs(Wx) * (2.0 / n_fft), np.asarray(ssq_freqs).squeeze()


def main():
    save_dir = Path('hmst_figures')
    save_dir.mkdir(exist_ok=True)

    # ── 设备选择: 有 GPU 则整条 HMST (STFT + squeeze kernel) 走 CUDA ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        from models.sast import _load_hmst_cuda
        ext = _load_hmst_cuda()
        if ext is not None:
            print("  CUDA squeeze kernel: 已加载 ✓")
        else:
            print("  CUDA squeeze kernel: 不可用 → GPU 上跑 Python fallback (慢)")
            print("    提示: 在 MSVC 环境先执行 "
                  "`python deploy/compile_hmst_cuda.py` 预编译 kernel")
    else:
        print("  ⚠ 无 CUDA, HMST 在 CPU 上运行 (Python squeeze fallback, 较慢)")

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
    print(f"Computing HMST (M={M_SQUEEZE}) + baselines for {len(samples)} samples...")
    hmst_data = {}
    wsst_data = {}
    sst_data = {}
    for c, idx, signal in samples:
        x = torch.from_numpy(signal).float().unsqueeze(0).to(device)
        x_np = signal
        T_in = len(signal)

        tfr1, _, _ = compute_hmst(x, fs, n_fft=n_fft, hop_length=hop_length,
                                   order=1, M=M_SQUEEZE, sigma=sigma)
        tfr2, _, _ = compute_hmst(x, fs, n_fft=n_fft, hop_length=hop_length,
                                   order=2, M=M_SQUEEZE, sigma=sigma)
        hmst_data[c] = (tfr1, tfr2)

        try:
            w_mag, w_freqs = compute_wsst_baseline(x_np, fs, nv=32)
            wsst_data[c] = (w_mag, w_freqs, T_in)
        except Exception:
            wsst_data[c] = None

        try:
            s_mag, s_freqs = compute_sst_stft_baseline(x_np, fs, n_fft, hop_length)
            sst_data[c] = (s_mag, s_freqs, T_in)
        except Exception:
            sst_data[c] = None

    # ── 公共坐标 ──
    T_if = hmst_data[0][0].shape[2]
    freqs_hz = np.linspace(0, fs / 2, hmst_data[0][0].shape[1])
    t_hmst = np.arange(T_if) * hop_length / fs + n_fft / (2 * fs)

    def baseline_db(bl):
        """基线 (WSST / STFT-SST) → (t_ax, f_ax, db); 缺失则返回占位。"""
        if bl is not None:
            mag, freqs, T_in = bl
            t_ax = np.linspace(0, T_in / fs, mag.shape[1])
            return t_ax, freqs, 10 * np.log10(mag + 1e-12)
        return t_hmst, freqs_hz, np.full((len(freqs_hz), len(t_hmst)), -120.0)

    # ── 逐类 1×4 图 ──
    print("Plotting per-class figures...")
    for c, idx, _ in samples:
        tfr1, tfr2 = hmst_data[c]
        db1 = 10 * np.log10(tfr1[0].cpu().numpy() * stft_norm + 1e-12)
        db2 = 10 * np.log10(tfr2[0].cpu().numpy() * stft_norm + 1e-12)
        tw, fw, dbw = baseline_db(wsst_data[c])
        ts, fsq, dbs = baseline_db(sst_data[c])

        fig, axes = plt.subplots(1, 4, figsize=(28, 6))

        configs = [
            (tw, fw, dbw, f'WSST (Wavelet SST)\nClass {c} (#{idx})'),
            (ts, fsq, dbs, f'STFT\nClass {c} (#{idx})'),
            (t_hmst, freqs_hz, db1, f'HMST N=1 M={M_SQUEEZE}\nClass {c} (#{idx})'),
            (t_hmst, freqs_hz, db2, f'HMST N=2 M={M_SQUEEZE}\nClass {c} (#{idx})'),
        ]
        for ax, (t_ax, f_ax, db, title) in zip(axes, configs):
            ax.pcolormesh(t_ax, f_ax, db, shading='gouraud',
                          cmap=CMAP, vmin=vmin, vmax=vmax)
            ax.set_ylim(0, freq_max)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title(title)

        tags = ['CWT log scale', 'STFT linear\n(hann window)',
                'STFT linear\n(1st-order IF)', 'STFT linear\n(2nd-order IF)']
        for ax, tag in zip(axes, tags):
            ax.text(0.98, 0.02, tag, transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=7, color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.4))

        plt.suptitle(f'WSST vs STFT vs HMST N=1/2 (M={M_SQUEEZE}) '
                     f'— Class {c} (Sample #{idx})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fn = save_dir / f'hmst_class{c}_sample{idx}.png'
        plt.savefig(fn, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {fn}')

    # ── IF 叠加图 (N=1 vs N=2) ──
    print("\nPlotting IF overlay...")
    c1_idx = np.where(y == 1)[0]
    energies = np.sum(X[c1_idx] ** 2, axis=1)
    best_c1 = c1_idx[np.argmax(energies)]
    signal = X[best_c1]
    x = torch.from_numpy(signal).float().unsqueeze(0).to(device)

    IF1, mag = compute_hmst_if(x, fs, n_fft=n_fft, hop_length=hop_length,
                                order=1, sigma=sigma)
    IF2, _ = compute_hmst_if(x, fs, n_fft=n_fft, hop_length=hop_length,
                              order=2, sigma=sigma)

    mag_np = mag[0, :, 1:-1].cpu().numpy()
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
                       cmap=CMAP, vmin=vmin, vmax=vmax)

        for f_idx in range(8, min(120, IF.shape[1]), 4):
            if_val = IF[0, f_idx].cpu().numpy()
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

    # ── 全类汇总 5×4 ──
    print("\nPlotting summary...")
    fig, axes = plt.subplots(5, 4, figsize=(28, 28))

    for ci, (c, idx, _) in enumerate(samples):
        tfr1, tfr2 = hmst_data[c]
        db1 = 10 * np.log10(tfr1[0].cpu().numpy() * stft_norm + 1e-12)
        db2 = 10 * np.log10(tfr2[0].cpu().numpy() * stft_norm + 1e-12)
        tw, fw, dbw = baseline_db(wsst_data[c])
        ts, fsq, dbs = baseline_db(sst_data[c])

        for col, (t_ax, f_ax, db, title) in enumerate([
            (tw, fw, dbw, 'WSST\n(Wavelet SST)'),
            (ts, fsq, dbs, 'STFT\n(ssqueezepy)'),
            (t_hmst, freqs_hz, db1, f'HMST N=1 M={M_SQUEEZE}\n(1st-order IF)'),
            (t_hmst, freqs_hz, db2, f'HMST N=2 M={M_SQUEEZE}\n(2nd-order IF)'),
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

    plt.suptitle(f'WSST vs STFT vs HMST N=1/2 (M={M_SQUEEZE}) '
                 f'(5 classes × 4 methods)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fn = save_dir / 'hmst_summary_5x4.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fn}')

    print(f"\nDone! All figures saved to {save_dir}/")


if __name__ == '__main__':
    main()
