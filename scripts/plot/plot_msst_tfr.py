"""
绘制每个样本的 WSST2 / ssqueezepy WSST / MSST / FFT 2x2 对比图
=================================================================
布局:
  左上: WSST2 (our morse wavelet)    右上: ssqueezepy WSST
  左下: MSST (STFT, num=3)           右下: FFT 频谱 + 峰值标注

每个样本单独保存为一张图片。
用法: python plot_msst_tfr.py
      python plot_msst_tfr.py --cls 3          # 仅绘制指定类别
      python plot_msst_tfr.py --cls 3 --max-samples 5  # 限制样本数
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from scipy.signal import find_peaks
import ssqueezepy as ssq
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.tfr import wsst2, msst, compute_renyi

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 7
CMAP = 'jet'
SAVE_DIR = Path('hmst_figures/msst_comparison')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FS = 1000
TOTAL = 2.0
FMAX = 500
HLENGTH = 320

CLASS_NAMES = {
    0: 'Class 0: No-load',
    1: 'Class 1: Low load',
    2: 'Class 2: Mid load',
    3: 'Class 3: High load',
    4: 'Class 4: Pumping',
}


def find_dominant_freqs(tfr, freq_axis, fmax=500, n_peaks=8,
                        prominence=0.02, min_dist=8):
    """从 TFR 时间平均谱中提取主导频率。"""
    mask = freq_axis <= fmax
    f = freq_axis[mask]
    spec = np.mean(tfr[mask, :], axis=1)
    spec = spec / (spec.max() + 1e-12)

    peaks, props = find_peaks(spec, prominence=prominence, distance=min_dist)
    if len(peaks) == 0:
        peaks, props = find_peaks(spec, prominence=prominence * 0.5,
                                  distance=min_dist // 2)
    if len(peaks) == 0:
        peaks = [np.argmax(spec)]

    sort_idx = np.argsort(spec[peaks])[::-1]
    peaks = peaks[sort_idx][:n_peaks]
    peaks_hz = f[peaks]
    peaks_amp = spec[peaks]
    return peaks_hz, peaks_amp


def plot_one_sample(x, fs, idx, cls, save_path):
    """为单个样本生成三栏 TFR + FFT 频谱 2×2 对比图并保存。"""
    x = x.astype(np.float64)
    N = len(x)
    t_axis = np.arange(N) / fs
    cls_name = CLASS_NAMES.get(cls, f'Class {cls}')

    # ── 1. WSST2 (our) ──
    r_wsst2 = wsst2(x, fs, mywav='morse')
    f_w2 = r_wsst2['freqs']
    mag_w2 = np.abs(r_wsst2['WSST2'])
    mask_w2 = f_w2 <= FMAX
    re_w2 = compute_renyi(mag_w2[mask_w2, :])
    dom_w2, _ = find_dominant_freqs(mag_w2, f_w2, FMAX, prominence=0.03, n_peaks=8)

    # ── 2. ssqueezepy WSST ──
    Tx_ssq, Wx, ssq_freqs, scales = ssq.ssq_cwt(x, fs=fs)
    f_ssq = np.asarray(ssq_freqs).squeeze()
    mag_ssq = np.abs(Tx_ssq)
    if f_ssq[0] > f_ssq[-1]:
        f_ssq = f_ssq[::-1]
        mag_ssq = mag_ssq[::-1, :]
    mask_ssq = f_ssq <= FMAX
    re_ssq = compute_renyi(mag_ssq[mask_ssq, :])

    # ── 3. MSST (our) ──
    r_msst = msst(x, fs, hlength=HLENGTH, num=1)
    f_ms = r_msst['freqs']
    mag_ms = r_msst['MSST']
    mask_ms = f_ms <= FMAX
    re_ms = compute_renyi(mag_ms[mask_ms, :])
    dom_ms, _ = find_dominant_freqs(mag_ms, f_ms, FMAX, prominence=0.03, n_peaks=8)

    # ── 4. FFT spectrum ──
    fft_freqs = np.fft.rfftfreq(N, 1 / fs)
    fft_mag = np.abs(np.fft.rfft(x))
    mask_fft = fft_freqs <= FMAX
    fft_dom, fft_dom_amp = find_dominant_freqs(
        fft_mag[mask_fft].reshape(-1, 1), fft_freqs[mask_fft],
        fmax=FMAX, prominence=0.03, n_peaks=8)

    # ═══════════════════════════════════════════
    # 绘图: 2×2
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 三个 TFR 面板
    tfr_panels = [
        (axes[0, 0], f'WSST2 (our, morse)\nR={re_w2:.2f}', mag_w2, f_w2, mask_w2, dom_w2),
        (axes[0, 1], f'ssqueezepy WSST\nR={re_ssq:.2f}', mag_ssq, f_ssq, mask_ssq, dom_w2),
        (axes[1, 0], f'MSST (STFT, num=3)\nR={re_ms:.2f}', mag_ms, f_ms, mask_ms, dom_ms),
    ]

    for ax, title, mag, f_ax, mask, dom_freqs in tfr_panels:
        vmax = mag[mask, :].max() * 0.5
        ax.pcolormesh(t_axis, f_ax[mask], mag[mask, :],
                      shading='gouraud', cmap=CMAP, vmax=vmax)

        dom_text = '\n'.join([f'{f:6.1f} Hz' for f in dom_freqs[:6]])
        ax.text(0.985, 0.97, f'Dominant:\n{dom_text}',
                transform=ax.transAxes, fontsize=5.5, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.75, edgecolor='gray', lw=0.5),
                family='monospace')

        ax.set_ylim(0, FMAX)
        ax.set_xlim(0, TOTAL)
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Freq (Hz)')

    # ── 右下: FFT 频谱 ──
    ax_fft = axes[1, 1]
    ax_fft.plot(fft_freqs[mask_fft], fft_mag[mask_fft], 'k-', lw=0.6)
    ax_fft.set_xlim(0, FMAX)

    # 标注主导频率
    y_max = fft_mag[mask_fft].max()
    for fhz in fft_dom[:8]:
        ax_fft.axvline(fhz, color='red', lw=0.6, ls='--', alpha=0.5)
        ax_fft.annotate(f'{fhz:.1f}',
                        xy=(fhz, y_max * 0.92),
                        fontsize=5.5, ha='center', va='top',
                        color='red', rotation=90,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                  alpha=0.7, edgecolor='red', lw=0.3))

    # FFT 主导频率文字标注
    fft_dom_text = '\n'.join([f'{f:6.1f} Hz  ({a:.3f})'
                              for f, a in zip(fft_dom[:6], fft_dom_amp[:6])])
    ax_fft.text(0.985, 0.97, f'FFT Peaks:\n{fft_dom_text}',
                transform=ax_fft.transAxes, fontsize=5.5, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.75, edgecolor='gray', lw=0.5),
                family='monospace')

    ax_fft.set_title(f'FFT Spectrum (0-{FMAX} Hz)', fontsize=8, fontweight='bold')
    ax_fft.set_xlabel('Freq (Hz)')
    ax_fft.set_ylabel('Magnitude')
    ax_fft.grid(True, alpha=0.2)

    plt.suptitle(f'{cls_name}  |  Sample idx={idx}  |  '
                 f'WSST2 R={re_w2:.2f}  ssq R={re_ssq:.2f}  MSST R={re_ms:.2f}',
                 fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    return re_w2, re_ssq, re_ms


def main():
    parser = argparse.ArgumentParser(description='Plot WSST2 vs ssqueezepy vs MSST per sample')
    parser.add_argument('--cls', type=int, default=None,
                        help='Only plot specified class (0-4)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per class')
    parser.add_argument('--data', default='5_dataset.npz',
                        help='Dataset NPZ path')
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    train_X = data['train_X']
    train_y = data['train_y']

    classes = [args.cls] if args.cls is not None else sorted(np.unique(train_y))

    print(f"{'='*70}")
    print(f"MSST TFR Comparison — {len(classes)} class(es)")
    print(f"Output dir: {SAVE_DIR}")
    print(f"{'='*70}")

    summary = []

    for cls in classes:
        idxs = np.where(train_y == cls)[0]
        if args.max_samples:
            idxs = idxs[:args.max_samples]

        for idx in idxs:
            x = train_X[idx]
            fname = f'class{cls}_sample{idx}.png'
            save_path = SAVE_DIR / fname
            print(f'  Class {cls} idx={idx}...', end=' ', flush=True)
            re_w2, re_ssq, re_ms = plot_one_sample(x, FS, idx, cls, save_path)
            print(f'WSST2={re_w2:.2f}  ssq={re_ssq:.2f}  MSST={re_ms:.2f}')
            summary.append((cls, idx, re_w2, re_ssq, re_ms))

    # 汇总
    print(f"\n{'='*70}")
    print(f"{'Class':<22s} {'Idx':>5s} {'WSST2':>7s} {'ssq':>7s} {'MSST':>7s} {'W2-ssq':>7s} {'W2-MSST':>8s}")
    print('-' * 70)
    for cls, idx, re_w2, re_ssq, re_ms in summary:
        d1 = re_w2 - re_ssq
        d2 = re_w2 - re_ms
        print(f"{CLASS_NAMES[cls]:<22s} {idx:5d} {re_w2:7.2f} {re_ssq:7.2f} {re_ms:7.2f} {d1:+7.2f} {d2:+8.2f}")

    print(f"\nDone. {len(summary)} figures saved to {SAVE_DIR}/")


if __name__ == '__main__':
    main()
