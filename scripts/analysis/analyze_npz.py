"""
Comprehensive analysis of normalized_data.npz
Task 1: Data structure
Task 2: Per-class FFT analysis
Task 3: Time-frequency visualization (tfr_analysis.png)
Task 4: Wide vs sharp peak bandwidth analysis
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.tfr import msst

# ============================================================
# Task 1: Data Structure
# ============================================================
print("=" * 70)
print("TASK 1: Data Structure")
print("=" * 70)

data = np.load('normalized_data.npz')
print(f"\nKeys: {list(data.keys())}")

for k in data.keys():
    arr = data[k]
    print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")

train_X = data['train_X']  # (2812, 2000)
train_y = data['train_y']  # (2812,)
test_X  = data['test_X']   # (704, 2000)
test_y  = data['test_y']   # (704,)

fs = 1000  # Hz (from train_sast.py)
T_samples = train_X.shape[1]  # 2000
T_seconds = T_samples / fs    # 2.0 s

print(f"\nSignal length: T = {T_samples} samples ({T_seconds:.1f} s)")
print(f"Sampling rate: fs = {fs} Hz")
print(f"Nyquist frequency: {fs/2} Hz")

# Class distribution
all_y = np.concatenate([train_y, test_y])
classes, counts = np.unique(all_y, return_counts=True)
print(f"\nClasses found: {classes}")
print(f"Total samples: {len(all_y)} ({len(train_y)} train, {len(test_y)} test)")
print(f"Class distribution (all):")
for c, n in zip(classes, counts):
    train_n = (train_y == c).sum()
    test_n = (test_y == c).sum()
    print(f"  Class {c}: {n:5d} total  ({train_n:5d} train, {test_n:5d} test)")

# ============================================================
# Task 2: Per-class FFT analysis
# ============================================================
print("\n" + "=" * 70)
print("TASK 2: Per-class FFT Analysis")
print("=" * 70)

# Use all data for FFT analysis
all_X = np.concatenate([train_X, test_X], axis=0)  # (3516, 2000)
all_y_combined = np.concatenate([train_y, test_y])

# Compute FFT for each sample (positive frequencies only)
N_fft = T_samples
freqs_fft = np.fft.rfftfreq(N_fft, d=1/fs)  # 0 to 500 Hz
n_freqs = len(freqs_fft)

# Per-class mean FFT magnitude
class_fft_means = {}
class_fft_stds = {}

for c in sorted(classes):
    mask = all_y_combined == c
    X_class = all_X[mask]
    n_class = X_class.shape[0]

    # Compute FFT for each sample
    fft_mags = np.abs(np.fft.rfft(X_class, axis=1))
    # Normalize by signal length
    fft_mags = fft_mags / T_samples

    mean_fft = fft_mags.mean(axis=0)
    std_fft = fft_mags.std(axis=0)

    class_fft_means[c] = mean_fft
    class_fft_stds[c] = std_fft

    print(f"\nClass {c} ({n_class} samples):")
    print(f"  Mean FFT peak value: {mean_fft.max():.4f} at {freqs_fft[mean_fft.argmax()]:.1f} Hz")

# Frequency band analysis
bands = {
    'LOW_FREQ (0-20 Hz)': (0, 20),
    'BPF (40-60 Hz)': (40, 60),
    '2xBPF (90-110 Hz)': (90, 110),
    '140-160 Hz': (140, 160),
    '200+ Hz': (200, 500),
}

print("\n" + "-" * 70)
print("Frequency Band Analysis (mean FFT amplitude per band, per class)")
print("-" * 70)

band_energies = {}
for band_name, (f_low, f_high) in bands.items():
    mask = (freqs_fft >= f_low) & (freqs_fft <= f_high)
    print(f"\n{band_name}:")
    for c in sorted(classes):
        energy = class_fft_means[c][mask].sum()
        peak_val = class_fft_means[c][mask].max()
        peak_freq = freqs_fft[mask][class_fft_means[c][mask].argmax()]
        print(f"  Class {c}: total_energy={energy:.4f}, peak={peak_val:.4f} @ {peak_freq:.1f} Hz")
        if band_name not in band_energies:
            band_energies[band_name] = {}
        band_energies[band_name][c] = {'energy': energy, 'peak': peak_val, 'peak_freq': peak_freq}

# Peak detection for each class (global, 0-500 Hz)
print("\n" + "-" * 70)
print("Global Peak Detection (0-500 Hz, per class)")
print("-" * 70)

for c in sorted(classes):
    mean_fft = class_fft_means[c]
    # Smooth slightly for peak detection
    smoothed = gaussian_filter1d(mean_fft, sigma=1.0)
    # Dynamic threshold: 5% of max
    threshold = 0.05 * smoothed.max()
    peaks, properties = find_peaks(smoothed, height=threshold, distance=5)

    print(f"\nClass {c}: {len(peaks)} peaks found (threshold={threshold:.4f})")
    # Sort by peak height descending
    sorted_idx = np.argsort(properties['peak_heights'])[::-1]
    for rank, idx in enumerate(sorted_idx[:10]):
        p_freq = freqs_fft[peaks[idx]]
        p_height = properties['peak_heights'][idx]
        # Width at half prominence
        try:
            widths_half = peak_widths(smoothed, [peaks[idx]], rel_height=0.5)
            fwhm_hz = widths_half[0][0] * (freqs_fft[1] - freqs_fft[0])
        except:
            fwhm_hz = np.nan
        print(f"  #{rank+1}: {p_freq:7.1f} Hz  height={p_height:.4f}  FWHM={fwhm_hz:.2f} Hz")

# Class-by-class difference in key bands
print("\n" + "-" * 70)
print("Class-by-Class Differences (BPF band: 40-60 Hz)")
print("-" * 70)
bpf_mask = (freqs_fft >= 40) & (freqs_fft <= 60)
for c in sorted(classes):
    mean_bpf = class_fft_means[c][bpf_mask]
    peak_idx_local = mean_bpf.argmax()
    peak_f = freqs_fft[bpf_mask][peak_idx_local]
    peak_v = mean_bpf[peak_idx_local]
    # relative to class 0
    print(f"  Class {c}: BPF peak {peak_v:.4f} @ {peak_f:.1f} Hz")

# ============================================================
# Task 3: Time-Frequency Visualization
# ============================================================
print("\n" + "=" * 70)
print("TASK 3: Time-Frequency Visualization")
print("=" * 70)

# Pick 1 representative sample per class (highest RMS energy)
print("\nSelecting representative samples (highest RMS energy per class)...")
representative_samples = {}

for c in sorted(classes):
    mask = all_y_combined == c
    X_class = all_X[mask]
    rms = np.sqrt(np.mean(X_class**2, axis=1))
    best_idx = np.argmax(rms)
    # Get the original index in all_X
    orig_indices = np.where(mask)[0]
    representative_samples[c] = {
        'signal': X_class[best_idx],
        'rms': rms[best_idx],
        'orig_idx': orig_indices[best_idx],
        'global_idx': orig_indices[best_idx],
    }
    print(f"  Class {c}: idx={representative_samples[c]['orig_idx']}, RMS={rms[best_idx]:.4f}")

# Create the multi-panel figure
print("\nComputing MSST/STFT for representative samples...")
n_classes = len(classes)
fig, axes = plt.subplots(n_classes, 2, figsize=(16, 4 * n_classes))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

freq_max = 200  # Hz for display
class_names = [f'Class {c}' for c in sorted(classes)]

for row, c in enumerate(sorted(classes)):
    signal = representative_samples[c]['signal'].astype(np.float64)
    rms_val = representative_samples[c]['rms']

    # Compute MSST/STFT
    result = msst(signal, fs, num=3)
    stft_complex = result['STFT']  # [F, T] complex
    stft_mag = np.abs(stft_complex)
    msst_freqs = result['freqs']
    msst_t = result['t']

    # Left panel: STFT spectrogram (dB scale, 0-200 Hz)
    ax_left = axes[row, 0]
    stft_db = 10 * np.log10(stft_mag + 1e-12)
    # Clip to reasonable range
    vmin = -40
    vmax = stft_db.max()
    mask_freq = msst_freqs <= freq_max
    im = ax_left.pcolormesh(msst_t, msst_freqs[mask_freq], stft_db[mask_freq, :],
                            shading='gouraud', cmap='jet',
                            vmin=vmin, vmax=vmax)
    ax_left.set_ylabel('Frequency [Hz]', fontsize=10)
    ax_left.set_title(f'{class_names[row]} — STFT Spectrogram (RMS={rms_val:.3f})', fontsize=11, fontweight='bold')
    ax_left.set_ylim(0, freq_max)
    if row == n_classes - 1:
        ax_left.set_xlabel('Time [s]', fontsize=10)
    plt.colorbar(im, ax=ax_left, label='dB')

    # Annotate frequency bands
    for band_name, f_lo, f_hi, color, alpha in [
        ('LOW', 0, 20, 'cyan', 0.15),
        ('BPF', 40, 60, 'yellow', 0.12),
        ('2xBPF', 90, 110, 'lime', 0.12),
    ]:
        ax_left.axhspan(f_lo, f_hi, alpha=alpha, color=color)
        ax_left.text(msst_t[-1]*0.02, (f_lo+f_hi)/2, band_name,
                    fontsize=7, va='center', color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # Right panel: FFT magnitude spectrum (0-200 Hz)
    ax_right = axes[row, 1]
    fft_mag = np.abs(np.fft.rfft(signal)) / T_samples
    mask_fft = freqs_fft <= freq_max
    ax_right.plot(freqs_fft[mask_fft], fft_mag[mask_fft], color='#1f77b4', lw=1.0)
    ax_right.fill_between(freqs_fft[mask_fft], 0, fft_mag[mask_fft], alpha=0.3, color='#1f77b4')
    ax_right.set_ylabel('Magnitude', fontsize=10)
    ax_right.set_title(f'{class_names[row]} — FFT Spectrum (0-{freq_max} Hz)', fontsize=11, fontweight='bold')
    ax_right.set_xlim(0, freq_max)
    if row == n_classes - 1:
        ax_right.set_xlabel('Frequency [Hz]', fontsize=10)

    # Annotate frequency bands on FFT
    for band_name, f_lo, f_hi, color in [
        ('LOW', 0, 20, 'cyan'),
        ('BPF', 40, 60, 'yellow'),
        ('2xBPF', 90, 110, 'lime'),
    ]:
        ax_right.axvspan(f_lo, f_hi, alpha=0.12, color=color)

    # Mark peaks in FFT
    smoothed_fft = gaussian_filter1d(fft_mag[mask_fft], sigma=1.0)
    threshold_fft = 0.05 * smoothed_fft.max()
    peaks_fft, props_fft = find_peaks(smoothed_fft, height=threshold_fft, distance=5)
    if len(peaks_fft) > 0:
        sorted_peaks = np.argsort(props_fft['peak_heights'])[::-1][:5]
        for pi in sorted_peaks:
            pf = freqs_fft[mask_fft][peaks_fft[pi]]
            ph = props_fft['peak_heights'][pi]
            ax_right.annotate(f'{pf:.0f} Hz', xy=(pf, ph*1.05),
                            fontsize=7, ha='center', color='red',
                            arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    ax_right.grid(True, alpha=0.3)

plt.suptitle('Time-Frequency Analysis — normalized_data.npz\n'
             'Left: STFT Spectrogram (MSST) | Right: FFT Magnitude Spectrum',
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

output_path = 'd:/a_task1/sast-hmst/tfr_analysis.png'
fig.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {output_path}")

# ============================================================
# Task 4: Wide vs Sharp Peak Analysis (Class 0 vs Class 1)
# ============================================================
print("\n" + "=" * 70)
print("TASK 4: Wide vs Sharp Peak Analysis")
print("=" * 70)

# Focus on BPF (40-60 Hz) and 2xBPF (90-110 Hz) for Class 0 and Class 1
for c in [0, 1]:
    mask = all_y_combined == c
    X_class = all_X[mask]
    n_class = X_class.shape[0]

    # Compute mean FFT and also individual FFTs for per-sample bandwidth
    all_ffts = np.abs(np.fft.rfft(X_class, axis=1)) / T_samples

    print(f"\nClass {c} ({n_class} samples):")
    print("-" * 40)

    # BPF region (40-60 Hz)
    bpf_mask = (freqs_fft >= 40) & (freqs_fft <= 60)
    bpf2x_mask = (freqs_fft >= 90) & (freqs_fft <= 110)

    # Mean FFT for this class
    mean_fft = class_fft_means[c]

    # --- BPF Peak ---
    bpf_region = mean_fft[bpf_mask]
    bpf_freqs_region = freqs_fft[bpf_mask]
    bpf_peak_idx_local = bpf_region.argmax()
    bpf_peak_freq = bpf_freqs_region[bpf_peak_idx_local]
    bpf_peak_val = bpf_region[bpf_peak_idx_local]

    # -3dB bandwidth of BPF peak
    half_max = bpf_peak_val / np.sqrt(2)  # -3dB in amplitude
    # Find where mean FFT crosses half_max around the BPF peak
    above = mean_fft >= half_max
    # Find the contiguous region around the BPF peak in the full frequency range
    bpf_peak_global_idx = np.where(freqs_fft == bpf_peak_freq)[0][0]

    # Walk left and right from peak to find -3dB crossings
    bpf_left, bpf_right = bpf_peak_global_idx, bpf_peak_global_idx
    while bpf_left > 0 and mean_fft[bpf_left] >= half_max:
        bpf_left -= 1
    while bpf_right < len(freqs_fft) - 1 and mean_fft[bpf_right] >= half_max:
        bpf_right += 1

    bpf_bw_hz = freqs_fft[bpf_right] - freqs_fft[bpf_left]
    print(f"  BPF peak: {bpf_peak_freq:.1f} Hz, amplitude={bpf_peak_val:.4f}")
    print(f"  BPF -3dB bandwidth: {bpf_bw_hz:.2f} Hz  (from {freqs_fft[bpf_left]:.1f} to {freqs_fft[bpf_right]:.1f} Hz)")

    # --- 2xBPF Peak (around 100 Hz) ---
    bpf2x_region = mean_fft[bpf2x_mask]
    bpf2x_freqs_region = freqs_fft[bpf2x_mask]
    bpf2x_peak_idx_local = bpf2x_region.argmax()
    bpf2x_peak_freq = bpf2x_freqs_region[bpf2x_peak_idx_local]
    bpf2x_peak_val = bpf2x_region[bpf2x_peak_idx_local]

    half_max_2x = bpf2x_peak_val / np.sqrt(2)
    bpf2x_peak_global_idx = np.where(freqs_fft == bpf2x_peak_freq)[0][0]

    bpf2x_left, bpf2x_right = bpf2x_peak_global_idx, bpf2x_peak_global_idx
    while bpf2x_left > 0 and mean_fft[bpf2x_left] >= half_max_2x:
        bpf2x_left -= 1
    while bpf2x_right < len(freqs_fft) - 1 and mean_fft[bpf2x_right] >= half_max_2x:
        bpf2x_right += 1

    bpf2x_bw_hz = freqs_fft[bpf2x_right] - freqs_fft[bpf2x_left]
    print(f"  2xBPF peak: {bpf2x_peak_freq:.1f} Hz, amplitude={bpf2x_peak_val:.4f}")
    print(f"  2xBPF -3dB bandwidth: {bpf2x_bw_hz:.2f} Hz  (from {freqs_fft[bpf2x_left]:.1f} to {freqs_fft[bpf2x_right]:.1f} Hz)")

    # Ratio
    ratio = bpf_bw_hz / bpf2x_bw_hz if bpf2x_bw_hz > 0 else float('inf')
    print(f"  Ratio (BPF_BW / 2xBPF_BW): {ratio:.3f}  (\"how much wider BPF is\")")

    # --- Per-sample bandwidth statistics ---
    bpf_bws_per_sample = []
    bpf2x_bws_per_sample = []
    for i in range(min(n_class, 200)):  # sample up to 200 for efficiency
        sample_fft = all_ffts[i]

        # BPF bandwidth per sample
        bpf_region_sample = sample_fft[bpf_mask]
        if bpf_region_sample.max() > 0:
            hm = bpf_region_sample.max() / np.sqrt(2)
            above_s = bpf_region_sample >= hm
            if above_s.any():
                left_s = np.argmax(above_s)
                right_s = len(above_s) - 1 - np.argmax(above_s[::-1])
                bpf_bws_per_sample.append(bpf_freqs_region[right_s] - bpf_freqs_region[left_s])

        # 2xBPF bandwidth per sample
        bpf2x_region_sample = sample_fft[bpf2x_mask]
        if bpf2x_region_sample.max() > 0:
            hm_2x = bpf2x_region_sample.max() / np.sqrt(2)
            above_2x = bpf2x_region_sample >= hm_2x
            if above_2x.any():
                left_2x = np.argmax(above_2x)
                right_2x = len(above_2x) - 1 - np.argmax(above_2x[::-1])
                bpf2x_bws_per_sample.append(bpf2x_freqs_region[right_2x] - bpf2x_freqs_region[left_2x])

    if bpf_bws_per_sample:
        bpf_bw_mean = np.mean(bpf_bws_per_sample)
        bpf_bw_std = np.std(bpf_bws_per_sample)
        print(f"\n  Per-sample BPF bandwidth:  {bpf_bw_mean:.2f} +/- {bpf_bw_std:.2f} Hz (n={len(bpf_bws_per_sample)})")
    if bpf2x_bws_per_sample:
        bpf2x_bw_mean = np.mean(bpf2x_bws_per_sample)
        bpf2x_bw_std = np.std(bpf2x_bws_per_sample)
        print(f"  Per-sample 2xBPF bandwidth: {bpf2x_bw_mean:.2f} +/- {bpf2x_bw_std:.2f} Hz (n={len(bpf2x_bws_per_sample)})")

    if bpf_bws_per_sample and bpf2x_bws_per_sample:
        per_sample_ratios = [b / x for b, x in zip(bpf_bws_per_sample, bpf2x_bws_per_sample) if x > 0]
        if per_sample_ratios:
            ratio_mean = np.mean(per_sample_ratios)
            ratio_std = np.std(per_sample_ratios)
            print(f"  Per-sample ratio (BPF_BW / 2xBPF_BW): {ratio_mean:.3f} +/- {ratio_std:.3f}")

# Create a zoomed comparison plot for Class 0 and Class 1
print("\nCreating zoomed bandwidth comparison plot...")
fig_bw, axes_bw = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, c in enumerate([0, 1]):
    ax = axes_bw[ax_idx]
    mean_fft = class_fft_means[c]
    std_fft = class_fft_stds[c]

    # Plot full 0-200 Hz with focus annotations
    mask_200 = freqs_fft <= 200
    ax.plot(freqs_fft[mask_200], mean_fft[mask_200], color='#1f77b4', lw=1.5, label='Mean FFT')
    ax.fill_between(freqs_fft[mask_200],
                    mean_fft[mask_200] - std_fft[mask_200],
                    mean_fft[mask_200] + std_fft[mask_200],
                    alpha=0.2, color='#1f77b4', label='+/-1 std')

    # Highlight BPF and 2xBPF regions
    ax.axvspan(40, 60, alpha=0.12, color='orange', label='BPF (40-60 Hz)')
    ax.axvspan(90, 110, alpha=0.12, color='green', label='2xBPF (90-110 Hz)')

    # Mark -3dB points for BPF
    bpf_mask_plot = (freqs_fft >= 40) & (freqs_fft <= 60)
    bpf_region_plot = mean_fft[bpf_mask_plot]
    bpf_peak_idx = np.argmax(bpf_region_plot)
    bpf_peak_f = freqs_fft[bpf_mask_plot][bpf_peak_idx]
    bpf_peak_v = bpf_region_plot[bpf_peak_idx]
    hm_bpf = bpf_peak_v / np.sqrt(2)

    bpf_global_idx = np.where(np.abs(freqs_fft - bpf_peak_f) < 0.1)[0][0]
    left = bpf_global_idx
    while left > 0 and mean_fft[left] >= hm_bpf:
        left -= 1
    right = bpf_global_idx
    while right < len(freqs_fft) - 1 and mean_fft[right] >= hm_bpf:
        right += 1
    bw_bpf = freqs_fft[right] - freqs_fft[left]

    ax.axhline(y=hm_bpf, color='orange', ls='--', lw=0.8)
    ax.axvline(x=freqs_fft[left], color='orange', ls=':', lw=0.8)
    ax.axvline(x=freqs_fft[right], color='orange', ls=':', lw=0.8)

    # Mark -3dB points for 2xBPF
    bpf2x_mask_plot = (freqs_fft >= 90) & (freqs_fft <= 110)
    bpf2x_region_plot = mean_fft[bpf2x_mask_plot]
    bpf2x_peak_idx = np.argmax(bpf2x_region_plot)
    bpf2x_peak_f = freqs_fft[bpf2x_mask_plot][bpf2x_peak_idx]
    bpf2x_peak_v = bpf2x_region_plot[bpf2x_peak_idx]
    hm_2x = bpf2x_peak_v / np.sqrt(2)

    bpf2x_global_idx = np.where(np.abs(freqs_fft - bpf2x_peak_f) < 0.1)[0][0]
    left2 = bpf2x_global_idx
    while left2 > 0 and mean_fft[left2] >= hm_2x:
        left2 -= 1
    right2 = bpf2x_global_idx
    while right2 < len(freqs_fft) - 1 and mean_fft[right2] >= hm_2x:
        right2 += 1
    bw_2x = freqs_fft[right2] - freqs_fft[left2]

    ax.axhline(y=hm_2x, color='green', ls='--', lw=0.8)
    ax.axvline(x=freqs_fft[left2], color='green', ls=':', lw=0.8)
    ax.axvline(x=freqs_fft[right2], color='green', ls=':', lw=0.8)

    ratio_val = bw_bpf / bw_2x if bw_2x > 0 else float('inf')

    ax.set_xlabel('Frequency [Hz]', fontsize=11)
    ax.set_ylabel('Magnitude', fontsize=11)
    ax.set_title(f'Class {c} — BPF vs 2xBPF Bandwidth\n'
                 f'BPF BW={bw_bpf:.2f} Hz | 2xBPF BW={bw_2x:.2f} Hz | Ratio={ratio_val:.2f}x',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 200)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle('Wide vs Sharp Peak Analysis: BPF (40-60 Hz) vs 2xBPF (90-110 Hz)\n'
             'Dashed lines = -3dB threshold | Dotted lines = -3dB bandwidth bounds',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

bw_output_path = 'd:/a_task1/sast-hmst/bandwidth_analysis.png'
fig_bw.savefig(bw_output_path, dpi=150, bbox_inches='tight')
plt.close(fig_bw)
print(f"Saved: {bw_output_path}")

print("\n" + "=" * 70)
print("DONE — All tasks completed")
print("=" * 70)
