"""
绘制 5_dataset.npz 中每个类别一个样本的 WSST2 vs ssqueezepy WSST 对比时频图
=========================================================================
布局: 5 行 × 2 列 — 行=类别, 左=WSST2, 右=ssqueezepy WSST
每张子图标注主导频率列表。
用法: python plot_dataset_tfr.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.signal import find_peaks
import ssqueezepy as ssq
sys.path.insert(0, str(Path(__file__).parent))
from models.tfr import wsst2, renyi_entropy

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
CMAP = 'jet'
SAVE = Path('hmst_figures')
SAVE.mkdir(exist_ok=True)

FS = 1000
TOTAL = 2.0  # 2000 / 1000
FMAX = 500   # display cutoff

CLASS_NAMES = {
    0: 'Class 0: No-load',
    1: 'Class 1: Low load',
    2: 'Class 2: Mid load',
    3: 'Class 3: High load',
    4: 'Class 4: Pumping',
}


def find_dominant_freqs(tfr, freq_axis, fmax=500, n_peaks=8,
                        prominence=0.02, min_dist=8):
    """
    从 TFR 的时间平均谱中提取主导频率。

    Args:
        tfr:        [F, T] 幅度谱
        freq_axis:  [F] 频率轴 (Hz)
        fmax:       最大频率
        n_peaks:    最多返回的峰值数
        prominence: 峰显著度阈值 (相对于 max)
        min_dist:   峰最小间距 (bin)

    Returns:
        peaks_hz:   峰值频率列表 (Hz), 从高到低排序
    """
    mask = freq_axis <= fmax
    f = freq_axis[mask]
    # 时间平均谱 + 平滑
    spec = np.mean(tfr[mask, :], axis=1)
    spec = spec / (spec.max() + 1e-12)  # 归一化到 [0,1]

    peaks, props = find_peaks(spec, prominence=prominence,
                              distance=min_dist)
    if len(peaks) == 0:
        peaks, props = find_peaks(spec, prominence=prominence * 0.5,
                                  distance=min_dist // 2)
    if len(peaks) == 0:
        peaks = [np.argmax(spec)]

    # 按幅值降序排列
    sort_idx = np.argsort(spec[peaks])[::-1]
    peaks = peaks[sort_idx][:n_peaks]
    peaks_hz = f[peaks]
    peaks_amp = spec[peaks]

    return peaks_hz, peaks_amp


# ── 加载数据 ──
data = np.load('5_dataset.npz', allow_pickle=True)
train_X = data['train_X']
train_y = data['train_y']

# ── 每类取一个样本 ──
sample_indices = {}
for cls in sorted(np.unique(train_y)):
    idx = np.where(train_y == cls)[0][0]
    sample_indices[cls] = idx

print("=" * 60)
print(f"5-Dataset TFR Comparison — fs={FS}, N={train_X.shape[1]} samples")
print("=" * 60)



t_axis = np.arange(2000) / FS

tfr_results = {}

for cls, idx in sample_indices.items():
    x = train_X[idx].astype(np.float64)
    print(f"\nClass {cls} (train_X[{idx}])...", flush=True)

    # ── Our WSST2 ──
    print("  WSST2 (morse)...", end=" ", flush=True)
    r = wsst2(x, FS, gamma=0.01, mywav='morse', nv=32)
    f_wsst2 = r['freqs']
    mag_wsst2 = np.abs(r['WSST2'])
    mask_w = f_wsst2 <= FMAX
    re_wsst2 = renyi_entropy(mag_wsst2[mask_w, :])
    print(f"Renyi={re_wsst2:.2f}")

    # ── 主导频率 ──
    dom_hz, dom_amp = find_dominant_freqs(mag_wsst2, f_wsst2, FMAX,
                                          prominence=0.03, n_peaks=8)
    print(f"  Dominant freqs (Hz): {np.round(dom_hz, 1).tolist()}")

    # ── ssqueezepy WSST ──
    print("  ssqueezepy WSST...", end=" ", flush=True)
    Tx_ssq, Wx, ssq_freqs, scales = ssq.ssq_cwt(x, fs=FS)
    f_ssq = np.asarray(ssq_freqs).squeeze()
    mag_ssq = np.abs(Tx_ssq)
    if f_ssq[0] > f_ssq[-1]:
        f_ssq = f_ssq[::-1]
        mag_ssq = mag_ssq[::-1, :]
    mask_s = f_ssq <= FMAX
    re_ssq = renyi_entropy(mag_ssq[mask_s, :])
    print(f"Renyi={re_ssq:.2f}  Δ={re_wsst2 - re_ssq:+.2f}")

    tfr_results[cls] = {
        'wsst2': mag_wsst2,   'f_wsst2': f_wsst2,  're_wsst2': re_wsst2,
        'ssq':  mag_ssq,      'f_ssq':  f_ssq,     're_ssq':  re_ssq,
        'dom_hz': dom_hz,     'dom_amp': dom_amp,
        'x': x, 'idx': idx,
    }

# ═══════════════════════════════════════════════════════════════
# 绘图: 5 rows × 2 cols
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 2, figsize=(15, 20))

for row, cls in enumerate(sorted(tfr_results.keys())):
    r = tfr_results[cls]

    # ── 左: WSST2 ──
    ax_l = axes[row, 0]
    f_l = r['f_wsst2']
    mask_l = f_l <= FMAX
    ax_l.pcolormesh(t_axis, f_l[mask_l], r['wsst2'][mask_l, :],
                    shading='gouraud', cmap=CMAP)
    # 画主导频率水平虚线
    for dhz in r['dom_hz'][:6]:
        ax_l.axhline(dhz, color='white', lw=0.5, ls='--', alpha=0.5)
    # 频率标注框
    dom_text = '\n'.join([f'{f:6.1f} Hz' for f in r['dom_hz'][:6]])
    ax_l.text(0.985, 0.97, f'Dominant:\n{dom_text}',
              transform=ax_l.transAxes, fontsize=6, va='top', ha='right',
              bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                        alpha=0.7, edgecolor='gray', lw=0.5),
              family='monospace')

    ax_l.set_ylim(0, FMAX)
    ax_l.set_xlim(0, TOTAL)
    ax_l.set_title(f'{CLASS_NAMES[cls]}  |  WSST2  R={r["re_wsst2"]:.2f}',
                   fontsize=9, fontweight='bold')
    if row == 4:
        ax_l.set_xlabel('Time (s)')
    ax_l.set_ylabel('Freq (Hz)')

    # ── 右: ssqueezepy WSST ──
    ax_r = axes[row, 1]
    f_r = r['f_ssq']
    mask_r = f_r <= FMAX
    ax_r.pcolormesh(t_axis, f_r[mask_r], r['ssq'][mask_r, :],
                    shading='gouraud', cmap=CMAP)
    # 同样标注主导频率
    for dhz in r['dom_hz'][:6]:
        ax_r.axhline(dhz, color='white', lw=0.5, ls='--', alpha=0.5)

    ax_r.set_ylim(0, FMAX)
    ax_r.set_xlim(0, TOTAL)
    ax_r.set_title(f'{CLASS_NAMES[cls]}  |  ssqueezepy WSST  R={r["re_ssq"]:.2f}',
                   fontsize=9, fontweight='bold')
    if row == 4:
        ax_r.set_xlabel('Time (s)')

plt.suptitle('WSST2 vs ssqueezepy WSST — 5 Dataset Classes  |  dominant frequencies annotated',
             fontsize=12, fontweight='bold', y=0.996)
plt.tight_layout()
path = SAVE / 'dataset_5class_comparison.png'
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {path}")

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print("\n── Summary ──")
print(f"{'Class':<22s} {'WSST2':>7s} {'ssq':>7s} {'Delta':>7s}  Dominant freqs (Hz)")
print("-" * 80)
for cls in sorted(tfr_results.keys()):
    r = tfr_results[cls]
    d = r['re_wsst2'] - r['re_ssq']
    freqs_str = ', '.join([f'{f:.1f}' for f in r['dom_hz'][:5]])
    print(f"{CLASS_NAMES[cls]:<22s} {r['re_wsst2']:7.2f} {r['re_ssq']:7.2f} {d:+7.2f}  [{freqs_str}]")

print("\nDone.")
