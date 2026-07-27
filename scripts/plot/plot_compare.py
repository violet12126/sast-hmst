"""
WSST2 对比: ssqueezepy WSST | Our CWT-WSST | Our CWT-WSST2
==========================================================
fs=1024 (2^10), N=1024 samples.
信号: Bao et al. 2023, eq 22-23 — x1 (300Hz FM) + x2 (75→125Hz chirp)

用法: python plot_compare.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.tfr import wsst2, renyi_entropy

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9
CMAP = 'jet'
SAVE = Path('hmst_figures')
SAVE.mkdir(exist_ok=True)

FS = 1024
T_GEN = 1.0
GAMMA = 0.01


def renyi(x, a=3):
    return renyi_entropy(np.abs(np.asarray(x)), a)


# ═══════════════════════════════════════════════════════════════
# 1. 信号生成
# ═══════════════════════════════════════════════════════════════

def gen_paper_signal(fs=FS, T=T_GEN):
    """Bao et al. 2023 eq 22-23: x1(300Hz FM) + x2(75→125Hz chirp)."""
    t = np.arange(int(fs * T)) / fs
    x1 = np.sin(2 * np.pi * (300 * t - 1.5 * np.cos(14 * np.pi * t)))
    x2 = np.sin(2 * np.pi * (75 * t + 25 * t**2))
    IF1 = 300 + 21 * np.pi * np.sin(14 * np.pi * t)
    IF2 = 75 + 50 * t
    return t, x1 + x2, IF1, IF2


t, x, IF1, IF2 = gen_paper_signal()
n = len(x)
t_cwt = t

# ═══════════════════════════════════════════════════════════════
# 2. 计算 TFR
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print(f"WSST2 Comparison — fs={FS} (power of 2), N={n} samples")
print("=" * 60)

# ── Our CWT WSST/WSST2 ──
print("Our WSST/WSST2 (morse)...", end=" ", flush=True)
r = wsst2(x, FS, gamma=GAMMA, mywav='morse', nv=32)
f_cwt = r['freqs']
print(f"OK  na={len(f_cwt)}, f=[{f_cwt[0]:.1f}, {f_cwt[-1]:.1f}] Hz")

# ── ssqueezepy WSST (matching tt1.py exactly) ──
print("ssqueezepy WSST...", end=" ", flush=True)
import ssqueezepy as ssq
Tx_ssq, Wx, ssq_freqs, scales = ssq.ssq_cwt(x, fs=FS)
f_ssq = np.asarray(ssq_freqs).squeeze()
if f_ssq[0] > f_ssq[-1]:
    f_ssq = f_ssq[::-1]
    Tx_ssq = Tx_ssq[::-1, :]
has_ssq = True
print(f"OK  f=[{f_ssq[0]:.1f}, {f_ssq[-1]:.1f}] Hz")

# ── 幅度 (匹配 tt1.py: np.abs) ──
mag_ssq   = np.abs(Tx_ssq)
mag_wsst  = np.abs(r['WSST'])
mag_wsst2 = np.abs(r['WSST2'])

# ── Rényi ──
print("\n── Rényi Entropy (α=3) ──")
print(f"  {'ssqueezepy WSST (default):':<28s} {renyi(mag_ssq):.2f}")
print(f"  {'Our CWT-WSST (morse, N=1):':<28s} {renyi(mag_wsst):.2f}")
print(f"  {'Our CWT-WSST2 (morse, N=2):':<28s} {renyi(mag_wsst2):.2f}"
      f"  Δ={renyi(mag_wsst2)-renyi(mag_wsst):+.2f}")

# ═══════════════════════════════════════════════════════════════
# 构建 panels — 统一直线幅值, 无 dB/压缩
# ═══════════════════════════════════════════════════════════════
panels = [
    ('ssqueezepy WSST\n(default morlet)', mag_ssq,   t_cwt, f_ssq),
    ('Our CWT-WSST\n(Morse, N=1)',        mag_wsst,  t_cwt, f_cwt),
    ('Our CWT-WSST2\n(Morse, N=2)',       mag_wsst2, t_cwt, f_cwt),
]
n_cols = len(panels)

# ═══════════════════════════════════════════════════════════════
# Figure 1: Full comparison
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5.0))
if n_cols == 1:
    axes = [axes]

for ax, (title, img, t_ax, f_ax) in zip(axes, panels):
    ax.pcolormesh(t_ax, f_ax, img, shading='gouraud', cmap=CMAP)
    ax.plot(t, IF1, 'r--', lw=0.8, alpha=0.8, label='IF1 (300 Hz FM)')
    ax.plot(t, IF2, 'cyan', lw=0.6, alpha=0.7, label='IF2 (75→125 Hz)')
    ax.set_ylim(0, 500)
    ax.set_xlim(0, T_GEN)
    ax.set_xlabel('Time (s)')
    ax.set_title(title, fontsize=10, fontweight='bold')

axes[0].set_ylabel('Frequency (Hz)')
axes[0].legend(fontsize=6, loc='upper right')
plt.suptitle('WSST2 Comparison',
             fontsize=12, fontweight='bold')
plt.tight_layout()
path = SAVE / 'wsst2_full_comparison.png'
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {path}")

# ═══════════════════════════════════════════════════════════════
# Figure 2: Zoom — x1 FM ridge (200-420 Hz, 0.3-0.8 s)
# ═══════════════════════════════════════════════════════════════
ZOOM_Y = (200, 420)
ZOOM_X = (0.3, 0.8)

fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.8))
if n_cols == 1:
    axes = [axes]

for ax, (title, img, t_ax, f_ax) in zip(axes, panels):
    ax.pcolormesh(t_ax, f_ax, img, shading='gouraud', cmap=CMAP)
    ax.plot(t, IF1, 'r-', lw=0.8)
    ax.set_ylim(*ZOOM_Y)
    ax.set_xlim(*ZOOM_X)
    ax.set_xlabel('Time (s)')
    ax.set_title(title, fontsize=10, fontweight='bold')

axes[0].set_ylabel('Frequency (Hz)')
plt.suptitle('Zoom: x1 FM signal (200–420 Hz, 0.3–0.8 s)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
path = SAVE / 'wsst2_zoom.png'
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")

print("\nDone.")
