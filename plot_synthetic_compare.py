"""
SST / WSST / WSST2 / MSST 四栏对比 — 双频余弦信号
==================================================
信号:
  x(t) = cos(2π·0.5t) + 0.3·cos(2π·45t)

IF 理论值:
  分量1:  IF₁(t) = 0.5 Hz  (低频稳态)
  分量2:  IF₂(t) = 45 Hz   (高频稳态, 幅度 0.3)

用法: python plot_synthetic_compare.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import ssqueezepy as ssq
sys.path.insert(0, str(Path(__file__).parent))
from models.tfr import wsst2, msst, compute_renyi

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 7.5
CMAP = 'jet'
SAVE_DIR = Path('hmst_figures')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FS = 1024
T_END = 3.0
FMAX = 100
HLENGTH = 250  # MSST window

# ═══════════════════════════════════════════
# 信号生成
# ═══════════════════════════════════════════
t = np.arange(0, T_END, 1 / FS)
N = len(t)

sig = np.cos(2 * np.pi * 2.5 * t) + 0.3 * np.cos(2 * np.pi * 45 * t)

# 理论 IF (稳态常数)
if1 = np.full_like(t, 2.5)   # 分量1: 2.5 Hz
if2 = np.full_like(t, 45.0)  # 分量2: 45 Hz

print(f"Synthetic signal: N={N}, fs={FS}, T={T_END}s")
print(f"  x(t) = cos(2π·2.5t) + 0.3·cos(2π·45t)")
print(f"  Component 1: 2.5 Hz (amplitude 1.0)")
print(f"  Component 2: 45 Hz  (amplitude 0.3)")
print()

# ═══════════════════════════════════════════
# 计算四种 TFR
# ═══════════════════════════════════════════

# ── SST (STFT first-order, = MSST num=1) ──
print("SST (STFT, 1st-order)...", end=" ", flush=True)
r_sst = msst(sig, FS, hlength=HLENGTH, num=1)
f_sst = r_sst['freqs']
mag_sst = r_sst['MSST']
mask_sst = f_sst <= FMAX
re_sst = compute_renyi(mag_sst[mask_sst, :])
print(f"R={re_sst:.2f}")

# ── WSST (CWT first-order, ssqueezepy) ──
print("WSST (CWT, 1st-order)...", end=" ", flush=True)
Tx_wsst, Wx, f_wsst_raw, scales = ssq.ssq_cwt(sig, fs=FS)
f_wsst = np.asarray(f_wsst_raw).squeeze()
mag_wsst = np.abs(Tx_wsst)
if f_wsst[0] > f_wsst[-1]:
    f_wsst = f_wsst[::-1]
    mag_wsst = mag_wsst[::-1, :]
mask_wsst = f_wsst <= FMAX
re_wsst = compute_renyi(mag_wsst[mask_wsst, :])
print(f"R={re_wsst:.2f}")

# ── WSST2 (CWT second-order, our) ──
print("WSST2 (CWT, 2nd-order)...", end=" ", flush=True)
r_wsst2 = wsst2(sig, FS, gamma=0.01, mywav='morse', nv=32)
f_w2 = r_wsst2['freqs']
mag_w2 = np.abs(r_wsst2['WSST2'])
mask_w2 = f_w2 <= FMAX
re_w2 = compute_renyi(mag_w2[mask_w2, :])
print(f"R={re_w2:.2f}")

# ── MSST (STFT, num=3) ──
print("MSST (STFT, num=3)...", end=" ", flush=True)
r_msst = msst(sig, FS, hlength=HLENGTH, num=3)
f_ms = r_msst['freqs']
mag_ms = r_msst['MSST']
mask_ms = f_ms <= FMAX
re_ms = compute_renyi(mag_ms[mask_ms, :])
print(f"R={re_ms:.2f}")

# ═══════════════════════════════════════════
# 能量统计: 各方法在理论IF附近(±0.5 Hz)的重排后能量 vs FFT能量
# ═══════════════════════════════════════════
BAND_HALF = 2.5  # Hz
TARGETS = {'2.5 Hz': 2.5, '45 Hz': 45.0}

# FFT 参考能量 (在 band 内积分)
fft_freqs = np.fft.rfftfreq(N, 1 / FS)
fft_mag = np.abs(np.fft.rfft(sig))
fft_power = fft_mag ** 2
df_fft = FS / N

def band_energy_fft(f_target):
    """FFT 能量 (|X|² sum) 在 f_target ± BAND_HALF Hz 内"""
    in_band = np.abs(fft_freqs - f_target) <= BAND_HALF
    return fft_power[in_band].sum()

def band_energy_tfr(mag, f_ax, f_target):
    """TFR 能量 (|T|² sum over time & freq) 在 f_target ± BAND_HALF Hz 内。
       mag: (n_freqs, n_times); f_ax: (n_freqs,)"""
    in_band = np.abs(f_ax - f_target) <= BAND_HALF
    if not in_band.any():
        return 0.0
    return (mag[in_band, :] ** 2).sum()

# 只统计 FMAX 以内的部分（与绘图一致）
mask_sst  = f_sst  <= FMAX
mask_wsst = f_wsst <= FMAX
mask_w2   = f_w2   <= FMAX
mask_ms   = f_ms   <= FMAX

methods_energy = [
    ('SST (STFT 1st)',  mag_sst,  f_sst,  mask_sst),
    ('WSST (CWT 1st)',  mag_wsst, f_wsst, mask_wsst),
    ('WSST2 (CWT 2nd)', mag_w2,   f_w2,   mask_w2),
    ('MSST (STFT num=3)', mag_ms, f_ms,   mask_ms),
]

print(f"\n{'='*85}")
print(f"  Band-energy comparison: +/-{BAND_HALF} Hz around theoretical IF")
print(f"  (TFR energy = sum|T(t,f)|^2 over band & time;  FFT energy = sum|X(f)|^2 over band)")
print(f"{'='*85}")

for label in TARGETS:
    f0 = TARGETS[label]
    e_fft = band_energy_fft(f0)
    print(f"\n  ── Target: {label} ({f0} Hz) ──")
    print(f"  {'Method':<22s} {'TFR Energy':>14s} {'FFT Energy':>14s} {'Ratio (TFR/FFT)':>18s} {'Scaled Ratio':>14s}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*18} {'-'*14}")
    ratios = []
    for mname, mag, fax, mask in methods_energy:
        e_tfr = band_energy_tfr(mag[mask, :], fax[mask], f0)
        ratio = e_tfr / e_fft if e_fft > 0 else 0
        ratios.append(ratio)
        print(f"  {mname:<22s} {e_tfr:14.4e} {e_fft:14.4e} {ratio:18.6f}")
    # 相对于 SST 的归一化比例
    if ratios[0] > 0:
        print(f"  {'  (vs SST=1.0)':<22s} {'':>14s} {'':>14s} {'':>18s}", end="")
        for i, (mname, *_) in enumerate(methods_energy):
            print(f" {mname.split('(')[0].strip()}={ratios[i]/ratios[0]:.3f}", end="")
        print()

print()

# ═══════════════════════════════════════════
# 绘图: 2×2
# ═══════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

panels = [
    (axes[0, 0], f'SST (STFT, 1st-order)\nR={re_sst:.2f}', mag_sst, f_sst, mask_sst),
    (axes[0, 1], f'WSST (CWT, 1st-order)\nR={re_wsst:.2f}', mag_wsst, f_wsst, mask_wsst),
    (axes[1, 0], f'WSST2 (CWT, 2nd-order)\nR={re_w2:.2f}', mag_w2, f_w2, mask_w2),
    (axes[1, 1], f'MSST (STFT, num=3)\nR={re_ms:.2f}', mag_ms, f_ms, mask_ms),
]

for ax, title, mag, f_ax, mask in panels:
    vmax = mag[mask, :].max() * 0.5
    ax.pcolormesh(t, f_ax[mask], mag[mask, :],
                  shading='gouraud', cmap=CMAP, vmax=vmax)

    ax.set_ylim(0, FMAX)
    ax.set_xlim(0, T_END)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Freq (Hz)')

# FFT 小图 (右上角嵌入)
ax_fft = fig.add_axes([0.78, 0.57, 0.17, 0.16])
fft_freqs = np.fft.rfftfreq(N, 1 / FS)
fft_mag = np.abs(np.fft.rfft(sig))
mask_fft = fft_freqs <= FMAX
ax_fft.plot(fft_freqs[mask_fft], fft_mag[mask_fft], 'k-', lw=0.5)
ax_fft.set_xlim(0, FMAX)
ax_fft.set_title('FFT', fontsize=7)
ax_fft.set_xlabel('Hz', fontsize=6)
ax_fft.tick_params(labelsize=5)

plt.suptitle('SST vs WSST vs WSST2 vs MSST  |  '
             'x(t) = cos(2π·0.5t) + 0.3·cos(2π·45t)',
             fontsize=10, fontweight='bold', y=1.01)
plt.tight_layout()

path = SAVE_DIR / 'synthetic_sst_wsst_wsst2_msst.png'
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()

# ── Summary ──
print(f"\n{'='*55}")
print(f"{'Method':<25s} {'Renyi':>8s}")
print('-' * 40)
for name, re in [('SST (STFT 1st)', re_sst),
                  ('WSST (CWT 1st)', re_wsst),
                  ('WSST2 (CWT 2nd)', re_w2),
                  ('MSST (STFT num=3)', re_ms)]:
    print(f"{name:<25s} {re:8.2f}")

print(f"\nSaved: {path}")
