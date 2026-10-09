"""绘制高斯重排机制示意图: STFT 模糊峰 → IF 指向 → 高斯摊开 → 锐化脊线."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ═══════════════════════════════════════════════════════════════
# 构造数据
# ═══════════════════════════════════════════════════════════════
fs = 1000
N = 2000
f_res = fs / N          # 0.5 Hz/bin
freqs_hz = np.arange(500) * f_res   # 0-249.5 Hz

# 一个源 bin 在 f=60 (30Hz), 能量 E=2.0, IF 估计 omega_hat=98 (49Hz, BPF)
src_bin = 60
omega_hat = 98
E = 2.0
sigma_narrow = 1.0      # w_i 大 → σ 小 → 锐
sigma_wide = 4.0        # w_i 小 → σ 大 → 糊

def gauss_redistribute(omega_hat, E, sig, K=45):
    """模拟 kernel 的高斯摊开: 返回 (target_bins, energies)."""
    targets = np.arange(omega_hat - K, omega_hat + K + 1)
    w = np.exp(-0.5 * (np.arange(-K, K + 1) / sig) ** 2)
    w = w / w.sum()  # Z 归一化
    return targets, w * E

t_n, e_n = gauss_redistribute(omega_hat, E, sigma_narrow)
t_w, e_w = gauss_redistribute(omega_hat, E, sigma_wide)

# ═══════════════════════════════════════════════════════════════
# 图 (3 行): 源 → 重排 → 结果
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                         gridspec_kw={'height_ratios': [1, 2, 1]})

# ── Panel 1: 源 STFT 能量 (模糊宽峰) ──
ax = axes[0]
f_range = np.arange(0, 200)
# 模拟一个 STFT 模糊峰: 中心在 30Hz, 宽 (STFT 高斯窗导致的)
source = 2.0 * np.exp(-0.5 * ((f_range - src_bin) / 12.0) ** 2)
ax.plot(f_range * f_res, source, 'b-', lw=2, label='Source STFT magnitude (fuzzy peak)')
ax.fill_between(f_range * f_res, 0, source, color='blue', alpha=0.15)
ax.axvline(freqs_hz[src_bin], color='blue', ls='--', lw=1, alpha=0.7)
ax.annotate(f'source bin f={src_bin}\n({freqs_hz[src_bin]:.1f} Hz)',
            xy=(freqs_hz[src_bin], source.max()), xytext=(5, 1.9),
            fontsize=8, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue'))
ax.axvline(freqs_hz[omega_hat], color='red', ls='--', lw=1.5, alpha=0.8)
ax.annotate(f'IF estimate omega_hat={omega_hat}\n({freqs_hz[omega_hat]:.1f} Hz, BPF)',
            xy=(freqs_hz[omega_hat], 0.05), xytext=(freqs_hz[omega_hat]-8, 1.5),
            fontsize=8, color='red',
            arrowprops=dict(arrowstyle='->', color='red'))
ax.set_ylabel('Magnitude')
ax.set_title('Step 1: Fuzzy STFT peak + IF estimate points where energy "belongs"',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)

# ── Panel 2: 高斯摊开 (σ 对比) ──
ax = axes[1]
ax.fill_between(t_n * f_res, 0, e_n, color='green', alpha=0.35,
                label=f'sigma={sigma_narrow} (w_i high: sharp squeeze)')
ax.plot(t_n * f_res, e_n, 'g-', lw=1.5)
ax.fill_between(t_w * f_res, 0, -e_w, color='orange', alpha=0.35,
                label=f'sigma={sigma_wide} (w_i low: conservative spread)')
ax.plot(t_w * f_res, -e_w, 'orange', lw=1.5)
ax.axhline(0, color='black', lw=0.5)
# 标注 E 总能量守恒
ax.annotate(f'Sum = E = {E:.0f} (energy conserved)', xy=(0.05, 1.8),
            fontsize=9, fontweight='bold')
# 标 k 偏移
for k in [-2, -1, 0, 1, 2]:
    b = omega_hat + k
    ax.annotate(f'k={k:+d}', xy=(freqs_hz[b], 0.1 if k >= 0 else -0.1),
                ha='center', fontsize=7, color='darkgreen')
ax.set_ylabel('Energy distributed')
ax.set_title('Step 2: Gaussian kernel spreads E around omega_hat +/- K (K=45), sigma controls sharpness',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(-2.2, 2.2)
ax.grid(alpha=0.3)

# ── Panel 3: 结果对比 (原始模糊 vs 锐化) ──
ax = axes[2]
# 锐化结果: 能量聚集在 omega_hat 附近
result = np.zeros(200)
t_contrib = t_n[(t_n >= 0) & (t_n < 200)]
e_contrib = e_n[(t_n >= 0) & (t_n < 200)]
result[t_contrib] += e_contrib
# 原始宽峰 (缩放对比)
ax.plot(f_range * f_res, source, 'b--', lw=1.2, alpha=0.6, label='Original fuzzy peak (STFT)')
ax.plot(f_range * f_res, result, 'r-', lw=2, label='After Gaussian reassignment (SAST)')
ax.fill_between(f_range * f_res, 0, result, color='red', alpha=0.25)
ax.set_ylabel('Magnitude')
ax.set_xlabel('Frequency [Hz]')
ax.set_title('Step 3: Energy moved to IF ridge -> sharp line where the signal really is',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(0, 2.2)
ax.grid(alpha=0.3)

plt.tight_layout()
out = 'deploy/reassign_illustration.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'[Saved] {out}')
