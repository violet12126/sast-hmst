"""
SAST 自适应软核挤压机制验证脚本
==================================================
验证核心：挤压核宽度 σ 对稳定谐波与漂移滑差的不同影响。

信号设计:
  1. 高频稳态: 45 Hz 纯正弦波 (模拟稳定谐波)
  2. 低频滑差: 12 Hz 中心, ±2 Hz 漂移的正弦调制 (模拟涡带)
  3. 背景噪声: 高斯白噪声
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import time

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] # 支持中文
plt.rcParams['axes.unicode_minus'] = False

# ═══════════════════════════════════════════
# 1. 信号生成
# ═══════════════════════════════════════════
fs = 500
T = 2.0
t = np.arange(0, T, 1/fs)

# 分量 1: 稳定谐波 45 Hz
sig_stable = 1.0 * np.cos(2 * np.pi * 45 * t)

# 分量 2: 漂移滑差 12 Hz (频率随时间做 ±2Hz 的正弦波动)
# 瞬时频率 IF = 12 + 2*sin(2*pi*1.5*t)
# 相位 = 积分(IF) = 12*t - (2/(2*pi*1.5))*cos(2*pi*1.5*t)
phase_drift = 12 * t - (2 / (2 * np.pi * 1.5)) * np.cos(2 * np.pi * 1.5 * t)
sig_drift = 1.2 * np.cos(2 * np.pi * phase_drift)

# 混合信号 + 噪声
sig = sig_stable + sig_drift + 0.3 * np.random.randn(len(t))

# ═══════════════════════════════════════════
# 2. 基础 STFT 与 瞬时频率 (IF) 估计
# ═══════════════════════════════════════════
print("计算 STFT 与 IF 估计...")
nperseg = 256
f_ax, t_ax, Zxx = stft(sig, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg-1)
Z_mag = np.abs(Zxx)

# 通过相位差计算精确的瞬时频率 (Phase Vocoder 算法)
dt = 1 / fs
dphase = np.diff(np.angle(Zxx), axis=-1)
# 减去理论中心频率产生的相位提前量
expected_advance = 2 * np.pi * f_ax[:, None] * dt
dp_wrapped = (dphase - expected_advance + np.pi) % (2 * np.pi) - np.pi
if_est = f_ax[:, None] + dp_wrapped / (2 * np.pi * dt)
# 补齐最后一帧
if_est = np.pad(if_est, ((0, 0), (0, 1)), mode='edge')

# ═══════════════════════════════════════════
# 3. 高斯软核能量重分配引擎 (向量化加速)
# ═══════════════════════════════════════════
def gaussian_reassignment(mag, f_axis, if_estimate, sigma_map):
    """
    mag: STFT 幅值 (F, T)
    f_axis: 频率轴 (F,)
    if_estimate: 估计的瞬时频率 (F, T)
    sigma_map: 逐 TF bin 的挤压核宽度 (F, T)
    """
    # 扩展维度以进行矩阵广播: (目标频率F, 源频率F, 时间T)
    f_target = f_axis[:, None, None]     
    if_source = if_estimate[None, :, :]  
    sig_source = sigma_map[None, :, :]   
    mag_source = mag[None, :, :]         

    # 计算高斯权重: exp( - (f_target - IF)^2 / (2 * sigma^2) )
    weights = np.exp(-0.5 * ((f_target - if_source) / sig_source)**2)
    # 能量守恒归一化
    weights /= np.sum(weights, axis=0, keepdims=True) 
    
    # 将能量分配到新网格并沿源频率轴求和
    reassigned_mag = np.sum(mag_source * weights, axis=1)
    return reassigned_mag

print("执行能量重分配 (硬挤压/软挤压/自适应)...")
t0 = time.time()

# 策略 A: 全局硬挤压 (类 MSST) - Kernel 极窄
sigma_hard = np.full_like(Z_mag, 0.2)
TFR_hard = gaussian_reassignment(Z_mag, f_ax, if_est, sigma_hard)

# 策略 B: 全局软挤压 - Kernel 较宽
sigma_soft = np.full_like(Z_mag, 1.5)
TFR_soft = gaussian_reassignment(Z_mag, f_ax, if_est, sigma_soft)

# 策略 C: SAST 自适应挤压 - 高频硬挤，低频软留
# 模拟 GAT 输出: >30Hz 区域 C_i 极高(硬)，<30Hz 区域 C_i 极低(软)
sigma_sast = np.full_like(Z_mag, 1.5)        # 默认软核
sigma_sast[f_ax > 30, :] = 0.2               # 高频硬核
TFR_sast = gaussian_reassignment(Z_mag, f_ax, if_est, sigma_sast)

print(f"重分配完成，耗时: {time.time() - t0:.2f}s")

# ═══════════════════════════════════════════
# 4. 绘图对比
# ═══════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
vmax = np.percentile(Z_mag, 99.5)

panels = [
    (axes[0, 0], '1. 原始 STFT (基准)', Z_mag),
    (axes[0, 1], '2. 全局硬挤压 (类似 MSST)\n[45Hz完美锐化，12Hz滑差严重阶梯化破碎]', TFR_hard),
    (axes[1, 0], '3. 全局软挤压\n[12Hz滑差完美保留，45Hz依旧模糊]', TFR_soft),
    (axes[1, 1], '4. SAST 自适应挤压\n[45Hz硬挤锐化 + 12Hz软挤保护滑差]', TFR_sast)
]

for ax, title, tfr in panels:
    ax.pcolormesh(t_ax, f_ax, tfr, shading='gouraud', cmap='jet', vmax=vmax)
    ax.set_ylim(0, 60)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    
    # 标出两个分量的理论轨迹作为参考线
    ax.plot(t, np.full_like(t, 45), 'w--', alpha=0.5, lw=1)
    ax.plot(t, 12 + 2 * np.sin(2 * np.pi * 1.5 * t), 'w--', alpha=0.5, lw=1)

plt.tight_layout()
plt.savefig('sast_kernel_comparison.png', dpi=200)
print("图像已保存为 sast_kernel_comparison.png")