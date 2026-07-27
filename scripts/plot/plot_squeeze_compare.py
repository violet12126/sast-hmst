"""
MSST 挤压方式性能对比
=====================
对比三种挤压方式的速度和输出质量:
  1. Numpy 全流程 MSST (纯 CPU, 基线)
  2. Numpy MSST (STFT+IF) + CUDA 硬最近邻挤压
  3. Numpy MSST (STFT+IF) + CUDA 线性插值挤压

输出: hmst_figures/squeeze_compare/
  - speed_comparison.png   (速度对比柱状图)
  - tfr_comparison.png     (TFR 输出对比)
  - timing_breakdown.png   (各步骤耗时分解)

用法: python plot_squeeze_compare.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'deploy'))

from models.tfr import msst, compute_renyi
from ssqueezepy import ssq_stft

# Import CUDA kernels
import msst_squeeze_hard
import msst_squeeze_linear

plt.rcParams['font.family'] = 'sans-serif'
SAVE_DIR = Path('hmst_figures/squeeze_compare')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FS = 1000


# ═══════════════════════════════════════════════════════════════
# 1. MSST Pipeline Variants
# ═══════════════════════════════════════════════════════════════

def compute_fsst(x, fs=FS):
    """FSST (Fourier-based Synchrosqueezing Transform) via ssqueezepy."""
    t0 = time.perf_counter()
    Tx, _, ssq_freqs, _ = ssq_stft(x.astype(np.float64), fs=fs)
    elapsed = time.perf_counter() - t0
    return Tx, ssq_freqs, elapsed


def msst_numpy_full(x, fs=FS, num=3):
    """纯 numpy MSST 全流程 (基线). hlength 使用 models/tfr.py 的默认值."""
    t0 = time.perf_counter()
    r = msst(x, fs, num=num, save_trajectory=True)
    elapsed = time.perf_counter() - t0
    return r, elapsed


def msst_step1_stft(x, fs=FS, hlength=None):
    """仅 Step 1: STFT (numpy).

    hlength=None 时使用 min(N, 512), 与 ssqueezepy 对齐,
    避免 N/8 太短导致低频脊线呈波浪形.
    """
    N = len(x)
    if hlength is None:
        hlength = min(N, 512)
    hlength = hlength + 1 - (hlength % 2)
    ht = np.linspace(-0.5, 0.5, hlength)
    h = np.exp(-np.pi / 0.32**2 * ht**2)
    Lh = (hlength - 1) // 2
    tcol = N
    neta = int(round(N / 2))
    tfr_pre = np.zeros((N, tcol), dtype=np.complex128)

    t0 = time.perf_counter()
    for icol in range(tcol):
        ti = icol
        tau_min = -min(neta - 1, Lh, ti)
        tau_max = min(neta - 1, Lh, N - 1 - ti)
        if tau_min > tau_max:
            continue
        tau = np.arange(tau_min, tau_max + 1)
        indices = (N + tau) % N
        tfr_pre[indices, icol] = x[ti + tau] * np.conj(h[Lh + tau])
    tfr = np.fft.fft(tfr_pre, axis=0)
    tfr = tfr[:neta, :]
    elapsed = time.perf_counter() - t0
    return tfr, elapsed, N, neta, tcol


def msst_step2_if(tfr, N, neta, tcol):
    """仅 Step 2: IF 估计 (numpy)."""
    t0 = time.perf_counter()
    omega = np.zeros((neta, tcol - 1))
    for i in range(neta):
        phase = np.unwrap(np.angle(tfr[i, :]))
        omega[i, :] = np.diff(phase) * N / (2.0 * np.pi)
    omega = np.column_stack([omega, omega[:, -1]])
    omega = np.round(omega).astype(np.int32)
    elapsed = time.perf_counter() - t0
    return omega, elapsed


def msst_step3_refine(omega, neta, tcol, num=3):
    """仅 Step 3: IF 迭代精化 (numpy)."""
    t0 = time.perf_counter()
    omegas = [omega.copy()]
    omega_cur = omega
    for _ in range(num - 1):
        omega2 = np.zeros((neta, tcol), dtype=np.int32)
        valid = (omega_cur >= 1) & (omega_cur <= neta)
        eta_idx, b_idx = np.nonzero(valid)
        k_vals = omega_cur[eta_idx, b_idx] - 1
        omega2[eta_idx, b_idx] = omega_cur[k_vals, b_idx]
        omega_cur = omega2.copy()
        omegas.append(omega_cur.copy())
    elapsed = time.perf_counter() - t0
    return omega_cur, omegas, elapsed


def msst_step4_squeeze_numpy(tfr, omega, neta, tcol, N):
    """仅 Step 4: 硬挤压 (numpy)."""
    t0 = time.perf_counter()
    Ts = np.zeros((neta, tcol), dtype=np.complex128)
    threshold = 0.0001
    for b in range(tcol):
        for eta in range(neta):
            if np.abs(tfr[eta, b]) > threshold:
                k = omega[eta, b]
                if 1 <= k <= neta:
                    Ts[k - 1, b] += tfr[eta, b]
    Ts = Ts / (N / 2.0)
    elapsed = time.perf_counter() - t0
    return Ts, elapsed


def msst_step4_squeeze_cuda_hard(tfr_np, omega_np, device='cuda'):
    """Step 4: CUDA 硬最近邻挤压."""
    tfr_norm = tfr_np / (len(tfr_np.shape) if tfr_np.ndim == 1 else 1)
    mag = torch.from_numpy(np.abs(tfr_np)).float().unsqueeze(0).to(device)
    omega_t = torch.from_numpy(omega_np.copy()).int().unsqueeze(0).to(device)

    N_sig = tfr_np.shape[1] * 2  # approximate
    tfr_norm2 = tfr_np / (N_sig / 2.0)

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    Tx = msst_squeeze_hard.msst_squeeze_hard(mag, omega_t, 1e-4)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return Tx.cpu().numpy().squeeze(0), elapsed


def msst_step4_squeeze_cuda_linear(tfr_np, omega_final_np, freqs_hz, fs, device='cuda'):
    """Step 4: CUDA 线性插值挤压."""
    N_sig = 2 * tfr_np.shape[0]
    # Convert integer bin omega to continuous Hz
    if_hz = np.zeros_like(omega_final_np, dtype=np.float32)
    valid = omega_final_np >= 1
    if_hz[valid] = (omega_final_np[valid] - 1) * fs / N_sig

    mag = torch.from_numpy(np.abs(tfr_np)).float().unsqueeze(0).to(device)
    if_hz_t = torch.from_numpy(if_hz).float().unsqueeze(0).to(device)
    freqs_t = torch.from_numpy(freqs_hz).float().to(device)

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    Tx = msst_squeeze_linear.msst_squeeze_linear(mag, if_hz_t, freqs_t, 1e-6)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return Tx.cpu().numpy().squeeze(0), elapsed


# ═══════════════════════════════════════════════════════════════
# 2. Benchmark
# ═══════════════════════════════════════════════════════════════

def benchmark_all(signals, device='cuda'):
    """完整基准测试."""
    N_SAMPLES = len(signals)
    N_WARMUP = 2

    print("=" * 70)
    print(f"MSST Squeeze Benchmark: {N_SAMPLES} samples, T={signals.shape[1]}")
    print(f"Device: {device}, CUDA available: {torch.cuda.is_available()}")
    print("=" * 70)

    # Warmup
    _ = msst(signals[0], FS, num=3, save_trajectory=True)

    results = {
        'stft': [], 'if_est': [], 'refine': [],
        'squeeze_np': [], 'squeeze_cuda_hard': [], 'squeeze_cuda_linear': [],
        'total_np': [], 'total_cuda_hard': [], 'total_cuda_linear': [],
        'renyi_np': [], 'renyi_hard': [], 'renyi_linear': [],
        'max_diff_hard': [], 'mean_diff_hard': [],
        'max_diff_linear': [], 'mean_diff_linear': [],
    }

    for i in range(min(N_SAMPLES, 20)):
        x = signals[i].astype(np.float64)
        N = len(x)

        # ── Numpy full pipeline ──
        r_np, t_total_np = msst_numpy_full(x, FS, num=3)
        tfr_np = r_np['MSST']
        freqs_hz = r_np['freqs']

        # ── Step-by-step numpy for breakdown ──
        tfr, t_stft, N_sig, neta, tcol = msst_step1_stft(x, FS)
        omega_1, t_if = msst_step2_if(tfr, N_sig, neta, tcol)
        omega_f, omegas, t_refine = msst_step3_refine(omega_1, neta, tcol, num=3)
        Ts_np, t_sqz_np = msst_step4_squeeze_numpy(tfr, omega_f, neta, tcol, N_sig)

        # ── CUDA hard squeeze ──
        Ts_hard, t_sqz_hard = msst_step4_squeeze_cuda_hard(tfr, omega_f, device)
        t_total_hard = t_stft + t_if + t_refine + t_sqz_hard

        # ── CUDA linear squeeze ──
        Ts_linear, t_sqz_linear = msst_step4_squeeze_cuda_linear(
            tfr, omega_f, freqs_hz, FS, device)
        t_total_linear = t_stft + t_if + t_refine + t_sqz_linear

        # ── Quality metrics ──
        re_np = compute_renyi(tfr_np)
        re_hard = compute_renyi(Ts_hard)
        re_linear = compute_renyi(Ts_linear)

        diff_hard = np.abs(tfr_np - Ts_hard)
        diff_linear = np.abs(tfr_np - Ts_linear)

        # Store
        results['stft'].append(t_stft)
        results['if_est'].append(t_if)
        results['refine'].append(t_refine)
        results['squeeze_np'].append(t_sqz_np)
        results['squeeze_cuda_hard'].append(t_sqz_hard)
        results['squeeze_cuda_linear'].append(t_sqz_linear)
        results['total_np'].append(t_total_np)
        results['total_cuda_hard'].append(t_total_hard)
        results['total_cuda_linear'].append(t_total_linear)
        results['renyi_np'].append(re_np)
        results['renyi_hard'].append(re_hard)
        results['renyi_linear'].append(re_linear)
        results['max_diff_hard'].append(diff_hard.max())
        results['mean_diff_hard'].append(diff_hard.mean())
        results['max_diff_linear'].append(diff_linear.max())
        results['mean_diff_linear'].append(diff_linear.mean())

        if i == 0:
            # Save one sample for TFR comparison plot
            # Also compute FSST for the sample
            Tx_fsst, fsst_freqs, t_fsst = compute_fsst(x, FS)
            sample_data = {
                'x': x, 'tfr_np': tfr_np, 'Ts_hard': Ts_hard,
                'Ts_linear': Ts_linear, 'freqs': freqs_hz,
                'fsst': np.abs(Tx_fsst), 'fsst_freqs': fsst_freqs,
                'stft': np.abs(tfr) / (N_sig / 2.0),
            }

    # ── Compute averages ──
    avg = {k: np.mean(v) for k, v in results.items() if 'renyi' not in k and 'diff' not in k}
    avg['renyi_np'] = np.mean(results['renyi_np'])
    avg['renyi_hard'] = np.mean(results['renyi_hard'])
    avg['renyi_linear'] = np.mean(results['renyi_linear'])
    avg['max_diff_hard'] = np.mean(results['max_diff_hard'])
    avg['mean_diff_hard'] = np.mean(results['mean_diff_hard'])
    avg['max_diff_linear'] = np.mean(results['max_diff_linear'])
    avg['mean_diff_linear'] = np.mean(results['mean_diff_linear'])

    return avg, sample_data


# ═══════════════════════════════════════════════════════════════
# 3. Plotting
# ═══════════════════════════════════════════════════════════════

def plot_speed_comparison(avg, save_path):
    """速度对比柱状图."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: Total pipeline time ──
    ax = axes[0]
    methods = ['Numpy\n(full CPU)', 'Numpy+CUDA\nHard NN', 'Numpy+CUDA\nLinear']
    times = [avg['total_np'] * 1000, avg['total_cuda_hard'] * 1000,
             avg['total_cuda_linear'] * 1000]
    colors = ['#d62728', '#2ca02c', '#1f77b4']
    bars = ax.bar(methods, times, color=colors, edgecolor='white', linewidth=0.8)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{t:.1f} ms', ha='center', fontsize=10, fontweight='bold')
    speedup_hard = avg['total_np'] / avg['total_cuda_hard']
    speedup_linear = avg['total_np'] / avg['total_cuda_linear']
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.set_title(f'Total MSST Pipeline Time\n'
                 f'CUDA Hard: {speedup_hard:.1f}x faster on squeeze step',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # ── Right: Step breakdown ──
    ax = axes[1]
    steps = ['STFT', 'IF Estimate', 'IF Refine', 'Squeeze']
    x = np.arange(len(steps))
    width = 0.25

    np_times = [avg['stft'] * 1000, avg['if_est'] * 1000,
                avg['refine'] * 1000, avg['squeeze_np'] * 1000]
    hard_times = [avg['stft'] * 1000, avg['if_est'] * 1000,
                  avg['refine'] * 1000, avg['squeeze_cuda_hard'] * 1000]
    linear_times = [avg['stft'] * 1000, avg['if_est'] * 1000,
                    avg['refine'] * 1000, avg['squeeze_cuda_linear'] * 1000]

    ax.bar(x - width, np_times, width, label='Numpy squeeze', color='#d62728')
    ax.bar(x, hard_times, width, label='CUDA Hard', color='#2ca02c')
    ax.bar(x + width, linear_times, width, label='CUDA Linear', color='#1f77b4')

    ax.set_xticks(x)
    ax.set_xticklabels(steps)
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.set_title('Per-Step Time Breakdown', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tfr_comparison(sample, save_path):
    """TFR 输出对比: FSST + 3 种 MSST."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    FS = 1000
    FMAX = 200

    # ── FSST has its own freq/time axes ──
    t_fsst = np.arange(sample['fsst'].shape[1]) / FS
    f_fsst = sample['fsst_freqs']
    mask_fsst = f_fsst <= FMAX
    fsst_renyi = compute_renyi(sample["fsst"])

    # ── MSST variants share the same freq axis ──
    t_msst = np.arange(sample['tfr_np'].shape[1]) / FS
    mask_msst = sample['freqs'] <= FMAX
    freqs_msst = sample['freqs'][mask_msst]

    np_renyi = compute_renyi(sample["tfr_np"])
    hard_renyi = compute_renyi(sample["Ts_hard"])
    linear_renyi = compute_renyi(sample["Ts_linear"])

    panels = [
        (axes[0, 0], 'FSST (ssqueezepy)', sample['fsst'][mask_fsst, :],
         f'R={fsst_renyi:.2f}', t_fsst, f_fsst[mask_fsst]),
        (axes[0, 1], 'MSST — Numpy Hard NN', sample['tfr_np'][mask_msst, :],
         f'R={np_renyi:.2f}', t_msst, freqs_msst),
        (axes[1, 0], 'MSST — CUDA Hard NN', sample['Ts_hard'][mask_msst, :],
         f'R={hard_renyi:.2f}', t_msst, freqs_msst),
        (axes[1, 1], 'MSST — CUDA Linear Interp', sample['Ts_linear'][mask_msst, :],
         f'R={linear_renyi:.2f}', t_msst, freqs_msst),
    ]

    for ax, title, tfr, re_str, t_ax, f_ax in panels:
        vmax = tfr.max() * 0.5
        im = ax.pcolormesh(t_ax, f_ax, tfr,
                          shading='gouraud', cmap='jet', vmax=vmax)
        ax.set_ylim(0, FMAX)
        ax.set_title(f'{title}\n{re_str}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')

    plt.suptitle('TFR Comparison: FSST vs MSST (Hard NN vs Linear Interpolation)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_timing_breakdown(avg, save_path):
    """各步骤耗时占比饼图 vs 加速比."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: Numpy time breakdown (pie) ──
    ax = axes[0]
    labels = ['STFT', 'IF Estimate', 'IF Refine', 'Squeeze (Hard NN)']
    sizes = [avg['stft'], avg['if_est'], avg['refine'], avg['squeeze_np']]
    colors_pie = ['#ff7f0e', '#9467bd', '#8c564b', '#d62728']
    explode = (0, 0, 0, 0.05)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
           autopct='%1.0f%%', startangle=90, textprops={'fontsize': 9})
    ttl_np_ms = avg['total_np'] * 1000
    ax.set_title(f'Numpy MSST: Total {ttl_np_ms:.0f} ms',
                 fontsize=11, fontweight='bold')

    # ── Right: Squeeze step speedup (bar) ──
    ax = axes[1]
    squeeze_methods = ['Numpy\nHard NN', 'CUDA\nHard NN', 'CUDA\nLinear']
    squeeze_times = [avg['squeeze_np'] * 1000, avg['squeeze_cuda_hard'] * 1000,
                     avg['squeeze_cuda_linear'] * 1000]
    bars = ax.bar(squeeze_methods, squeeze_times,
                  color=['#d62728', '#2ca02c', '#1f77b4'],
                  edgecolor='white', linewidth=0.8)

    for bar, t in zip(bars, squeeze_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{t:.2f} ms', ha='center', fontsize=10, fontweight='bold')

    speedup_h = avg['squeeze_np'] / avg['squeeze_cuda_hard']
    speedup_l = avg['squeeze_np'] / avg['squeeze_cuda_linear']
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.set_title(f'Squeeze Step Only\n'
                 f'CUDA Hard: {speedup_h:.0f}x  |  '
                 f'CUDA Linear: {speedup_l:.0f}x',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("MSST Squeeze Kernel Performance Comparison")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("[ERROR] CUDA not available — cannot test CUDA kernels")
        return

    # ── Generate test signals (short for benchmark speed) ──
    print("\nGenerating test signals...")
    T_SIG = 500  # shorter for faster benchmark
    t = np.arange(0, T_SIG / FS, 1 / FS)
    base_sig = (np.sin(2 * np.pi * 48 * t + 0.15 * np.sin(2 * np.pi * 3 * t)) +
                0.6 * np.sin(2 * np.pi * 96 * t) +
                0.25 * np.sin(2 * np.pi * 12 * t))
    base_sig = base_sig.astype(np.float64)
    N_TEST = 15
    signals = np.tile(base_sig, (N_TEST, 1))
    signals += np.random.randn(*signals.shape) * 0.01
    print(f"  {N_TEST} signals, T={T_SIG}, fs={FS} Hz")

    # ── Run benchmark ──
    print("\nRunning benchmark...")
    avg, sample = benchmark_all(signals, device)

    # ── Generate longer signal for TFR comparison ──
    print("\nGenerating longer signal for TFR comparison...")
    T_LONG = 1000  # 1.0 second for better visualization
    t_long = np.arange(0, T_LONG / FS, 1 / FS)
    x_long = (np.sin(2 * np.pi * 48 * t_long + 0.15 * np.sin(2 * np.pi * 3 * t_long)) +
              0.6 * np.sin(2 * np.pi * 96 * t_long) +
              0.25 * np.sin(2 * np.pi * 12 * t_long))
    x_long = x_long.astype(np.float64)

    # Compute all TFR variants on the longer signal
    print("  Computing FSST...")
    Tx_fsst, fsst_freqs, _ = compute_fsst(x_long, FS)
    print("  Computing MSST (numpy full)...")
    r_long, _ = msst_numpy_full(x_long, FS, num=3)
    tfr_long = r_long['MSST']
    freqs_long = r_long['freqs']
    omega_long = r_long['omega_final']

    # Step-by-step for CUDA squeeze variants
    N_long = len(x_long)
    tfr_raw, _, _, neta_long, tcol_long = msst_step1_stft(x_long, FS)
    omega_1_long, _ = msst_step2_if(tfr_raw, N_long, neta_long, tcol_long)
    omega_f_long, _, _ = msst_step3_refine(omega_1_long, neta_long, tcol_long, num=3)
    Ts_hard_long, _ = msst_step4_squeeze_cuda_hard(tfr_raw, omega_f_long, device)
    Ts_linear_long, _ = msst_step4_squeeze_cuda_linear(
        tfr_raw, omega_f_long, freqs_long, FS, device)

    sample_long = {
        'x': x_long,
        'tfr_np': tfr_long,
        'Ts_hard': Ts_hard_long,
        'Ts_linear': Ts_linear_long,
        'freqs': freqs_long,
        'fsst': np.abs(Tx_fsst),
        'fsst_freqs': fsst_freqs,
        'stft': np.abs(tfr_raw) / (N_long / 2.0),
    }
    print(f"  Long signal: T={T_LONG} ({T_LONG/FS:.1f}s), done.")

    # ── Print results ──
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  {'Metric':<35s} {'Numpy':>10s} {'CUDA Hard':>10s} {'CUDA Linear':>10s}")
    print(f"  {'-'*65}")
    print(f"  {'Total pipeline (ms)':<35s} {avg['total_np']*1000:>10.1f} "
          f"{avg['total_cuda_hard']*1000:>10.1f} {avg['total_cuda_linear']*1000:>10.1f}")
    print(f"  {'Squeeze step only (ms)':<35s} {avg['squeeze_np']*1000:>10.1f} "
          f"{avg['squeeze_cuda_hard']*1000:>10.2f} {avg['squeeze_cuda_linear']*1000:>10.2f}")
    print(f"  {'Squeeze speedup':<35s} {'1.0x':>10s} "
          f"{avg['squeeze_np']/avg['squeeze_cuda_hard']:>9.0f}x "
          f"{avg['squeeze_np']/avg['squeeze_cuda_linear']:>9.0f}x")
    print()
    print(f"  {'Renyi entropy (lower=better)':<35s} {avg['renyi_np']:>10.2f} "
          f"{avg['renyi_hard']:>10.2f} {avg['renyi_linear']:>10.2f}")
    print(f"  {'Max diff vs Numpy':<35s} {'(baseline)':>10s} "
          f"{avg['max_diff_hard']:>10.6f} {avg['max_diff_linear']:>10.4f}")
    print(f"  {'Mean diff vs Numpy':<35s} {'(baseline)':>10s} "
          f"{avg['mean_diff_hard']:>10.8f} {avg['mean_diff_linear']:>10.6f}")
    print()

    print(f"  Step breakdown (ms):")
    print(f"    {'STFT:':<20s} {avg['stft']*1000:>8.1f} ms")
    print(f"    {'IF Estimate:':<20s} {avg['if_est']*1000:>8.1f} ms")
    print(f"    {'IF Refine:':<20s} {avg['refine']*1000:>8.1f} ms")
    print(f"    {'Squeeze (numpy):':<20s} {avg['squeeze_np']*1000:>8.1f} ms")
    print(f"    {'Squeeze (CUDA H):':<20s} {avg['squeeze_cuda_hard']*1000:>8.2f} ms")
    print(f"    {'Squeeze (CUDA L):':<20s} {avg['squeeze_cuda_linear']*1000:>8.2f} ms")

    # ── Generate plots ──
    print("\nGenerating plots...")
    plot_speed_comparison(avg, SAVE_DIR / 'speed_comparison.png')
    plot_tfr_comparison(sample_long, SAVE_DIR / 'tfr_comparison.png')
    plot_timing_breakdown(avg, SAVE_DIR / 'timing_breakdown.png')

    print(f"\nDone. Figures saved to {SAVE_DIR}/")


if __name__ == '__main__':
    main()
