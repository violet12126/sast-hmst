"""
对比 numpy MSST (手动 STFT) vs torch.stft 的 IF 估计差异
=========================================================
验证 torch.stft 能否替代手动 STFT 作为 MSST 全流程 GPU 化的第一步。

用法: python test_torch_stft_compare.py
"""

import numpy as np
import torch
import time
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from models.tfr import msst, compute_renyi

FS = 1000
SAVE_DIR = Path('hmst_figures/torch_stft_compare')
SAVE_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.family'] = 'sans-serif'


# ═══════════════════════════════════════════════════════════════
# 1. Numpy MSST (baseline)
# ═══════════════════════════════════════════════════════════════

def msst_stft_numpy(x, hlength=None):
    """手动 STFT (当前 msst() 的 Step 1)，返回 omega_1 + 中间量。"""
    x = np.asarray(x, dtype=np.float64).ravel()
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
    t_stft = time.perf_counter() - t0

    # IF estimation (Step 2)
    t0 = time.perf_counter()
    omega = np.zeros((neta, tcol - 1))
    for i in range(neta):
        phase = np.unwrap(np.angle(tfr[i, :]))
        omega[i, :] = np.diff(phase) * N / (2.0 * np.pi)
    omega = np.column_stack([omega, omega[:, -1]])
    omega = np.round(omega).astype(np.int32)
    t_if = time.perf_counter() - t0

    return tfr, omega, t_stft, t_if, N, neta, tcol


# ═══════════════════════════════════════════════════════════════
# 2. Torch STFT (GPU 替代)
# ═══════════════════════════════════════════════════════════════

def gaussian_window_torch(hlength, device='cpu'):
    """与手动实现相同的 Gaussian window: exp(-pi*t^2/0.32^2)."""
    t = torch.linspace(-0.5, 0.5, hlength, device=device, dtype=torch.float64)
    return torch.exp(-np.pi / 0.32**2 * t**2)


def msst_stft_torch(x_np, hlength=None, device='cpu'):
    """torch.stft 替代手动 STFT + IF 估计。

    Args:
        x_np: [T] numpy array (float64)
        hlength: window length

    Returns:
        tfr_np:    [neta, tcol] complex128 numpy
        omega_np:  [neta, tcol] int32 numpy
        t_stft:    STFT 耗时 (s)
        t_if:      IF 估计耗时 (s)
    """
    N = len(x_np)

    if hlength is None:
        hlength = min(N, 512)
    hlength = hlength + 1 - (hlength % 2)

    # Window
    window = gaussian_window_torch(hlength, device=device)
    # 镜像填充到 N (n_fft) — torch.stft 的 win_length < n_fft 时会自动补零
    # 但我们希望窗口完全填满 n_fft 以避免 torch.stft 内部裁剪
    # 实际上 torch.stft 会自动 pad win_length -> n_fft，所以我们直接用 win_length=hlength

    x_t = torch.from_numpy(x_np).to(device=device, dtype=torch.float64)

    # n_fft 需要 >= win_length (torch 约束)
    n_fft = max(N, hlength)

    # STFT via torch.stft
    # hop_length=1, win_length=hlength, center=True
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    X = torch.stft(
        x_t.unsqueeze(0),           # [1, T]
        n_fft=n_fft,
        hop_length=1,
        win_length=hlength,
        window=window.float(),
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    # X: [1, N//2+1, T_frames]
    if device == 'cuda':
        torch.cuda.synchronize()
    t_stft = time.perf_counter() - t0

    neta = n_fft // 2 + 1
    tcol_stft = X.shape[2]

    # IF estimation (torch — GPU)
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # X: [1, neta, tcol] -> squeeze -> [neta, tcol]
    X_sq = X.squeeze(0).to(torch.complex128)  # [neta, tcol]

    # phase diff
    phase = torch.angle(X_sq)                              # [neta, tcol]
    phase_unwrapped = torch.from_numpy(
        np.unwrap(phase.cpu().numpy(), axis=1)
    ).to(device=device, dtype=torch.float64)               # [neta, tcol]
    diff_phase = torch.diff(phase_unwrapped, dim=1)        # [neta, tcol-1]
    omega = diff_phase * N / (2.0 * np.pi)                 # [neta, tcol-1]
    omega = torch.round(omega).to(torch.int32)

    # 末列填充
    last_col = omega[:, -1:].clone()
    omega = torch.cat([omega, last_col], dim=1)            # [neta, tcol]

    if device == 'cuda':
        torch.cuda.synchronize()
    t_if = time.perf_counter() - t0

    # -> numpy
    tfr_np = X_sq.cpu().numpy()
    omega_np = omega.cpu().numpy()

    return tfr_np, omega_np, t_stft, t_if


# ═══════════════════════════════════════════════════════════════
# 3. MSST IF 迭代 (共享, numpy)
# ═══════════════════════════════════════════════════════════════

def msst_refine(omega, neta, tcol, num=3):
    """IF 迭代精化 (numpy). 返回 omega_final + omegas + 耗时."""
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


# ═══════════════════════════════════════════════════════════════
# 4. Comparison
# ═══════════════════════════════════════════════════════════════

def compare_all(T=1000, device='cuda'):
    """全面对比 numpy vs torch STFT 的 MSST 各步骤输出."""

    print("=" * 70)
    print(f"MSST STFT: Numpy (manual) vs Torch (torch.stft)")
    print(f"T={T}, fs={FS} Hz, device={device}")
    print("=" * 70)

    # ── Test signal ──
    t_axis = np.arange(T) / FS
    x = (np.sin(2 * np.pi * 48 * t_axis + 0.15 * np.sin(2 * np.pi * 3 * t_axis)) +
         0.6 * np.sin(2 * np.pi * 96 * t_axis) +
         0.25 * np.sin(2 * np.pi * 12 * t_axis))
    x = x.astype(np.float64)

    # ── 1. Numpy baseline (完整 msst) ──
    print("\n[1] Numpy MSST (full manual)")
    t0 = time.perf_counter()
    r_np = msst(x, FS, num=3, save_trajectory=True)
    t_np = time.perf_counter() - t0
    tfr_np = r_np['STFT']          # [neta, tcol] complex
    omega1_np = r_np['omegas'][0]   # [neta, tcol] int32
    omega_f_np = r_np['omega_final']
    msst_np = r_np['MSST']
    print(f"  Total: {t_np*1000:.1f} ms, Rényi: {compute_renyi(msst_np):.2f}")

    # ── 2. Numpy manual STFT + IF (step-by-step) ──
    print("\n[2] Numpy manual STFT + IF (step-by-step)")
    tfr_np2, omega1_np2, t_stft_np, t_if_np, N, neta, tcol = msst_stft_numpy(x)
    omega_f_np2, omegas_np2, t_refine = msst_refine(omega1_np2, neta, tcol, num=3)
    print(f"  STFT: {t_stft_np*1000:.1f} ms, IF: {t_if_np*1000:.1f} ms, "
          f"Refine: {t_refine*1000:.1f} ms")

    # ── 3. Torch STFT + IF ──
    print("\n[3] Torch STFT + IF (torch.stft)")
    tfr_t, omega1_t, t_stft_t, t_if_t = msst_stft_torch(x, device=device)

    # Torch STFT may produce different neta (N//2+1 vs manually rounded N/2)
    neta_t = tfr_t.shape[0]
    tcol_t = tfr_t.shape[1]
    print(f"  STFT: {t_stft_t*1000:.2f} ms, IF: {t_if_t*1000:.2f} ms")
    print(f"  Shape: [{neta_t}, {tcol_t}] (numpy: [{neta}, {tcol}])")

    # IF refinement on torch omega
    omega_f_t, omegas_t, t_refine_t = msst_refine(omega1_t, neta_t, tcol_t, num=3)
    print(f"  Refine: {t_refine_t*1000:.1f} ms")

    # ── 4. Compare ──
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # 对齐形状 (torch may have neta = N//2+1 vs numpy neta = round(N/2))
    neta_min = min(neta, neta_t)
    tcol_min = min(tcol, tcol_t)

    # ── STFT 幅度 ──
    stft_np_abs = np.abs(tfr_np2[:neta_min, :tcol_min])
    stft_t_abs = np.abs(tfr_t[:neta_min, :tcol_min])
    stft_diff = np.abs(stft_np_abs - stft_t_abs)
    stft_rel_diff = stft_diff / (stft_np_abs.max() + 1e-8)

    print(f"\n  STFT magnitude:")
    print(f"    Max diff:    {stft_diff.max():.6f}")
    print(f"    Mean diff:   {stft_diff.mean():.6f}")
    print(f"    Rel max:     {stft_rel_diff.max()*100:.2f}%")
    print(f"    Correlation: {np.corrcoef(stft_np_abs.ravel(), stft_t_abs.ravel())[0,1]:.6f}")

    # ── Omega_1 (一阶 IF) ──
    o1_np = omega1_np2[:neta_min, :tcol_min]
    o1_t = omega1_t[:neta_min, :tcol_min]
    # Only compare where both are valid (>=1)
    valid_mask = (o1_np >= 1) & (o1_t >= 1)
    if valid_mask.sum() > 0:
        o1_match = (o1_np[valid_mask] == o1_t[valid_mask]).mean()
        o1_diff = np.abs(o1_np[valid_mask].astype(float) -
                         o1_t[valid_mask].astype(float))
        print(f"\n  Omega_1 (1st-order IF):")
        print(f"    Exact match:  {o1_match*100:.1f}%")
        print(f"    Mean |diff|:  {o1_diff.mean():.2f} bins (when valid)")
        print(f"    Max |diff|:   {o1_diff.max():.0f} bins")
        # Distribution of differences
        diffs, counts = np.unique(o1_diff.astype(int), return_counts=True)
        for d, c in zip(diffs[:8], counts[:8]):
            print(f"      diff={d} bin: {c:>8d} ({c/valid_mask.sum()*100:.1f}%)")

    # ── Omega_final (after 3 refinements) ──
    of_np = omega_f_np2[:neta_min, :tcol_min]
    of_t = omega_f_t[:neta_min, :tcol_min]
    valid_f = (of_np >= 1) & (of_t >= 1)
    if valid_f.sum() > 0:
        of_match = (of_np[valid_f] == of_t[valid_f]).mean()
        of_diff = np.abs(of_np[valid_f].astype(float) - of_t[valid_f].astype(float))
        print(f"\n  Omega_final (after 3 refinements):")
        print(f"    Exact match:  {of_match*100:.1f}%")
        print(f"    Mean |diff|:  {of_diff.mean():.2f} bins")
        print(f"    Max |diff|:   {of_diff.max():.0f} bins")
        diffs_f, counts_f = np.unique(of_diff.astype(int), return_counts=True)
        for d, c in zip(diffs_f[:8], counts_f[:8]):
            print(f"      diff={d} bin: {c:>8d} ({c/valid_f.sum()*100:.1f}%)")

    # ── TFR shape alignment ──
    freqs_np = r_np['freqs']
    freqs_t = np.arange(neta_t) / N * FS  # torch STFT freq axis

    # ── Speed ──
    speedup_stft = t_stft_np / max(t_stft_t, 1e-6)
    speedup_if = t_if_np / max(t_if_t, 1e-6)
    speedup_total = (t_stft_np + t_if_np) / max(t_stft_t + t_if_t, 1e-6)
    print(f"\n  Speed:")
    print(f"    STFT:    numpy {t_stft_np*1000:.1f} ms -> torch {t_stft_t*1000:.2f} ms "
          f"({speedup_stft:.0f}x)")
    print(f"    IF est:  numpy {t_if_np*1000:.1f} ms -> torch {t_if_t*1000:.2f} ms "
          f"({speedup_if:.0f}x)")
    print(f"    Total:   {speedup_total:.0f}x speedup for STFT+IF")

    # ── 5. Plot ──
    print("\n[4] Generating comparison plots...")
    plot_comparison(x, stft_np_abs, stft_t_abs, stft_diff,
                    o1_np, o1_t, of_np, of_t,
                    freqs_np, freqs_t, neta_min, tcol_min, valid_f,
                    msst_np, N, t_axis)
    print(f"  Saved to {SAVE_DIR}/")

    return dict(
        stft_corr=np.corrcoef(stft_np_abs.ravel(), stft_t_abs.ravel())[0,1],
        omega1_match=o1_match if valid_mask.sum() > 0 else 0,
        omega_f_match=of_match if valid_f.sum() > 0 else 0,
        omega_f_mean_diff=of_diff.mean() if valid_f.sum() > 0 else 0,
        speedup_stft=speedup_stft,
        speedup_if=speedup_if,
        speedup_total=speedup_total,
    )


# ═══════════════════════════════════════════════════════════════
# 5. Plotting
# ═══════════════════════════════════════════════════════════════

def plot_comparison(x, stft_np, stft_t, stft_diff,
                    o1_np, o1_t, of_np, of_t,
                    freqs_np, freqs_t, neta_min, tcol_min, valid_f,
                    msst_np, N, t_axis):
    FMAX = 200

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Mask frequencies
    mask_np = freqs_np[:neta_min] <= FMAX
    mask_t = freqs_t[:neta_min] <= FMAX

    # Time axis for torch (may be different due to padding)
    t_np = t_axis[:tcol_min]

    # Panel 1: Numpy STFT
    ax = axes[0, 0]
    ax.pcolormesh(t_np, freqs_np[:neta_min][mask_np], stft_np[mask_np, :],
                  shading='gouraud', cmap='jet', vmax=stft_np.max()*0.5)
    ax.set_ylim(0, FMAX)
    ax.set_title('Numpy Manual STFT', fontweight='bold')
    ax.set_ylabel('Frequency (Hz)')

    # Panel 2: Torch STFT
    ax = axes[0, 1]
    ax.pcolormesh(t_np, freqs_t[:neta_min][mask_t], stft_t[mask_t, :],
                  shading='gouraud', cmap='jet', vmax=stft_np.max()*0.5)
    ax.set_ylim(0, FMAX)
    ax.set_title('Torch STFT', fontweight='bold')

    # Panel 3: |diff|
    ax = axes[0, 2]
    im = ax.pcolormesh(t_np, freqs_np[:neta_min][mask_np],
                       stft_diff[mask_np, :],
                       shading='gouraud', cmap='hot', vmax=stft_np.max()*0.05)
    ax.set_ylim(0, FMAX)
    ax.set_title('|Numpy - Torch| STFT Difference', fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Panel 4: Omega_1 comparison (scatter of valid bins)
    ax = axes[1, 0]
    sample_cols = np.linspace(0, tcol_min-1, min(8, tcol_min), dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_cols)))
    for i, col in enumerate(sample_cols):
        valid_col = (o1_np[:, col] >= 1) & (o1_t[:, col] >= 1)
        if valid_col.sum() > 2:
            ax.scatter(o1_np[valid_col, col], o1_t[valid_col, col],
                      s=3, alpha=0.6, color=colors[i], label=f't={col/FS:.2f}s')
    lims = [0, max(o1_np.max(), o1_t.max())]
    ax.plot(lims, lims, 'k--', linewidth=0.8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Numpy Omega_1 (bin)')
    ax.set_ylabel('Torch Omega_1 (bin)')
    ax.set_title(f'1st-order IF: {len(sample_cols)} time slices', fontweight='bold')
    ax.legend(fontsize=6, loc='lower right')

    # Panel 5: Omega_final comparison
    ax = axes[1, 1]
    for i, col in enumerate(sample_cols):
        valid_col = (of_np[:, col] >= 1) & (of_t[:, col] >= 1)
        if valid_col.sum() > 2:
            ax.scatter(of_np[valid_col, col], of_t[valid_col, col],
                      s=3, alpha=0.6, color=colors[i], label=f't={col/FS:.2f}s')
    lims_f = [0, max(of_np.max(), of_t.max())]
    ax.plot(lims_f, lims_f, 'k--', linewidth=0.8)
    ax.set_xlim(lims_f); ax.set_ylim(lims_f)
    ax.set_xlabel('Numpy Omega_final (bin)')
    ax.set_ylabel('Torch Omega_final (bin)')
    ax.set_title('Final IF (3 refinements)', fontweight='bold')
    ax.legend(fontsize=6, loc='lower right')

    # Panel 6: IF diff histogram
    ax = axes[1, 2]
    if valid_f.sum() > 0:
        of_diff = np.abs(of_np[valid_f].astype(float) - of_t[valid_f].astype(float))
        ax.hist(of_diff, bins=range(0, 21), edgecolor='white', alpha=0.8)
        ax.axvline(x=of_diff.mean(), color='red', linestyle='--',
                   label=f'Mean = {of_diff.mean():.2f} bin')
        ax.set_xlabel('|Omega_final diff| (bin)')
        ax.set_ylabel('Count')
        ax.set_title(f'IF Final Diff Distribution\n'
                     f'Exact match: {(of_diff==0).mean()*100:.1f}%',
                     fontweight='bold')
        ax.legend()

    plt.suptitle('Numpy Manual STFT vs Torch STFT — MSST IF Estimation Comparison',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'stft_torch_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("[WARN] CUDA not available — using CPU (speedup will be limited)")

    # Test with T=500 (faster) and T=1000 (our standard)
    for T_test in [500, 1000]:
        print(f"\n{'='*70}")
        print(f"Testing T={T_test}")
        print(f"{'='*70}")
        results = compare_all(T=T_test, device=device)

    print("\n" + "=" * 70)
    print("DONE — check hmst_figures/torch_stft_compare/")
    print("=" * 70)
