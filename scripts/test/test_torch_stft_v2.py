"""
Torch STFT + Squeeze vs Numpy: 速度与时频图对比
================================================
T=2000 (真实信号长度), 仅对比 STFT 和 挤压两步。
IF 估计共用 numpy MSST 的 omega_final (控制变量)。

用法: python test_torch_stft_v2.py
"""

import numpy as np
import torch
import time
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.tfr import msst, compute_renyi

FS = 1000
SAVE_DIR = Path('hmst_figures/torch_stft_compare')
SAVE_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.family'] = 'sans-serif'


# ═══════════════════════════════════════════════════════════════
# 1. Numpy STFT (手动, 与 msst() Step 1 完全一致)
# ═══════════════════════════════════════════════════════════════

def stft_numpy(x, hlength=None):
    """手动 modulated-form STFT (models/tfr.py msst() 的 Step 1)."""
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
    elapsed = time.perf_counter() - t0

    return tfr, elapsed


# ═══════════════════════════════════════════════════════════════
# 2. Torch STFT (GPU, torch.stft)
# ═══════════════════════════════════════════════════════════════

def stft_torch(x_np, hlength=None, device='cuda'):
    """torch.stft GPU STFT."""
    x_np = np.asarray(x_np, dtype=np.float64).ravel()
    N = len(x_np)

    if hlength is None:
        hlength = min(N, 512)
    hlength = hlength + 1 - (hlength % 2)

    n_fft = max(N, hlength)

    t_win = torch.linspace(-0.5, 0.5, hlength, device=device, dtype=torch.float32)
    window = torch.exp(-np.pi / 0.32**2 * t_win**2).float()

    x_t = torch.from_numpy(x_np).float().to(device).unsqueeze(0)  # [1, T]

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    X = torch.stft(x_t, n_fft=n_fft, hop_length=1, win_length=hlength,
                   window=window, center=True, pad_mode='reflect',
                   normalized=False, onesided=True, return_complex=True)

    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return X.squeeze(0).cpu().numpy(), elapsed


# ═══════════════════════════════════════════════════════════════
# 3. Numpy Squeeze (硬挤压, 与 msst() Step 4 完全一致)
# ═══════════════════════════════════════════════════════════════

def squeeze_numpy(tfr, omega, N_sig):
    """硬最近邻挤压 (numpy)."""
    neta, tcol = omega.shape
    t0 = time.perf_counter()
    Ts = np.zeros((neta, tcol), dtype=np.complex128)
    threshold = 0.0001
    for b in range(tcol):
        for eta in range(neta):
            if np.abs(tfr[eta, b]) > threshold:
                k = omega[eta, b]
                if 1 <= k <= neta:
                    Ts[k - 1, b] += tfr[eta, b]
    Ts = Ts / (N_sig / 2.0)
    elapsed = time.perf_counter() - t0
    return Ts, elapsed


# ═══════════════════════════════════════════════════════════════
# 4. Torch Squeeze (GPU scatter_add, 同算法)
# ═══════════════════════════════════════════════════════════════

def squeeze_torch(tfr_np, omega_np, N_sig, device='cuda'):
    """硬最近邻挤压 (torch GPU scatter_add)."""
    neta, tcol = omega_np.shape
    tfr_np_c64 = tfr_np.astype(np.complex64)
    tfr_t = torch.from_numpy(tfr_np_c64).to(device=device)
    omega_t = torch.from_numpy(omega_np.copy()).to(device=device, dtype=torch.long)

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # mask: valid IF + above threshold
    mag_t = tfr_t.abs()
    valid = (omega_t >= 1) & (omega_t <= neta) & (mag_t > 1e-4)

    f_idx, t_idx = torch.where(valid)  # [F, T] -> (row_idx, col_idx)
    k_idx = omega_t[f_idx, t_idx] - 1  # 1-indexed -> 0-indexed

    Tx = torch.zeros(neta, tcol, dtype=torch.cfloat, device=device)
    Tx.index_put_((k_idx, t_idx), tfr_t[f_idx, t_idx], accumulate=True)
    Tx = Tx / (N_sig / 2.0)

    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return Tx.cpu().numpy(), elapsed


# ═══════════════════════════════════════════════════════════════
# 5. Torch IF estimation (phase diff, matching numpy's Step 2)
# ═══════════════════════════════════════════════════════════════

def unwrap_torch(phase, discont=np.pi):
    d = torch.diff(phase, dim=-1)
    dd = torch.where(d > discont, d - 2*np.pi, d)
    dd = torch.where(dd < -discont, dd + 2*np.pi, dd)
    return torch.cat([phase[..., :1], phase[..., :1] + torch.cumsum(dd, dim=-1)], dim=-1)


def if_estimation_torch(tfr_np, N_sig, device='cuda'):
    """相位差分 IF 估计 (torch GPU)."""
    tfr_t = torch.from_numpy(tfr_np.astype(np.complex64)).to(device=device)
    neta, tcol = tfr_t.shape

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    phase = torch.angle(tfr_t)
    phase_uw = unwrap_torch(phase)
    d_phase = torch.diff(phase_uw, dim=-1)
    omega = d_phase * N_sig / (2.0 * np.pi)
    omega = torch.round(omega).to(torch.int32)
    last_col = omega[:, -1:].clone()
    omega = torch.cat([omega, last_col], dim=-1)

    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return omega.cpu().numpy(), elapsed


# ═══════════════════════════════════════════════════════════════
# 6. Main comparison
# ═══════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T_SIG = 2000  # 真实信号长度

    print(f"STFT + Squeeze: Numpy vs Torch GPU")
    print(f"T={T_SIG}, fs={FS} Hz, device={device}")
    print("=" * 60)

    # ── Test signal ──
    t_axis = np.arange(T_SIG) / FS
    x = (np.sin(2*np.pi*48*t_axis + 0.15*np.sin(2*np.pi*3*t_axis)) +
         0.6*np.sin(2*np.pi*96*t_axis) +
         0.25*np.sin(2*np.pi*12*t_axis))
    x = x.astype(np.float64)
    N_sig = len(x)
    print(f"Signal: {T_SIG} samples, {T_SIG/FS:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # Numpy path: STFT(numpy) + IF(numpy) + Squeeze(numpy)
    # ═══════════════════════════════════════════════════════════
    print("\n── Numpy Path ──")

    # STFT
    tfr_np, t_stft_np = stft_numpy(x)
    F_np, T_np = tfr_np.shape
    print(f"  STFT:    {t_stft_np*1000:.1f} ms, [{F_np}, {T_np}]")

    # IF (numpy — extract from full msst to get exact omega)
    r_np = msst(x, FS, num=1, save_trajectory=True)
    omega_np = r_np['omegas'][0][:F_np, :T_np]  # [F, T] int32, 1-indexed

    # Squeeze
    Ts_np, t_sqz_np = squeeze_numpy(tfr_np, omega_np, N_sig)
    re_np = compute_renyi(np.abs(Ts_np))
    t_total_np = t_stft_np + t_sqz_np
    print(f"  Squeeze: {t_sqz_np*1000:.1f} ms, Renyi={re_np:.2f}")
    print(f"  Total:   {t_total_np*1000:.1f} ms")

    # ═══════════════════════════════════════════════════════════
    # Torch path: STFT(torch) + IF(torch) + Squeeze(torch)
    # ═══════════════════════════════════════════════════════════
    print("\n── Torch Path ──")

    # STFT
    tfr_t, t_stft_t = stft_torch(x, device=device)
    F_t, T_t = tfr_t.shape
    print(f"  STFT:    {t_stft_t*1000:.2f} ms, [{F_t}, {T_t}]")

    # IF (torch GPU)
    omega_t, t_if_t = if_estimation_torch(tfr_t, N_sig, device=device)
    print(f"  IF est:  {t_if_t*1000:.2f} ms")

    # Squeeze (torch GPU)
    Ts_t, t_sqz_t = squeeze_torch(tfr_t, omega_t, N_sig, device=device)
    re_t = compute_renyi(np.abs(Ts_t))
    t_total_t = t_stft_t + t_if_t + t_sqz_t
    print(f"  Squeeze: {t_sqz_t*1000:.2f} ms, Renyi={re_t:.2f}")
    print(f"  Total:   {t_total_t*1000:.2f} ms")

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  {'Step':<15s} {'Numpy (ms)':>12s} {'Torch (ms)':>12s} {'Speedup':>10s}")
    print(f"  {'-'*49}")
    print(f"  {'STFT':<15s} {t_stft_np*1000:>12.1f} {t_stft_t*1000:>12.2f} "
          f"{t_stft_np/max(t_stft_t,1e-6):>9.0f}x")
    print(f"  {'IF est':<15s} {'(in STFT)':>12s} {t_if_t*1000:>12.2f} {'-':>10s}")
    print(f"  {'Squeeze':<15s} {t_sqz_np*1000:>12.1f} {t_sqz_t*1000:>12.2f} "
          f"{t_sqz_np/max(t_sqz_t,1e-6):>9.0f}x")
    print(f"  {'STFT+IF+Sqz':<15s} {t_total_np*1000:>12.1f} {t_total_t*1000:>12.2f} "
          f"{t_total_np/max(t_total_t,1e-6):>9.0f}x")

    # ═══════════════════════════════════════════════════════════
    # TFR comparison
    # ═══════════════════════════════════════════════════════════
    # Align to common F/T for comparison
    F_min = min(F_np, F_t)
    T_min = min(T_np, T_t)

    ts_np_abs = np.abs(Ts_np[:F_min, :T_min])
    ts_t_abs = np.abs(Ts_t[:F_min, :T_min])

    stft_np_abs = np.abs(tfr_np[:F_min, :T_min])
    stft_t_abs = np.abs(tfr_t[:F_min, :T_min])
    stft_corr = np.corrcoef(stft_np_abs.ravel(), stft_t_abs.ravel())[0, 1]
    tfr_corr = np.corrcoef(ts_np_abs.ravel(), ts_t_abs.ravel())[0, 1]

    print(f"\n  STFT magnitude corr:   {stft_corr:.6f}")
    print(f"  Squeezed TFR corr:     {tfr_corr:.6f}")
    print(f"  Renyi (numpy):         {re_np:.2f}")
    print(f"  Renyi (torch):         {re_t:.2f}")

    # ── Plot (use aligned shapes) ──
    print("\n[Plot] Generating...")
    plot_comparison(tfr_np[:F_min, :T_min], tfr_t[:F_min, :T_min],
                    Ts_np[:F_min, :T_min], Ts_t[:F_min, :T_min],
                    r_np['freqs'][:F_min], T_SIG, FS, F_min, T_min)
    print(f"  Saved to {SAVE_DIR}/")


def plot_comparison(stft_np, stft_t, sqz_np, sqz_t, freqs, T_sig, FS, F_min, T_min):
    """2x3 对比图: STFT(2) + Sqz TFR(2) + diff(2)."""
    FMAX = 200
    t_axis = np.arange(T_min) / FS
    mask = freqs[:F_min] <= FMAX

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    vmax_stft = max(np.abs(stft_np).max(), np.abs(stft_t).max()) * 0.5
    vmax_sqz = max(np.abs(sqz_np).max(), np.abs(sqz_t).max()) * 0.5

    stft_np_abs = np.abs(stft_np)
    stft_t_abs = np.abs(stft_t)
    sqz_np_abs = np.abs(sqz_np)
    sqz_t_abs = np.abs(sqz_t)

    panels = [
        (axes[0, 0], 'Numpy STFT (manual)', stft_np_abs, vmax_stft),
        (axes[0, 1], 'Torch STFT (torch.stft)', stft_t_abs, vmax_stft),
        (axes[0, 2], '|STFT diff|', np.abs(stft_np_abs - stft_t_abs),
         vmax_stft * 0.1),
        (axes[1, 0], 'Numpy Squeeze TFR', sqz_np_abs, vmax_sqz),
        (axes[1, 1], 'Torch Squeeze TFR', sqz_t_abs, vmax_sqz),
        (axes[1, 2], '|Squeezed diff|', np.abs(sqz_np_abs - sqz_t_abs),
         vmax_sqz * 0.1),
    ]

    for ax, title, data, vm in panels:
        cmap = 'hot' if 'diff' in title.lower() else 'jet'
        im = ax.pcolormesh(t_axis, freqs[:F_min][mask], data[mask, :],
                          shading='gouraud', cmap=cmap, vmax=vm)
        ax.set_ylim(0, FMAX)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        if 'diff' in title.lower():
            plt.colorbar(im, ax=ax, fraction=0.046)

    # 加 Rényi 标注
    re_np = compute_renyi(sqz_np_abs)
    re_t = compute_renyi(sqz_t_abs)
    axes[1,0].text(0.02, 0.98, f'R={re_np:.2f}', transform=axes[1,0].transAxes,
                   fontsize=11, fontweight='bold', va='top', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    axes[1,1].text(0.02, 0.98, f'R={re_t:.2f}', transform=axes[1,1].transAxes,
                   fontsize=11, fontweight='bold', va='top', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

    plt.suptitle(f'STFT + Squeeze: Numpy (manual) vs Torch (GPU) — T={T_sig} ({T_sig/FS:.1f}s)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'stft_squeeze_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
