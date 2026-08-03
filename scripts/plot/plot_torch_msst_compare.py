"""
对比 numpy MSST (CPU, models/tfr.py) vs torch MSST (GPU, models/msst_torch.py)
=============================================================================
验证 torch 重写是否匹配 numpy (算法正确性) + 速度对比 (GPU 提速).

两个版本算法一致 (modulated STFT + 相位差分 IF + 迭代精化 + 硬挤压):
  - numpy: 纯 numpy, CPU, per-frame 循环
  - torch: 纯 torch, GPU, 向量化 (torch.fft + scatter)

输出:
  - numpy_vs_torch_msst.png: 6 面板时频图 (STFT/MSST/差异)
  - 控制台: STFT 数值差异 + 速度对比

用法:
  python scripts/plot/plot_torch_msst_compare.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.tfr import msst as msst_numpy
from models.msst_torch import msst_torch


def make_signal(fs: int = 1000, N: int = 2000, seed: int = 0) -> np.ndarray:
    """合成测试信号: BPF(48Hz, FM) + 2xBPF(96Hz) + LOW_FREQ(12Hz, AM)."""
    rng = np.random.RandomState(seed)
    t = np.arange(N) / fs
    sig = (np.sin(2 * np.pi * 48 * t + 0.15 * np.sin(2 * np.pi * 3 * t)) +
           0.6 * np.sin(2 * np.pi * 96 * t) +
           0.25 * np.sin(2 * np.pi * 12 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t)) +
           0.05 * rng.randn(N))
    return sig.astype(np.float64)


def main():
    fs, N, num = 1000, 2000, 4
    sig = make_signal(fs, N)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Signal: N={N}, fs={fs}, num={num}")
    print(f"Device (torch): {device}")
    print("=" * 60)

    # ── numpy MSST (CPU) ──
    print("\n[1] numpy MSST (CPU, per-frame loop)...")
    t0 = time.perf_counter()
    r_np = msst_numpy(sig, fs, num=num, save_trajectory=True)
    t_np = time.perf_counter() - t0
    print(f"    time: {t_np:.3f}s")

    # ── torch MSST (GPU) ──
    print("\n[2] torch MSST (GPU, vectorized)...")
    x = torch.from_numpy(sig).to(device)
    # warmup (首次 FFT/cuFFT 初始化开销)
    _ = msst_torch(x, fs, num=num)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    r_torch = msst_torch(x, fs, num=num, save_trajectory=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_torch = time.perf_counter() - t0
    print(f"    time: {t_torch:.3f}s")
    print(f"\n    Speedup: {t_np / t_torch:.1f}x  (numpy {t_np:.3f}s -> torch {t_torch:.3f}s)")

    # ── 提取 ──
    stft_np = np.abs(r_np['STFT'])                  # [F, T]
    stft_torch = r_torch['STFT'].abs().cpu().numpy()
    msst_np = r_np['MSST']                          # [F, T]
    msst_torch_v = r_torch['MSST'].cpu().numpy()
    freqs = r_np['freqs']
    t_axis = r_np['t']

    # ── 数值差异 (算法正确性) ──
    stft_diff = np.abs(stft_np - stft_torch)
    stft_rel = stft_diff.max() / (stft_np.max() + 1e-12)
    print("\n" + "=" * 60)
    print("STFT magnitude diff (numpy vs torch):")
    print(f"  max abs diff:  {stft_diff.max():.2e}")
    print(f"  mean abs diff: {stft_diff.mean():.2e}")
    print(f"  relative:      {stft_rel:.2e}  "
          f"{'-> 匹配 ✓ (数值精度, float64 vs complex64)' if stft_rel < 1e-4 else '-> 差异偏大, 需检查'}")

    msst_diff = np.abs(msst_np - msst_torch_v)
    msst_rel = msst_diff.max() / (msst_np.max() + 1e-12)
    print(f"\nMSST magnitude diff (numpy vs torch):")
    print(f"  max abs diff: {msst_diff.max():.2e}, relative: {msst_rel:.2e}")

    # omega 一致性
    omega_np = r_np['omega_final']
    omega_torch = r_torch['omega_final'].cpu().numpy()
    omega_mismatch = (omega_np != omega_torch).sum()
    print(f"\nomega_final mismatch: {omega_mismatch}/{omega_np.size} "
          f"({100*omega_mismatch/omega_np.size:.2f}%)")

    # ── 画图 (6 面板) ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    freq_max = 200
    f_mask = freqs <= freq_max

    def _plot(ax, data, title, vmax=None):
        db = 10 * np.log10(data + 1e-12)
        im = ax.pcolormesh(t_axis, freqs[f_mask], db[f_mask], shading='gouraud',
                           cmap='jet', vmin=-40, vmax=10 if vmax is None else vmax)
        ax.set_ylim(0, freq_max)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='dB')

    _plot(axes[0, 0], stft_np, f'(a) numpy STFT (CPU, {t_np:.2f}s)')
    _plot(axes[0, 1], stft_torch, f'(b) torch STFT (GPU, {t_torch:.2f}s)')
    _plot(axes[0, 2], stft_diff, '(c) |numpy - torch| STFT', vmax=stft_diff.max())

    _plot(axes[1, 0], msst_np, '(d) numpy MSST')
    _plot(axes[1, 1], msst_torch_v, '(e) torch MSST')
    _plot(axes[1, 2], msst_diff, '(f) |numpy - torch| MSST', vmax=msst_diff.max())

    plt.suptitle(
        f'numpy MSST (CPU) vs torch MSST (GPU)   N={N}, fs={fs}, num={num}\n'
        f'STFT max diff: {stft_diff.max():.2e} (rel {stft_rel:.2e})  |  '
        f'Speed: {t_np:.2f}s -> {t_torch:.2f}s ({t_np / t_torch:.1f}x)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = Path(__file__).parent.parent.parent / 'numpy_vs_torch_msst.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
