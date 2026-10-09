"""
"挤压权衡" 示意图 (支撑 Word 文档 第 5 节的核心论点)
====================================================

论点: Rényi 熵并非越小越好. 同步压缩类方法把能量向脊线聚集, 必然降低 Rényi
      熵; 但"聚集"与"失真"是同一枚硬币的两面:
  (1) 低频 (涡带) 分量在物理上本来就是 10-15 Hz 量级的**宽带**脉动.
      硬挤压把它压成一根细线 -> 丢掉了真实的带宽信息, 低频特征在图上"变小/变小,
      不利于观察;
  (2) 真正宽带的成分 (水力噪声带) 被强行压成若干伪谱线 -> 制造出原始信号中
      不存在的"锐利谱线", 这是失真.

两个指标都必须是**尺度无关**的 (否则会被各方法的绝对幅度标定污染):
  - 带内有效带宽  eff_bw = exp(-Σ p·ln p) · Δf   (Hz)   —— 越大 = 越保留带宽
  - 带内峰值集中度 p_peak = 峰值 bin 能量 / 带内总能量  —— 越大 = 越"线状化"

本脚本生成两张图 (默认 results/squeeze_tradeoff/):
  lowfreq_loss.png         真实数据: STFT / MSST4(硬挤压) / SAST, 下排 0-40 Hz 放大
  broadband_distortion.png 合成信号: 50 Hz 真实谱线 + 120-180 Hz 宽带噪声

用法:
  python scripts/plot/plot_squeeze_tradeoff.py \
      --checkpoint sast_checkpoints/sast_v3_model.pt --device cuda --class 1
  python scripts/plot/plot_squeeze_tradeoff.py --scan          # 扫全部类, 只打印指标
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tfr_common import (
    FS, HLENGTH, MSST_NUM, CMAP, load_samples, hard_squeeze_mag, renyi_entropy,
    apply_plot_style,
)
from models.tfr import msst

apply_plot_style()

LOW_BAND = (0.5, 25.0)      # 涡带/低频关注带 (Hz)
WIDE_BAND = (120.0, 180.0)  # 合成信号中的宽带噪声带 (Hz)
FMAX = 500.0


# ═══════════════════════════════════════════════════════════════
# 计算
# ═══════════════════════════════════════════════════════════════

def load_sast(checkpoint_path, device):
    import torch
    from sast_model import load_checkpoint as _lc
    model, _, _ = _lc(checkpoint_path, torch.device(device))

    def infer(x):
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            res = model(xt, return_all=True, training=False)
        return dict(mag=res['tfr_enhanced'][0].cpu().numpy(),
                    freqs=np.asarray(res['freqs'].cpu().numpy()).ravel(),
                    t=np.asarray(res['t_axis'].cpu().numpy()).ravel())

    return infer


def band_stats(mag, freqs, band):
    """尺度无关的带内统计: (有效带宽 Hz, 峰值集中度, 带内 Rényi 熵).

    先在带内对时间取平均功率谱 (抑制单帧噪声起伏), 再算分布的熵.
    """
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    mag = np.asarray(mag, dtype=np.float64)
    if freqs[0] > freqs[-1]:
        freqs, mag = freqs[::-1], mag[::-1, :]

    b = (freqs >= band[0]) & (freqs <= band[1])
    idx = np.where(b)[0]
    if idx.size < 2:
        return float('nan'), float('nan'), float('nan')

    df = float(np.median(np.diff(freqs[idx])))
    power = mag[idx, :] ** 2
    spec = power.mean(axis=1)
    p = spec / (spec.sum() + 1e-12)

    eff_bw = float(np.exp(-np.sum(p * np.log(p + 1e-12))) * df)
    p_peak = float(p.max())
    renyi = float(-np.log(np.sum(p ** 3)) / 2.0)   # α=3 的 Rényi 熵
    return eff_bw, p_peak, renyi


def method_panels(x, sast_infer):
    """返回 [(label, mag, freqs, t), ...] : STFT / MSST4 / SAST."""
    r = msst(x, FS, hlength=HLENGTH, num=MSST_NUM, save_trajectory=True)
    stft = r['STFT']
    msst4 = hard_squeeze_mag(stft, r['omega_final'], stft.shape[0])
    out = [
        ('STFT (无挤压)', np.abs(stft), r['freqs'], r['t']),
        ('MSST num=4 (硬挤压)', msst4, r['freqs'], r['t']),
    ]
    if sast_infer is not None:
        s = sast_infer(x)
        out.append(('SAST (自适应挤压)', s['mag'], s['freqs'], s['t']))
    return out


def report(panels, band, title):
    print(f'\n{title}  (带 = {band[0]}-{band[1]} Hz)')
    print(f"{'method':<26s} {'Rényi(全带)':>12s} {'带内有效带宽':>14s} "
          f"{'峰值集中度':>12s}")
    print('-' * 68)
    rows = []
    for label, mag, freqs, _ in panels:
        freqs = np.asarray(freqs, dtype=np.float64).ravel()
        m = freqs <= FMAX
        r = renyi_entropy(mag[m, :])
        eb, pk, _ = band_stats(mag, freqs, band)
        rows.append((label, r, eb, pk))
        print(f'{label:<26s} {r:12.2f} {eb:11.2f} Hz {pk * 100:10.1f}%')
    return rows


# ═══════════════════════════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════════════════════════

def _panel(ax, mag, freqs, t, title, fmax, vmax):
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    mag = np.asarray(mag, dtype=np.float64)
    if freqs[0] > freqs[-1]:
        freqs, mag = freqs[::-1], mag[::-1, :]
    m = freqs <= fmax
    ax.pcolormesh(t, freqs[m], mag[m, :], shading='gouraud',
                  cmap=CMAP, vmax=vmax)
    ax.set_ylim(0, fmax)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Freq [Hz]')
    ax.set_title(title, fontsize=8, fontweight='bold')


def plot_lowfreq(panels, band, cls, idx, save_path, fmax=FMAX):
    m500 = np.asarray(panels[0][2]).ravel() <= fmax
    vmax_full = max(p[1][m500, :].max() for p in panels) * 0.5

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 8.6))
    axes = np.atleast_2d(axes)

    for j, (label, mag, freqs, t) in enumerate(panels):
        freqs = np.asarray(freqs).ravel()
        r = renyi_entropy(mag[freqs <= fmax, :])
        eb, pk, _ = band_stats(mag, freqs, band)
        _panel(axes[0, j], mag, freqs, t,
               f'{label}\n全带 Rényi={r:.2f}', fmax, vmax_full)

        zmax = band[1] * 1.6
        vmax_zoom = mag[freqs <= zmax, :].max() * 0.5
        _panel(axes[1, j], mag, freqs, t,
               f'{label} — 0-{zmax:.0f} Hz 放大\n'
               f'带内有效带宽={eb:.1f} Hz   峰值集中度={pk * 100:.0f}%',
               zmax, vmax_zoom)
        axes[1, j].axhline(band[1], color='white', ls='--', lw=0.9, alpha=0.85)

    fig.suptitle('挤压权衡 (真实数据): 硬挤压把低频涡带压成一根细线, '
                 '丢失真实的 10-15 Hz 带宽\n'
                 f'class {cls}, sample idx={idx}', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] {save_path}')


def make_broadband_signal(seed=0):
    """窄带谱线 (50 Hz) + 宽带水力噪声带 (120-180 Hz)."""
    rng = np.random.default_rng(seed)
    N = int(2.0 * FS)
    t = np.arange(N) / FS

    tone = 1.0 * np.sin(2 * np.pi * 50.0 * t)

    noise = rng.standard_normal(N)
    f = np.fft.rfftfreq(N, 1.0 / FS)
    H = ((f >= WIDE_BAND[0]) & (f <= WIDE_BAND[1])).astype(float)
    wide = np.fft.irfft(np.fft.rfft(noise) * H, n=N)
    wide = 2.0 * wide / (np.abs(wide).max() + 1e-12)
    return tone + wide


def plot_broadband(panels, save_path, fmax=FMAX):
    m500 = np.asarray(panels[0][2]).ravel() <= fmax
    vmax = max(p[1][m500, :].max() for p in panels) * 0.5

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8))
    axes = np.atleast_1d(axes)
    for ax, (label, mag, freqs, t) in zip(axes, panels):
        freqs = np.asarray(freqs).ravel()
        r = renyi_entropy(mag[freqs <= fmax, :])
        eb, pk, _ = band_stats(mag, freqs, WIDE_BAND)
        _panel(ax, mag, freqs, t,
               f'{label}\n全带 Rényi={r:.2f}   |   '
               f'{WIDE_BAND[0]:.0f}-{WIDE_BAND[1]:.0f} Hz 有效带宽={eb:.1f} Hz',
               fmax, vmax)
        ax.axhline(WIDE_BAND[0], color='white', ls='--', lw=0.9, alpha=0.85)
        ax.axhline(WIDE_BAND[1], color='white', ls='--', lw=0.9, alpha=0.85)

    fig.suptitle('宽带失真 (合成信号): 硬挤压把 120-180 Hz 宽带噪声压成若干伪谱线, '
                 'Rényi 熵同时被压低\n'
                 '(50 Hz 真实谱线 + 120-180 Hz 宽带噪声)', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] {save_path}')


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='挤压权衡示意图')
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--class', dest='cls', type=int, default=1)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--outdir', default='results/squeeze_tradeoff')
    ap.add_argument('--scan', action='store_true',
                    help='扫全部 5 类, 只打印指标 (不画图)')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sast_infer = load_sast(args.checkpoint, args.device) if args.checkpoint else None

    # ── 扫描模式: 找最能说明问题的类 ──
    if args.scan:
        xs, ys, idxs, classes = load_samples(args.data, per_class=1)
        for c in classes:
            k = [i for i, cc in enumerate(ys) if cc == c][0]
            panels = method_panels(xs[k], sast_infer)
            print(f'\n{"=" * 70}\nclass {c}  (idx={idxs[k]})')
            report(panels, LOW_BAND, f'低频带')
        return

    # ── 真实数据 ──
    xs, ys, idxs, _ = load_samples(args.data, per_class=1)
    k = [i for i, c in enumerate(ys) if c == args.cls][0]
    print(f'真实样本: class {args.cls}, idx={idxs[k]}')
    panels = method_panels(xs[k], sast_infer)
    report(panels, LOW_BAND, '低频带 (真实数据)')
    plot_lowfreq(panels, LOW_BAND, args.cls, idxs[k],
                 outdir / 'lowfreq_loss.png')

    # ── 合成宽带 ──
    x_syn = make_broadband_signal()
    panels_s = method_panels(x_syn, sast_infer)
    report(panels_s, WIDE_BAND, '宽带 (合成信号)')
    plot_broadband(panels_s, outdir / 'broadband_distortion.png')

    print(f'\n输出目录: {outdir}')


if __name__ == '__main__':
    main()