"""
谐波梳齿细查 (回答"100 Hz 是不是 2×BPF"与"50 Hz 谱线在哪")
=============================================================

背景矛盾:
  普查发现 100.1 / 149.9 / 200.2 Hz 三条线都精确落在 Z_r·f_rot 的
  整数倍上 (2×,3×,4× = 99.99/149.99/199.98), 偏差 <=0.22 Hz;
  但 42-55 Hz 带内检测到的峰却在 45.9~49.8 Hz, 而不是理论的 50.00 Hz.

  两种可能:
    (a) 50 Hz 谱线确实存在, 只是被更强的 46~48 Hz 分量盖过 / 被峰检测
        的 5 Hz 最小间距规则合并掉了;
    (b) 机器在带负荷工况下并非严格电网同步 (333.3 rpm 是额定值而非实际值),
        实际转速约 9x47 = 423 rpm? (不合常理) 或 BPF 不是 50 Hz.

细查内容 (全部在 38-60 Hz, 24-30 Hz 两个窗口内):
  1. 类均值谱的局部极大 (无最小间距限制, 只要求 prominence)
  2. 逐样本: 每个样本在窗口内的最强峰的频率 -> 分布直方图
  3. 逐样本: 50.0 Hz 处的窄带能量 (dB) 与窗口内最强峰能量 (dB) 之差
     -> 若 50 Hz 是一条独立谱线, 这个差值应在一个很窄的范围内集中
  4. 空转工况单独检查 24-30 Hz 窗口 —— 用于**证伪** "空转 45.65% 转速 ->
     BPF=22.82 Hz" 假设: 该窗口实测为均匀噪声底 (能量占比仅 1.16%), 无独立峰;
     空转真实主峰在 45.90-46.39 Hz, 属工况漂移分量 (详见 docs §14.4)。

输出 (results/band_census/):
  comb_probe.csv       每类: 窗口内局部极大 (f, level)
  bp_line_check.csv    每类: 50Hz 能量 / 窗口最强 / 最强峰频率中位数
  comb_probe.png       38-60 Hz 放大谱 (5 类叠加) + 理论梳齿

用法:
  python scripts/analysis/probe_harmonic_comb.py --samples 300
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / 'scripts' / 'plot'))

from tfr_common import apply_plot_style       # noqa: E402

apply_plot_style()
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

FS = 1000.0
NPERSEG = 512
NOVERLAP = 384
NFFT = 4096              # 细网格 df = 0.244 Hz (真分辨率仍 1.953 Hz)
CLASS_NAMES = ['空转', '低负荷', '中负荷', '高负荷', '抽水']

N_RATED, Z_R, Z_G = 333.3, 9, 20
F_ROT = N_RATED / 60.0
# F_IDLE_ROT (0.4565·F_ROT) 已删除: 45.65% 转速假设被 20-30 Hz 窗口实测证伪
# (均匀噪声底, 无独立峰), 见文件头 4. 与 docs §14.4。

WINDOWS = [(38.0, 60.0), (20.0, 30.0)]
COMB = [('f_rot', F_ROT), ('2f_rot', 2 * F_ROT), ('BPF=9f_rot', Z_R * F_ROT),
        ('2×BPF', 2 * Z_R * F_ROT), ('3×BPF', 3 * Z_R * F_ROT),
        ('4×BPF', 4 * Z_R * F_ROT),
        # 注: 20-30 Hz 窗口保留用于证伪检查, 但不再画"空转BPF=22.82 Hz"参考线
        ]


def db(p, ref=None):
    ref = p.max() if ref is None else ref
    return 10.0 * np.log10(np.maximum(p, 1e-30) / max(ref, 1e-30))


def main():
    ap = argparse.ArgumentParser(description='谐波梳齿细查')
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--samples', type=int, default=300)
    ap.add_argument('--outdir', default='results/band_census')
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    X, y = d['train_X'], d['train_y'].ravel().astype(np.int64)
    if X.ndim == 3:
        X = X[:, :, 0]
    classes = sorted(np.unique(y).tolist())

    print(f'理论梳齿 (f_rot={F_ROT:.4f} Hz):')
    for nm, fq in COMB:
        print(f'  {nm:<10s} {fq:8.3f} Hz')

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    cmap = plt.get_cmap('tab10')
    comb_rows, line_rows = [], []

    for wi, (lo, hi) in enumerate(WINDOWS):
        ax = axes[wi]
        print(f'\n{"=" * 84}\n窗口 {lo:.0f}-{hi:.0f} Hz')
        print(f'{"class":<8s} {"窗口内局部极大 (f Hz / 相对级 dB)":<66s}')
        print('-' * 84)
        for ci, c in enumerate(classes):
            cidx = np.where(y == c)[0]
            if len(cidx) > args.samples:
                cidx = np.random.RandomState(42).choice(cidx, args.samples, replace=False)
            xs = X[cidx].astype(np.float64)
            acc = None
            for x in xs:
                _, p = welch(x, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT)
                acc = p if acc is None else acc + p
            acc /= len(xs)
            m = (acc <= 0) | ~np.isfinite(acc)
            acc = np.where(m, 1e-30, acc)
            f = np.fft.rfftfreq(NFFT, 1 / FS)
            sel = (f >= lo) & (f <= hi)
            fs_, aq = f[sel], acc[sel]
            pdb = db(aq)
            pk, prop = find_peaks(pdb, prominence=1.5)
            if len(pk) == 0:
                pk = np.array([int(np.argmax(pdb))])
                prop = {'prominences': np.array([pdb[pk[0]]])}
            order = np.argsort(pdb[pk])[::-1]
            tops = [(float(fs_[pk[i]]), float(pdb[pk[i]])) for i in order[:8]]
            for fq_, lv_ in tops:
                comb_rows.append((c, CLASS_NAMES[c], f'{lo:.0f}-{hi:.0f}',
                                  f'{fq_:.3f}', f'{lv_:.2f}'))
            print(f'{CLASS_NAMES[c]:<8s} '
                  + '  '.join(f'{q:.2f}Hz/{v:.1f}dB' for q, v in tops))

            ax.plot(fs_, pdb, lw=0.9, color=cmap(ci), label=CLASS_NAMES[c], alpha=0.85)

            # 50 Hz (或窗口中心理论线) 窄带能量 vs 窗口最强峰
            if wi == 0:
                tgt = Z_R * F_ROT
                m50 = (fs_ >= tgt - 1.0) & (fs_ <= tgt + 1.0)
                lv50 = float(np.max(pdb[m50])) if m50.any() else float('nan')
                imax = int(np.argmax(pdb))
                line_rows.append((c, CLASS_NAMES[c], '50.00', f'{lv50:.2f}',
                                  f'{pdb[imax]:.2f}', f'{fs_[imax]:.3f}',
                                  f'{lv50 - pdb[imax]:.2f}'))

        for nm, fq in COMB:
            if lo <= fq <= hi:
                ax.axvline(fq, color='#999999', lw=0.6, ls=':')
                ax.annotate(f'{nm}\n{fq:.2f}', (fq, ax.get_ylim()[1]),
                            fontsize=6, rotation=90, va='top', ha='right',
                            color='#666666')
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('相对级 (dB)')
        ax.set_ylim(-45, 3)
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(fontsize=8, ncol=5)
        ax.set_title(f'{lo:.0f}-{hi:.0f} Hz 放大 (5 类类均值谱, 细网格 nfft={NFFT})')

    fig.tight_layout()
    fig.savefig(outdir / 'comb_probe.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    with open(outdir / 'comb_probe.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['class', 'class_name', 'window_hz', 'peak_hz', 'rel_level_db'])
        w.writerows(comb_rows)

    print(f'\n{"=" * 84}\n50 Hz 谱线专项: 窄带级 vs 42-55Hz 窗口最强峰')
    print(f'{"class":<8s} {"50Hz处级":>9s} {"窗口最强":>9s} {"最强峰f":>9s} '
          f'{"差值":>8s}')
    print('-' * 52)
    for r in line_rows:
        print(f'{r[1]:<8s} {r[3]:>9s} {r[4]:>9s} {r[5]:>9s} {r[6]:>8s}')

    with open(outdir / 'bp_line_check.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['class', 'class_name', 'target_hz', 'level_at_target_db',
                    'window_peak_db', 'window_peak_hz', 'delta_db'])
        w.writerows(line_rows)

    print(f'\n输出目录: {outdir}')


if __name__ == '__main__':
    main()