"""
时频算法版本挑选脚本 (实际数据集, 每类抽样一个样本)
====================================================

在"实际数据集"上对 SST / MSST / WSST 等算法的不同版本做并排对比,
供肉眼确定「哪一个算法用哪一个版本」. 正式对比见
scripts/plot/compare_tfr_quality.py 与 compare_baselines.py.

绘图样式与 scripts/plot/plot_msst_tfr.py 完全一致 (tfr_common.plot_panel):
  - 线性幅度, vmax = 0.5·(0-500Hz 内峰值), jet 色标
  - 频率轴 0-500 Hz, 标主主导频率 + Rényi 熵

算法版本 (SST / WSST 一阶用 ssqueezepy 库):
  STFT 域:  stft | sst1(ssqueezepy) | sst2(Oberlin) | msst1..4(models.tfr)
  CWT  域:  wsst1(ssqueezepy) | wsst2(morse, models.tfr)

输出 (默认 tfr_version_probe/):
  - 每个 (类, 版本) 单独一张 PNG:  class{c}_v{nn}_{name}.png
  - 每个类一张汇总大图 (网格):      contact_class{c}.png
  - 数值汇总表:                     summary.csv

用法:
  python scripts/plot/plot_tfr_versions.py                  # 全部 5 类
  python scripts/plot/plot_tfr_versions.py --cls 0          # 只跑 class 0 (冒烟)
  python scripts/plot/plot_tfr_versions.py --fmax 500
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ssqueezepy as ssq

from tfr_common import (
    FS, HLENGTH, MSST_NUM, plot_panel, save_tfr_png,
    load_samples, hard_squeeze_mag, renyi_entropy, ridge_energy_ratio,
)
from models.tfr import wsst2, msst, sst2_stft

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 7
FMAX = 500.0        # 频率上限 (Hz)

CLASS_NAMES = {
    0: 'Class 0: No-load', 1: 'Class 1: Low load', 2: 'Class 2: Mid load',
    3: 'Class 3: High load', 4: 'Class 4: Pumping',
}


# ═══════════════════════════════════════════════════════════════
# 各版本 TFR 计算 (统一 canonical 参数; SST/WSST 一阶用 ssqueezepy)
# ═══════════════════════════════════════════════════════════════

def compute_all_versions(x):
    """计算单个样本的所有版本, 返回 {name: dict(mag=, freqs=, t=)}."""
    N = len(x)
    t = np.arange(N) / FS

    # STFT + MSST 家族 (一次 msst save_trajectory, 向量化挤压出 4 阶)
    r = msst(x, FS, hlength=HLENGTH, num=MSST_NUM, save_trajectory=True)
    stft = r['STFT']
    f_stft = r['freqs']
    omegas = r['omegas']   # 长度 = num = 4

    # 二阶 SST (Oberlin)
    r_sst2 = sst2_stft(x, FS)

    # ssqueezepy 一阶 SST (STFT 域) 与 一阶 WSST (CWT 域)
    Tx_sst, _, f_sst, _ = ssq.ssq_stft(x, fs=FS)
    Tx_wsst, _, f_wsst, _ = ssq.ssq_cwt(x, fs=FS)

    # 二阶 WSST (本文 morse)
    r_wsst2 = wsst2(x, FS, mywav='morse')

    versions = {
        'stft':  dict(mag=np.abs(stft), freqs=f_stft, t=t),
        'sst1':  dict(mag=np.abs(Tx_sst), freqs=np.asarray(f_sst).ravel(), t=t),
        'sst2':  dict(mag=np.abs(r_sst2['SST2']), freqs=r_sst2['freqs'], t=t),
        'wsst1': dict(mag=np.abs(Tx_wsst), freqs=np.asarray(f_wsst).ravel(), t=t),
        'wsst2': dict(mag=np.abs(r_wsst2['WSST2']), freqs=r_wsst2['freqs'], t=t),
    }
    for k, omega in enumerate(omegas):
        versions[f'msst{k + 1}'] = dict(
            mag=hard_squeeze_mag(stft, omega, stft.shape[0]), freqs=f_stft, t=t)
    return versions


VERSIONS = [
    ('stft',  'STFT (raw)'),
    ('sst1',  'SST1 (ssqueezepy)'),
    ('sst2',  'SST2 (Oberlin)'),
    ('msst1', 'MSST num=1'),
    ('msst2', 'MSST num=2'),
    ('msst3', 'MSST num=3'),
    ('msst4', 'MSST num=4'),
    ('wsst1', 'WSST1 (ssqueezepy)'),
    ('wsst2', 'WSST2 (morse)'),
]


def plot_contact_sheet(versions, cls, idx, save_path, fmax):
    """一个类的所有版本汇总成一张网格大图 (线性幅度样式)."""
    n = len(VERSIONS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, title) in zip(axes, VERSIONS):
        res = versions[name]
        mask = res['freqs'] <= fmax
        r = renyi_entropy(res['mag'][mask, :])
        plot_panel(ax, res['mag'], res['freqs'], res['t'],
                   f"{title}\nR={r:.2f}", fmax=fmax)

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle(f'{CLASS_NAMES.get(cls, f"Class {cls}")}  |  sample idx={idx}\n'
                 f'{n} TFR versions (linear mag, 0-{fmax:.0f} Hz)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='TFR 版本挑选 (实际数据抽样)')
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--cls', type=int, default=None, help='只跑指定类别')
    ap.add_argument('--fmax', type=float, default=FMAX)
    ap.add_argument('--outdir', default='tfr_version_probe')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fmax = args.fmax

    xs, ys, idxs, classes = load_samples(args.data, per_class=1)
    if args.cls is not None:
        keep = [i for i, c in enumerate(ys) if c == args.cls]
        xs = [xs[i] for i in keep]; ys = [ys[i] for i in keep]; idxs = [idxs[i] for i in keep]

    print(f"{'='*70}")
    print(f"TFR 版本挑选 — {len(xs)} 个样本, 每个 {len(VERSIONS)} 个版本")
    print(f"输出目录: {outdir}   (线性幅度样式, 0-{fmax:.0f} Hz)")
    print(f"{'='*70}")

    summary_rows = []

    for si, (x, cls, idx) in enumerate(zip(xs, ys, idxs)):
        print(f"\n[{si+1}/{len(xs)}] Class {cls} idx={idx} ... 计算各版本 ...")
        versions = compute_all_versions(x)

        for vi, (name, title) in enumerate(VERSIONS):
            res = versions[name]
            mask = res['freqs'] <= fmax
            r = renyi_entropy(res['mag'][mask, :])
            rr = ridge_energy_ratio(res['mag'][mask, :])
            summary_rows.append((cls, idx, name, r, rr))

            fname = f'class{cls}_v{vi:02d}_{name}.png'
            save_tfr_png(res['mag'], res['freqs'], res['t'],
                         outdir / fname, f"{title} (class {cls}, idx {idx})",
                         fmax=fmax)

        contact = outdir / f'contact_class{cls}.png'
        plot_contact_sheet(versions, cls, idx, contact, fmax)
        print(f"  保存 class {cls} 的 {len(VERSIONS)} 个版本 + 汇总图 -> {contact.name}")

    # summary.csv
    csv_path = outdir / 'summary.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'sample_idx', 'version', 'renyi_alpha3', 'ridge_energy_ratio'])
        w.writerows(summary_rows)
    print(f"\n数值汇总 -> {csv_path}")

    # 控制台均值表 (每版本跨类平均 Rényi, 越低越好)
    print(f"\n{'version':<10s} {'mean Rényi':>11s} {'mean ridge':>11s}")
    print('-' * 34)
    agg = {}
    for cls, idx, name, r, rr in summary_rows:
        agg.setdefault(name, []).append(r)
    for name, _ in VERSIONS:
        rs = agg.get(name, [])
        rrs = [rr for c, i, n, r, rr in summary_rows if n == name]
        print(f"{name:<10s} {np.mean(rs):11.2f} {np.mean(rrs):11.3f}")

    print("\nDone. 请打开 tfr_version_probe/ 对比各版本后决定采用哪套.")


if __name__ == '__main__':
    main()