"""
时频图质量对比脚本 (待补充实验清单 §2.2(2))
============================================

对比 canonical 5 方法 + SAST(本文):
  SST1(ssqueezepy) / WSST1(ssqueezepy) / WSST2(morse) / MSST1 / MSST4  /  [SAST]

指标: Rényi 熵 (α=3) + 脊线能量占比, **按物理分量分区统计**:
  full : 0-500 Hz           全带
  harm : 42-105 Hz          BPF + 2xBPF —— 应该挤压, 越集中越好
  low  : 2-25 Hz            LOW_FREQ 涡带 —— 不应挤压, 应保留带宽

分区的理由: SAST 故意不在低频挤压, 若只报全带指标, 它的低频策略反而会
拉低全带 Rényi 熵分数. 汇报时应比较"该挤压的地方"的表现 (harm),
以及"不该挤压的地方"是否被保住 (low).
  注: Rényi 熵并非越小越好 —— 见 Word 文档第 5.3 节.

统一实现: 全部走 tfr_common.CANONICAL_METHODS (canonical 参数),
          绘图统一为 plot_msst_tfr.py 的线性 0-500 Hz 样式.

输出 (默认 results/tfr_quality/):
  - tfr_quality.csv        每 (region, 方法, 样本) 一行
  - tfr_quality_agg.csv    每 (region, 方法) mean ± std
  - renyi_comparison.png   全带 Rényi 柱状图 (保持原文件名)
  - renyi_region_<r>.png   各分区 Rényi 柱状图
  - renyi_by_region.png    三分区并排总览
  - tfr_grid_<method>.png  (--plot-grid) 每方法一张多样本并排 TFR

用法:
  python scripts/plot/compare_tfr_quality.py --per-class 10
  python scripts/plot/compare_tfr_quality.py --checkpoint sast_checkpoints/sast_v3_model.pt \
      --device cuda --per-class 10
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tfr_common import (
    CANONICAL_METHODS, FMAX_DEFAULT, REGIONS, REGION_LABELS, region_mask,
    load_samples, renyi_entropy, ridge_energy_ratio, plot_panel,
    apply_plot_style,
)

apply_plot_style()


def _load_sast(checkpoint_path, device):
    """加载 SAST 模型, 返回 method 函数 x -> dict(mag, freqs, t)."""
    import torch
    from sast_model import load_checkpoint
    model, freq_encoder, meta = load_checkpoint(checkpoint_path, torch.device(device))
    model = model.to(device)

    def infer(x):
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            res = model(xt, return_all=True, training=False)
        return dict(
            mag=res['tfr_enhanced'][0].cpu().numpy(),
            freqs=np.asarray(res['freqs'].cpu().numpy()).ravel(),
            t=np.asarray(res['t_axis'].cpu().numpy()).ravel(),
        )

    return infer


def _bar_chart(ax, names, means, stds, title, ylabel='Rényi entropy α=3',
               highlight='SAST'):
    colors = ['#C44E52' if n == highlight else '#4C72B0' for n in names]
    ax.bar(names, means, yerr=stds, capsize=4, color=colors, alpha=0.88)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description='时频图质量对比 (按物理分区)')
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--per-class', type=int, default=10, help='每类样本数')
    ap.add_argument('--fmax', type=float, default=FMAX_DEFAULT)
    ap.add_argument('--checkpoint', default=None, help='SAST checkpoint (.pt), 缺省跳过 SAST')
    ap.add_argument('--device', default='cuda', help='SAST 用设备 (cuda/cpu)')
    ap.add_argument('--outdir', default='results/tfr_quality')
    ap.add_argument('--plot-grid', action='store_true', help='每方法绘制多样本并排 TFR')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    methods = list(CANONICAL_METHODS)
    if args.checkpoint:
        try:
            methods.append(('SAST', _load_sast(args.checkpoint, args.device)))
        except Exception as e:
            print(f'⚠ 加载 SAST 失败, 跳过: {e}')

    xs, ys, idxs, classes = load_samples(args.data, per_class=args.per_class)
    names = [m[0] for m in methods]
    print(f'样本: {len(xs)} 个 ({len(classes)} 类 × {args.per_class})')
    print(f'方法: {names}')
    print(f'分区: {list(REGIONS)}')

    rows = []                       # (region, method, cls, idx, renyi, ridge)
    agg = {r: {n: {'renyi': [], 'ridge': []} for n in names} for r in REGIONS}
    grid_mags = {n: [] for n in names}

    for name, fn in methods:
        for x, cls, idx in zip(xs, ys, idxs):
            res = fn(x)
            freqs = np.asarray(res['freqs']).ravel()
            for region, band in REGIONS.items():
                m = region_mask(freqs, band)
                if not m.any():
                    continue
                sub = res['mag'][m, :]
                r = renyi_entropy(sub)
                rr = ridge_energy_ratio(sub)
                rows.append((region, name, cls, idx, r, rr))
                agg[region][name]['renyi'].append(r)
                agg[region][name]['ridge'].append(rr)
            grid_mags[name].append((res, cls, idx))
        print(f'  {name:<6s} 完成 ({len(grid_mags[name])} 样本)')

    # ── CSV ──
    csv_path = outdir / 'tfr_quality.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['region', 'method', 'class', 'sample_idx',
                    'renyi_alpha3', 'ridge_energy_ratio'])
        w.writerows(rows)

    agg_path = outdir / 'tfr_quality_agg.csv'
    with open(agg_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['region', 'method', 'renyi_mean', 'renyi_std',
                    'ridge_mean', 'ridge_std'])
        for region in REGIONS:
            for name in names:
                d = agg[region][name]
                w.writerow([region, name,
                            np.mean(d['renyi']), np.std(d['renyi']),
                            np.mean(d['ridge']), np.std(d['ridge'])])

    # ── 控制台表 (每个分区一张) ──
    for region in REGIONS:
        print(f"\n{'=' * 66}")
        print(f"{REGION_LABELS[region]}   [{region}]")
        print(f"{'method':<8s} {'Rényi (mean±std)':>20s} {'ridge (mean±std)':>20s}")
        print('-' * 52)
        for name in sorted(names, key=lambda n: np.mean(agg[region][n]['renyi'])):
            d = agg[region][name]
            print(f"{name:<8s} {np.mean(d['renyi']):>9.2f} ±{np.std(d['renyi']):<8.2f} "
                  f"{np.mean(d['ridge']):>9.3f} ±{np.std(d['ridge']):<8.3f}")

    # ── 柱状图: 每个分区一张 + 并排总览 ──
    for region in REGIONS:
        means = [np.mean(agg[region][n]['renyi']) for n in names]
        stds = [np.std(agg[region][n]['renyi']) for n in names]
        fig, ax = plt.subplots(figsize=(0.9 * len(names), 4.5))
        _bar_chart(ax, names, means, stds,
                   f'Rényi entropy — {REGION_LABELS[region]}')
        fig.tight_layout()
        fname = 'renyi_comparison.png' if region == 'full' else f'renyi_region_{region}.png'
        fig.savefig(outdir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)

    region_list = list(REGIONS)
    fig, axes = plt.subplots(1, len(region_list),
                             figsize=(5.6 * len(region_list), 4.6), squeeze=False)
    for ax, region in zip(axes[0], region_list):
        means = [np.mean(agg[region][n]['renyi']) for n in names]
        stds = [np.std(agg[region][n]['renyi']) for n in names]
        _bar_chart(ax, names, means, stds, REGION_LABELS[region])
        ax.tick_params(axis='x', rotation=30)
    fig.suptitle('TFR quality by physical region (Rényi α=3, mean ± std)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / 'renyi_by_region.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 多样本并排网格 (可选) ──
    if args.plot_grid:
        for name, _ in methods:
            entries = grid_mags[name]
            ncols = min(5, len(entries))
            nrows = int(np.ceil(len(entries) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows))
            axes = np.atleast_1d(axes).ravel()
            for ax, (res, cls, idx) in zip(axes, entries):
                plot_panel(ax, res['mag'], res['freqs'], res['t'],
                           f'class {cls} idx={idx}', fmax=args.fmax,
                           annotate_dominant=False)
            for ax in axes[len(entries):]:
                ax.axis('off')
            fig.suptitle(f'{name} — {len(entries)} samples (0-{args.fmax:.0f} Hz)',
                         fontsize=11, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(outdir / f'tfr_grid_{name}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"\n结果: {csv_path}")
    print(f"聚合: {agg_path}")
    print(f"图表: {outdir / 'renyi_comparison.png'}, {outdir / 'renyi_by_region.png'}")


if __name__ == '__main__':
    main()