"""
基线方法对比脚本 (待补充实验清单 §2.2(3))
========================================

对比三组挤压策略:
  1. canonical 5 方法       : SST1 / WSST1 / WSST2 / MSST1 / MSST4 (基线)
  2. fixed-σ MSST           : 软高斯重排, σ 固定 (扫描 σ ∈ {0.5,1,2,3} bin)
  3. SAST (本文)            : σ 由 PPM+GAT 从信号学习 (需 checkpoint)

对比维度:
  - TFR 质量指标: Rényi 熵 + 脊线能量占比 (全带 0-500 Hz)
  - 可分性: 每样本 TFR 的频域特征 (mean/max over time) → PCA 降维后
            silhouette 系数 + 1-NN 分类准确率 + t-SNE 可视化
            按物理分区统计: full / harm (42-105 Hz, 应挤压) / low (2-25 Hz, 不应挤压)

统一实现: 复用 tfr_common.CANONICAL_METHODS / canonical 参数 / 线性绘图.

用法:
  python scripts/plot/compare_baselines.py --checkpoint sast_checkpoints/sast_v3_model.pt \
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
    FS, HLENGTH, MSST_NUM, FMAX_DEFAULT, CANONICAL_METHODS, band_mask,
    REGIONS, REGION_LABELS, region_mask, apply_plot_style,
    load_samples, renyi_entropy, ridge_energy_ratio,
)
from models.tfr import msst

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

apply_plot_style()

SIGMA_SWEEP = [0.5, 1.0, 2.0, 3.0]


# ═══════════════════════════════════════════════════════════════
# 软高斯重排 (固定 σ) — 与 SAST ReassignerFunction 同公式, 仅 σ 常数
# ═══════════════════════════════════════════════════════════════

def soft_gaussian_reassign(tfr_mag, omega_final, sigma):
    """
    固定 σ 的软高斯重排 (匹配 models/sast.py SparseGaussianReassigner 前向).

    Args:
        tfr_mag:     [F, T] STFT 幅值
        omega_final: [F, T] int 收敛 IF (1-indexed, 0=无效)
        sigma:       固定核宽 (bin)

    Returns:
        [F, T] 重排后幅度
    """
    F, T = tfr_mag.shape
    K = int(np.ceil(3.0 * sigma))
    offsets = np.arange(-K, K + 1)
    Z = np.exp(-0.5 * (offsets / sigma) ** 2).sum()

    valid = (omega_final >= 1) & (omega_final <= F)
    idx = np.clip(omega_final - 1, 0, F - 1).astype(np.int64)
    cols = np.broadcast_to(np.arange(T, dtype=np.int64)[None, :], (F, T))

    out = np.zeros((F, T), dtype=np.float64)
    src = tfr_mag * valid
    for k in offsets:
        w = np.exp(-0.5 * (k / sigma) ** 2) / Z
        target = np.clip(idx + k, 0, F - 1)
        np.add.at(out, (target, cols), src * w)
    return out


def _fixed_sigma(x, sigma):
    """固定 σ 软重排, 返回 method dict(mag, freqs, t)."""
    r = msst(x, FS, hlength=HLENGTH, num=MSST_NUM, save_trajectory=True)
    mag = soft_gaussian_reassign(np.abs(r['STFT']), r['omega_final'], sigma)
    return dict(mag=mag, freqs=r['freqs'], t=r['t'])


def _load_sast(checkpoint_path, device):
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


# ═══════════════════════════════════════════════════════════════
# 可分性 (频域特征 -> PCA -> silhouette + 1-NN + t-SNE)
# ═══════════════════════════════════════════════════════════════

def freq_feature(mag):
    """每样本 TFR -> 频域特征 [mean over time ; max over time]."""
    return np.concatenate([mag.mean(axis=1), mag.max(axis=1)])


def _knn_accuracy(Z, y):
    """1-NN 留一法准确率."""
    from scipy.spatial.distance import cdist
    D = cdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    pred = y[np.argmin(D, axis=1)]
    return float((pred == y).mean())


def separability(mags, ys, outdir, tag, region='full', make_tsne=False):
    """mags: list of [F,T] 幅度谱 (已在某一频段内切片); 返回 (silhouette, knn_acc)."""
    X = np.stack([freq_feature(m) for m in mags])
    y = np.array(ys)
    n = len(ys)
    n_comp = min(10, n - 1, X.shape[1])

    pca = PCA(n_components=n_comp)
    Z = pca.fit_transform(X)

    n_labels = len(set(y))
    sil = silhouette_score(Z, y) if n_comp >= 2 and 1 < n_labels < n else float('nan')
    knn = _knn_accuracy(Z, y) if n >= 2 else float('nan')

    if make_tsne and n >= 8:
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, n - 1)),
                    init='pca', random_state=42)
        E = tsne.fit_transform(Z)
        fig, ax = plt.subplots(figsize=(6, 5))
        for c in sorted(set(y)):
            ax.scatter(E[y == c, 0], E[y == c, 1], s=30,
                       label=f'class {c}', alpha=0.8)
        ax.set_title(f'{tag} — {REGION_LABELS[region]}\n'
                     f'silhouette={sil:.3f}, 1-NN={knn:.3f}')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(outdir / f'tsne_{region}_{tag}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    return sil, knn


def main():
    ap = argparse.ArgumentParser(description='基线方法对比 (canonical 5 + 固定σ + SAST)')
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--fmax', type=float, default=FMAX_DEFAULT)
    ap.add_argument('--checkpoint', default=None, help='SAST checkpoint (.pt), 缺省跳过 SAST')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--outdir', default='results/baseline_compare')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xs, ys, idxs, classes = load_samples(args.data, per_class=args.per_class)
    print(f'样本: {len(xs)} 个 ({len(classes)} 类 × {args.per_class})')

    quality_rows = []   # (method, cls, idx, renyi, ridge)
    sep_entries = []    # (tag, {region: [mag, ...]})

    def run_method(tag, fn, collect_sep=False):
        mags = []
        per_region = {r: [] for r in REGIONS}
        for x, cls, idx in zip(xs, ys, idxs):
            res = fn(x)
            freqs = np.asarray(res['freqs']).ravel()
            m = band_mask(freqs, args.fmax)
            full = res['mag'][m, :]
            mags.append(full)
            for region, band in REGIONS.items():
                rb = region_mask(freqs, band)
                if rb.any():
                    per_region[region].append(res['mag'][rb, :])
            quality_rows.append((tag, cls, idx,
                                 renyi_entropy(full), ridge_energy_ratio(full)))
        if collect_sep:
            sep_entries.append((tag, per_region))
        print(f'  {tag:<14s} 完成 ({len(mags)} 样本)')
        return mags, per_region

    # ── 1) canonical 5 方法 (全部参与可分性) ──
    for name, fn in CANONICAL_METHODS:
        run_method(name, fn, collect_sep=True)

    # ── 2) 固定 σ 扫描 ──
    fixed_mags, fixed_region = {}, {}
    for s in SIGMA_SWEEP:
        fixed_mags[s], fixed_region[s] = run_method(
            f'fixed-σ={s:g}', lambda x, s=s: _fixed_sigma(x, s))

    # ── 3) SAST ──
    if args.checkpoint:
        try:
            run_method('SAST', _load_sast(args.checkpoint, args.device),
                       collect_sep=True)
        except Exception as e:
            print(f'⚠ 加载 SAST 失败, 跳过: {e}')

    # ── 质量汇总表 ──
    print(f"\n{'method':<14s} {'Rényi (mean±std)':>20s} {'ridge (mean±std)':>20s}")
    print('-' * 56)
    agg = {}
    for method, cls, idx, r, rr in quality_rows:
        agg.setdefault(method, {'renyi': [], 'ridge': []})
        agg[method]['renyi'].append(r)
        agg[method]['ridge'].append(rr)
    for method in sorted(agg, key=lambda m: np.mean(agg[m]['renyi'])):
        d = agg[method]
        print(f"{method:<14s} {np.mean(d['renyi']):>9.2f} ±{np.std(d['renyi']):<8.2f} "
              f"{np.mean(d['ridge']):>9.3f} ±{np.std(d['ridge']):<8.3f}")

    # ── 可分性 (按物理分区) ──
    # 固定σ: 用 Rényi 最优的那个 σ 做代表
    best_s = min(SIGMA_SWEEP, key=lambda s: np.mean([renyi_entropy(m) for m in fixed_mags[s]]))
    sep_entries.append((f'fixed-σ={best_s:g}', fixed_region[best_s]))

    sep_results = []
    for region in REGIONS:
        print(f"\n{'=' * 48}")
        print(f"可分性 — {REGION_LABELS[region]}   [{region}]")
        print(f"{'tag':<16s} {'silhouette':>12s} {'1-NN acc':>10s}")
        print('-' * 40)
        for tag, per_region in sep_entries:
            mags = per_region.get(region, [])
            if not mags:
                continue
            # t-SNE 只画谐波带 (汇报关注的"应挤压"区域), 避免文件爆炸
            sil, knn = separability(mags, ys, outdir, tag, region=region,
                                    make_tsne=(region == 'harm'))
            sep_results.append((region, tag, sil, knn))
            print(f"{tag:<16s} {sil:>12.3f} {knn:>10.3f}")

    # ── CSV ──
    with open(outdir / 'baseline_quality.csv', 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['method', 'class', 'sample_idx', 'renyi_alpha3', 'ridge_energy_ratio'])
        w.writerows(quality_rows)
    with open(outdir / 'baseline_separability.csv', 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['region', 'method', 'silhouette', 'knn_accuracy'])
        w.writerows(sep_results)

    print(f"\n结果: {outdir}/baseline_quality.csv")
    print(f"可分性: {outdir}/baseline_separability.csv")


if __name__ == '__main__':
    main()