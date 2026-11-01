"""
节点 IF 聚合方式 与 "分量存在度" 体检
======================================
对每个工况抽样, 检查 MSSTNodeExtractor 的四个物理节点
(B4 拆节点后: LOW_FREQ / BPF_DRIFT / BPF_LINE / 2xBPF):

  1. IF 估计的三种口径对比
       IF_med      现行做法: 带内 omega_final 的无权重中位数
       f_centroid  带内能量质心   Σ(f·E)/ΣE      (逐帧)
       f_argmax    带内能量最大 bin 对应的频率   (逐帧)
     目的: 检验 node_if 是否落在"能量实际所在的位置"。

  2. 分量存在度 e_n ∈ [0,1]  (用于给 w_range 下界 / 频率锚 做门控)
       c = 带内能量最高的 k 个 bin 的能量占比,  k = ceil(0.1·F_n)
       e = clip((c - k/F_n) / (1 - k/F_n), 0, 1)
       平坦宽带谱 -> c≈k/F_n -> e≈0 ; 谱线 -> e->1

输出 results/band_census/node_if_probe.csv

用法:
  python scripts/analysis/probe_node_if.py --per-class 12
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ))

from models.sast_nodes import (MSSTNodeExtractor, PUMP_TURBINE_REGIONS,  # noqa: E402
                               region_weights)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CLASS_NAMES = ['空转', '低负荷', '中负荷', '高负荷', '抽水']
N_RATED, Z_R = 333.3, 9.0
F_ROT = N_RATED / 60.0
# 绝对锚直接取节点定义 (B4 后: BPF_LINE 49.995±1.5, 2xBPF 99.990±1.5; 其余无锚)
ANCHOR = [r.f_anchor for r in PUMP_TURBINE_REGIONS]


def presence(band_spec: np.ndarray, w: np.ndarray | None = None,
             frac: float = 0.1) -> float:
    """带内谱 (按 bin, 时间已合并) 的能量集中度 -> 存在度 e。

    w: 区域频带权重 (region_weights)。平坦谱基线取"权重最大的 k 个 bin 的
    权重占比", 硬边界 (w≡1) 下退化为 k/F_n。None 表示历史硬矩形口径。
    """
    e = np.maximum(band_spec, 0.0)
    tot = e.sum()
    Fn = e.size
    if tot <= 0 or Fn == 0:
        return 0.0
    k = max(1, int(np.ceil(frac * Fn)))
    if w is None:
        p = k / Fn
    else:
        p = float(np.sort(w)[::-1][:k].sum() / max(1e-12, float(w.sum())))
    c = float(np.sort(e)[::-1][:k].sum() / tot)
    return float(np.clip((c - p) / max(1e-8, 1.0 - p), 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser(description='节点 IF 聚合 / 存在度体检')
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--per-class', type=int, default=12)
    ap.add_argument('--outdir', default='results/band_census')
    ap.add_argument('--legacy', action='store_true',
                    help='用**历史**口径 (BPF_DRIFT[42,48] + BPF_LINE[48,53] + '
                         '两端皆闭硬矩形 + p=k/F_n) 复现改前行为, 用于对照')
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 直接用规范表 + region_weights。
    # ⚠ --legacy 已失效: 它依赖 B4 拆分出的 BPF_DRIFT / BPF_LINE 两个节点,
    #   而该拆分已撤回 (41 Hz 主峰落在原 BPF_DRIFT 左界之外)。历史结果保留在
    #   results/band_census/ 下, 不再可复现 —— 这里显式报错, 不静默退化成
    #   "跑新口径却打印历史口径"。
    if args.legacy:
        raise SystemExit(
            '--legacy 不可用: 4 节点 (BPF_DRIFT/BPF_LINE) 定义已随 B4 撤回删除。\n'
            '历史输出见 results/band_census/。若确需复现, 请在'
            ' PUMP_TURBINE_REGIONS 上临时手工还原 42/48.5 边界后再跑。')
    regions = PUMP_TURBINE_REGIONS

    d = np.load(args.data, allow_pickle=True)
    X, y = d['train_X'], d['train_y'].ravel().astype(np.int64)
    if X.ndim == 3:
        X = X[:, :, 0]

    ex = MSSTNodeExtractor(fs=1000.0)
    rows = []

    print(f'节点定义: ' + '  '.join(
        f'{r.name}[{r.f_min:.1f},{r.f_max:.1f})' for r in regions))
    print(f'绝对锚: ' + '  '.join(
        f'{r.name}→{"—" if a is None else f"{a:.3f}Hz"}'
        for r, a in zip(regions, ANCHOR)))
    print()

    for c in sorted(np.unique(y).tolist()):
        idx = np.where(y == c)[0]
        if len(idx) > args.per_class:
            idx = np.random.RandomState(0).choice(idx, args.per_class, replace=False)
        Xs = X[idx].astype(np.float64)

        acc = {}
        for n, r in enumerate(regions):
            acc[n] = {'med': [], 'cen': [], 'arg': [], 'e': [], 'bw': [], 'p': []}

        for x in Xs:
            nf = ex(x)
            freqs = nf.freqs
            E = np.abs(nf.tfr_stft) ** 2                     # [F, T]
            for n, r in enumerate(regions):
                if args.legacy:
                    # 历史口径: 两端皆闭的硬矩形
                    m = (freqs >= r.f_min) & (freqs <= r.f_max)
                    w_in = None
                    Eb = E[m]                                # [F_n, T]
                else:
                    # 与训练侧同源: 半开区间 + 共享内部边单位分解过渡带
                    w_in = region_weights(freqs, [r])[0]     # [F]
                    m = w_in > 0.0
                    Eb = E[m] * w_in[m][:, None]             # [F_n, T]
                fb = freqs[m]
                col = Eb.sum(axis=1)                         # 带内 bin 谱
                tot = col.sum()
                if tot <= 0:
                    continue
                cen = float((fb * col).sum() / tot)
                arg = float(fb[int(np.argmax(col))])
                acc[n]['med'].append(float(np.mean(nf.if_hz[n])))
                acc[n]['cen'].append(cen)
                acc[n]['arg'].append(arg)
                acc[n]['e'].append(presence(col, w_in))
                acc[n]['bw'].append(float(np.mean(nf.bandwidth[n])))
                acc[n]['p'].append(float(nf.persistence[n]))

        print(f'{"=" * 78}')
        print(f'{CLASS_NAMES[c]}  (n={len(Xs)})')
        print(f'{"node":<10s} {"IF_med":>8s} {"E质心":>8s} {"E峰f":>8s} '
              f'{"|锚-f_med|":>11s} {"|锚-E峰|":>10s} {"存在度e":>8s} {"bw":>6s}')
        for n, r in enumerate(regions):
            med = float(np.mean(acc[n]['med']))
            cen = float(np.mean(acc[n]['cen']))
            arg = float(np.mean(acc[n]['arg']))
            e = float(np.mean(acc[n]['e']))
            bw = float(np.mean(acc[n]['bw']))
            a = ANCHOR[n]
            da_m = f'{abs(a - med):.3f}' if a else '—'
            da_a = f'{abs(a - arg):.3f}' if a else '—'
            print(f'{r.name:<10s} {med:>8.3f} {cen:>8.3f} {arg:>8.3f} '
                  f'{da_m:>11s} {da_a:>10s} {e:>8.3f} {bw:>6.2f}')
            rows.append((c, CLASS_NAMES[c], r.name, f'{r.f_min:.1f}', f'{r.f_max:.1f}',
                         f'{med:.4f}', f'{cen:.4f}', f'{arg:.4f}',
                         f'{e:.4f}', f'{bw:.4f}', f'{float(np.mean(acc[n]["p"])):.4f}',
                         '' if a is None else f'{a:.4f}'))

    with open(outdir / 'node_if_probe.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['class', 'class_name', 'node', 'f_min', 'f_max',
                    'if_med_hz', 'f_centroid_hz', 'f_argmax_hz',
                    'presence_e', 'bandwidth_hz', 'persistence', 'anchor_hz'])
        w.writerows(rows)

    print(f'\n输出: {outdir / "node_if_probe.csv"}')


if __name__ == '__main__':
    main()