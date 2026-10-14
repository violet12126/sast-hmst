"""
按 4 节点重算工况静态原型 STATIC_PROTOTYPES
============================================
B4 拆节点后, 旧的 5x4 原型 (E_total = E_LF + E_BPF族 + E_2xBPF) 与新的
4 节点能量划分语义不再匹配; 且旧原型存在两处归一化/换算不一致:

  1. 旧原型前三列之和 = 0.273/0.911/0.620/0.819/0.806, 而在线
     compute_v_obs 的 E_total = Σ 带能量 使前三列恒和为 1.0。
     余弦相似度对"前三列整体 vs log 比值列"的相对权重敏感, 故旧 alpha
     是被系统性扭曲的。
  2. 旧代码 E_lin = exp(log1p(x)) = 1 + x, 每个节点被加了 +1 地板。

本脚本**复用模型同一条计算路径** (models.sast.compute_v_obs) 重算, 由构造
保证在线/离线一致。计算路径不依赖 STATIC_PROTOTYPES 本身, 故无循环依赖。

特征 (6 维):
  V_obs = [ R_LF, R_DRIFT, R_LINE, R_2x,
            log10(E_DRIFT / E_LINE),
            log10((E_DRIFT + E_LINE) / E_2x) ]

数据划分: 用 sast_data.load_dataset 取**与训练完全相同**的 train 划分
(同 seed / val_split), 确保原型不看到验证集。

用法:
  python scripts/analysis/build_static_prototypes.py --per-class 400
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ))

from models.sast_nodes import MSSTNodeExtractor, PUMP_TURBINE_REGIONS  # noqa: E402
from models.sast import compute_v_obs                                  # noqa: E402
from sast_data import load_dataset                                     # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CLASS_NAMES = ['空转', '低负荷', '中负荷', '高负荷', '抽水']

# 旧表 (5x4, 3 节点时期 EDA) — 仅用于并列对照, 不参与任何计算
OLD_PROTOTYPES = np.array([
    [0.037636, 0.208980, 0.026644, 0.986240],
    [0.068068, 0.832580, 0.010149, 1.987957],
    [0.046170, 0.269881, 0.303809, 0.001379],
    [0.013093, 0.153476, 0.652464, -0.651019],
    [0.021357, 0.685910, 0.099116, 0.850605],
])

V_COLS = ['R_LF', 'R_DRIFT', 'R_LINE', 'R_2x',
          'log10(E_DRIFT/E_LINE)', 'log10((E_DRIFT+E_LINE)/E_2x)']


def node_energy_for(X: np.ndarray, ex: MSSTNodeExtractor,
                    batch: int = 32, device: str = 'cpu') -> np.ndarray:
    """[N, T] 信号 -> [N, N_phys, T] 节点对数能量。

    优先走 GPU 向量化路径 (extract_gpu), 失败则退回 numpy CPU 路径。
    """
    outs = []
    for i in range(0, len(X), batch):
        chunk = X[i:i + batch].astype(np.float32)
        if device == 'cuda':
            try:
                res = ex.extract_gpu(torch.from_numpy(chunk).to(device))
                outs.append(res['node_energy'].float().cpu().numpy())
                continue
            except Exception as e:                     # pragma: no cover
                print(f'  [warn] GPU 路径失败 ({type(e).__name__}: {e}), '
                      f'退回 CPU')
                device = 'cpu'
        outs.append(np.stack([ex(x.astype(np.float64)).energy for x in chunk]))
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser(description='按 4 节点重算静态原型')
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--val_split', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--per-class', '--per_class', dest='per_class',
                    type=int, default=400, help='每类抽样样本数 (0 = 全部)')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--out', default='results/band_census/static_prototypes.csv')
    args = ap.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    ds = load_dataset(args.data, val_split=args.val_split, seed=args.seed)
    X = ds['train_X']
    y = ds['train_y']
    print(f'数据: {args.data}  train 划分 N={len(X)}  '
          f'(val_split={args.val_split}, seed={args.seed}, 已排除验证集)')
    print(f'设备: {device}')
    print(f'节点: ' + '  '.join(
        f'{r.name}[{r.f_min:.1f},{r.f_max:.1f})' for r in PUMP_TURBINE_REGIONS))

    rng = np.random.RandomState(args.seed)
    ex = MSSTNodeExtractor(fs=float(ds['fs']))

    per_class_v = {}
    for c in sorted(np.unique(y).tolist()):
        idx = np.where(y == c)[0]
        if args.per_class and len(idx) > args.per_class:
            idx = rng.choice(idx, args.per_class, replace=False)
        print(f'  class {c} ({CLASS_NAMES[c]}): {len(idx)} 样本 ...', flush=True)
        E = node_energy_for(X[idx], ex, batch=args.batch, device=device)
        # [N, N_phys, T] -> [N, T, 6], 再跨样本与时间取中位数 (对离群样本稳健)
        V = compute_v_obs(torch.from_numpy(E).float())      # [N, T, 6]
        per_class_v[c] = V.reshape(-1, V.shape[-1]).numpy()  # [N*T, 6]

    # 逐点校验: compute_v_obs 的 R 块对每个 (样本, 帧) 恒和为 1
    row_sum_pt = np.concatenate([per_class_v[c][:, :4].sum(axis=1)
                                 for c in sorted(per_class_v)])
    print(f'逐点 R 块求和: min={row_sum_pt.min():.9f}, '
          f'max={row_sum_pt.max():.9f} (应为 1)')

    # 逐列取中位数。中位数不是线性泛函, 故"各行中位数之和" ≠ 1 ——
    # 而在线 V_obs 的 R 块恒和为 1。余弦相似度对 R 块 vs log 列的相对权重
    # 敏感, 所以这里把原型 R 块**重新归一化**回 1, 使其与任何在线样本
    # 落在同一流形上 (这正是重算原型要根治的问题)。
    proto_raw = np.stack([np.median(per_class_v[c], axis=0)
                          for c in sorted(per_class_v)], axis=0)   # [5, 6]
    proto = proto_raw.copy()
    proto[:, :4] /= proto[:, :4].sum(axis=1, keepdims=True)
    print(f'逐列中位数的 R 块和  : {np.round(proto_raw[:, :4].sum(axis=1), 6)}'
          f'  -> 重归一化后: {np.round(proto[:, :4].sum(axis=1), 6)}')

    # ── 并列对照 ──
    print('\n' + '=' * 78)
    print('新原型 (5x6, 4 节点重算):')
    print(f'{"cls":>3s} {"name":<8s} ' + ' '.join(f'{n:>22s}' for n in V_COLS))
    for c in sorted(per_class_v):
        print(f'{c:>3d} {CLASS_NAMES[c]:<8s} ' +
              ' '.join(f'{v:>22.6f}' for v in proto[c]))

    print('\n前三列 (R_LF+R_DRIFT+R_LINE+R_2x) 之和 —— 应恒为 1.0:')
    rsum = proto[:, :4].sum(axis=1)
    for c in sorted(per_class_v):
        print(f'  class {c} ({CLASS_NAMES[c]}): {rsum[c]:.9f}')

    print('\n旧原型 (5x4) 前三列之和 —— 恒 < 1, 与在线归一化不一致:')
    for c in range(5):
        print(f'  class {c} ({CLASS_NAMES[c]}): '
              f'{OLD_PROTOTYPES[c, :3].sum():.6f}')

    print('\n漂移/线 判别子 log10(E_DRIFT/E_LINE) (>0 = 漂移占优, <0 = 线占优):')
    for c in sorted(per_class_v):
        print(f'  class {c} ({CLASS_NAMES[c]}): {proto[c, 4]:+8.4f}')

    # ── 输出 CSV ──
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['class', 'class_name'] + V_COLS +
                   [f'old_{n}' for n in ['R_LF', 'R_BPF', 'R_2x', 'log10(Eb/E2x)']])
        for c in sorted(per_class_v):
            w.writerow([c, CLASS_NAMES[c]] + [f'{v:.6f}' for v in proto[c]] +
                       [f'{v:.6f}' for v in OLD_PROTOTYPES[c]])
    print(f'\n输出: {out}')

    # ── 可直接粘贴的 Python 字面量 ──
    print('\n' + '=' * 78)
    print('粘贴到 models/sast.py 的 STATIC_PROTOTYPES (注意 dtype=torch.float64):')
    print('STATIC_PROTOTYPES = torch.tensor([')
    for c in sorted(per_class_v):
        vals = ', '.join(f'{v:.6f}' for v in proto[c])
        print(f'    [{vals}],  # Class {c}: {CLASS_NAMES[c]}')
    print('], dtype=torch.float64)')


if __name__ == '__main__':
    main()