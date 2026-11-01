"""逐帧 presence 分布探针 (只读): 检查缺席门限 0.10/0.20 是否在帧级干净分离.

关心的是: 加载工况 (类 1-4) 的 2xBPF 逐帧 presence 会不会有相当比例掉到 0.20
以下 —— 若有, 缺席门控会在那些帧把上界收拢到 0.20, 把加载工况的硬挤压打掉。

输出: results/band_census/absence_frames.csv + 控制台汇总
用法: conda run -n my_work python scripts/analysis/probe_absence_frames.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.msst_torch import msst_torch          # noqa: E402
from models.sast_losses import node_presence      # noqa: E402
from models.sast_nodes import PUMP_TURBINE_REGIONS  # noqa: E402
from sast_config import SastConfig                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='results/band_census/absence_frames.csv')
    a = ap.parse_args()

    cfg = SastConfig()
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    d = np.load(a.data, allow_pickle=True)
    X, y = d['train_X'], d['train_y']
    rng = np.random.default_rng(cfg.seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0],
                                     size=min(a.per_class, int((y == c).sum())),
                                     replace=False)
                          for c in np.unique(y)])
    Xs = torch.from_numpy(X[idx]).float()

    print(f'样本 {len(idx)} 个, 设备 {dev}')
    e_all = []
    with torch.no_grad():
        for i in range(len(Xs)):
            r = msst_torch(Xs[i].to(dev), cfg.fs,
                           hlength=cfg.msst_hlength, num=cfg.msst_num,
                           save_trajectory=False, skip_squeeze=True)
            s_mag = r['STFT'].abs().unsqueeze(0)          # [1, F, T]
            Fb = s_mag.shape[1]
            freqs = torch.arange(Fb, device=dev, dtype=torch.float32) \
                / (2.0 * Fb) * cfg.fs
            e = node_presence(s_mag, freqs, PUMP_TURBINE_REGIONS,
                              frac=cfg.presence_frac)
            e_all.append(e.cpu())
    E = torch.cat(e_all)                                # [N, 3, T]
    yy = y[idx]

    rows = ['class,node,mean,std,p10,p25,median,p75,p90,frac_below_0.20']
    print(f'\n{"类":>3s} {"节点":<10s} {"mean":>6s} {"std":>6s} '
          f'{"p10":>6s} {"med":>6s} {"p90":>6s} {"<0.20":>7s}')
    for c in np.unique(yy):
        m = yy == c
        for n, r in enumerate(PUMP_TURBINE_REGIONS):
            v = E[m, n, :].reshape(-1).numpy()
            frac = float((v < 0.20).mean())
            print(f'{c:3d} {r.name:<10s} {v.mean():6.3f} {v.std():6.3f} '
                  f'{np.percentile(v, 10):6.3f} {np.median(v):6.3f} '
                  f'{np.percentile(v, 90):6.3f} {frac:7.3f}')
            rows.append(f'{c},{r.name},{v.mean():.4f},{v.std():.4f},'
                        f'{np.percentile(v, 10):.4f},{np.percentile(v, 25):.4f},'
                        f'{np.median(v):.4f},{np.percentile(v, 75):.4f},'
                        f'{np.percentile(v, 90):.4f},{frac:.4f}')
        print()

    # ── 门限候选: 2xBPF 的缺席度均值 a 与"加载工况误判率" ──
    print('门限候选 (2xBPF): 逐帧缺席度 a 的类均值 / 加载类 a>0.05 的帧占比')
    print(f'{"lo":>5s} {"hi":>5s}  ' + '  '.join(f'{c:>12d}' for c in range(5)))
    for lo, hi in [(0.10, 0.20), (0.10, 0.16), (0.12, 0.18),
                   (0.14, 0.19), (0.15, 0.20), (0.13, 0.17)]:
        line = f'{lo:5.2f} {hi:5.2f}  '
        for c in range(5):
            v = E[yy == c, 2, :].reshape(-1)
            aa = ((hi - v) / (hi - lo)).clamp(0, 1)
            line += f'  a={aa.mean():.2f}/{float((aa > 0.05).float().mean()):.2f}'
        print(line)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print(f'已写入 {a.out}')


if __name__ == '__main__':
    main()