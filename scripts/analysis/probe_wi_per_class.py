"""逐类 w_i 统计 (只读): 验证缺席门控是否真的带来"空转 / 加载"之间的自适应切换.

输出每类每节点的 w_i 时间均值、样本间标准差, 以及缺席度先验 (presence)。
用法:
  conda run -n my_work python scripts/analysis/probe_wi_per_class.py \
      -c sast_checkpoints/gate3node/sast_v3_e020.pt --per-class 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.sast_losses import node_presence                  # noqa: E402
from models.sast_nodes import PUMP_TURBINE_REGIONS            # noqa: E402
from sast_model import load_checkpoint                        # noqa: E402

CLASS_NAMES = {0: '空转', 1: '低负荷', 2: '中负荷', 3: '高负荷', 4: '抽水'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--checkpoint', required=True)
    ap.add_argument('-d', '--data', default='5_dataset.npz')
    ap.add_argument('--per-class', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    model, _freq_encoder, meta = load_checkpoint(a.checkpoint, dev)
    cfg = meta['config']
    model.eval()
    print(f'checkpoint: {a.checkpoint}')

    d = np.load(a.data, allow_pickle=True)
    X, y = d['test_X'], d['test_y']
    rng = np.random.default_rng(cfg.seed)

    names = [r.name for r in PUMP_TURBINE_REGIONS]
    print(f'\n{"类":<8s}' + ''.join(f'{n:>16s}' for n in names)
          + f'{"presence 2xBPF":>16s}')
    print('-' * (8 + 16 * (len(names) + 1)))
    rows = ['class,' + ','.join(names) + ',presence_2xBPF']
    for c in np.unique(y):
        idx = rng.choice(np.where(y == c)[0],
                         size=min(a.per_class, int((y == c).sum())),
                         replace=False)
        xb = torch.from_numpy(X[idx]).float().to(dev)
        with torch.no_grad():
            res = model(xb, return_all=True, training=False)
            w = res['w_i'].mean(dim=-1)                    # [B, N] 时间均值
            pres = node_presence(res['tfr_raw'], res['freqs'],
                                 PUMP_TURBINE_REGIONS,
                                 frac=cfg.presence_frac).mean(dim=-1)
        m = w.mean(dim=0).cpu().numpy()
        s = w.std(dim=0).cpu().numpy()
        p2 = pres[:, 2].mean().item()
        print(f'{CLASS_NAMES.get(int(c), str(c)):<8s}'
              + ''.join(f'{m[i]:9.3f}±{s[i]:5.3f}' for i in range(len(names)))
              + f'{p2:16.3f}')
        rows.append(f'{c},' + ','.join(f'{v:.4f}' for v in m) + f',{p2:.4f}')

    out = Path('results/band_census/wi_per_class.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print(f'\n已写入 {out}')


if __name__ == '__main__':
    main()