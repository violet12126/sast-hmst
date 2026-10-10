"""
汇总 SAST 在 5 个工况类上的自适应挤压行为 (供 Word 文档表格使用)
==================================================================
对每类抽取若干样本, 统计:
  - 各物理节点的 IF 均值 (Hz)
  - 各物理节点的信任度 w_i 均值
  - 各物理节点的核宽 sigma_sq 均值 (bin, 由 w_i 映射)
  - 各物理节点的挤压轮数 lambda_i = round(w_i * n_sqz_max)

用法:
  python scripts/analysis/summarize_sast_per_class.py \
      --checkpoint sast_checkpoints/sast_v3_model.pt --device cuda --per-class 8
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from sast_model import load_checkpoint          # noqa: E402
from sast_config import SastConfig              # noqa: E402
from models.sast_nodes import PUMP_TURBINE_REGIONS   # noqa: E402

# 节点名直接由节点定义导出 (B4 拆节点后 4 个: LOW_FREQ/BPF_DRIFT/BPF_LINE/2xBPF)
NODE_NAMES = [f'{r.name} ({r.f_min:.0f}-{r.f_max:.0f}Hz)'
              for r in PUMP_TURBINE_REGIONS]
CLASS_NAMES = ['空转', '低负荷', '中负荷', '高负荷', '抽水']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--checkpoint', default='sast_checkpoints/sast_v3_model.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--per-class', type=int, default=8)
    ap.add_argument('--outdir', default='results/sast_behavior')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, _, meta = load_checkpoint(args.checkpoint, dev)
    args_ck = meta.get('args', {}) if isinstance(meta, dict) else {}
    cfg = SastConfig.from_dict(args_ck)
    n_sqz_max = cfg.n_sqz_max

    d = np.load(args.data, allow_pickle=True)
    X, y = d['train_X'], d['train_y'].ravel().astype(np.int64)
    if X.ndim == 3:
        X = X[:, :, 0]
    classes = sorted(np.unique(y).tolist())

    rows = []
    for c in classes:
        cidx = np.where(y == c)[0]
        en = (X[cidx] ** 2).sum(axis=1)
        chosen = cidx[np.argsort(en)[::-1][:args.per_class]]

        w_list, if_list, sg_list, cls_hat = [], [], [], []
        for idx in chosen:
            xt = torch.from_numpy(X[idx].astype(np.float32)).unsqueeze(0).to(dev)
            with torch.no_grad():
                res = model(xt, return_all=True, training=False)
            w_list.append(res['w_i'][0].cpu().numpy())           # [N_phys, T]
            if_list.append(res['node_if'][0].cpu().numpy())      # [N_phys, T]
            sg_list.append(res['sigma_sq'][0].cpu().numpy())     # [F, T]

        w = np.stack(w_list)        # [n, N_phys, T]
        fif = np.stack(if_list)     # [n, N_phys, T]

        w_mean = w.mean(axis=(0, 2))            # [N_phys]
        if_mean = fif.mean(axis=(0, 2))         # [N_phys]
        lam = np.round(w_mean * n_sqz_max).astype(int)

        print(f'\n{"=" * 74}')
        print(f'class {c} ({CLASS_NAMES[c]})  n={len(chosen)}  '
              f'w_i spread(max-min)={w_mean.ptp():.3f}')
        print(f'{"node":<22s} {"IF(Hz)":>8s} {"w_i":>7s} {"σ(bin)":>9s} {"λ":>3s}')
        print('-' * 54)
        for n in range(len(NODE_NAMES)):
            sigma = cfg.sigma_min + (1 - w_mean[n]) * (cfg.sigma_max - cfg.sigma_min)
            print(f'{NODE_NAMES[n]:<22s} {if_mean[n]:8.1f} {w_mean[n]:7.3f} '
                  f'{sigma:9.2f} {lam[n]:3d}')
            rows.append((c, CLASS_NAMES[c], NODE_NAMES[n], float(if_mean[n]),
                         float(w_mean[n]), float(sigma), int(lam[n])))

    # utf-8-sig: 带 BOM, 否则中文 Windows 的 Excel 会按 GBK 解码而看到乱码
    with open(outdir / 'sast_node_behavior.csv', 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['class', 'class_name', 'node', 'if_hz', 'w_i', 'sigma_bin', 'lambda'])
        w.writerows(rows)
    print(f'\n输出: {outdir / "sast_node_behavior.csv"}')


if __name__ == '__main__':
    main()