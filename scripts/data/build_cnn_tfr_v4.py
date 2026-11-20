"""用训练好的 SAST v4 生成下游 CNN 的时频图集
=============================================

对应 ``build_cnn_tfr_dataset.py`` 的 v4 版本。区别只有一处:
TFR 由 ``models.sast_v4.SASTv4`` (+ ``sast_checkpoints/v4_gat2/best.pt``)
产生, 而不是 v3 的 ``sast_inference.infer_sast``。
β 斜坡取 ``models.sast_v4`` 的模块默认 [10, 12) Hz (2026-11-14 定稿); 权重
默认取 ``v4_gat2`` —— 定稿诊断图 (``infer_out_v4_r10_12/``) 用的就是这一对。

数据切分与 v3 版**逐样本一致** (同一个 ``build_split`` / SEED=42), 因此两套
图集可以直接对比, 下游 CNN 的划分也完全对齐。

manifest 里的 ``arm`` 仍写 ``'sast'``, 这样 ``train_cnn_tfr.py`` 无需改动即可
用 ``--arm sast`` 训练; 另加 ``source='sast_v4'`` 列用于溯源。

``--arm wsst`` 复用 v3 版的 ``WsstArm`` (即 ``tfr_common.method_wsst1``,
ssqueezepy 的 ssq_cwt, 零参数), 走**同一个** ``save_tfr_jpg`` 渲染规格,
切分也来自同一个 ``build_split`` —— 这样两臂的图集严格同源, 下游 CNN 的
对比才是干净的。写 manifest 时按 arm 合并: 只替换本次重建的 arm 的行,
其余 arm 的已有行原样保留。

用法
----
  python -m scripts.data.build_cnn_tfr_v4                          # 只出 sast
  python -m scripts.data.build_cnn_tfr_v4 --arm wsst               # 只补 wsst
  python -m scripts.data.build_cnn_tfr_v4 --arm all                # 两臂全出
  python -m scripts.data.build_cnn_tfr_v4 --splits train,val --limit 20

输出
----
  cnn_tfr_dataset_v4/sast/{train,val,test}/*.jpg
  cnn_tfr_dataset_v4/wsst/{train,val,test}/*.jpg
  cnn_tfr_dataset_v4/manifest_full.csv   (两臂合并)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch                                           # noqa: E402

from models.sast_v4 import SASTv4                      # noqa: E402
from scripts.plot.tfr_render import save_tfr_jpg       # noqa: E402
from scripts.data.build_cnn_tfr_dataset import (       # noqa: E402
    CLASS_NAMES,
    N_CLASS,
    build_split,
    load_pools,
    summarize,
)

FS = 1000.0


FIELDS = ['img_path', 'arm', 'split', 'class', 'class_name',
          'pool_index', 'source']
SPLIT_ORDER = {'train': 0, 'val': 1, 'test': 2}


def load_v4(ckpt: Path, device: torch.device) -> SASTv4:
    model = SASTv4(use_gat=True).to(device).eval()
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    sd = ck.get('state_dict', ck)
    model.load_state_dict(sd)
    print(f'v4 权重 {ckpt}  (epoch {ck.get("epoch", "?")}, '
          f'val_loss {ck.get("val_loss", float("nan"))})')
    return model


def sast_tfr_fn(model: SASTv4, device: torch.device):
    """SAST v4 前向 -> (mag[F,T], freqs[F], t[T])。"""
    def fn(x):
        xt = torch.from_numpy(np.asarray(x, np.float32)).unsqueeze(0).to(device)
        T = model(xt)[0].cpu().numpy()
        freqs = np.arange(T.shape[0]) / 2000.0 * FS
        t = np.arange(T.shape[1], dtype=np.float64) / FS
        return T, freqs, t
    return fn


def wsst_tfr_fn():
    """WSST1 -> (mag, freqs, t)。复用 v3 版 WsstArm, 保证两臂同源。"""
    from scripts.data.build_cnn_tfr_dataset import WsstArm
    arm = WsstArm()

    def fn(x):
        mag, freqs, t = arm.tfr(np.asarray(x))
        return np.asarray(mag), np.asarray(freqs).ravel(), np.asarray(t)
    return fn


def write_manifest(man: Path, new_rows, rebuilt_arms, merge: bool):
    """按 arm 合并写 manifest: 只替换 rebuilt_arms 的行, 其余原样保留。"""
    old = []
    if merge and man.exists():
        with open(man, encoding='utf-8') as fh:
            old = list(csv.DictReader(fh))
        print(f'  合并: 已有 {len(old)} 行, 替换 arm ∈ {sorted(rebuilt_arms)}')
    kept = [r for r in old if r.get('arm') not in rebuilt_arms]
    rows = [{k: r.get(k, '') for k in FIELDS} for r in kept + list(new_rows)]
    rows.sort(key=lambda r: (r['arm'], SPLIT_ORDER.get(r['split'], 9),
                             int(r['class']), int(r['pool_index'])))
    with open(man, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    per = {}
    for r in rows:
        per.setdefault(r['arm'], {}).setdefault(r['split'], 0)
        per[r['arm']][r['split']] += 1
    print(f'manifest -> {man}  ({len(rows)} 行)')
    for a, d in per.items():
        print(f'  {a}: ' + '  '.join(f'{k}={d.get(k, 0)}'
                                     for k in ('train', 'val', 'test')))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / '5_dataset.npz')
    ap.add_argument('--sast_ckpt', type=Path,
                    default=ROOT / 'sast_checkpoints' / 'v4_gat2' / 'best.pt')
    ap.add_argument('--out', type=Path, default=ROOT / 'cnn_tfr_dataset_v4')
    ap.add_argument('--arm', default='sast', choices=['sast', 'wsst', 'all'])
    ap.add_argument('--splits', default='train,val,test')
    ap.add_argument('--limit', type=int, default=0,
                    help='每个 split 每类只出 N 张 (冒烟用; 此时不合并 manifest)')
    ap.add_argument('--tag', default='full')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tr, y_tr, X_te, y_te = load_pools(args.data)
    splits = build_split(y_tr, y_te)
    summarize(splits)

    arms = ['sast', 'wsst'] if args.arm == 'all' else [args.arm]
    models = {}
    if 'sast' in arms:
        models['sast'] = load_v4(args.sast_ckpt, device)

    wanted = [s.strip() for s in args.splits.split(',') if s.strip()]
    rows = []
    for arm in arms:
        fn = (sast_tfr_fn(models['sast'], device) if arm == 'sast'
              else wsst_tfr_fn())
        source = 'sast_v4' if arm == 'sast' else 'wsst1'
        print(f'\n[{arm}] 就绪', flush=True)
        for split in wanted:
            items = splits[split]
            if args.limit:
                kept, seen = [], {}
                for c, i in items:
                    if seen.get(c, 0) < args.limit:
                        kept.append((c, i))
                        seen[c] = seen.get(c, 0) + 1
                items = kept
            for k, (c, i) in enumerate(items):
                x = X_tr[i] if split != 'test' else X_te[i]
                mag, freqs, t = fn(x)
                out = args.out / arm / split / f'c{c}_p{i:05d}.jpg'
                save_tfr_jpg(mag, freqs, t, out)
                rows.append({
                    # ⚠ 相对 ``args.out``, 不是 ``args.out.parent``。本脚本把
                    # manifest 写在 args.out 内部 (v3 版写在 out_dir.parent),
                    # 而 train_cnn_tfr.py 用 ``root = manifest.parent`` 拼路径。
                    # 按 out.parent 写会拼成 cnn_tfr_dataset_v4/cnn_tfr_dataset_v4/...
                    'img_path': str(out.relative_to(args.out)).replace('\\', '/'),
                    'arm': arm, 'split': split, 'class': c,
                    'class_name': CLASS_NAMES[c], 'pool_index': i,
                    'source': source,
                })
                if (k + 1) % 50 == 0 or k + 1 == len(items):
                    print(f'  {arm}/{split}: {k + 1}/{len(items)}', flush=True)

    write_manifest(args.out / f'manifest_{args.tag}.csv', rows, set(arms),
                   merge=not args.limit)


if __name__ == '__main__':
    main()
