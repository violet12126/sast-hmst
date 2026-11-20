"""
第 6 章 CNN 输入时频图数据集生成器
====================================

两臂 (arm) 同源、同切分、同渲染规格, 唯一差别是时频表示:

  sast : gate3node_full 检查点前向 (squeeze_scope='full') -> tfr_calibrated
  wsst : method_wsst1 (ssqueezepy ssq_cwt), 零参数

切分口径见 papers/毕业论文_第6章_CNN工况识别验证.md 表 6-1:
  训练池按类 8.5:1.5 划分, 单类上限 训练 600 / 验证 120, seed = 42;
  测试集取自测试池, 单类上限 200。
实测得到 训练 2396 / 验证 450 / 测试 752, 与表 6-1 逐类一致。

渲染规格 (与 §6.2.2 一致):
  875x656 像素、全幅 axes、jet、vmin=0、vmax=逐图峰值、
  频率轴 0-200 Hz 线性映射到 656 行、JPEG 质量 95。

用法
----
  # 预览: 每类 2 张 (训练集), 两臂都出, 另加一张拼版总览
  python scripts/data/build_cnn_tfr_dataset.py --preview

  # 全量: 两臂 x 三切分
  python scripts/data/build_cnn_tfr_dataset.py --arm all --splits train,val,test
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

import matplotlib                                     # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # noqa: E402

from scripts.plot.tfr_render import (                  # noqa: E402
    F_MAX, IMG_W, IMG_H, save_tfr_jpg,
)

FS = 1000.0

CLASS_NAMES = ['空转', '低负荷', '中负荷', '高负荷', '抽水']
N_CLASS = 5

CAP_TRAIN, CAP_VAL, CAP_TEST = 600, 120, 200
VAL_FRAC = 0.15
SEED = 42

DEFAULT_DATA = ROOT / '5_dataset.npz'
DEFAULT_CKPT = ROOT / 'sast_checkpoints' / 'gate3node_full' / 'sast_v3_e020.pt'


# ═══════════════════════════════════════════════════════════════
# 1. 数据切分 (表 6-1)
# ═══════════════════════════════════════════════════════════════
def load_pools(data_path: Path):
    data = np.load(data_path, allow_pickle=True)
    X_tr, y_tr = data['train_X'], data['train_y'].ravel().astype(np.int64)
    X_te, y_te = data['test_X'], data['test_y'].ravel().astype(np.int64)
    if X_tr.ndim == 3:
        X_tr = X_tr[:, :, 0]
    if X_te.ndim == 3:
        X_te = X_te[:, :, 0]
    return X_tr, y_tr, X_te, y_te


def dedup_train_pool(X_tr, y_tr):
    """按内容哈希剔除训练池内部的重复副本 (保留首次出现)。

    数据集普查显示训练池内部有 198 份重复副本; 它们会在 build_split 里随机
    落到 train 与 val 两侧, 造成 train/val 泄漏。返回 (X, y, 剔除数)。
    """
    import hashlib
    seen, keep = set(), []
    for i in range(len(y_tr)):
        h = hashlib.md5(np.ascontiguousarray(
            X_tr[i], dtype=np.float32).tobytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        keep.append(i)
    keep = np.asarray(keep)
    return X_tr[keep], y_tr[keep], len(y_tr) - len(keep)


def build_split(y_tr: np.ndarray, y_te: np.ndarray, seed: int = SEED):
    """返回 {split: [(class, pool_index), ...]}，pool_index 指训练池/测试池内下标。"""
    rng = np.random.default_rng(seed)
    out = {'train': [], 'val': [], 'test': []}

    for c in range(N_CLASS):
        idx = np.where(y_tr == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = min(CAP_TRAIN, int(round(n * (1.0 - VAL_FRAC))))
        n_val = min(CAP_VAL, n - n_train)
        out['train'].extend((c, int(i)) for i in idx[:n_train])
        out['val'].extend((c, int(i)) for i in idx[n_train:n_train + n_val])

        tidx = np.where(y_te == c)[0]
        rng.shuffle(tidx)
        out['test'].extend((c, int(i)) for i in tidx[:CAP_TEST])

    for k in out:
        out[k].sort(key=lambda t: (t[0], t[1]))
    return out


def summarize(splits):
    print('切分统计 (应为 训练 2396 / 验证 450 / 测试 752):')
    for k in ('train', 'val', 'test'):
        per = np.bincount([c for c, _ in splits[k]], minlength=N_CLASS)
        print(f'  {k:5s} n={len(splits[k]):5d}  per-class={per.tolist()}')


# ═══════════════════════════════════════════════════════════════
# 2. 渲染 (走 scripts/plot/tfr_render.py 的唯一口径)
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# 3. 两臂
# ═══════════════════════════════════════════════════════════════
class SastArm:
    name = 'sast'

    def __init__(self, ckpt: Path, device: str = 'cuda',
                 scope: str | None = None):
        import torch
        from sast_model import load_checkpoint
        from sast_inference import infer_sast
        self._torch = torch
        self._infer = infer_sast
        dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model, _, self.meta = load_checkpoint(str(ckpt), dev)
        if scope is not None:
            self.model.squeeze_scope = scope
        self.device = dev

    def tfr(self, x: np.ndarray):
        r = self._infer(self.model, x, return_all=True)
        # tfr_hybrid = 物理带内 SAST 挤压 + 带外 MSST, 两份都是未标定重排输出,
        # 幅度约定一致。不再用 tfr_calibrated (它按 γ≈6.11 把带内噪声底压暗,
        # 与带外对不上 —— 实测见 models/sast.py 中 tfr_hybrid 处的注释)。
        return r['tfr_hybrid'], r['freqs'], r['t_axis']


class WsstArm:
    name = 'wsst'

    def __init__(self):
        from scripts.plot.tfr_common import method_wsst1
        self._fn = method_wsst1

    def tfr(self, x: np.ndarray):
        r = self._fn(x.astype(np.float64))
        return r['mag'], r['freqs'], r['t']


# ═══════════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════════
def make_arm(name: str, ckpt: Path):
    return SastArm(ckpt) if name == 'sast' else WsstArm()


def run(arm_names, splits_wanted, splits, pools, out_dir: Path,
        per_class: int | None, tag: str, scope: str | None = None):
    X_tr, _, X_te, _ = pools
    rows = []
    for arm_name in arm_names:
        arm = SastArm(DEFAULT_CKPT, scope=scope) if arm_name == 'sast' \
            else WsstArm()
        print(f'\n[{arm_name}] 就绪', flush=True)

        for split in splits_wanted:
            items = splits[split]
            if per_class is not None:
                kept, seen = [], {}
                for c, i in items:
                    if seen.get(c, 0) < per_class:
                        kept.append((c, i))
                        seen[c] = seen.get(c, 0) + 1
                items = kept

            for k, (c, i) in enumerate(items):
                x = X_tr[i] if split != 'test' else X_te[i]
                mag, freqs, t = arm.tfr(x)
                out = out_dir / arm_name / split / f'c{c}_p{i:05d}.jpg'
                save_tfr_jpg(mag, freqs, t, out)
                rows.append({
                    'img_path': str(out.relative_to(out_dir.parent)).replace('\\', '/'),
                    'arm': arm_name, 'split': split, 'class': c,
                    'class_name': CLASS_NAMES[c], 'pool_index': i,
                })
                if (k + 1) % 25 == 0 or k + 1 == len(items):
                    print(f'  {arm_name}/{split}: {k + 1}/{len(items)}', flush=True)

    man = out_dir.parent / f'manifest_{tag}.csv'
    with open(man, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nmanifest -> {man}  ({len(rows)} 行)')


def contact_sheet(out_dir: Path, dst: Path):
    """两臂 x 五类的拼版总览, 每格取该类第一张预览图。"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(N_CLASS, 2, figsize=(9.0, 4.6 * N_CLASS), dpi=100)
    for r, c in enumerate(range(N_CLASS)):
        for col, arm in enumerate(('sast', 'wsst')):
            cands = sorted((out_dir / arm / 'train').glob(f'c{c}_p*.jpg'))
            ax = axes[r, col]
            ax.set_axis_off()
            if cands:
                ax.imshow(plt.imread(cands[0]))
            ax.set_title(f'{arm.upper()} — {c} {CLASS_NAMES[c]}'
                         + (f'  ({cands[0].name})' if cands else ''),
                         fontsize=10, pad=3)
    fig.tight_layout()
    fig.savefig(dst, dpi=100)
    plt.close(fig)
    print(f'总览图 -> {dst}')


def main():
    global DEFAULT_CKPT
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    ap.add_argument('--out', type=Path, default=ROOT / 'cnn_tfr_dataset')
    ap.add_argument('--arm', default='sast', choices=['sast', 'wsst', 'all'])
    ap.add_argument('--splits', default='train,val,test')
    ap.add_argument('--preview', action='store_true',
                    help='每类只出 2 张训练集样本, 输出到 cnn_tfr_preview/')
    ap.add_argument('--preview-per-class', type=int, default=2)
    ap.add_argument('--scope', default='msst',
                    choices=['msst', 'full', 'bands'],
                    help="SAST 挤压作用域; 图集默认 'msst' "
                         "(物理带内 SAST, 带外与 MSST 一致)")
    ap.add_argument('--tag', default='full',
                    help='manifest 文件名后缀 -> manifest_{tag}.csv '
                         '(默认 full, 会覆盖同名文件)')
    args = ap.parse_args()

    DEFAULT_CKPT = args.ckpt

    X_tr, y_tr, X_te, y_te = load_pools(args.data)
    print(f'训练池 {len(y_tr)}  测试池 {len(y_te)}')
    print(f'  训练池分布 {np.bincount(y_tr, minlength=N_CLASS).tolist()}')
    print(f'  测试池分布 {np.bincount(y_te, minlength=N_CLASS).tolist()}')
    splits = build_split(y_tr, y_te)
    summarize(splits)

    if args.preview:
        out_dir = ROOT / 'cnn_tfr_preview'
        arms = ['sast', 'wsst'] if args.arm == 'all' else [args.arm]
        if args.arm == 'sast':
            arms = ['sast', 'wsst']          # 预览默认两臂都出, 便于同时校验两条渲染路径
        run(arms, ['train'], splits, (X_tr, y_tr, X_te, y_te),
            out_dir, args.preview_per_class, 'preview', scope=args.scope)
        contact_sheet(out_dir, ROOT / 'cnn_tfr_preview' / 'contact_sheet.png')
    else:
        arms = ['sast', 'wsst'] if args.arm == 'all' else [args.arm]
        sp = [s.strip() for s in args.splits.split(',') if s.strip()]
        run(arms, sp, splits, (X_tr, y_tr, X_te, y_te),
            args.out, None, args.tag, scope=args.scope)


if __name__ == '__main__':
    main()