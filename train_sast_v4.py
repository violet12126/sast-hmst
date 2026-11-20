"""SAST v4 训练 —— 纯物理损失, 与 CNN 完全解耦
=============================================

训练目标 (无 SupCon / 无标签分类项, 见 ``models/sast_v4_losses``):

    L = 1.0·RE_2D(锁频带) + 0.2·(L_anchor + L_ratio) + 0.5·L_soft_ent
        + 0.05·mean(z_w²)                     # mlp_w 未激活 logit 的 L2

(2026-11-13 删去原 L_absence: w 头改为真区间映射后该罚项恒为 0。)

**2026-11-14 两处训练动力学修正** (否则 RE_2D 单调推力会让 w 贴死在区间端点
并把 mlp_w 梯度压到 1e-7, 上一轮 60 epoch 的 w_BPF 全程 0.9800):

  * ``--lambda_z 0.05``: L2 把工作点约束回 |z_w| ≤ 2.5 的高梯度线性区
  * ``--presence_weight 1``: RE_2D 的逐节点推力 ∝ 带内存在度, 一阶平衡点
    2λ_z·z* = −presence_n·∂RE_2D/∂z_n 因而随信噪比/浓度自然分化

只训练 GAT (``MinimalPhysicsGAT``, 39k 参数); 双窗前端 / 重排核都是
无参算子, 随 GAT 输出一起前向。数据切分复用 ``build_split``
(val = 每类 15%, 与下游 CNN 同一划分, SEED=42)。

报告口径 (按定稿, **不看 KNN**):
  * 每 epoch: 验证集总损失 + 各项分解 + 逐节点 mean w
  * 最优 epoch: 验证集**逐类**带内集中度与 Rényi-2D 熵
    (conc 越高 / h2 越低 = 脊线越锐; LOW_FREQ 只作"没被压瘪"的对照)

用法
----
  python train_sast_v4.py                       # 完整 60 epoch
  python train_sast_v4.py --limit 8 --epochs 2  # 冒烟

输出
----
  sast_checkpoints/v4_gat/best.pt
  sast_checkpoints/v4_gat/history.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch                                              # noqa: E402

from models.sast_v4 import SASTv4                         # noqa: E402
from models.sast_v4_losses import (                       # noqa: E402
    sast_v4_loss,
    tf_quality,
)
from models.sast_nodes import PUMP_TURBINE_REGIONS        # noqa: E402
from scripts.data.build_cnn_tfr_dataset import (          # noqa: E402
    CLASS_NAMES,
    N_CLASS,
    build_split,
    dedup_train_pool,
    load_pools,
)

NODE_NAMES = [r.name for r in PUMP_TURBINE_REGIONS]


def batch_indices(pairs, batch: int, rng: np.random.Generator):
    """[(class, idx), ...] -> 打乱后的 batch 下标列表。"""
    order = rng.permutation(len(pairs))
    for s in range(0, len(order), batch):
        yield [pairs[i] for i in order[s:s + batch]]


def run_batches(model, X, pairs, batch, rng, device, opt=None,
                lambda_z: float = 0.002,
                lambda_tear_bpf: float = 5.0,
                lambda_tear_2x: float = 60.0):
    """跑一批数据; opt 为 None 时只评估。返回 (loss_sum, parts_sum, n)。"""
    tot, parts_sum, n = 0.0, {}, 0
    train = opt is not None
    model.train(train)
    for chunk in batch_indices(pairs, batch, rng):
        idx = [i for _, i in chunk]
        x = torch.from_numpy(np.asarray(X[idx], np.float32)).to(device)
        with torch.set_grad_enabled(train):
            T, d = model(x, return_detail=True, detach_frontend=True)
            loss, parts = sast_v4_loss(
                T, d['S_base'], d['w_node'], d['freqs'],
                lambda_tear_bpf=lambda_tear_bpf,
                lambda_tear_2x=lambda_tear_2x,
                z_w=d.get('z_w'), lambda_z=lambda_z)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        b = len(idx)
        tot += float(loss) * b
        for k, v in parts.items():
            parts_sum[k] = parts_sum.get(k, 0.0) + v * b
        n += b
    if n == 0:
        return 0.0, {}, 0
    return tot / n, {k: v / n for k, v in parts_sum.items()}, n


@torch.no_grad()
def per_class_quality(model, X, pairs, device):
    """逐类时频质量 (conc / h2 / if), 每类取该类全部验证样本的均值。"""
    model.eval()
    acc = {}
    for c in range(N_CLASS):
        idx = [i for cc, i in pairs if cc == c]
        if not idx:
            continue
        q_sum, n = {}, 0
        for s in range(0, len(idx), 4):
            x = torch.from_numpy(
                np.asarray(X[idx[s:s + 4]], np.float32)).to(device)
            T, d = model(x, return_detail=True, detach_frontend=True)
            q = tf_quality(T, d['freqs'])
            for k, v in q.items():
                q_sum[k] = q_sum.get(k, 0.0) + v * x.shape[0]
            n += x.shape[0]
        acc[CLASS_NAMES[c]] = {k: v / n for k, v in q_sum.items()}
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / '5_dataset.npz')
    ap.add_argument('--out', type=Path,
                    default=ROOT / 'sast_checkpoints' / 'v4_gat')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0,
                    help='每类只取前 N 条 (冒烟用)')
    ap.add_argument('--no_gat', action='store_true',
                    help='消融: 用固定 C_prior 而非 GAT (此时无可训练参数)')
    ap.add_argument('--lambda_z', type=float, default=0.002,
                    help='mlp_w 未激活 logit 的 L2 护栏权重 (只防爆, 默认 0.002)')
    ap.add_argument('--lambda_tear_bpf', type=float, default=5.0,
                    help='BPF 带撕裂权重 (实测 λ=5 → β*≈0.90-0.95)')
    ap.add_argument('--lambda_tear_2x', type=float, default=60.0,
                    help='2xBPF 带撕裂权重 (实测 λ=60 → 空转~0.35, 加载 0.80-1.0)')
    ap.add_argument('--use_absence_gate', action='store_true',
                    help='启用旧的手工缺席门控 (默认关闭: 固定区间 + bias=0)')
    ap.add_argument('--no_delta_omega', action='store_true',
                    help='不建 Δω 头 (实测 mean|Δω|≤0.10 Hz)')
    ap.add_argument('--no_edges', action='store_true',
                    help='消融: 跳过图消息传递 (等价 3 节点独立 MLP)')
    ap.add_argument('--dedup', action='store_true',
                    help='切分前按内容哈希去重 (训练池内部 + test vs train/val)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, _, _ = load_pools(args.data)
    if args.dedup:
        X, y, nd = dedup_train_pool(X, y)
        print(f'去重: 训练池剔除 {nd} 份重复副本 ({len(y)} 条保留)')
    # 只用训练池; build_split 的 test 分支此处不使用 (传 y 仅为满足签名),
    # 目的是让 train/val 与下游 CNN 的划分逐样本一致 (SEED=42)。
    splits = build_split(y, y)
    tr, va = splits['train'], splits['val']
    if args.limit:
        def cap(ps):
            out, cnt = [], {}
            for c, i in ps:
                if cnt.get(c, 0) < args.limit:
                    out.append((c, i))
                    cnt[c] = cnt.get(c, 0) + 1
            return out
        tr, va = cap(tr), cap(va)

    print(f'device={device}  train={len(tr)}  val={len(va)}  '
          f'节点={NODE_NAMES}')

    model = SASTv4(use_gat=not args.no_gat,
                   gat_kwargs=dict(use_absence_gate=args.use_absence_gate,
                                   no_edges=args.no_edges,
                                   use_delta_omega=not args.no_delta_omega)
                   ).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'可训练参数 = {n_par:,}'
          + ('  (--no_gat: 无参数, 仅前向诊断)' if args.no_gat else ''))

    opt = (torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
           if n_par else None)
    rng = np.random.default_rng(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    best, best_ep, hist = float('inf'), -1, []
    t0 = time.time()

    kw = dict(lambda_z=args.lambda_z,
              lambda_tear_bpf=args.lambda_tear_bpf,
              lambda_tear_2x=args.lambda_tear_2x)
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_parts, _ = run_batches(model, X, tr, args.batch, rng,
                                           device, opt, **kw)
        va_loss, va_parts, _ = run_batches(model, X, va, args.batch, rng,
                                           device, None, **kw)
        w_str = ' '.join(f'{k.split("_", 1)[1]}={va_parts.get(k, 0):.2f}'
                         for k in va_parts if k.startswith('w_'))
        z_str = ' '.join(f'{k.split("_", 1)[1]}={va_parts.get(k, 0):.2f}'
                         for k in va_parts
                         if k.startswith('z_') and k != 'z_l2')
        rec = {'epoch': ep, 'train_loss': tr_loss, 'val_loss': va_loss,
               'val_parts': va_parts, 'lr': args.lr}
        hist.append(rec)
        star = ''
        if va_loss < best:
            best, best_ep = va_loss, ep
            if n_par:
                torch.save({'state_dict': model.state_dict(), 'epoch': ep,
                            'val_loss': best, 'args': vars(args)},
                           args.out / 'best.pt')
            star = '  *'
        print(f'  ep {ep:2d}/{args.epochs}  train={tr_loss:.4f}  '
              f'val={va_loss:.4f}  [re={va_parts.get("re_2d", 0):.3f} '
              f'tearB={va_parts.get("tear_bpf", 0):.4f} '
              f'tear2x={va_parts.get("tear_2x", 0):.4f} '
              f'z2={va_parts.get("z_l2", 0):.4f}]  w({w_str})  z({z_str})  '
              f'({time.time()-t0:.0f}s){star}')
        if n_par and ep - best_ep >= args.patience:
            print(f'  早停: {args.patience} 轮验证损失未下降')
            break

    # 用最优权重报告逐类质量
    if n_par and (args.out / 'best.pt').exists():
        ck = torch.load(args.out / 'best.pt', map_location=device,
                        weights_only=False)
        model.load_state_dict(ck['state_dict'])
    q = per_class_quality(model, X, va, device)
    print(f'\n验证集逐类时频质量 (best epoch {best_ep}, val_loss={best:.4f})')
    hdr = f'{"class":>6}  ' + '  '.join(f'{n:>10}' for n in NODE_NAMES)
    print('  ' + hdr)
    for metric in ('conc', 'h2', 'if'):
        print(f'  {metric}:')
        for cname, d in q.items():
            vals = '  '.join(f'{d.get(f"{metric}_{n}", float("nan")):10.3f}'
                             for n in NODE_NAMES)
            print(f'  {cname:>6}  {vals}')

    with open(args.out / 'history.json', 'w', encoding='utf-8') as fh:
        json.dump({'args': {k: (str(v) if isinstance(v, Path) else v)
                            for k, v in vars(args).items()},
                   'best_epoch': best_ep, 'best_val_loss': best,
                   'history': hist, 'val_quality': q,
                   'elapsed_s': time.time() - t0},
                  fh, ensure_ascii=False, indent=2)
    print(f'\n输出 -> {args.out}')


if __name__ == '__main__':
    main()
