"""
SAST Data Loading
=================
load_dataset - .npz dataset loading + train/val split.
build_balanced_batch_indices - P×K 类别均衡 batch 索引生成器.
"""
from typing import Dict, Optional

import numpy as np


def build_balanced_batch_indices(y: np.ndarray, p: int, k: int,
                                 n_batches: int, rng=None,
                                 n_classes: Optional[int] = None
                                 ) -> np.ndarray:
    """P×K 类别均衡采样: 返回 [n_batches, p*k_eff] 的 int64 索引矩阵。

    每个 batch 含 k 个类 × 每类 p 个样本。用于替代 `torch.randperm` 均匀切片
    —— 后者在类别极不均衡 (训练集 234/2110/703/4219/469) 且 batch 小时, 最稀少
    的两类在多数 batch 内**根本没有正样本对**, SupCon 拿不到梯度 (B=2 时约
    61.5% 的 batch 对空转/抽水零梯度; 见 docs §14.3)。

    规则:
      - 类内**不放回**抽 p 个 (放回会让同一样本自配成假正样本对);
      - 类池 < p 时退化为放回 (否则该类无法凑够 p 个, batch 大小不齐);
      - batch 内索引最后打乱, 避免同类样本成块排列影响 BatchNorm 类统计。
      - 类池为空的类自动跳过 (k_eff = min(k, 非空类数))。

    Args:
        y:         [N] 标签 (连续 0..n_classes-1)
        p:         每类样本数 (须 >= 2, 否则 batch 内每类仅 1 个 → 无正样本对)
        k:         每 batch 类别数
        n_batches: 生成的 batch 数
        rng:       numpy Generator / RandomState
        n_classes: 类别数 (None = y.max()+1)

    Returns:
        [n_batches, p*k_eff] int64 索引矩阵; 直接用 X_train[idx] 索引即可。
    """
    y = np.asarray(y).ravel().astype(np.int64)
    if rng is None:
        rng = np.random.default_rng(0)
    if n_classes is None:
        n_classes = int(y.max()) + 1 if y.size else 0

    pools = [np.where(y == c)[0] for c in range(n_classes)]
    nonempty = np.array([c for c, pool in enumerate(pools) if pool.size > 0],
                        dtype=np.int64)
    if nonempty.size == 0:
        raise ValueError('build_balanced_batch_indices: y 为空')
    k_eff = int(min(k, nonempty.size))

    out = np.empty((n_batches, p * k_eff), dtype=np.int64)
    for b in range(n_batches):
        cls = rng.choice(nonempty, size=k_eff, replace=False)
        cols = []
        for c in cls:
            pool = pools[int(c)]
            cols.append(rng.choice(pool, size=p,
                                   replace=pool.size < p))
        row = np.concatenate(cols)
        rng.shuffle(row)
        out[b] = row
    return out


def load_dataset(data_path: str,
                 max_samples: Optional[int] = None,
                 max_len: int = 2000,
                 val_split: float = 0.0,
                 seed: int = 42) -> Dict:
    """
    Load and preprocess the .npz dataset.

    Args:
        data_path:   path to .npz file (expects 'train_X', 'train_y')
        max_samples: cap total samples (None = all)
        max_len:     truncate time dimension
        val_split:   fraction for validation (0 = no split)
        seed:        random seed for split

    Returns:
        dict with keys: train_X, train_y, [val_X, val_y], N_total, T, fs (inferred=1000)
    """
    data = np.load(data_path, allow_pickle=True)
    X, y = data['train_X'], data['train_y']
    if X.ndim == 3:
        X = X[:, :, 0]
    y = y.ravel().astype(np.int64)

    if max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    X = X[:, :max_len]

    N_total, T = X.shape

    result = {
        'N_total': N_total,
        'T': T,
        'fs': 1000,
        'n_classes': len(np.unique(y)),
        'class_counts': np.bincount(y),
    }

    if val_split > 0:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(N_total)
        n_val = max(1, int(N_total * val_split))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        result['train_X'] = X[train_idx]
        result['train_y'] = y[train_idx]
        result['val_X'] = X[val_idx]
        result['val_y'] = y[val_idx]
        result['N_train'] = len(train_idx)
        result['N_val'] = len(val_idx)
    else:
        result['train_X'] = X
        result['train_y'] = y

    return result
