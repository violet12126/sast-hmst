"""
SAST Data Loading
=================
load_dataset - .npz dataset loading + train/val split.
"""
from typing import Dict, Optional

import numpy as np


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
