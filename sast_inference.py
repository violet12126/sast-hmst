"""
SAST Inference Helpers
======================
infer_sast, predict_class, compute_class_centroids, evaluate_accuracy.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from models.sast import SAST
from models.sast_losses import FreqEncoder


@torch.no_grad()
def infer_sast(model: SAST, x, return_all: bool = False) -> Dict[str, np.ndarray]:
    """
    Run SAST inference on a single signal.

    Args:
        model:      SAST model (eval mode)
        x:          [T] or [1, T] numpy array or torch tensor
        return_all: if True, return all diagnostics; else just tfr_enhanced

    Returns:
        dict of numpy arrays (always with batch dim squeezed)
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.to(next(model.parameters()).device)

    results = model(x, return_all=True, training=False)

    # Convert to numpy, squeezing batch dim
    out = {}
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[0].cpu().numpy() if v.shape[0] == 1 else v.cpu().numpy()
        else:
            out[k] = v

    if not return_all:
        return {'tfr_enhanced': out['tfr_enhanced']}
    return out


@torch.no_grad()
def predict_class(model: SAST, freq_encoder: FreqEncoder, x,
                  centroids: torch.Tensor) -> Tuple[int, np.ndarray]:
    """
    Predict class via nearest class centroid in z_freq space (KNN).

    Args:
        model:        SAST (eval)
        freq_encoder: FreqEncoder (eval)
        x:            [T] or [1,T] signal
        centroids:    [n_classes, embed_dim] L2-normalized class centroids

    Returns:
        pred_class: int
        sims:       [n_classes] cosine similarities to centroids
    """
    model.eval()
    freq_encoder.eval()

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device)

    result = model(x, return_all=True)
    z = freq_encoder(result['tfr_enhanced'])  # [1, D]
    centroids = centroids.to(device)
    sims = torch.matmul(z, centroids.T)[0]    # [n_classes]
    pred = int(sims.argmax())
    return pred, sims.cpu().numpy()


@torch.no_grad()
def compute_class_centroids(model: SAST, freq_encoder: FreqEncoder,
                            X: np.ndarray, y: np.ndarray,
                            batch_size: int = 4,
                            device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Compute per-class z_freq centroids (L2-normalized mean) from a labeled set.
    用于 KNN 评估 (SupCon 无分类头).

    Returns:
        centroids: [n_classes, embed_dim] L2-normalized
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    freq_encoder.eval()
    n_classes = len(np.unique(y))
    z_sum = None
    counts = torch.zeros(n_classes, device=device)
    N = len(X)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_x = torch.from_numpy(X[start:end]).float().to(device)
        batch_y = torch.from_numpy(y[start:end]).long().to(device)
        result = model(batch_x, return_all=True)
        z = freq_encoder(result['tfr_enhanced'])  # [B, D]
        if z_sum is None:
            z_sum = torch.zeros(n_classes, z.shape[1], device=device)
        for c in range(n_classes):
            mask = (batch_y == c)
            if mask.any():
                z_sum[c] += z[mask].sum(dim=0)
                counts[c] += mask.sum()
    centroids = z_sum / counts.clamp(min=1).unsqueeze(-1)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)
    return centroids


@torch.no_grad()
def evaluate_accuracy(model: SAST, freq_encoder: FreqEncoder,
                      X: np.ndarray, y: np.ndarray,
                      centroids: torch.Tensor,
                      batch_size: int = 4,
                      device: Optional[torch.device] = None) -> Dict:
    """
    Evaluate KNN accuracy: z_freq nearest class centroid.

    Returns:
        dict with 'accuracy', 'n_correct', 'n_total', 'per_class_correct', 'per_class_total'
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    freq_encoder.eval()
    centroids = centroids.to(device)
    n_classes = centroids.shape[0]
    per_class_correct = np.zeros(n_classes, dtype=int)
    per_class_total = np.zeros(n_classes, dtype=int)
    N = len(X)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_x = torch.from_numpy(X[start:end]).float().to(device)
        batch_y = y[start:end]
        result = model(batch_x, return_all=True)
        z = freq_encoder(result['tfr_enhanced'])       # [B, D]
        sims = torch.matmul(z, centroids.T)            # [B, n_classes]
        preds = sims.argmax(dim=-1).cpu().numpy()
        for i, true_label in enumerate(batch_y):
            per_class_total[true_label] += 1
            if preds[i] == true_label:
                per_class_correct[true_label] += 1
    n_correct = per_class_correct.sum()
    n_total = per_class_total.sum()
    accuracy = n_correct / max(1, n_total)
    return {
        'accuracy': accuracy,
        'n_correct': int(n_correct),
        'n_total': int(n_total),
        'per_class_correct': per_class_correct.tolist(),
        'per_class_total': per_class_total.tolist(),
    }
