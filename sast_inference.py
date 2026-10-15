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
from models.sast_nodes import physics_band_mask


def _masked_tfr(result):
    """把增强谱限制在 4 个物理带内再做 z_freq 编码 (与训练侧同一处理)。

    与 total_sast_loss 里的掩码同源 (physics_band_mask), 保证验证指标与训练
    优化的输入分布一致 —— 否则训练时编码器看带内、验证时看全带, 指标会失真。
    """
    tfr, freqs = result['tfr_enhanced'], result['freqs']
    return tfr * physics_band_mask(freqs).to(tfr.dtype).view(1, -1, 1)


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
    z = freq_encoder(_masked_tfr(result))      # [1, D]
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
        z = freq_encoder(_masked_tfr(result))     # [B, D]
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
        z = freq_encoder(_masked_tfr(result))          # [B, D]
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


@torch.no_grad()
def evaluate_tf_quality(model: SAST,
                        X: np.ndarray,
                        batch_size: int = 4,
                        device: Optional[torch.device] = None) -> Dict:
    """
    验证集上的**时频质量**指标 (与 scripts/analysis/probe_if_layers.py 同口径).

    WHY: λ_supcon 由 1.0 降到 0.2 后, KNN 精度 (由 z_freq 池化特征定义, 测的是
    *工况可分辨性*) 不再能反映模型好坏 —— 优化重心已回到时频分布本身。本函数
    补上一个直接测「时频表示有多忠实/多清晰」的验证指标。

    逐节点三项 (带内, 用 region_weights 加权, 与训练侧 renyi_2d_loss 同加权口径):
      renyi2d        带内 Rényi-2D 熵 (α=2, 越低越集中)
      concentration  带内能量最高的 10% bin 的能量占比 (越高越集中)
      band_spread_hz 带内能量加权频率标准差 (Hz, 越小越窄)

    另给 hard / soft 分组均值, 便于看「该挤的挤了、该留的留了」这一分工是否成立
    (hard = BLADE_PASS/BLADE_HARMONIC, soft = HYDRAULIC/BLADE_DRIFT, 见 SOFT_F_TYPES)。

    Args:
        X:          [N, T] 验证信号
        batch_size: 批大小

    Returns:
        dict: 逐节点均值 + 'hard' / 'soft' 分组均值
    """
    from models.sast_nodes import (PUMP_TURBINE_REGIONS, region_weights,
                                   SOFT_F_TYPES)
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    regions = PUMP_TURBINE_REGIONS
    n_nodes = len(regions)
    eps = 1e-8
    N = len(X)
    acc = {k: np.zeros(n_nodes) for k in ('renyi2d', 'concentration',
                                          'band_spread_hz')}
    n_batches = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_x = torch.from_numpy(X[start:end]).float().to(device)
        res = model(batch_x, training=False)
        tfr, freqs = res['tfr_enhanced'], res['freqs']      # [B, F, T], [F]
        B = tfr.shape[0]

        for n, r in enumerate(regions):
            w = region_weights(freqs, [r])[0]
            m = w > 0.0
            band = tfr[:, m, :] * w[m].view(1, -1, 1)          # [B, Fn, T]
            fb = freqs[m]                                       # [Fn]
            # Rényi-2D (逐样本 (F,T) 联合分布)
            p = band.clamp(min=eps)
            p = p / p.sum(dim=(1, 2), keepdim=True).clamp(min=eps)
            h = torch.log((p ** 2).sum(dim=(1, 2)).clamp(min=eps)) / (1.0 - 2)
            acc['renyi2d'][n] += h.mean().item() * B
            # 集中度 (时间边际谱的 top-10% bin 占比)
            col = band.sum(dim=1)                               # [B, Fn]
            k = max(1, int(np.ceil(0.1 * col.shape[1])))
            conc = (col.topk(k, dim=1).values.sum(dim=1) /
                    col.sum(dim=1).clamp(min=eps))
            acc['concentration'][n] += conc.mean().item() * B
            # 带内能量加权频率标准差 (逐帧对频率求加权方差, 再对时间取均值)
            wsum = band.sum(dim=1).clamp(min=eps)                # [B, T]
            mu = (fb.view(1, -1, 1) * band).sum(dim=1) / wsum    # [B, T]
            var = ((fb.view(1, -1, 1) - mu.unsqueeze(1)) ** 2 * band
                   ).sum(dim=1) / wsum
            acc['band_spread_hz'][n] += \
                torch.sqrt(var.clamp(min=eps)).mean().item() * B
        n_batches += B

    out: Dict = {}
    for k in acc:
        out[k] = (acc[k] / max(1, n_batches)).tolist()
    out['nodes'] = [r.name for r in regions]
    soft_idx = [i for i, r in enumerate(regions) if r.f_type in SOFT_F_TYPES]
    hard_idx = [i for i, r in enumerate(regions) if r.f_type not in SOFT_F_TYPES]
    for key in ('renyi2d', 'concentration', 'band_spread_hz'):
        v = np.asarray(out[key])
        out[f'{key}_hard'] = float(v[hard_idx].mean()) if hard_idx else float('nan')
        out[f'{key}_soft'] = float(v[soft_idx].mean()) if soft_idx else float('nan')
    out['soft_nodes'] = [regions[i].name for i in soft_idx]
    out['hard_nodes'] = [regions[i].name for i in hard_idx]
    return out
