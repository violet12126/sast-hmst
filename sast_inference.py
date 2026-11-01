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


# ═══════════════════════════════════════════════════════════════
# Q_IF — 算子级 IF 质量 (只作评估指标, 不进损失)
# ═══════════════════════════════════════════════════════════════
# 两次只读探针 (scripts/analysis/probe_q_if.py / probe_q_if_norm.py) 的结论:
#   · 指标本身有效: 纯 50 Hz 单音 Q_raw ≈ 0.92, 纯白噪声 ≈ 0.45。
#   · **噪声地板不是 0 而是 ≈0.45** —— 白噪声的 ω̂ 在邻 bin 随机游走,
#     E[(Δ_{2bin}ω̂)²] ≈ 1.4 bin² ⇒ q_vert 期望 exp(−0.7) ≈ 0.5。因此绝对
#     阈值无意义, 必须先扣地板。
#   · 扣地板后四个物理锚点排序自洽 (空转 2xBPF < LOW_FREQ < BPF < 加载 2xBPF),
#     但 **BPF 随负荷仍然倒挂**: 空转 0.46 → 高负荷 0.18 (τ=0.25 口径)。
#     机理: [40,53) 内 41 Hz 流动峰与 50 Hz 锁频线共存, 两簇脊线交界处 ω̂
#     逐 bin 跳变拉低 q_vert —— 指标度量的是"带内谱纯度", 不是"锁频线的 IF
#     质量"。故按事前判据**放弃作为损失项**, 只保留为评估指标。
# 公式与标定细节见 docs §5.1。
Q_IF_SIGMA_V = 1.0
Q_IF_SIGMA_T = 1.5
Q_IF_DT = 20
# 同口径白噪声标定 (hlength=1024, num=1, F=1000, 8 seeds, full-band 口径)
Q_IF_NOISE_FLOOR = {'LOW_FREQ': 0.455, 'BPF': 0.465, '2xBPF': 0.447}
Q_IF_NOISE_FLOOR_DEFAULT = 0.455


def q_if_band(tfr_mag: torch.Tensor, omega: torch.Tensor, freqs: torch.Tensor,
              region, q_noise: Optional[float] = None,
              eps: float = 1e-8) -> torch.Tensor:
    """单频带的 Q_IF, 逐样本一个标量 [B] (不含 presence 门控, 已扣噪声地板)。

    q_vert(f,t) = exp(-(ω[f+1,t]-ω[f-1,t])² / (2·σ_v²))
    q_temp(f,t) = exp(-(ω[f,t+dt]-ω[f,t-dt])² / (2·σ_t²))
    Q_raw(n,t)  = Σ_f |S|²·q_vert·q_temp / Σ_f |S|²      (f ∈ 该带, 权重加权)
    Q_IF        = clamp((Q_raw - q_noise)/(1 - q_noise), 0, 1)

    Args:
        tfr_mag: [B, F, T] 原始 (未增强) 幅值谱
        omega:   [B, F, T] int32, 1-indexed bin, 0 = invalid
        freqs:   [F]
        region:  FreqRegion
    """
    from models.sast_nodes import region_weights
    B, F, T = tfr_mag.shape
    E = tfr_mag ** 2
    valid = omega > 0

    q_v = torch.zeros_like(E)
    if F > 2:
        dv = (omega[:, 2:] - omega[:, :-2]).float()
        ok = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
        q_v[:, 1:F - 1] = torch.exp(-dv ** 2 / (2.0 * Q_IF_SIGMA_V ** 2)) * ok

    q_t = torch.zeros_like(E)
    if T > 2 * Q_IF_DT:
        dw = (omega[:, :, 2 * Q_IF_DT:] - omega[:, :, :-2 * Q_IF_DT]).float()
        ok = valid[:, :, 2 * Q_IF_DT:] & valid[:, :, :-2 * Q_IF_DT]
        q_t[:, :, Q_IF_DT:T - Q_IF_DT] = \
            torch.exp(-dw ** 2 / (2.0 * Q_IF_SIGMA_T ** 2)) * ok

    q = q_v * q_t
    w = region_weights(freqs, [region])[0].view(1, -1, 1)
    sl = slice(Q_IF_DT, T - Q_IF_DT)
    den = (E[:, :, sl] * w).sum(dim=1) + eps
    q_raw = (E[:, :, sl] * w * q[:, :, sl]).sum(dim=1) / den

    qn = q_noise if q_noise is not None else \
        Q_IF_NOISE_FLOOR.get(region.name, Q_IF_NOISE_FLOOR_DEFAULT)
    return ((q_raw - qn) / (1.0 - qn)).clamp(0.0, 1.0).mean(dim=-1)


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

    逐节点四项 (带内, 用 region_weights 加权, 与训练侧 renyi_2d_loss 同加权口径):
      renyi2d        带内 Rényi-2D 熵 (α=2, 越低越集中)
      concentration  带内能量最高的 10% bin 的能量占比 (越高越集中)
      band_spread_hz 带内能量加权频率标准差 (Hz, 越小越窄)
      q_if           算子级 IF 质量 (相位驻点共识度, 已扣白噪声地板 ≈0.45,
                     不含 presence 门控; 定义见 q_if_band 与 docs §5.1)
                     —— 它衡量"这条脊线在原始 STFT 里是否真的存在且平稳",
                     与 Rényi 衡量"增强谱有多集中"是两件事, 可以背离:
                     增强谱可以把底噪压成尖峰 (Rényi 变好) 而 Q_IF 不动。
                     ⚠ 算在 tfr_raw 上, **与模型参数无关**, 故对固定验证集是
                     常数参考 (衡量频带/工况固有脊线质量), 不随 epoch 变化。

    另给 hard / soft 分组均值, 便于看「该挤的挤了、该留的留了」这一分工是否成立
    (hard = BLADE_PASS/BLADE_HARMONIC, soft = HYDRAULIC, 见 SOFT_F_TYPES)。
    ⚠ 3 节点: soft 只有 LOW_FREQ; BPF 带内的两条脊线 (41 Hz 流动峰 0.686 Hz /
      50 Hz 锁定线 0.584 Hz) 本征展宽同属窄带量级, 故整条 BPF 归 hard。

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
                                          'band_spread_hz', 'q_if')}
    n_batches = 0
    have_if = True

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_x = torch.from_numpy(X[start:end]).float().to(device)
        res = model(batch_x, training=False)
        tfr, freqs = res['tfr_enhanced'], res['freqs']      # [B, F, T], [F]
        tfr_raw = res.get('tfr_raw')
        omega = res.get('omega_if')                         # [B, F, T] int32
        if omega is None or tfr_raw is None:
            have_if = False
        B = tfr.shape[0]

        for n, r in enumerate(regions):
            w = region_weights(freqs, [r])[0]
            m = w > 0.0
            band = tfr[:, m, :] * w[m].view(1, -1, 1)          # [B, Fn, T]
            fb = freqs[m]                                       # [Fn]
            if have_if:
                acc['q_if'][n] += q_if_band(tfr_raw, omega, freqs, r
                                            ).mean().item() * B
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
    for key in ('renyi2d', 'concentration', 'band_spread_hz', 'q_if'):
        v = np.asarray(out[key])
        out[f'{key}_hard'] = float(v[hard_idx].mean()) if hard_idx else float('nan')
        out[f'{key}_soft'] = float(v[soft_idx].mean()) if soft_idx else float('nan')
    out['soft_nodes'] = [regions[i].name for i in soft_idx]
    out['hard_nodes'] = [regions[i].name for i in hard_idx]
    return out
