"""
SAST Shared Utilities — config, data loading, model factory, visualization
==========================================================================

Extracted from train_sast.py and infer_sast.py to eliminate duplication.
Single source of truth for model creation, data loading, and diagnostic plots.

Usage:
  from sast_utils import SastConfig, load_dataset, create_model, load_checkpoint
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.sast import SAST
from models.sast_losses import FreqEncoder
from models.sast_graph import NODE_NAMES, get_graph_summary

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8


# ═══════════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class SastConfig:
    """Unified SAST configuration — single source of truth for all hyperparams."""

    # ── Signal ──
    fs: int = 1000
    max_len: int = 2000

    # ── Model architecture ──
    d_h: int = 96
    n_heads: int = 4
    n_layers: int = 2
    sigma_min: float = 0.5
    sigma_max: float = 15.0
    msst_num: int = 4
    d_cond: int = 32
    f_type_embed_dim: int = 16
    ppn_temperature: float = 0.08
    prototype_temperature: float = 0.1
    dropout: float = 0.1

    # ── Training ──
    epochs: int = 20
    batch_size: int = 2
    lr: float = 0.001
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 42

    # ── Loss weights ──
    lambda_supcon: float = 1.0
    lambda_entropy: float = 0.1
    lambda_physics: float = 0.5
    lambda_smooth: float = 0.05
    lambda_balance: float = 0.5
    lambda_var: float = 0.5

    # ── Data ──
    data_path: str = '5_dataset.npz'
    val_split: float = 0.15
    max_samples: Optional[int] = None

    # ── Output ──
    save_dir: str = 'sast_checkpoints'
    viz_every: int = 5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'SastConfig':
        """Create config from dict (e.g. checkpoint args), ignoring unknown keys."""
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ═══════════════════════════════════════════════════════════════
# 2. Reproducibility
# ═══════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════
# 3. Data loading
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# 4. Model factory
# ═══════════════════════════════════════════════════════════════

def create_model(config: SastConfig, device: torch.device) -> SAST:
    """Create a fresh SAST model from config."""
    model = SAST(
        fs=config.fs,
        d_h=config.d_h,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        msst_num=config.msst_num,
        d_cond=config.d_cond,
        f_type_embed_dim=config.f_type_embed_dim,
        ppn_temperature=config.ppn_temperature,
        prototype_temperature=config.prototype_temperature,
        dropout=config.dropout,
    ).to(device)
    return model


def create_freq_encoder(n_freq_bins: int,
                        embed_dim: int = 128,
                        device: Optional[torch.device] = None) -> FreqEncoder:
    """Create a FreqEncoder (TFR -> z_freq) for SupCon."""
    enc = FreqEncoder(n_freq_bins=n_freq_bins, embed_dim=embed_dim)
    if device is not None:
        enc = enc.to(device)
    return enc


def get_freq_bins(model: SAST, sample_signal: np.ndarray,
                  device: torch.device) -> int:
    """Run a dummy forward pass to determine F_bins from model output."""
    dummy = torch.from_numpy(sample_signal).float().unsqueeze(0).to(device)
    with torch.no_grad():
        result = model(dummy, return_all=True)
    return result['freqs'].shape[0]


def load_checkpoint(checkpoint_path: str,
                    device: torch.device
                    ) -> Tuple[SAST, FreqEncoder, dict]:
    """
    Load model + freq_encoder from a training checkpoint.

    Returns:
        model, freq_encoder, metadata (dict with config + C_prior)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})

    config = SastConfig.from_dict(ckpt_args)

    model = create_model(config, device)
    model.load_state_dict(ckpt['model_state_dict'])

    # Infer F_bins from encoder state dict
    enc_weight = ckpt['encoder_state_dict']['encoder.0.weight']
    F_bins = enc_weight.shape[1]

    freq_encoder = create_freq_encoder(F_bins, device=device)
    freq_encoder.load_state_dict(ckpt['encoder_state_dict'])

    model.eval()
    freq_encoder.eval()

    C_prior = model.get_C_prior().cpu().numpy()

    print(f"Loaded SAST checkpoint: {checkpoint_path}")
    print(f"  d_h={config.d_h}, n_heads={config.n_heads}, n_layers={config.n_layers}")
    print(f"  fs={config.fs} Hz, F_bins={F_bins}")
    print(f"  C_prior: {C_prior}")
    print(get_graph_summary())

    metadata = {
        'config': config,
        'C_prior': C_prior,
        'F_bins': F_bins,
        'checkpoint_args': ckpt_args,
    }
    return model, freq_encoder, metadata


# ═══════════════════════════════════════════════════════════════
# 5. Inference helpers
# ═══════════════════════════════════════════════════════════════

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

    results = model(x, return_all=True)

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


# ═══════════════════════════════════════════════════════════════
# 6. Visualization (consolidated — single source of truth)
# ═══════════════════════════════════════════════════════════════

def _plot_stft_panel(ax, tfr, t_axis, freqs, freq_max, title, add_colorbar=False):
    """Plot a single STFT/TFR panel with dB scale."""
    db = 10 * np.log10(tfr + 1e-12)
    im = ax.pcolormesh(t_axis, freqs, db, shading='gouraud', cmap='jet',
                       vmin=-30, vmax=10)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title(title)
    if add_colorbar:
        plt.colorbar(im, ax=ax, label='dB')
    return im


def _plot_sigma_panel(ax, sigma_sq, t_axis, freqs, freq_max):
    """Plot squeeze bandwidth panel."""
    im = ax.pcolormesh(t_axis, freqs, sigma_sq, shading='gouraud',
                       cmap='RdYlGn_r', vmin=0, vmax=15)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]')
    ax.set_title('(c) Squeeze Bandwidth σ_sq [bins]\nRed=Broad(Soft), Green=Narrow(Hard)')
    plt.colorbar(im, ax=ax, label='σ_sq')


def _plot_w_i_panel(ax, w_i, t_axis, colors, node_names):
    """Plot w_i(t) per physics node."""
    N = w_i.shape[0]
    for n in range(N):
        ax.plot(t_axis, w_i[n], color=colors[n], lw=1.2,
                label=node_names[n] if n < len(node_names) else f'N{n}')
    ax.axhline(y=0.5, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('w_i')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(d) IF Trust w_i(t)\nw_i→1: Trust & squeeze  |  w_i→0: Conservative')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(alpha=0.3)


def _plot_gate_panel(ax, gate_edge, t_axis):
    """Plot edge gates panel."""
    M = gate_edge.shape[0]
    for m in range(M):
        ax.plot(t_axis, gate_edge[m], lw=0.5, alpha=0.6)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('gate_edge')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'(e) Edge Ratio Gates\n{M} physics edges')
    ax.grid(alpha=0.3)


def _plot_attention_panel(ax, A_ij):
    """Plot attention matrix panel."""
    M = A_ij.shape[0]
    im = ax.imshow(A_ij, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Edge Index')
    ax.set_title(f'(f) Edge Attention A_ij\n{M} edges → diagnostic probe')
    plt.colorbar(im, ax=ax, label='A_ij')


def _plot_enhanced_tfr_panel(ax, tfr_enhanced, t_axis, freqs, freq_max,
                              node_if, w_i, colors, node_names):
    """Plot SAST TFR with node IF traces."""
    db_enh = 10 * np.log10(tfr_enhanced + 1e-12)
    ax.pcolormesh(t_axis, freqs, db_enh, shading='gouraud', cmap='jet',
                  vmin=-30, vmax=10)
    ax.set_ylim(0, freq_max)
    N = w_i.shape[0]
    for n in range(N):
        c_mean = w_i[n].mean()
        name = node_names[n] if n < len(node_names) else f'N{n}'
        ax.plot(t_axis, node_if[n], lw=0.8, alpha=0.5 + 0.5 * c_mean,
                color=colors[n], label=f'{name} (w={c_mean:.2f})')
    ax.set_xlabel('Time [s]')
    ax.set_title('(b) SAST Enhanced TFR + Node IF')
    ax.legend(fontsize=6, loc='upper right')


def plot_tfr(results: Dict[str, np.ndarray],
             save_path: str,
             freq_max: float = 200,
             title: Optional[str] = None):
    """
    3-panel plot: Raw STFT | SAST TFR + Node IF | w_i(t)

    Args:
        results:  from infer_sast(..., return_all=True)
        save_path: output .png path
        freq_max:  y-axis frequency limit
        title:     optional suptitle override
    """
    tfr_raw = results['tfr_raw']
    tfr_enhanced = results['tfr_enhanced']
    w_i = results['w_i']
    node_if = results['node_if']
    freqs = results['freqs']
    t_axis = results['t_axis']

    N = w_i.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(N, 3)))
    # Use physical node names (skip OP if w_i only has phys nodes)
    if N <= 3:
        node_names = NODE_NAMES[1:]  # skip OP
    else:
        node_names = NODE_NAMES

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # (a) Raw STFT
    _plot_stft_panel(axes[0], tfr_raw, t_axis, freqs, freq_max,
                     '(a) Raw STFT\nAll components blurred by finite window')

    # (b) SAST TFR + node IF
    _plot_enhanced_tfr_panel(axes[1], tfr_enhanced, t_axis, freqs, freq_max,
                             node_if, w_i, colors, node_names)

    # (c) w_i(t)
    _plot_w_i_panel(axes[2], w_i, t_axis, colors, node_names)

    suptitle = title or 'SAST: Structure-Aware Synchrosqueezing\nPhysics Graph + Ratio-Gated PPM → Adaptive Squeeze'
    plt.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Plot] Saved: {save_path}')


def plot_full_diagnostics(results: Dict[str, np.ndarray],
                          save_path: str,
                          epoch: Optional[int] = None,
                          freq_max: float = 200):
    """
    6-panel full diagnostic plot.

    Args:
        results:   from infer_sast(..., return_all=True)
        save_path: output .png path
        epoch:     optional epoch number for title
        freq_max:  y-axis frequency limit
    """
    tfr_raw = results['tfr_raw']
    tfr_enhanced = results['tfr_enhanced']
    sigma_sq = results['sigma_sq']
    w_i = results['w_i']
    gate_edge = results['gate_edge']
    A_ij = results['A_ij'].mean(axis=0)  # average over heads
    node_if = results['node_if']
    freqs = results['freqs']
    t_axis = results['t_axis']

    N = w_i.shape[0]
    M_edges = gate_edge.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(N, 3)))
    if N <= 3:
        node_names = NODE_NAMES[1:]
    else:
        node_names = NODE_NAMES

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # (a) Raw STFT
    _plot_stft_panel(axes[0, 0], tfr_raw, t_axis, freqs, freq_max,
                     '(a) Raw STFT (No Squeeze)', add_colorbar=True)

    # (b) SAST TFR + Node IF
    _plot_enhanced_tfr_panel(axes[0, 1], tfr_enhanced, t_axis, freqs, freq_max,
                             node_if, w_i, colors, node_names)

    # (c) σ_sq bandwidth
    _plot_sigma_panel(axes[0, 2], sigma_sq, t_axis, freqs, freq_max)

    # (d) w_i(t)
    _plot_w_i_panel(axes[1, 0], w_i, t_axis, colors, node_names)

    # (e) gate_edge(t)
    _plot_gate_panel(axes[1, 1], gate_edge, t_axis)

    # (f) A_ij attention
    _plot_attention_panel(axes[1, 2], A_ij)

    title = 'SAST v3'
    if epoch is not None:
        title += f' — Epoch {epoch}'
    title += '\nPhysics Graph + Ratio-Gated PPM → IF Trust w_i → Adaptive Squeeze'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Plot] Saved: {save_path}')


# ═══════════════════════════════════════════════════════════════
# 7. Inference summary printer
# ═══════════════════════════════════════════════════════════════

def print_inference_summary(results: Dict[str, np.ndarray],
                            C_prior: Optional[np.ndarray] = None,
                            class_label: Optional[int] = None,
                            sample_idx: Optional[int] = None,
                            pred_class: Optional[int] = None,
                            probs: Optional[np.ndarray] = None):
    """Print a formatted inference summary table."""
    w_i = results['w_i']
    node_if = results['node_if']
    gate_node = results.get('gate_node')

    N = w_i.shape[0]
    if N <= 3:
        node_names = NODE_NAMES[1:]
    else:
        node_names = NODE_NAMES

    print(f"\n{'='*70}")
    header = "SAST Inference Summary"
    if class_label is not None:
        header += f" — Class {class_label}"
    if sample_idx is not None:
        header += f", Sample #{sample_idx}"
    print(header)
    print(f"{'='*70}")

    if pred_class is not None and probs is not None:
        print(f"Predicted: class {pred_class}  |  "
              + "  ".join(f"P({c})={probs[c]:.3f}" for c in range(len(probs))))

    print(f"{'Node':<15s} {'IF(Hz)':>8s} {'C_prior':>7s} {'w_i':>7s} {'gate':>7s}  Interpretation")
    print(f"{'-'*70}")

    for n in range(N):
        f_mean = node_if[n].mean()
        w_mean = w_i[n].mean()
        g_mean = gate_node[n].mean() if gate_node is not None else 0.0
        c_prior_val = C_prior[n] if C_prior is not None else float('nan')

        if w_mean > 0.7:
            interp = 'TRUSTED — aggressive squeeze'
        elif w_mean > 0.5:
            interp = 'Moderate trust'
        elif g_mean > 0.5:
            interp = 'Matched prior, low w_i'
        else:
            interp = 'CONSERVATIVE — soft squeeze'

        name = node_names[n] if n < len(node_names) else f'N{n}'
        c_str = f"{c_prior_val:7.3f}" if not np.isnan(c_prior_val) else "     N/A"
        print(f"  {name:<15s} {f_mean:8.1f} {c_str} {w_mean:7.3f} {g_mean:7.3f}  {interp}")

    w_spread = w_i.mean(axis=1).ptp()
    print(f"\n  w_i spread (max-min across nodes): {w_spread:.3f}")
    if w_spread > 0.2:
        print("  → SAST is actively differentiating frequency components")
    else:
        print("  → w_i not yet differentiated (need more training?)")
