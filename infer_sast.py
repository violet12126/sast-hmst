"""
SAST 推理 + 可视化脚本
=======================

加载训练好的 SAST 模型, 对信号做自适应同步压缩, 输出增强 TFR + 诊断面板。

用法:
  # 1. 绘制增强 TFR (三面板: STFT + SAST + C_i 曲线)
  python infer_sast.py --checkpoint sast_checkpoints/sast_v2_model.pt \\
      --data 5_dataset.npz --class 1 --output sast_tfr.png

  # 2. 全诊断面板 (六面板)
  python infer_sast.py --checkpoint sast_checkpoints/sast_v2_model.pt \\
      --data 5_dataset.npz --class 1 --mode full --output sast_full.png

  # 3. 批量导出增强 TFR 特征
  python infer_sast.py --checkpoint sast_checkpoints/sast_v2_model.pt \\
      --data 5_dataset.npz --mode batch --output sast_features.npy

  # 4. Python API
  from infer_sast import load_sast, infer_sast
  model = load_sast('sast_v2_model.pt')
  tfr = infer_sast(model, signal)
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent))
from models.sast import SAST, NODE_NAMES
from models.sast_graph import PHYSICS_EDGES, get_graph_summary

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9


# ═══════════════════════════════════════════════════════════════
# 1. Model loading
# ═══════════════════════════════════════════════════════════════

def load_sast(checkpoint_path: str, device: str = 'cuda'):
    """
    从 checkpoint 加载训练好的 SAST 模型。

    Args:
        checkpoint_path: .pt 文件路径
        device: 'cuda' 或 'cpu'

    Returns:
        model: SAST (eval mode)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    args = ckpt.get('args', {})

    model = SAST(
        fs=1000,
        d_h=args.get('d_h', 96),
        n_heads=args.get('n_heads', 4),
        n_layers=args.get('n_layers', 2),
        sigma_min=0.5, sigma_max=15.0,
        msst_num=args.get('msst_num', 3),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Loaded SAST from {checkpoint_path}")
    print(f"  d_h={model.d_h}")
    print(f"  C_prior: {model.get_C_prior().cpu().numpy()}")
    print(get_graph_summary())
    return model


# ═══════════════════════════════════════════════════════════════
# 2. Inference
# ═══════════════════════════════════════════════════════════════

def infer_sast(model, x):
    """SAST 增强 TFR (单信号)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.to(next(model.parameters()).device)

    with torch.no_grad():
        result = model(x, return_all=True)
    return result['tfr_enhanced'][0].cpu().numpy()


@torch.no_grad()
def infer_sast_full(model, x):
    """SAST 完整诊断输出."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.to(next(model.parameters()).device)

    results = model(x, return_all=True)

    # 全部转 numpy
    out = {}
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            if v.shape[0] == 1:
                out[k] = v[0].cpu().numpy()
            else:
                out[k] = v.cpu().numpy()
        else:
            out[k] = v
    return out


# ═══════════════════════════════════════════════════════════════
# 3. Visualization
# ═══════════════════════════════════════════════════════════════

def plot_enhanced_tfr(results, save_path, freq_max=200):
    """
    三面板图: Raw STFT → SAST TFR + node IF → C_i(t)
    """
    tfr_raw = results['tfr_raw']
    tfr_enhanced = results['tfr_enhanced']
    C_i = results['C_i']          # [N, T]
    node_if = results['node_if']   # [N, T]
    freqs = results['freqs']
    t_axis = results['t_axis']

    N = C_i.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # (a) Raw STFT
    ax = axes[0]
    db_raw = 10 * np.log10(tfr_raw + 1e-12)
    ax.pcolormesh(t_axis, freqs, db_raw, shading='gouraud',
                  cmap='jet', vmin=-30, vmax=10)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title('(a) Raw STFT\nAll components blurred by finite window')

    # (b) SAST TFR + node traces
    ax = axes[1]
    db_enh = 10 * np.log10(tfr_enhanced + 1e-12)
    ax.pcolormesh(t_axis, freqs, db_enh, shading='gouraud',
                  cmap='jet', vmin=-30, vmax=10)
    ax.set_ylim(0, freq_max)
    colors = plt.cm.tab10(np.linspace(0, 1, N))
    for n in range(N):
        c_mean = C_i[n].mean()
        ax.plot(t_axis, node_if[n], lw=0.8,
                alpha=0.4 + 0.6 * c_mean,
                color=colors[n])
    ax.set_xlabel('Time [s]')
    ax.set_title('(b) SAST Adaptive Squeeze + Node IF\n'
                 'Bright trace = high C_i (trusted)')

    # (c) C_i(t)
    ax = axes[2]
    for n in range(N):
        ax.plot(t_axis, C_i[n], color=colors[n], lw=1.2,
                label=f'{NODE_NAMES[n]}')
    ax.axhline(y=0.5, color='gray', ls='--', lw=0.8, alpha=0.7)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('C_i')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(c) Compressibility Token C_i(t)\n'
                 'C_i→1 = "trust this component, squeeze hard"\n'
                 'C_i→0 = "uncertain, preserve bandwidth"')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    plt.suptitle('SAST: Structure-Aware Synchrosqueezing\n'
                 'Physics Graph + Ratio-Gated PPM → Adaptive Squeeze',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_full_diagnostics(results, save_path, freq_max=200):
    """六面板完整诊断."""
    tfr_raw = results['tfr_raw']
    tfr_enhanced = results['tfr_enhanced']
    sigma_sq = results['sigma_sq']
    C_i = results['C_i']
    gate_edge = results['gate_edge']
    gate_node = results['gate_node']
    A_ij = results['A_ij'].mean(axis=0)  # [M, T] 多头平均
    node_if = results['node_if']
    freqs = results['freqs']
    t_axis = results['t_axis']

    N = C_i.shape[0]
    M_edges = gate_edge.shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # (a) Raw STFT
    ax = axes[0, 0]
    db = 10 * np.log10(tfr_raw + 1e-12)
    im = ax.pcolormesh(t_axis, freqs, db, shading='gouraud',
                       cmap='jet', vmin=-30, vmax=10)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Frequency [Hz]')
    ax.set_title('(a) Raw STFT (No Squeeze)')
    plt.colorbar(im, ax=ax, label='dB')

    # (b) SAST TFR
    ax = axes[0, 1]
    db_enh = 10 * np.log10(tfr_enhanced + 1e-12)
    im2 = ax.pcolormesh(t_axis, freqs, db_enh, shading='gouraud',
                        cmap='jet', vmin=-30, vmax=10)
    ax.set_ylim(0, freq_max)
    colors = plt.cm.tab10(np.linspace(0, 1, N))
    for n in range(N):
        ax.plot(t_axis, node_if[n], lw=0.5, alpha=0.4, color=colors[n])
    ax.set_xlabel('Time [s]')
    ax.set_title('(b) SAST Enhanced TFR + Node IF')
    plt.colorbar(im2, ax=ax, label='dB')

    # (c) Bandwidth
    ax = axes[0, 2]
    im3 = ax.pcolormesh(t_axis, freqs, sigma_sq, shading='gouraud',
                        cmap='RdYlGn_r', vmin=0, vmax=15)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]')
    ax.set_title('(c) Squeeze Bandwidth σ_sq [bins]')
    plt.colorbar(im3, ax=ax, label='σ_sq')

    # (d) C_i(t)
    ax = axes[1, 0]
    for n in range(N):
        ax.plot(t_axis, C_i[n], color=colors[n], lw=1.2, label=NODE_NAMES[n])
    ax.axhline(y=0.5, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('C_i')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(d) Compressibility Token C_i(t)')
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    # (e) Gate per edge
    ax = axes[1, 1]
    for m in range(M_edges):
        ax.plot(t_axis, gate_edge[m], lw=0.5, alpha=0.6)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('gate_edge')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'(e) Edge Ratio Gates ({M_edges} edges)')
    ax.grid(alpha=0.3)

    # (f) Attention
    ax = axes[1, 2]
    im4 = ax.imshow(A_ij, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Edge Index')
    ax.set_title(f'(f) Edge Attention A_ij ({M_edges} edges)')
    plt.colorbar(im4, ax=ax, label='A_ij')

    plt.suptitle('SAST Inference — Full Diagnostic Panel\n'
                 'Physics Graph + Ratio-Gated PPM → C_i → Adaptive Squeeze',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def export_features(model, X, device='cuda'):
    """批量导出 SAST 增强频域特征."""
    model = model.to(device)
    features = []
    for i in range(len(X)):
        x = torch.from_numpy(X[i]).float().unsqueeze(0).to(device)
        feat = model.get_freq_features(x)
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


# ═══════════════════════════════════════════════════════════════
# 4. CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='SAST Inference + Visualization')
    parser.add_argument('-c', '--checkpoint', required=True,
                        help='Model checkpoint (.pt)')
    parser.add_argument('-d', '--data', default='5_dataset.npz')
    parser.add_argument('--class', dest='target_class', type=int, default=1)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('--mode', choices=['tfr', 'full', 'batch'], default='tfr')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--freq-max', type=float, default=200)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_sast(args.checkpoint, device=device)

    data = np.load(args.data, allow_pickle=True)
    X, y = data['train_X'], data['train_y']
    if X.ndim == 3:
        X = X[:, :, 0]
    y = y.ravel()

    if args.mode == 'batch':
        print(f"Exporting features for {len(X)} samples...")
        feats = export_features(model, X, device=device)
        out_path = args.output or 'sast_features.npy'
        np.save(out_path, feats)
        labels_path = out_path.replace('.npy', '_labels.npy')
        np.save(labels_path, y)
        print(f"  Features: {out_path}  shape={feats.shape}")
        return

    # Select sample
    if args.sample is not None:
        sample_idx = args.sample
        signal = X[sample_idx]
    else:
        idx_c = np.where(y == args.target_class)[0]
        energies = np.sum(X[idx_c] ** 2, axis=1)
        best_local = idx_c[np.argmax(energies)]
        sample_idx = best_local
        signal = X[sample_idx]

    print(f"Sample: class={y[sample_idx]}, idx={sample_idx}, T={len(signal)}")
    print("Running SAST inference...")
    results = infer_sast_full(model, signal)

    # Plot
    out_path = args.output
    if args.mode == 'full':
        out_path = out_path or f'sast_full_class{y[sample_idx]}_s{sample_idx}.png'
        plot_full_diagnostics(results, out_path, freq_max=args.freq_max)
    else:
        out_path = out_path or f'sast_tfr_class{y[sample_idx]}_s{sample_idx}.png'
        plot_enhanced_tfr(results, out_path, freq_max=args.freq_max)

    # Summary
    C_i = results['C_i']
    node_if = results['node_if']
    gate_node = results['gate_node']
    c_prior = model.get_C_prior().cpu().numpy()

    print(f"\n{'='*65}")
    print(f"SAST Inference Summary — Class {y[sample_idx]}, Sample #{sample_idx}")
    print(f"{'='*65}")
    print(f"{'Node':<15s} {'IF(Hz)':>8s} {'C_prior':>7s} {'C_i':>7s} {'gate':>7s}  Interpretation")
    print(f"{'-'*65}")
    for n in range(C_i.shape[0]):
        f_mean = node_if[n].mean()
        c_mean = C_i[n].mean()
        g_mean = gate_node[n].mean()

        if c_mean > 0.7 and g_mean > 0.5:
            interp = 'TRUSTED — aggressive squeeze'
        elif c_mean > 0.5:
            interp = 'Moderate trust'
        elif g_mean > 0.5:
            interp = 'Matched prior, low C_i'
        else:
            interp = 'CONSERVATIVE — soft squeeze'

        print(f"  {NODE_NAMES[n]:<15s} {f_mean:8.1f} {c_prior[n]:7.3f} {c_mean:7.3f} {g_mean:7.3f}  {interp}")

    c_spread = C_i.mean(axis=1).ptp()
    print(f"\n  C_i spread (max-min across nodes): {c_spread:.3f}")
    if c_spread > 0.2:
        print("  → SAST is actively differentiating frequency components")
    else:
        print("  → C_i not yet differentiated (need more training?)")


if __name__ == '__main__':
    main()
