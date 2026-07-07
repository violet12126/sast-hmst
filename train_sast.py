"""
SAST 训练脚本 v2: 匿名全连接图 + Physics Prototype Memory → Compressibility Token

架构:
  Signal → BlindRidgeExtractor → Anonymous Graph
  → PhysicsPrototypeMemory → EdgeConditionedGAT → C_i
  → AdaptiveSqueeze → Physical TFR

损失 (纯自监督, 无任务损失):
  L = L_entropy + λ2·L_physics + λ3·L_smooth

用法:
  python train_sast.py --epochs 5 --lr 0.001 --batch_size 4
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.sast import (SAST, compute_stft, compute_hmst_if,
                         compute_sst_baseline)
from models.sast_losses import total_sast_loss

plt.rcParams['font.family'] = 'sans-serif'


def train_sast_step(model, signals, optimizer, device,
                    lambda_entropy=0.1, lambda_physics=0.5, lambda_smooth=0.01):
    """
    单步训练: Signal → SAST → Loss

    Returns: loss, losses_dict, results
    """
    results = model(signals, return_all=True)

    tfr_enhanced = results['tfr_enhanced']
    A_ij = results['A_ij']
    C_i = results['C_i']
    gate = results['gate']
    edge_feats = results['edge_feats']
    edge_src = results['edge_src']
    edge_dst = results['edge_dst']

    loss, losses_dict = total_sast_loss(
        tfr_enhanced, A_ij, C_i, gate,
        edge_feats, edge_src, edge_dst,
        lambda_entropy=lambda_entropy,
        lambda_physics=lambda_physics,
        lambda_smooth=lambda_smooth,
    )

    # 诊断统计
    losses_dict['C_mean'] = C_i.mean().item()
    losses_dict['gate_mean'] = gate.mean().item()

    high_mask = (gate > 0.5).float()
    low_mask = (gate <= 0.5).float()
    if high_mask.sum() > 0:
        losses_dict['C_matched'] = (C_i * high_mask).sum().item() / high_mask.sum().item()
    else:
        losses_dict['C_matched'] = 0.0
    if low_mask.sum() > 0:
        losses_dict['C_unmatched'] = (C_i * low_mask).sum().item() / low_mask.sum().item()
    else:
        losses_dict['C_unmatched'] = 0.0

    losses_dict['sigma_mean'] = results['sigma_i'].mean().item()

    return loss, losses_dict, results


def visualize_comparison(results, sample_idx, epoch, save_dir, freq_max=200):
    """六面板诊断可视化。"""
    tfr_raw = results['tfr_raw'][sample_idx].detach().cpu().numpy()
    tfr_enhanced = results['tfr_enhanced'][sample_idx].detach().cpu().numpy()
    sigma_sq = results['sigma_sq'][sample_idx].detach().cpu().numpy()
    C_i = results['C_i'][sample_idx].detach().cpu().numpy()
    gate = results['gate'][sample_idx].detach().cpu().numpy()
    A_ij = results['A_ij'][sample_idx].mean(dim=0).detach().cpu().numpy()
    ridge_freq = results['ridge_freq'][sample_idx].detach().cpu().numpy()
    freqs = results['freqs'].detach().cpu().numpy()
    t_frames = results['t_frames'].detach().cpu().numpy()

    K = C_i.shape[0]
    t_if = t_frames[1:-1]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # (a) STFT
    ax = axes[0, 0]
    im = ax.pcolormesh(t_frames, freqs, 10 * np.log10(tfr_raw + 1e-12),
                       shading='gouraud', cmap='inferno')
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title('(a) STFT (No Squeeze)')
    plt.colorbar(im, ax=ax, label='dB')

    # (b) SAST
    ax = axes[0, 1]
    im2 = ax.pcolormesh(t_frames, freqs, 10 * np.log10(tfr_enhanced + 1e-12),
                        shading='gouraud', cmap='inferno')
    ax.set_ylim(0, freq_max)
    for k in range(K):
        ax.plot(t_if, ridge_freq[k], lw=0.5, alpha=0.4)
    ax.set_xlabel('Time [s]')
    ax.set_title('(b) SAST Enhanced TFR + Tracked Ridges')
    plt.colorbar(im2, ax=ax, label='dB')

    # (c) σ_sq 带宽热力图
    ax = axes[0, 2]
    im3 = ax.pcolormesh(t_frames, freqs, sigma_sq,
                        shading='gouraud', cmap='RdYlGn_r', vmin=0, vmax=15)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title('(c) Squeeze Bandwidth σ_sq [bins]\n'
                 'Red=Broad(Soft), Green=Narrow(Hard)')
    plt.colorbar(im3, ax=ax, label='σ_sq [bins]')

    # (d) C_i 时间曲线
    ax = axes[1, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    for k in range(K):
        mean_f = ridge_freq[k].mean()
        ax.plot(t_if, C_i[k], color=colors[k], lw=1.2,
                label=f'R{k} (~{mean_f:.1f} Hz)')
    ax.axhline(y=0.5, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('C_i')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(d) Compressibility Token C_i(t)\n'
                 'C_i→1: Aggressive squeeze  |  C_i→0: Conservative')
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    # (e) gate 原型匹配
    ax = axes[1, 1]
    for k in range(K):
        ax.plot(t_if, gate[k], color=colors[k], lw=1.2, label=f'R{k}')
    ax.axhline(y=0.5, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('gate')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(e) Prototype Match Gate(t)\n'
                 'gate→1: Known prototype  |  gate→0: Anonymous')
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    # (f) A_ij 因果推理探针
    ax = axes[1, 2]
    im4 = ax.imshow(A_ij, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Edge Index (i→j)')
    ax.set_title('(f) Edge Attention A_ij (Diagnostic Probe)\n'
                 'Bright = strong message passing')
    plt.colorbar(im4, ax=ax, label='A_ij')

    plt.suptitle(f'SAST v2 — Epoch {epoch}\n'
                 'Anonymous Graph + Physics Prototype Memory → Compressibility Token',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fn = Path(save_dir) / f'sast_v2_epoch{epoch:03d}.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Viz] Saved: {fn}')


def main():
    parser = argparse.ArgumentParser(description='Train SAST v2')
    parser.add_argument('--data', type=str, default='5_dataset.npz')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--d_h', type=int, default=96)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--K_ridges', type=int, default=6)
    parser.add_argument('--lambda_entropy', type=float, default=0.1)
    parser.add_argument('--lambda_physics', type=float, default=0.5)
    parser.add_argument('--lambda_smooth', type=float, default=0.01)
    parser.add_argument('--n_fft', type=int, default=512)
    parser.add_argument('--hop_length', type=int, default=128)
    parser.add_argument('--hmst_order', type=int, default=2,
                        help='HMST IF 估计阶数 (1=标准一阶, 2=二阶对线性调频无偏)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='sast_checkpoints')
    parser.add_argument('--viz_every', type=int, default=2)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # ── 加载数据 ──
    print(f"\nLoading data from {args.data}...")
    data = np.load(args.data, allow_pickle=True)
    X, y = data['train_X'], data['train_y']
    if X.ndim == 3:
        X = X[:, :, 0]

    fs = 1000
    N, T_total = X.shape
    print(f"  N={N}, T={T_total}, fs={fs} Hz")
    print(f"  Classes: {np.unique(y)}")

    max_len = min(T_total, 2000)
    X = X[:, :max_len]
    X_tensor = torch.from_numpy(X).float()

    # ── 创建模型 ──
    print(f"\nCreating SAST v2 model...")
    print(f"  K_ridges={args.K_ridges} (anonymous, fully-connected)")
    print(f"  d_h={args.d_h}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"  HMST IF order: {args.hmst_order}")

    model = SAST(
        fs=fs, n_fft=args.n_fft, hop_length=args.hop_length,
        d_h=args.d_h, n_heads=args.n_heads, n_layers=args.n_layers,
        K_ridges=args.K_ridges,
        sigma_min=0.5, sigma_max=15.0,
        hmst_order=args.hmst_order,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Prototypes: {model.ppm.M_proto}")
    print(f"  Anonymous ridges: {model.K_ridges}")
    M = model.K_ridges * (model.K_ridges - 1)
    print(f"  Graph edges: {M} (fully connected)")

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 固定可视化样本 ──
    viz_sample = X_tensor[0:1].to(device)

    # ── SST 基线 ──
    print(f"\nComputing SST baseline (ssqueezepy)...")
    try:
        Tx, Wx, ssq_freqs = compute_sst_baseline(
            X[0], fs, n_fft=args.n_fft, hop_length=args.hop_length
        )
        print(f"  SST shape: {Tx.shape}, freq: [{ssq_freqs[0]:.1f}, {ssq_freqs[-1]:.1f}] Hz")
    except Exception as e:
        print(f"  WARNING: ssqueezepy SST failed: {e}")

    # ── 训练循环 ──
    print(f"\n{'='*60}")
    print(f"Training SAST v2 for {args.epochs} epochs")
    print(f"  λ_e={args.lambda_entropy} λ_p={args.lambda_physics} λ_s={args.lambda_smooth}")
    print(f"{'='*60}")

    model.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        n_batches = max(1, N // args.batch_size)
        epoch_losses = {}

        for bi in range(n_batches):
            idx = perm[bi * args.batch_size:(bi + 1) * args.batch_size]
            batch_x = X_tensor[idx].to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, losses_dict, _ = train_sast_step(
                model, batch_x, optimizer, device,
                lambda_entropy=args.lambda_entropy,
                lambda_physics=args.lambda_physics,
                lambda_smooth=args.lambda_smooth,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in losses_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

        scheduler.step()

        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)

        print(f"Epoch {epoch:3d}: "
              f"T={epoch_losses.get('total', 0):.4f} "
              f"E={epoch_losses.get('entropy', 0):.4f} "
              f"P={epoch_losses.get('physics', 0):.4f} "
              f"S={epoch_losses.get('smooth', 0):.4f} "
              f"Cmean={epoch_losses.get('C_mean', 0):.3f} "
              f"gate={epoch_losses.get('gate_mean', 0):.3f} "
              f"Cm={epoch_losses.get('C_matched', 0):.3f} "
              f"Cu={epoch_losses.get('C_unmatched', 0):.3f} "
              f"sigma={epoch_losses.get('sigma_mean', 0):.2f}")

        if epoch % args.viz_every == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                _, _, viz_results = train_sast_step(
                    model, viz_sample, optimizer, device,
                    lambda_entropy=args.lambda_entropy,
                    lambda_physics=args.lambda_physics,
                    lambda_smooth=args.lambda_smooth,
                )
            visualize_comparison(viz_results, 0, epoch, save_dir)

    # ── 保存 ──
    ckpt_path = save_dir / 'sast_v2_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'prototype_config': {
            'prototypes': model.ppm.prototypes,
            'temperature': model.ppm.temperature,
        },
    }, ckpt_path)
    print(f"\nModel saved to {ckpt_path}")
    print("Training complete.")


if __name__ == '__main__':
    main()
