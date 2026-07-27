"""
SAST v2 训练脚本: 物理图 + 比值门控 PPM + GAT → Compressibility Token

架构:
  Signal → MSST (CPU, numpy) → Node Features (frequency-region aggregation)
  → Physics Graph (fixed topology, typed edges)
  → PhysicsPrototypeMemory (ratio-gated) → EdgeConditionedGAT → C_i
  → AdaptiveSqueeze → Physical TFR → TFRClassifier → class prediction

损失 (五分量, 设计文档 §5.7):
  L = L_task + 0.1·L_entropy + 0.5·L_physics + 0.05·L_smooth + 0.01·L_balance

用法:
  python train_sast.py --epochs 20 --lr 0.001 --batch_size 2 --device cuda
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
import time

sys.path.insert(0, str(Path(__file__).parent))

from models.sast import SAST, NODE_NAMES
from models.sast_losses import total_sast_loss, TFRClassifier
from models.sast_graph import get_graph_summary

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8


def train_step(model, classifier, signals, labels, optimizer, device,
               lambda_task=1.0, lambda_entropy=0.1, lambda_physics=0.5,
               lambda_smooth=0.05, lambda_balance=0.01):
    """
    单步训练: Signal → SAST → Classifier → 5-component Loss

    Returns: loss, losses_dict, results
    """
    # SAST forward
    results = model(signals, return_all=True)

    tfr_raw = results['tfr_raw']
    tfr_enhanced = results['tfr_enhanced']
    C_i = results['C_i']
    A_ij = results['A_ij']
    gate_edge = results['gate_edge']
    edge_feats = results['edge_feats']
    edge_src = results['edge_src']
    edge_dst = results['edge_dst']
    node_if = results['node_if']
    freqs = results['freqs']

    # 5-component loss
    loss, losses_dict = total_sast_loss(
        tfr_raw, tfr_enhanced, C_i, labels, classifier,
        edge_feats, gate_edge, edge_src, edge_dst,
        node_if, freqs, A_ij,
        lambda_task=lambda_task,
        lambda_entropy=lambda_entropy,
        lambda_physics=lambda_physics,
        lambda_smooth=lambda_smooth,
        lambda_balance=lambda_balance,
    )

    # 额外诊断
    losses_dict['gate_mean'] = gate_edge.mean().item()
    C_per_node = C_i.mean(dim=-1)  # [B, N]
    losses_dict['C_matched'] = C_per_node.mean().item()
    losses_dict['C_prior_mean'] = model.get_C_prior().mean().item()

    return loss, losses_dict, results


def visualize_diagnostics(results, sample_idx, epoch, save_dir, freq_max=200):
    """
    六面板诊断可视化:
      (a) Raw STFT  (b) SAST TFR + node IF traces
      (c) σ_sq bandwidth  (d) C_i(t) per physics node
      (e) gate_edge(t) per edge  (f) A_ij attention
    """
    tfr_raw = results['tfr_raw'][sample_idx].detach().cpu().numpy()
    tfr_enhanced = results['tfr_enhanced'][sample_idx].detach().cpu().numpy()
    sigma_sq = results['sigma_sq'][sample_idx].detach().cpu().numpy()
    C_i = results['C_i'][sample_idx].detach().cpu().numpy()  # [N, T]
    gate_edge = results['gate_edge'][sample_idx].detach().cpu().numpy()  # [M, T]
    gate_node = results['gate_node'][sample_idx].detach().cpu().numpy()  # [N, T]
    A_ij = results['A_ij'][sample_idx].mean(dim=0).detach().cpu().numpy()  # [M, T]
    node_if = results['node_if'][sample_idx].detach().cpu().numpy()  # [N, T]
    freqs = results['freqs'].detach().cpu().numpy()
    t_axis = results['t_axis'].detach().cpu().numpy()

    N = C_i.shape[0]
    M_edges = gate_edge.shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # (a) Raw STFT
    ax = axes[0, 0]
    db = 10 * np.log10(tfr_raw + 1e-12)
    ax.pcolormesh(t_axis, freqs, db, shading='gouraud', cmap='jet',
                  vmin=-30, vmax=10)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title('(a) Raw STFT (No Squeeze)')

    # (b) SAST TFR + node IF traces
    ax = axes[0, 1]
    db_enh = 10 * np.log10(tfr_enhanced + 1e-12)
    ax.pcolormesh(t_axis, freqs, db_enh, shading='gouraud', cmap='jet',
                  vmin=-30, vmax=10)
    ax.set_ylim(0, freq_max)
    colors = plt.cm.tab10(np.linspace(0, 1, N))
    for n in range(N):
        c_mean = C_i[n].mean()
        ax.plot(t_axis, node_if[n], lw=0.8, alpha=0.5 + 0.5 * c_mean,
                color=colors[n], label=f'{NODE_NAMES[n]} (C={c_mean:.2f})')
    ax.set_xlabel('Time [s]')
    ax.set_title('(b) SAST Enhanced TFR + Node IF')
    ax.legend(fontsize=6, loc='upper right')

    # (c) σ_sq bandwidth
    ax = axes[0, 2]
    im3 = ax.pcolormesh(t_axis, freqs, sigma_sq, shading='gouraud',
                        cmap='RdYlGn_r', vmin=0, vmax=15)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]')
    ax.set_title("(c) Squeeze Bandwidth σ_sq [bins]\nRed=Broad(Soft), Green=Narrow(Hard)")
    plt.colorbar(im3, ax=ax, label='σ_sq')

    # (d) C_i(t) per physics node
    ax = axes[1, 0]
    for n in range(N):
        ax.plot(t_axis, C_i[n], color=colors[n], lw=1.2,
                label=NODE_NAMES[n])
    ax.axhline(y=0.5, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('C_i')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(d) Compressibility Token C_i(t)\nC_i→1: Trust & squeeze  |  C_i→0: Conservative')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(alpha=0.3)

    # (e) gate_edge(t)
    ax = axes[1, 1]
    for m in range(M_edges):
        ax.plot(t_axis, gate_edge[m], lw=0.5, alpha=0.6)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('gate_edge')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'(e) Edge Ratio Gates\n{M_edges} physics edges')
    ax.grid(alpha=0.3)

    # (f) A_ij attention matrix
    ax = axes[1, 2]
    im4 = ax.imshow(A_ij, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Edge Index')
    ax.set_title(f'(f) Edge Attention A_ij\n{M_edges} edges → diagnostic probe')
    plt.colorbar(im4, ax=ax, label='A_ij')

    plt.suptitle(f'SAST v2 — Epoch {epoch}\n'
                 'Physics Graph + Ratio-Gated PPM → Compressibility Token',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fn = Path(save_dir) / f'sast_v2_epoch{epoch:03d}.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Viz] Saved: {fn}')


def main():
    parser = argparse.ArgumentParser(description='Train SAST v2')
    parser.add_argument('--data', type=str, default='5_dataset.npz')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--d_h', type=int, default=96)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--lambda_task', type=float, default=1.0)
    parser.add_argument('--lambda_entropy', type=float, default=0.1)
    parser.add_argument('--lambda_physics', type=float, default=0.5)
    parser.add_argument('--lambda_smooth', type=float, default=0.05)
    parser.add_argument('--lambda_balance', type=float, default=0.01)
    parser.add_argument('--msst_num', type=int, default=3,
                        help='MSST 迭代次数')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='sast_checkpoints')
    parser.add_argument('--viz_every', type=int, default=5)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit dataset size (for quick testing)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(get_graph_summary())

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # ── 加载数据 ──
    print(f"\nLoading data from {args.data}...")
    data = np.load(args.data, allow_pickle=True)
    X, y = data['train_X'], data['train_y']
    if X.ndim == 3:
        X = X[:, :, 0]
    y = y.ravel().astype(np.int64)

    fs = 1000
    N_total, T_total = X.shape
    if args.max_samples:
        X = X[:args.max_samples]
        y = y[:args.max_samples]
        N_total = len(X)
    print(f"  N={N_total}, T={T_total}, fs={fs} Hz")
    print(f"  Classes: {np.unique(y)}, counts: {np.bincount(y)}")

    max_len = min(T_total, 2000)
    X = X[:, :max_len]

    # ── 创建模型 ──
    print(f"\nCreating SAST v2 model...")
    print(f"  d_h={args.d_h}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"  MSST num={args.msst_num}")

    model = SAST(
        fs=fs, d_h=args.d_h, n_heads=args.n_heads, n_layers=args.n_layers,
        sigma_min=0.5, sigma_max=15.0,
        msst_num=args.msst_num,
    ).to(device)

    # 用 dummy 数据初始化 squeeze (获取 F_bins)
    dummy_x = torch.from_numpy(X[0]).float().unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(dummy_x, return_all=True)

    # 分类器
    F_bins = len(model.node_extractor(X[0]).freqs)
    print(f"  F_bins={F_bins}")
    classifier = TFRClassifier(n_freq_bins=F_bins, n_classes=5).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cls_params = sum(p.numel() for p in classifier.parameters())
    print(f"  SAST params: {total_params:,} (trainable: {trainable_params:,})")
    print(f"  Classifier params: {cls_params:,}")

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # ── 固定可视化样本 ──
    viz_idx = 0
    viz_signal = torch.from_numpy(X[viz_idx]).float().unsqueeze(0).to(device)
    viz_label = torch.tensor([y[viz_idx]], device=device)

    print(f"\n{'='*60}")
    print(f"Training SAST v2 for {args.epochs} epochs")
    print(f"  λ_task={args.lambda_task} λ_e={args.lambda_entropy}")
    print(f"  λ_p={args.lambda_physics} λ_s={args.lambda_smooth}")
    print(f"  λ_b={args.lambda_balance}")
    print(f"{'='*60}")

    model.train()
    classifier.train()

    for epoch in range(args.epochs):
        perm = torch.randperm(N_total)
        n_batches = max(1, N_total // args.batch_size)
        epoch_losses = {}
        epoch_start = time.time()

        for bi in range(n_batches):
            idx = perm[bi * args.batch_size:(bi + 1) * args.batch_size]
            batch_x = torch.from_numpy(X[idx]).float().to(device)
            batch_y = torch.from_numpy(y[idx]).long().to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, losses_dict, _ = train_step(
                model, classifier, batch_x, batch_y, optimizer, device,
                lambda_task=args.lambda_task,
                lambda_entropy=args.lambda_entropy,
                lambda_physics=args.lambda_physics,
                lambda_smooth=args.lambda_smooth,
                lambda_balance=args.lambda_balance,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            for k, v in losses_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

        scheduler.step()
        epoch_time = time.time() - epoch_start

        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)

        print(f"Epoch {epoch:3d} ({epoch_time:.0f}s): "
              f"T={epoch_losses.get('total', 0):.4f} "
              f"task={epoch_losses.get('task', 0):.3f} "
              f"ent={epoch_losses.get('entropy', 0):.3f} "
              f"phy={epoch_losses.get('physics', 0):.4f} "
              f"smo={epoch_losses.get('smooth', 0):.4f} "
              f"bal={epoch_losses.get('balance', 0):.4f} "
              f"Cm={epoch_losses.get('C_mean', 0):.3f} "
              f"Cs={epoch_losses.get('C_spread', 0):.3f} "
              f"g={epoch_losses.get('gate_mean', 0):.3f}")

        if epoch % args.viz_every == 0 or epoch == args.epochs - 1:
            model.eval()
            classifier.eval()
            with torch.no_grad():
                viz_results = model(viz_signal, return_all=True)
                # compute losses for viz sample too
                _, viz_losses = total_sast_loss(
                    viz_results['tfr_raw'],
                    viz_results['tfr_enhanced'],
                    viz_results['C_i'],
                    viz_label,
                    classifier,
                    viz_results['edge_feats'],
                    viz_results['gate_edge'],
                    viz_results['edge_src'],
                    viz_results['edge_dst'],
                    viz_results['node_if'],
                    viz_results['freqs'],
                    viz_results['A_ij'],
                )
                viz_results['losses'] = viz_losses
            visualize_diagnostics(viz_results, 0, epoch, save_dir)
            model.train()
            classifier.train()

    # ── 保存 ──
    ckpt_path = save_dir / 'sast_v2_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'C_prior': model.get_C_prior().cpu(),
        'args': vars(args),
    }, ckpt_path)
    print(f"\nModel saved to {ckpt_path}")
    print(f"Final C_prior: {model.get_C_prior().cpu().numpy()}")

    # ── 打印 C_i 总结 ──
    model.eval()
    with torch.no_grad():
        final_results = model(viz_signal, return_all=True)
        C_i_final = final_results['C_i'][0].cpu().numpy()  # [N, T]
        print(f"\nFinal C_i (time-mean per node):")
        for n, name in enumerate(NODE_NAMES):
            print(f"  {name:<15s} {C_i_final[n].mean():.3f} ± {C_i_final[n].std():.3f}")
        print(f"  C_i spread: {C_i_final.mean(axis=1).ptp():.3f}")

    print("Training complete.")


if __name__ == '__main__':
    main()
