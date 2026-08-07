"""
SAST v3 训练脚本: 异构物理图 + 静态原型软匹配 + 自适应阶数 SST + 稀疏高斯重排

架构:
  Signal -> MSST (CPU, numpy) -> Node Features (frequency-region aggregation)
  -> StaticPrototypeMatcher -> cond_ctx
  -> PhysicsPrototypeMemory (ratio-gated, heterogeneous) -> EdgeConditionedGAT -> w_i
  -> SparseGaussianReassigner -> Physical TFR -> FreqEncoder -> z_freq (SupCon)

损失 (五分量, 自监督 SupCon):
  L = λ_sc·L_supcon + λ_e·RE_2D + λ_p·L_physics + λ_s·L_smooth + λ_b·L_balance

  L_supcon: 监督对比 (工况标签定义正负对, 但不分类) - 主监督
  其余: RE_2D 集中度 / L_physics 物理一致 / L_smooth 时序平滑 / L_balance 防退化

用法:
  python train_sast.py --epochs 20 --lr 0.001 --batch_size 8 --device cuda
  python train_sast.py --epochs 50 --d_h 128 --n_layers 3 --seed 123
  python train_sast.py --data my_data.npz --val_split 0.2 --max_samples 500
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from models.sast_losses import total_sast_loss
from models.sast_graph import get_graph_summary, NODE_NAMES

from sast_utils import (
    SastConfig, set_seed, load_dataset,
    create_model, create_freq_encoder, get_freq_bins,
    compute_class_centroids, plot_full_diagnostics, infer_sast, evaluate_accuracy,
)


def parse_args():
    """Parse CLI args and return (config, device_str)."""
    p = argparse.ArgumentParser(
        description='Train SAST v3 (SupCon self-supervised)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_sast.py --epochs 20 --device cuda
  python train_sast.py --epochs 50 --d_h 128 --n_layers 3 --seed 123
  python train_sast.py --data my_data.npz --val_split 0.2 --max_samples 500
  python train_sast.py --fs 2000 --sigma_min 1.0 --sigma_max 20.0
        """,
    )

    # ── Data ──
    p.add_argument('--data', type=str, default='5_dataset.npz')
    p.add_argument('--fs', type=int, default=1000, help='Sampling rate (Hz)')
    p.add_argument('--max_len', type=int, default=2000)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--val_split', type=float, default=0.15,
                   help='Validation split fraction (0 = no split)')

    # ── Model ──
    p.add_argument('--d_h', type=int, default=96)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--sigma_min', type=float, default=0.5)
    p.add_argument('--sigma_max', type=float, default=15.0)
    p.add_argument('--msst_num', type=int, default=4,
                   help='MSST 迭代次数 (=N_max=4)')
    p.add_argument('--d_cond', type=int, default=32)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--ppn_temperature', type=float, default=0.08)
    p.add_argument('--prototype_temperature', type=float, default=0.1)

    # ── Training ──
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=8,
                   help='SupCon 需 batch 内同工况正样本对, 建议 >=8')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)

    # ── Loss weights ──
    p.add_argument('--lambda_supcon', type=float, default=1.0)
    p.add_argument('--lambda_entropy', type=float, default=0.1)
    p.add_argument('--lambda_physics', type=float, default=0.5)
    p.add_argument('--lambda_smooth', type=float, default=0.05)
    p.add_argument('--lambda_balance', type=float, default=0.5)
    p.add_argument('--lambda_var', type=float, default=0.5)
    p.add_argument('--lambda_lowfreq', type=float, default=0.05,
                   help='低频锐化 loss 权重 (方法5, 0=禁用)')

    # ── New module settings ──
    p.add_argument('--smoother_kernel', type=int, default=15,
                   help='TemporalSmoother conv kernel size (odd, >=3)')
    p.add_argument('--n_sqz_max', type=int, default=4,
                   help='推理多轮挤压最大轮数')

    # ── Output ──
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save_dir', type=str, default='sast_checkpoints')
    p.add_argument('--viz_every', type=int, default=5)
    p.add_argument('--resume', type=str, default=None,
                   help='从 checkpoint 恢复训练 (路径, 如 sast_checkpoints/sast_v3_model.pt)')

    args = p.parse_args()

    config = SastConfig(
        fs=args.fs, max_len=args.max_len,
        d_h=args.d_h, n_heads=args.n_heads, n_layers=args.n_layers,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        msst_num=args.msst_num, d_cond=args.d_cond,
        ppn_temperature=args.ppn_temperature,
        prototype_temperature=args.prototype_temperature,
        dropout=args.dropout,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, seed=args.seed,
        lambda_supcon=args.lambda_supcon, lambda_entropy=args.lambda_entropy,
        lambda_physics=args.lambda_physics, lambda_smooth=args.lambda_smooth,
        lambda_balance=args.lambda_balance,
        lambda_var=args.lambda_var,
        lambda_lowfreq=args.lambda_lowfreq,
        data_path=args.data, val_split=args.val_split,
        max_samples=args.max_samples,
        save_dir=args.save_dir, viz_every=args.viz_every,
        smoother_kernel=args.smoother_kernel,
        n_sqz_max=args.n_sqz_max,
        resume=args.resume,
    )
    return config, args.device


def main():
    config, device_str = parse_args()

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    set_seed(config.seed)
    print(get_graph_summary())

    save_dir = Path(config.save_dir)
    save_dir.mkdir(exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # 1. Load data
    # ═══════════════════════════════════════════════════════════
    print(f"\nLoading data from {config.data_path}...")
    dataset = load_dataset(
        config.data_path,
        max_samples=config.max_samples,
        max_len=config.max_len,
        val_split=config.val_split,
        seed=config.seed,
    )
    X_train, y_train = dataset['train_X'], dataset['train_y']
    has_val = 'val_X' in dataset

    print(f"  Train: N={dataset.get('N_train', dataset['N_total'])}, T={dataset['T']}")
    if has_val:
        print(f"  Val:   N={dataset['N_val']}")
        X_val, y_val = dataset['val_X'], dataset['val_y']
    print(f"  Classes: {np.unique(y_train)}, counts: {np.bincount(y_train)}")

    # ═══════════════════════════════════════════════════════════
    # 2. Create model + freq_encoder
    # ═══════════════════════════════════════════════════════════
    print(f"\nCreating SAST v3 model (SupCon)...")
    print(f"  d_h={config.d_h}, n_heads={config.n_heads}, n_layers={config.n_layers}")
    print(f"  fs={config.fs} Hz, sigma=[{config.sigma_min}, {config.sigma_max}]")
    print(f"  MSST num={config.msst_num}")

    model = create_model(config, device)

    # Determine F_bins via dummy forward pass
    F_bins = get_freq_bins(model, X_train[0], device)
    print(f"  F_bins={F_bins}")

    freq_encoder = create_freq_encoder(F_bins, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_params = sum(p.numel() for p in freq_encoder.parameters())
    print(f"  SAST params: {total_params:,} (trainable: {trainable_params:,})")
    print(f"  Encoder params: {enc_params:,}")

    # ── Resume from checkpoint (fine-tune) ──
    start_epoch = 0
    if config.resume:
        print(f"\nResuming from checkpoint: {config.resume}")
        ckpt = torch.load(config.resume, map_location=device, weights_only=False)
        model_missing, model_unexpected = model.load_state_dict(
            ckpt['model_state_dict'], strict=False)
        enc_missing, enc_unexpected = freq_encoder.load_state_dict(
            ckpt['encoder_state_dict'], strict=False)
        if model_missing:
            print(f"  Model new params (random init): {model_missing}")
        if enc_missing:
            print(f"  Encoder new params: {enc_missing}")
        start_epoch = ckpt.get('epoch', 0)
        print(f"  Resuming from epoch {start_epoch}")

    # ═══════════════════════════════════════════════════════════
    # 3. Optimizer + scheduler
    # ═══════════════════════════════════════════════════════════
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(freq_encoder.parameters()),
        lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs,
    )

    # ═══════════════════════════════════════════════════════════
    # 4. Fixed visualization sample
    # ═══════════════════════════════════════════════════════════
    viz_idx = 0
    viz_signal = X_train[viz_idx]
    viz_label = y_train[viz_idx]

    print(f"\n{'='*60}")
    print(f"Training SAST v3 (SupCon) for {config.epochs} epochs")
    print(f"  Batch size: {config.batch_size}, LR: {config.lr}")
    print(f"  λ_sc={config.lambda_supcon}  λ_e={config.lambda_entropy}")
    print(f"  λ_p={config.lambda_physics}  λ_s={config.lambda_smooth}")
    print(f"  λ_b={config.lambda_balance}")
    if has_val:
        print(f"  Val split: {config.val_split:.0%} (KNN accuracy via z_freq centroids)")
    print(f"{'='*60}")

    # ═══════════════════════════════════════════════════════════
    # 5. Training loop
    # ═══════════════════════════════════════════════════════════
    N_train = len(X_train)
    best_val_acc = 0.0

    for epoch in range(start_epoch, start_epoch + config.epochs):
        model.train()
        freq_encoder.train()

        perm = torch.randperm(N_train)
        n_batches = max(1, N_train // config.batch_size)
        epoch_losses = {}
        epoch_start = time.time()

        for bi in range(n_batches):
            idx = perm[bi * config.batch_size:(bi + 1) * config.batch_size]
            batch_x = torch.from_numpy(X_train[idx]).float().to(device)
            batch_y = torch.from_numpy(y_train[idx]).long().to(device)

            # ── Forward ──
            results = model(batch_x, return_all=True)

            # ── 5-component loss (SupCon 主监督) ──
            loss, losses_dict = total_sast_loss(
                results['tfr_raw'],
                results['tfr_enhanced'],
                results['w_i'],
                batch_y,
                freq_encoder,
                results['edge_feats'],
                results['gate_edge'],
                results['edge_src'],
                results['edge_dst'],
                results['node_if'],
                results['freqs'],
                results['A_ij'],
                lambda_supcon=config.lambda_supcon,
                lambda_entropy=config.lambda_entropy,
                lambda_physics=config.lambda_physics,
                lambda_smooth=config.lambda_smooth,
                lambda_balance=config.lambda_balance,
                lambda_var=config.lambda_var,
                lambda_lowfreq=config.lambda_lowfreq,
            )

            # ── TemporalSmoother 正则: 防残差学成 delta ──
            smoother_reg = torch.tensor(0.0, device=device)
            if model.wi_smoother is not None:
                smoother_reg = model.wi_smoother.reg_loss()
                loss = loss + smoother_reg

            # Additional diagnostics
            losses_dict['gate_mean'] = results['gate_edge'].mean().item()
            losses_dict['C_prior_mean'] = model.get_C_prior().mean().item()
            losses_dict['smoother_reg'] = smoother_reg.item()

            # ── Backward ──
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(freq_encoder.parameters()),
                max_norm=config.grad_clip,
            )
            optimizer.step()

            for k, v in losses_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Average losses over batches
        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)

        # ── Logging ──
        print(f"Epoch {epoch:3d} ({epoch_time:.0f}s): "
              f"T={epoch_losses.get('total', 0):.4f} "
              f"sc={epoch_losses.get('supcon', 0):.3f} "
              f"ent={epoch_losses.get('entropy_2d', 0):.3f} "
              f"phy={epoch_losses.get('physics', 0):.4f} "
              f"smo={epoch_losses.get('smooth', 0):.4f} "
              f"bal={epoch_losses.get('balance', 0):.4f} "
              f"lf={epoch_losses.get('lowfreq_sharp', 0):.4f} "
              f"sr={epoch_losses.get('smoother_reg', 0):.4f} "
              f"w_m={epoch_losses.get('w_mean', 0):.3f} "
              f"w_s={epoch_losses.get('w_spread', 0):.3f} "
              f"g={epoch_losses.get('gate_mean', 0):.3f}")

        # ── Validation (KNN via z_freq centroids) ──
        if has_val:
            n_centroid = min(200, len(X_train))
            centroids = compute_class_centroids(
                model, freq_encoder, X_train[:n_centroid], y_train[:n_centroid],
                batch_size=max(1, config.batch_size), device=device,
            )
            val_metrics = evaluate_accuracy(
                model, freq_encoder, X_val, y_val, centroids,
                batch_size=max(1, config.batch_size), device=device,
            )
            acc = val_metrics['accuracy']
            marker = ' *' if acc > best_val_acc else ''
            if acc > best_val_acc:
                best_val_acc = acc
            print(f"        Val acc={acc:.3f} ({val_metrics['n_correct']}/{val_metrics['n_total']}){marker}")

        # ── Visualization ──
        if epoch % config.viz_every == 0 or epoch == config.epochs - 1:
            model.eval()
            freq_encoder.eval()
            viz_results = infer_sast(model, viz_signal, return_all=True)

            # Compute losses for viz sample
            with torch.no_grad():
                viz_signal_t = torch.from_numpy(viz_signal).float().unsqueeze(0).to(device)
                viz_label_t = torch.tensor([viz_label], device=device)
                viz_full = model(viz_signal_t, return_all=True)
                _, viz_losses = total_sast_loss(
                    viz_full['tfr_raw'], viz_full['tfr_enhanced'],
                    viz_full['w_i'], viz_label_t, freq_encoder,
                    viz_full['edge_feats'], viz_full['gate_edge'],
                    viz_full['edge_src'], viz_full['edge_dst'],
                    viz_full['node_if'], viz_full['freqs'], viz_full['A_ij'],
                    lambda_lowfreq=config.lambda_lowfreq,
                )
                # Attach losses to results for title context
                viz_results['_losses'] = viz_losses

            plot_full_diagnostics(
                viz_results,
                str(save_dir / f'sast_v3_epoch{epoch:03d}.png'),
                epoch=epoch,
            )

    # ═══════════════════════════════════════════════════════════
    # 6. Save checkpoint
    # ═══════════════════════════════════════════════════════════
    ckpt_path = save_dir / f'sast_v3_e{epoch+1:03d}.pt'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': freq_encoder.state_dict(),
        'C_prior': model.get_C_prior().cpu(),
        'args': config.to_dict(),
        'f_bins': F_bins,
    }, ckpt_path)
    print(f"\nModel saved to {ckpt_path}")
    print(f"  Final C_prior: {model.get_C_prior().cpu().numpy()}")

    if has_val:
        print(f"  Best val accuracy: {best_val_acc:.3f}")

    # ═══════════════════════════════════════════════════════════
    # 7. Final w_i summary
    # ═══════════════════════════════════════════════════════════
    model.eval()
    with torch.no_grad():
        viz_signal_t = torch.from_numpy(viz_signal).float().unsqueeze(0).to(device)
        final_results = model(viz_signal_t, return_all=True)
        w_i_final = final_results['w_i'][0].cpu().numpy()  # [N_phys, T]
        print(f"\nFinal w_i (time-mean per node):")
        phys_names = NODE_NAMES[1:]
        for n in range(w_i_final.shape[0]):
            name = phys_names[n] if n < len(phys_names) else f'N{n}'
            print(f"  {name:<15s} {w_i_final[n].mean():.3f} ± {w_i_final[n].std():.3f}")
        print(f"  w_i spread: {w_i_final.mean(axis=1).ptp():.3f}")

    print("Training complete.")


if __name__ == '__main__':
    main()
