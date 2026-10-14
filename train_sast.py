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
from models.sast_nodes import PUMP_TURBINE_REGIONS
from models.sast_graph import get_graph_summary, NODE_NAMES

from sast_config import SastConfig, set_seed
from sast_data import load_dataset, build_balanced_batch_indices
from sast_model import create_model, create_freq_encoder, get_freq_bins
from sast_inference import compute_class_centroids, infer_sast, evaluate_accuracy
from sast_viz import plot_full_diagnostics


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
                   help='仅在 --balanced_p 0 时生效; 否则由 P×K 覆盖 '
                        '(实际 batch = balanced_p * balanced_k)')
    # ── 类别均衡采样 (P×K) ──
    p.add_argument('--balanced_p', type=int, default=2,
                   help='每个类在 batch 内的样本数; 0 = 关闭均衡采样 '
                        '(退回 randperm 均匀切片)。须 >=2, 否则每类仅 1 个样本, '
                        'SupCon 无正样本对')
    p.add_argument('--balanced_k', type=int, default=5,
                   help='每个 batch 抽多少个类 (P×K 的 K)')
    p.add_argument('--freq_constraints_active', action='store_true',
                   help='方案 A 反向开关: 打开后 L_physics/L_anchor/L_ratio '
                        '重新进入 L_total (默认降级为诊断, 不参与反向传播)')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)

    # ── Loss weights ──
    p.add_argument('--lambda_supcon', type=float, default=1.0)
    p.add_argument('--lambda_entropy', type=float, default=0.1)
    p.add_argument('--lambda_physics', type=float, default=0.5)
    p.add_argument('--lambda_balance', type=float, default=0.5)
    p.add_argument('--lambda_var', type=float, default=0.5)
    p.add_argument('--lambda_consistency', type=float, default=0.3,
                   help='w_i 频段内时序一致性 loss 权重 (越高各帧越一致, 防跳变)')
    p.add_argument('--lambda_lowfreq', type=float, default=0.05,
                   help='低频锐化 loss 权重 (0=禁用)')
    p.add_argument('--lambda_range', type=float, default=0.3,
                   help='w_i per-node 逐帧范围约束 loss 权重')
    p.add_argument('--lambda_anchor', type=float, default=0.2,
                   help='绝对频率锚 loss 权重 (BPF_LINE 49.995±1.5Hz, 2xBPF 99.990±1.5Hz)')
    p.add_argument('--lambda_ratio', type=float, default=0.2,
                   help='谐波比值一致性 loss 权重 (作用在可微 if_hat 上, 0=禁用)')
    p.add_argument('--ratio_tol', type=float, default=0.06,
                   help='比值允许的相对偏差 (flat-bottom 半宽, 默认 6%%)')
    p.add_argument('--presence_frac', type=float, default=0.1,
                   help='分量存在度估计: 计入"峰"的带内 bin 比例')
    p.add_argument('--w_low_min', type=float, default=0.05,
                   help='分量缺席时 w_range/balance 下界松弛到该值')

    # ── New module settings ──
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
        balanced_p=args.balanced_p, balanced_k=args.balanced_k,
        freq_constraints_active=args.freq_constraints_active,
        lr=args.lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, seed=args.seed,
        lambda_supcon=args.lambda_supcon, lambda_entropy=args.lambda_entropy,
        lambda_physics=args.lambda_physics,
        lambda_balance=args.lambda_balance,
        lambda_var=args.lambda_var,
        lambda_consistency=args.lambda_consistency,
        lambda_lowfreq=args.lambda_lowfreq,
        lambda_range=args.lambda_range,
        lambda_anchor=args.lambda_anchor,
        lambda_ratio=args.lambda_ratio,
        ratio_tol=args.ratio_tol,
        presence_frac=args.presence_frac,
        w_low_min=args.w_low_min,
        data_path=args.data, val_split=args.val_split,
        max_samples=args.max_samples,
        save_dir=args.save_dir, viz_every=args.viz_every,
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

    # ── P×K 均衡采样: 有效 batch 大小 ──
    # 关闭时退回 torch.randperm 均匀切片 + config.batch_size。
    N_train = len(X_train)
    if config.balanced_p > 0:
        if config.balanced_p < 2:
            print(f"  [WARN] balanced_p={config.balanced_p} < 2: 每类在 batch 内"
                  f"仅 1 个样本 -> 无正样本对, SupCon 得不到梯度。")
        eff_batch = config.balanced_p * config.balanced_k
        n_batches = max(1, N_train // eff_batch)
    else:
        eff_batch = config.batch_size
        n_batches = max(1, N_train // eff_batch)
    print(f"  Batch: {'P×K 均衡' if config.balanced_p > 0 else 'randperm 均匀'}"
          f"  P={config.balanced_p} K={config.balanced_k}"
          f"  -> B={eff_batch}, {n_batches} batch/epoch, "
          f"{eff_batch * n_batches}/{N_train} 样本/epoch")

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
    print(f"  Batch size: {eff_batch}"
          f"{' (P×K)' if config.balanced_p > 0 else ''}, LR: {config.lr}")
    print(f"  λ_sc={config.lambda_supcon}  λ_e={config.lambda_entropy}")
    print(f"  λ_p={config.lambda_physics}"
          f"{'' if config.freq_constraints_active else ' (诊断, 见下)'}"
          f"  λ_b={config.lambda_balance}")
    print(f"  λ_var={config.lambda_var}  λ_c={config.lambda_consistency}  λ_lf={config.lambda_lowfreq}  λ_r={config.lambda_range}")
    print(f"  presence_frac={config.presence_frac}  w_low_min={config.w_low_min}")
    tag = ('【消融: 参与反向传播】' if config.freq_constraints_active
           else '【方案 A: 已降级为诊断, 不参与反向传播】')
    print(f"  λ_anchor={config.lambda_anchor}  λ_ratio={config.lambda_ratio}  "
          f"ratio_tol={config.ratio_tol}  {tag}")
    if not config.freq_constraints_active:
        print(f"      理由: 重排落点不可学, 三项只能经 σ 操纵测量窗内能量质心;"
              f" 且实测五类 BPF_LINE 质心跨 0.74 Hz < ±1.5 Hz 容差。")
    print(f"  节点 ({len(PUMP_TURBINE_REGIONS)} 物理节点, B4 拆节点):")
    for r in PUMP_TURBINE_REGIONS:
        anch = '—' if r.f_anchor is None else f'{r.f_anchor:.3f}±{r.anchor_tol}Hz'
        print(f"    {r.name:<10s} [{r.f_min:.1f},{r.f_max:.1f}) Hz  {r.f_type:<15s} "
              f"w∈[{r.w_range_low},{r.w_range_high}]  锚 {anch}")
    if has_val:
        print(f"  Val split: {config.val_split:.0%} (KNN accuracy via z_freq centroids)")
    print(f"{'='*60}")

    # ═══════════════════════════════════════════════════════════
    # 5. Training loop
    # ═══════════════════════════════════════════════════════════
    best_val_acc = 0.0

    if config.balanced_p > 0:
        # 每 epoch 用 seed+epoch 的独立 Generator -> 采样可复现且跨 epoch 不同。
        # 打印一次类直方图 + 各类每 epoch 期望出现次数 (供论文报告过采样倍数)。
        _demo = build_balanced_batch_indices(
            y_train, config.balanced_p, config.balanced_k, 1,
            rng=np.random.default_rng(config.seed))
        print(f"\n  [P×K] batch 0 类别直方图: "
              f"{np.bincount(y_train[_demo[0]], minlength=int(y_train.max()) + 1).tolist()}"
              f"  (共 {_demo.shape[1]} 个样本)")
        cls_pool = np.bincount(y_train, minlength=int(y_train.max()) + 1)
        n_cls = int((cls_pool > 0).sum())
        # 每个类被抽中的概率 = k/n_cls, 每次抽 p 个
        mult = n_batches * config.balanced_p * (config.balanced_k / max(1, n_cls)) \
            / np.maximum(cls_pool, 1)
        for c in range(len(cls_pool)):
            if cls_pool[c] == 0:
                continue
            print(f"        class {c}: 池 {cls_pool[c]:>5d} -> 每 epoch 约 "
                  f"{mult[c] * cls_pool[c]:>6.0f} 次 ({mult[c]:.2f}×)")

    for epoch in range(start_epoch, start_epoch + config.epochs):
        model.train()
        freq_encoder.train()

        if config.balanced_p > 0:
            batches = build_balanced_batch_indices(
                y_train, config.balanced_p, config.balanced_k, n_batches,
                rng=np.random.default_rng(config.seed + epoch))
        else:
            perm = torch.randperm(N_train)
        epoch_losses = {}
        epoch_start = time.time()

        for bi in range(n_batches):
            if config.balanced_p > 0:
                idx = batches[bi]
            else:
                idx = perm[bi * eff_batch:(bi + 1) * eff_batch]
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
                lambda_balance=config.lambda_balance,
                lambda_var=config.lambda_var,
                lambda_consistency=config.lambda_consistency,
                lambda_lowfreq=config.lambda_lowfreq,
                lambda_range=config.lambda_range,
                lambda_anchor=config.lambda_anchor,
                lambda_ratio=config.lambda_ratio,
                ratio_tol=config.ratio_tol,
                presence_frac=config.presence_frac,
                w_low_min=config.w_low_min,
                freq_constraints_active=config.freq_constraints_active,
            )

            # Additional diagnostics
            losses_dict['gate_mean'] = results['gate_edge'].mean().item()
            losses_dict['C_prior_mean'] = model.get_C_prior().mean().item()

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
              f"bal={epoch_losses.get('balance', 0):.4f} "
              f"wc={epoch_losses.get('w_consistency', 0):.4f} "
              f"wr={epoch_losses.get('w_range', 0):.4f} "
              f"lf={epoch_losses.get('lowfreq_sharp', 0):.4f} "
              f"w_m={epoch_losses.get('w_mean', 0):.3f} "
              f"w_s={epoch_losses.get('w_spread', 0):.3f} "
              f"g={epoch_losses.get('gate_mean', 0):.3f}")
        # 频率约束的诊断量 (方案 A 下不参与反向传播, 但须逐 epoch 记录:
        # 这正是"锚在不被约束时也天然满足"的论文证据)。wλ 为生效乘数。
        print(f"         [诊断] anc={epoch_losses.get('anchor', 0):.4f} "
              f"rat={epoch_losses.get('ratio', 0):.4f}  "
              f"wλ(p/a/r)={epoch_losses.get('w_physics', 0):.1f}/"
              f"{epoch_losses.get('w_anchor', 0):.1f}/"
              f"{epoch_losses.get('w_ratio', 0):.1f}")
        for _r in PUMP_TURBINE_REGIONS:
            _ih = epoch_losses.get(f'if_hat_{_r.name}', 0.0)
            _pr = epoch_losses.get(f'presence_{_r.name}', 0.0)
            _tail = (f"|锚-if_hat|={abs(_r.f_anchor - _ih):.3f}"
                     if _r.f_anchor is not None else "(无锚)")
            print(f"                {_r.name:<10s} if_hat={_ih:7.3f} Hz  "
                  f"presence={_pr:.3f}  {_tail}")

        # ── Validation (KNN via z_freq centroids) ──
        if has_val:
            n_centroid = min(200, len(X_train))
            centroids = compute_class_centroids(
                model, freq_encoder, X_train[:n_centroid], y_train[:n_centroid],
                batch_size=max(1, eff_batch), device=device,
            )
            val_metrics = evaluate_accuracy(
                model, freq_encoder, X_val, y_val, centroids,
                batch_size=max(1, eff_batch), device=device,
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
                viz_full = model(viz_signal_t, return_all=True, training=False)
                _, viz_losses = total_sast_loss(
                    viz_full['tfr_raw'], viz_full['tfr_enhanced'],
                    viz_full['w_i'], viz_label_t, freq_encoder,
                    viz_full['edge_feats'], viz_full['gate_edge'],
                    viz_full['edge_src'], viz_full['edge_dst'],
                    viz_full['node_if'], viz_full['freqs'], viz_full['A_ij'],
                    lambda_lowfreq=config.lambda_lowfreq,
                    lambda_range=config.lambda_range,
                    lambda_anchor=config.lambda_anchor,
                    lambda_ratio=config.lambda_ratio,
                    ratio_tol=config.ratio_tol,
                    presence_frac=config.presence_frac,
                    w_low_min=config.w_low_min,
                    freq_constraints_active=config.freq_constraints_active,
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
        final_results = model(viz_signal_t, return_all=True, training=False)
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
