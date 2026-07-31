"""
SAST v3 推理 + 可视化脚本
==========================

加载训练好的 SAST 模型, 对信号做自适应同步压缩, 输出增强 TFR + 诊断面板。

用法:
  # 1. 绘制增强 TFR (三面板: STFT + SAST + w_i 曲线)
  python infer_sast.py --checkpoint sast_checkpoints/sast_v3_model.pt \\
      --data 5_dataset.npz --class 1 --output sast_tfr.png

  # 2. 全诊断面板 (六面板)
  python infer_sast.py --checkpoint sast_checkpoints/sast_v3_model.pt \\
      --data 5_dataset.npz --class 1 --mode full --output sast_full.png

  # 3. 预测单个样本的类别
  python infer_sast.py --checkpoint sast_checkpoints/sast_v3_model.pt \\
      --data 5_dataset.npz --class 1 --mode predict

  # 4. 在验证集上评估准确率
  python infer_sast.py --checkpoint sast_checkpoints/sast_v3_model.pt \\
      --data 5_dataset.npz --mode eval --eval-split 0.15

  # 5. 批量导出增强 TFR 特征
  python infer_sast.py --checkpoint sast_checkpoints/sast_v3_model.pt \\
      --data 5_dataset.npz --mode batch --output sast_features.npy

  # 6. Python API
  from infer_sast import run_inference
  from sast_utils import load_checkpoint, infer_sast

  model, freq_encoder, meta = load_checkpoint('sast_v3_model.pt', device)
  results = infer_sast(model, signal, return_all=True)
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent))

from sast_utils import (
    SastConfig, set_seed, load_dataset,
    load_checkpoint, infer_sast, predict_class, evaluate_accuracy, compute_class_centroids,
    plot_tfr, plot_full_diagnostics, print_inference_summary,
)
from models.sast_graph import get_graph_summary


# ═══════════════════════════════════════════════════════════════
# High-level convenience API
# ═══════════════════════════════════════════════════════════════

def run_inference(checkpoint_path: str, signal,
                  device: str = 'cuda'):
    """
    Convenience: load model + run inference on a single signal.

    Args:
        checkpoint_path: path to .pt checkpoint
        signal:          [T] numpy array
        device:          'cuda' or 'cpu'

    Returns:
        results: dict of numpy arrays from SAST forward
    """
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model, freq_encoder, meta = load_checkpoint(checkpoint_path, dev)
    results = infer_sast(model, signal, return_all=True)
    return results


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='SAST v3 Inference + Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 3-panel TFR plot
  python infer_sast.py -c sast_v3_model.pt -d 5_dataset.npz --class 1

  # Full 6-panel diagnostics
  python infer_sast.py -c sast_v3_model.pt -d 5_dataset.npz --class 1 --mode full

  # Predict class for a sample
  python infer_sast.py -c sast_v3_model.pt -d 5_dataset.npz --class 1 --mode predict

  # Evaluate on held-out data
  python infer_sast.py -c sast_v3_model.pt -d 5_dataset.npz --mode eval --eval-split 0.2
        """,
    )

    p.add_argument('-c', '--checkpoint', required=True,
                   help='Model checkpoint (.pt)')
    p.add_argument('-d', '--data', default='5_dataset.npz')
    p.add_argument('--class', dest='target_class', type=int, default=None,
                   help='Target class to sample from (default: highest-energy sample overall)')
    p.add_argument('--sample', type=int, default=None,
                   help='Specific sample index (overrides --class)')
    p.add_argument('-o', '--output', default=None)
    p.add_argument('--mode', choices=['tfr', 'full', 'predict', 'eval', 'batch'],
                   default='tfr',
                   help='tfr=3-panel | full=6-panel | predict=classify | eval=accuracy | batch=export features')
    p.add_argument('--device', default='cuda')
    p.add_argument('--freq-max', type=float, default=200)
    p.add_argument('--eval-split', type=float, default=0.15,
                   help='Validation split for eval mode')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed (for eval split)')
    p.add_argument('--max-samples', type=int, default=None,
                   help='Max samples (for batch/eval mode)')

    return p.parse_args()


def _select_sample(X, y, target_class=None, sample_idx=None) -> tuple:
    """Select a sample from the dataset. Returns (signal, class_label, sample_idx)."""
    if sample_idx is not None:
        return X[sample_idx], int(y[sample_idx]), sample_idx

    if target_class is not None:
        idx_c = np.where(y == target_class)[0]
        if len(idx_c) == 0:
            raise ValueError(f"No samples found for class {target_class}. "
                           f"Available classes: {np.unique(y)}")
        energies = np.sum(X[idx_c] ** 2, axis=1)
        best_local = idx_c[np.argmax(energies)]
    else:
        # Default: highest energy sample overall
        energies = np.sum(X ** 2, axis=1)
        best_local = int(np.argmax(energies))

    return X[best_local], int(y[best_local]), best_local


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    print(f"Device: {device}")

    # ── Load model ──
    model, freq_encoder, meta = load_checkpoint(args.checkpoint, device)

    # ── Load data ──
    data = np.load(args.data, allow_pickle=True)
    X, y = data['train_X'], data['train_y']
    if X.ndim == 3:
        X = X[:, :, 0]
    y = y.ravel().astype(np.int64)

    if args.max_samples:
        X = X[:args.max_samples]
        y = y[:args.max_samples]
    X = X[:, :2000]  # truncate

    # ═══════════════════════════════════════════════════════════
    # Mode: batch export
    # ═══════════════════════════════════════════════════════════
    if args.mode == 'batch':
        print(f"Exporting features for {len(X)} samples...")
        features = []
        for i in range(len(X)):
            result = infer_sast(model, X[i], return_all=True)
            tfr = result['tfr_enhanced']
            freq_feat = tfr.mean(axis=-1) + tfr.max(axis=-1)
            features.append(freq_feat)
        feats = np.stack(features, axis=0)

        out_path = args.output or 'sast_features.npy'
        np.save(out_path, feats)
        labels_path = out_path.replace('.npy', '_labels.npy')
        np.save(labels_path, y[:len(feats)])
        print(f"  Features: {out_path}  shape={feats.shape}")
        print(f"  Labels:   {labels_path}")
        return

    # ═══════════════════════════════════════════════════════════
    # Mode: eval (accuracy on random split)
    # ═══════════════════════════════════════════════════════════
    if args.mode == 'eval':
        print(f"Evaluating accuracy with {args.eval_split:.0%} val split...")
        dataset = load_dataset(
            args.data, max_samples=args.max_samples,
            val_split=args.eval_split, seed=args.seed,
        )
        n_centroid = min(200, len(dataset['train_X']))
        centroids = compute_class_centroids(
            model, freq_encoder,
            dataset['train_X'][:n_centroid], dataset['train_y'][:n_centroid],
            batch_size=4, device=device,
        )
        metrics = evaluate_accuracy(
            model, freq_encoder,
            dataset['val_X'], dataset['val_y'], centroids,
            batch_size=4, device=device,
        )
        print(f"\n{'='*50}")
        print(f"Evaluation Results")
        print(f"{'='*50}")
        print(f"  Accuracy:  {metrics['accuracy']:.3f} "
              f"({metrics['n_correct']}/{metrics['n_total']})")
        print(f"  Per-class:")
        for c in range(len(metrics['per_class_total'])):
            correct = metrics['per_class_correct'][c]
            total = metrics['per_class_total'][c]
            acc_c = correct / max(1, total)
            print(f"    Class {c}: {acc_c:.3f} ({correct}/{total})")
        return

    # ═══════════════════════════════════════════════════════════
    # Modes: tfr / full / predict (single sample)
    # ═══════════════════════════════════════════════════════════
    signal, class_label, sample_idx = _select_sample(
        X, y, args.target_class, args.sample,
    )
    print(f"Sample: class={class_label}, idx={sample_idx}, T={len(signal)}")

    # Run inference
    print("Running SAST inference...")
    results = infer_sast(model, signal, return_all=True)

    # Predict if requested
    pred_class = None
    probs = None
    if args.mode == 'predict':
        n_centroid = min(200, len(X))
        centroids = compute_class_centroids(
            model, freq_encoder, X[:n_centroid], y[:n_centroid],
            batch_size=4, device=device,
        )
        pred_class, probs = predict_class(model, freq_encoder, signal, centroids)
        print(f"\nPredicted: class {pred_class}  |  True: class {class_label}")
        print("  " + "  ".join(f"P({c})={probs[c]:.3f}" for c in range(len(probs))))

    # Plot
    out_path = args.output
    if args.mode == 'full':
        out_path = out_path or f'sast_full_class{class_label}_s{sample_idx}.png'
        plot_full_diagnostics(results, out_path)
    else:
        out_path = out_path or f'sast_tfr_class{class_label}_s{sample_idx}.png'
        plot_tfr(results, out_path, freq_max=args.freq_max)

    # Summary table
    C_prior = meta['C_prior']
    print_inference_summary(
        results, C_prior=C_prior,
        class_label=class_label, sample_idx=sample_idx,
        pred_class=pred_class, probs=probs,
    )


if __name__ == '__main__':
    main()
