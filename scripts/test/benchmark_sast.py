"""
SAST 性能基准测试 — 评估各模块耗时, 估算训练时长
==============================================
用法: python benchmark_sast.py
"""
import numpy as np
import torch
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.tfr import msst
from models.sast_nodes import MSSTNodeExtractor
from models.sast_graph import compute_edge_features
from models.sast import SAST

FS = 1000
N_SAMPLES = 50          # 测试样本数
N_WARMUP = 3            # 预热轮数
N_REPEAT = 10           # 每轮重复次数
DATASET_PATH = '5_dataset.npz'


def fmt_time(seconds):
    """格式化时间."""
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    elif seconds >= 0.001:
        return f"{seconds*1000:.1f} ms"
    else:
        return f"{seconds*1e6:.1f} us"


def benchmark_msst(signals):
    """MSST 耗时 (numpy, CPU)."""
    print("\n[1] MSST (numpy, CPU)")
    print("-" * 60)

    total_t = 0.0
    for i in range(N_WARMUP):
        _ = msst(signals[0], FS, num=5, save_trajectory=True)

    for i in range(N_SAMPLES):
        t0 = time.perf_counter()
        r = msst(signals[i], FS, num=5, save_trajectory=True)
        total_t += time.perf_counter() - t0

    avg = total_t / N_SAMPLES
    print(f"  N_max=5, save_trajectory=True")
    print(f"  Per sample: {fmt_time(avg)}")
    print(f"  Throughput: {1.0/avg:.1f} samples/s")
    print(f"  TFR shape:  {r['MSST'].shape}")
    print(f"  omegas:     {len(r['omegas'])} trajectories")
    return avg


def benchmark_node_extract(signals):
    """节点特征提取耗时."""
    print("\n[2] MSSTNodeExtractor (MSST + per-region aggregation)")
    print("-" * 60)

    extractor = MSSTNodeExtractor(fs=FS, msst_num=5)

    total_t = 0.0
    for i in range(N_WARMUP):
        _ = extractor(signals[0])

    for i in range(N_SAMPLES):
        t0 = time.perf_counter()
        nodes = extractor(signals[i])
        total_t += time.perf_counter() - t0

    avg = total_t / N_SAMPLES
    print(f"  Per sample: {fmt_time(avg)}")
    print(f"  Throughput: {1.0/avg:.1f} samples/s")
    print(f"  Node if_hz: {nodes.if_hz.shape}")
    print(f"  omegas:     {len(nodes.omegas)} trajectories, each {nodes.omegas[0].shape}")
    return avg


def benchmark_edge_features(signals):
    """边特征计算耗时 (numpy)."""
    print("\n[3] Edge Feature Computation (numpy, per-sample)")
    print("-" * 60)

    extractor = MSSTNodeExtractor(fs=FS, msst_num=5)
    all_nodes = []
    for i in range(min(N_SAMPLES, 20)):
        all_nodes.append(extractor(signals[i]))

    total_t = 0.0
    for i in range(len(all_nodes)):
        nodes = all_nodes[i]
        node_bw = nodes.bandwidth
        t0 = time.perf_counter()
        _ = compute_edge_features(
            nodes.if_hz, nodes.energy, nodes.persistence,
            node_bw=node_bw, window_size=5, fs=FS,
        )
        total_t += time.perf_counter() - t0

    avg = total_t / len(all_nodes)
    print(f"  Per sample: {fmt_time(avg)}")
    print(f"  Output: [M=6, T, 5] edge features")
    return avg


def benchmark_sast_forward(signals, device):
    """SAST 完整前向耗时."""
    print(f"\n[4] SAST v3 Forward Pass ({device})")
    print("-" * 60)

    model = SAST(fs=FS, d_h=64, n_heads=4, n_layers=2, N_max=5, msst_num=5).to(device)
    model.eval()

    # 预热
    x0 = torch.from_numpy(signals[0]).float().unsqueeze(0).to(device)
    for _ in range(N_WARMUP):
        with torch.no_grad():
            _ = model(x0, training=True)

    # 单样本
    total_t = 0.0
    n_test = min(N_SAMPLES, 30)
    for i in range(n_test):
        x_t = torch.from_numpy(signals[i]).float().unsqueeze(0).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x_t, training=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_t += time.perf_counter() - t0

    avg_single = total_t / n_test
    print(f"  Single sample (B=1): {fmt_time(avg_single)}")
    print(f"  Single throughput:   {1.0/avg_single:.1f} samples/s")

    # 小批量 (B=4) — numpy MSST 串行, 其余并行
    batch_size = 4
    n_batches = min(8, n_test // batch_size)
    total_t_batch = 0.0
    for i in range(n_batches):
        start = i * batch_size
        batch_signals = signals[start:start + batch_size]
        x_batch = torch.from_numpy(batch_signals).float().to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x_batch, training=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_t_batch += time.perf_counter() - t0

    avg_batch = total_t_batch / n_batches / batch_size
    print(f"  Per-sample in B={batch_size}: {fmt_time(avg_batch)}")
    print(f"  Batch throughput:     {batch_size/avg_batch:.1f} samples/s")
    return avg_single, avg_batch


def estimate_training_time(msst_time, sast_time, device):
    """估算训练时长."""
    print(f"\n[5] Training Time Estimation ({device})")
    print("-" * 60)

    data = np.load(DATASET_PATH, allow_pickle=True)
    n_samples = data['train_X'].shape[0]
    n_epochs = 50
    batch_size = 8

    # 每样本总耗时 ≈ MSST + 边特征 (numpy) + SAST forward + backward (2x forward)
    per_sample_total = msst_time + sast_time  # forward only
    per_sample_train = per_sample_total * 3.0  # forward + backward ≈ 3x forward

    per_epoch = n_samples * per_sample_train
    total = per_epoch * n_epochs

    print(f"  Dataset:      {n_samples} samples")
    print(f"  Epochs:       {n_epochs}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Per sample:   {fmt_time(per_sample_total)} (forward)")
    print(f"  Per sample:   {fmt_time(per_sample_train)} (forward+backward est.)")
    print(f"  Per epoch:    {fmt_time(per_epoch)}")
    print(f"  Total (50 ep): {fmt_time(total)}")
    print()
    print(f"  Bottleneck breakdown:")
    print(f"    MSST (numpy):     {fmt_time(msst_time)} ({msst_time/per_sample_total*100:.0f}%)")
    print(f"    SAST+Edge+GAT:    {fmt_time(sast_time)} ({sast_time/per_sample_total*100:.0f}%)")

    # CUDA 加速潜力
    if device.type == 'cuda':
        print(f"\n  CUDA acceleration potential:")
        print(f"    MSST: numpy -> CUDA kernel (est. 10-50x speedup)")
        msst_cuda = msst_time / 10.0
        sast_cuda = sast_time * 0.8  # GAT already on GPU
        per_sample_cuda = msst_cuda + sast_cuda
        per_sample_train_cuda = per_sample_cuda * 3.0
        per_epoch_cuda = n_samples * per_sample_train_cuda
        total_cuda = per_epoch_cuda * n_epochs
        print(f"    With MSST CUDA kernel (10x):")
        print(f"      Per epoch: {fmt_time(per_epoch_cuda)}")
        print(f"      Total:     {fmt_time(total_cuda)}")


def main():
    print("=" * 70)
    print("SAST v3 Performance Benchmark")
    print("=" * 70)
    print(f"Config: N_samples={N_SAMPLES}, N_warmup={N_WARMUP}, fs={FS} Hz")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 生成合成信号 (或从数据集加载)
    try:
        data = np.load(DATASET_PATH, allow_pickle=True)
        signals = data['train_X'][:N_SAMPLES].astype(np.float64)
        print(f"Using real signals from {DATASET_PATH}")
    except (FileNotFoundError, KeyError):
        print("Generating synthetic signals (T=1000, 3 components)")
        t = np.arange(0, 1.0, 1 / FS)
        sig_template = (np.sin(2 * np.pi * 48 * t + 0.15 * np.sin(2 * np.pi * 3 * t)) +
                        0.6 * np.sin(2 * np.pi * 96 * t) +
                        0.25 * np.sin(2 * np.pi * 12 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t)))
        signals = np.tile(sig_template.astype(np.float64), (N_SAMPLES, 1))
        # 加微小噪声区分样本
        signals += np.random.randn(*signals.shape) * 0.01

    print(f"Signal shape: {signals.shape}")

    # ── 逐模块 benchmark ──
    t_msst = benchmark_msst(signals)
    t_node = benchmark_node_extract(signals)
    t_edge = benchmark_edge_features(signals)
    t_sast, t_sast_batch = benchmark_sast_forward(signals, device)

    estimate_training_time(t_node, t_sast, device)

    # ── 汇总 ──
    print(f"\n[6] Summary")
    print("-" * 60)
    print(f"  {'Module':<30s} {'Time':>10s} {'%':>6s}")
    print(f"  {'-'*47}")
    total = t_node + t_sast
    rows = [
        ("MSST (numpy)", t_msst),
        ("Node Extraction", t_node - t_msst),
        ("Edge Features (numpy)", t_edge),
        ("SAST forward (GAT+Squeeze)", t_sast - t_node),
        ("SAST forward (total)", t_sast),
        ("Total per-sample (forward)", t_node + t_sast),
    ]
    for name, t_val in rows:
        pct = t_val / total * 100 if total > 0 else 0
        print(f"  {name:<30s} {fmt_time(t_val):>10s} {pct:5.0f}%")

    print(f"\n  {'='*47}")
    print(f"  MSST (numpy) is the main bottleneck — optimize with CUDA kernel first.")


if __name__ == '__main__':
    main()
