"""
EDA: 离线统计 5 类工况的频段能量分布 → 硬编码为静态原型向量
=========================================================
对 5_dataset.npz 逐样本计算:
  - E_LF:    8-20 Hz   频段能量占比
  - E_BPF:   42-55 Hz  频段能量占比
  - E_2xBPF: 90-105 Hz 频段能量占比
  - log_ratio: log10(E_BPF / E_2xBPF)  — RSI 相对强弱

每类取均值 → 5 个 4 维原型向量 → 硬编码到 models/sast.py 的 StaticPrototypeMatcher。

用法: python eda_prototypes.py
"""
import numpy as np
from pathlib import Path

FS = 1000
CLASS_NAMES = {
    0: '空转 (No-load)',
    1: '低负荷 (Low load)',
    2: '中负荷 (Mid load)',
    3: '高负荷 (High load)',
    4: '抽水 (Pumping)',
}

BANDS = {
    'LF':     (8, 20),
    'BPF':    (42, 55),
    '2xBPF':  (90, 105),
}


def compute_energy_ratios(x, fs):
    """计算单样本的 4 维能量比例向量。"""
    N = len(x)
    fft_freqs = np.fft.rfftfreq(N, 1/fs)
    fft_mag = np.abs(np.fft.rfft(x))

    ratios = []
    for name, (f_min, f_max) in BANDS.items():
        mask = (fft_freqs >= f_min) & (fft_freqs <= f_max)
        E_band = np.sum(fft_mag[mask] ** 2)
        ratios.append(E_band)

    E_total = np.sum(fft_mag ** 2) + 1e-12
    R_LF = ratios[0] / E_total
    R_BPF = ratios[1] / E_total
    R_2xBPF = ratios[2] / E_total
    log_ratio = np.log10(max(ratios[1], 1e-12) / max(ratios[2], 1e-12))

    return np.array([R_LF, R_BPF, R_2xBPF, log_ratio], dtype=np.float64)


def main():
    data = np.load('5_dataset.npz', allow_pickle=True)
    train_X = data['train_X']
    train_y = data['train_y'].ravel()

    print("=" * 80)
    print("EDA: Per-Class Energy Distribution Prototypes")
    print("=" * 80)
    print(f"Dataset: 5_dataset.npz | {len(train_X)} samples | fs={FS} Hz")
    print(f"Bands: LF={BANDS['LF']}, BPF={BANDS['BPF']}, 2xBPF={BANDS['2xBPF']}")
    print()

    prototypes = {}
    all_stats = {}

    for cls in sorted(np.unique(train_y)):
        idxs = np.where(train_y == cls)[0]
        vectors = np.array([compute_energy_ratios(train_X[i], FS) for i in idxs])

        mean_vec = vectors.mean(axis=0)
        std_vec = vectors.std(axis=0)

        prototypes[cls] = mean_vec
        all_stats[cls] = {
            'mean': mean_vec,
            'std': std_vec,
            'n_samples': len(idxs),
        }

        print(f"Class {cls} — {CLASS_NAMES[cls]} ({len(idxs)} samples)")
        print(f"  R_LF:     {mean_vec[0]:.4f} ± {std_vec[0]:.4f}")
        print(f"  R_BPF:    {mean_vec[1]:.4f} ± {std_vec[1]:.4f}")
        print(f"  R_2xBPF:  {mean_vec[2]:.4f} ± {std_vec[2]:.4f}")
        print(f"  log10(BPF/2xBPF): {mean_vec[3]:+.3f} ± {std_vec[3]:.3f}")
        print()

    # ── 输出可复制到代码中的格式 ──
    print("=" * 80)
    print("Hardcoded prototypes (copy to models/sast.py):")
    print("=" * 80)
    print("STATIC_PROTOTYPES = torch.tensor([")
    for cls in sorted(prototypes.keys()):
        v = prototypes[cls]
        print(f"    [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}, {v[3]:.6f}],  # Class {cls}: {CLASS_NAMES[cls]}")
    print("], dtype=torch.float64)")
    print()

    # ── 原型间距离矩阵 ──
    print("=" * 80)
    print("Prototype pairwise distances (Euclidean):")
    print("=" * 80)
    proto_mat = np.stack([prototypes[c] for c in sorted(prototypes.keys())])
    # Normalize for distance computation
    proto_norm = proto_mat / (proto_mat.std(axis=0, keepdims=True) + 1e-12)
    print(f"{'':>12s}", end="")
    for c in sorted(prototypes.keys()):
        print(f"  Class {c:>6s}", end="")
    print()
    for i in sorted(prototypes.keys()):
        print(f"  Class {i:<5s}", end="")
        for j in sorted(prototypes.keys()):
            dist = np.linalg.norm(proto_norm[i] - proto_norm[j])
            print(f"  {dist:10.4f}", end="")
        print()
    print("\n(Larger distance = easier to distinguish via soft matching)")

    # ── 保存 ──
    out_path = Path('data/prototypes.npz')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             prototypes=proto_mat,
             class_names=[CLASS_NAMES[c] for c in sorted(prototypes.keys())])
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
