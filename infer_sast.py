"""
SAST 推理 + 可视化脚本
=======================
加载训练好的 SAST 模型, 对信号做自适应同步压缩, 输出 TFR + 诊断面板。

用法:
  # 1. 绘制增强 TFR (三面板: STFT vs HMST vs SAST)
  python infer_sast.py --checkpoint sast_checkpoints/sast_v2_model.pt \\
      --data 5_dataset.npz --class 1 --output sast_tfr.png

  # 2. 全诊断面板 (六面板: TFR + 带宽 + C_i + gate + A_ij)
  python infer_sast.py --checkpoint sast_checkpoints/sast_v2_model.pt \\
      --data 5_dataset.npz --class 1 --mode full --output sast_full.png

  # 3. 批量导出增强 TFR 用于下游任务
  python infer_sast.py --checkpoint sast_checkpoints/sast_v2_model.pt \\
      --data 5_dataset.npz --mode batch --output sast_features.npy

  # 4. Python API 调用
  from infer_sast import load_sast, infer_sast
  model = load_sast('sast_checkpoints/sast_v2_model.pt')
  tfr = infer_sast(model, signal)  # 仅返回增强 TFR
  results = model(signal, return_all=True)  # 返回完整诊断量
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
from models.sast import SAST, compute_stft

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9


# ============================================================
# 1. 模型加载
# ============================================================

def load_sast(checkpoint_path, device='cuda'):
    """
    从 checkpoint 加载训练好的 SAST 模型。

    Args:
        checkpoint_path: .pt 文件路径
        device: 'cuda' 或 'cpu'

    Returns:
        model: SAST (eval 模式)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 从 checkpoint 恢复配置
    args = ckpt['args']
    proto_cfg = ckpt.get('prototype_config', None)

    model = SAST(
        prototype_config=proto_cfg,
        fs=1000,  # 数据集固定参数
        n_fft=args.get('n_fft', 512),
        hop_length=args.get('hop_length', 128),
        d_h=args.get('d_h', 96),
        n_heads=args.get('n_heads', 4),
        n_layers=args.get('n_layers', 2),
        K_ridges=args.get('K_ridges', 6),
        sigma_min=0.5,
        sigma_max=15.0,
        hmst_order=args.get('hmst_order', 2),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Loaded SAST from {checkpoint_path}")
    print(f"  K_ridges={model.K_ridges}, hmst_order={model.hmst_order}")
    print(f"  d_h={model.d_h}, n_heads={model.gat.layers[0].n_heads}")
    print(f"  Prototypes: {model.ppm.M_proto}")
    return model


# ============================================================
# 2. 推理 (单信号 → TFR)
# ============================================================

def infer_sast(model, x):
    """
    用 SAST 处理单条信号, 返回增强 TFR。

    Args:
        model: SAST 模型 (eval 模式)
        x:      [T] 或 [1, T] 原始信号

    Returns:
        tfr_enhanced: [F_bins, T_frames] 自适应挤压 TFR (numpy)
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.to(next(model.parameters()).device)

    with torch.no_grad():
        tfr = model(x, return_all=False)

    return tfr[0].cpu().numpy()


@torch.no_grad()
def infer_sast_full(model, x):
    """
    用 SAST 处理单条信号, 返回完整诊断量。

    Returns: dict with keys:
        tfr_enhanced, tfr_raw, C_i, gate, sigma_i, sigma_sq,
        ridge_freq, A_ij, edge_feats, edge_src, edge_dst, freqs, t_frames
    """
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
            out[k] = v[0].cpu().numpy() if v.shape[0] == 1 else v.cpu().numpy()
        else:
            out[k] = v
    return out


# ============================================================
# 3. 可视化
# ============================================================

def plot_enhanced_tfr(results, save_path, freq_max=200, vmin=-30, vmax=10):
    """
    三面板图: STFT (原始) vs HMST IF 背景 vs SAST (增强 TFR)。

    展示 SAST 的核心价值:
      - 整数谐波 (BPF↔fr): C_i→1 → 窄挤压 → 锐利脊线
      - 滑差分量 (RSI↔fr):  C_i→0 → 宽挤压 → 保留物理展宽
    """
    tfr_raw = results['tfr_raw']       # [F, T_frames]
    tfr_enhanced = results['tfr_enhanced']  # [F, T_frames]
    C_i = results['C_i']               # [K, T_if]
    ridge_freq = results['ridge_freq']  # [K, T_if]
    freqs = results['freqs']           # [F]
    t_frames = results['t_frames']     # [T_frames]

    K = C_i.shape[0]
    t_if = t_frames[1:-1]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # (a) Raw STFT — 未挤压
    ax = axes[0]
    db_raw = 10 * np.log10(tfr_raw + 1e-12)
    ax.pcolormesh(t_frames, freqs, db_raw, shading='gouraud',
                  cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title('(a) Raw STFT (No Squeeze)\n'
                 'All frequency components are blurred')

    # (b) SAST Enhanced TFR — 自适应挤压
    ax = axes[1]
    db_enh = 10 * np.log10(tfr_enhanced + 1e-12)
    ax.pcolormesh(t_frames, freqs, db_enh, shading='gouraud',
                  cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_ylim(0, freq_max)

    # 脊线叠加, 颜色按 C_i 均值
    colors = plt.cm.RdYlGn(np.linspace(0.3, 1.0, K))
    for k in range(K):
        c_mean = C_i[k].mean()
        ax.plot(t_if, ridge_freq[k], lw=0.8, alpha=0.6,
                color=colors[k] if c_mean > 0.5 else 'gray')

    ax.set_xlabel('Time [s]')
    ax.set_title('(b) SAST Adaptive Squeeze\n'
                 'Green ridge = C_i→1 (sharp) | Gray ridge = C_i→0 (soft)')

    # (c) C_i 时间曲线 — 解释 SAST 的决策
    ax = axes[2]
    colors_tab = plt.cm.tab10(np.linspace(0, 1, K))
    for k in range(K):
        mean_f = ridge_freq[k].mean()
        ax.plot(t_if, C_i[k], color=colors_tab[k], lw=1.5,
                label=f'R{k} (~{mean_f:.1f} Hz)')
    ax.axhline(y=0.5, color='gray', ls='--', lw=0.8, alpha=0.7)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('C_i')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(c) Compressibility Token C_i(t)\n'
                 'C_i→1 = "trust this ridge, squeeze hard"\n'
                 'C_i→0 = "uncertain, keep soft"')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    plt.suptitle('SAST: Structure-Aware Synchrosqueezing\n'
                 'Adaptive squeeze bandwidth controlled by GAT trust assignment',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_full_diagnostics(results, save_path, freq_max=200):
    """
    六面板完整诊断: TFR × 2 + 带宽 + C_i + gate + A_ij。

    用于深入理解 SAST 的推理过程:
      (a) Raw STFT
      (b) SAST TFR + ridges
      (c) σ_sq 逐 bin 带宽
      (d) C_i(t) 可压缩性令牌
      (e) gate(t) 原型匹配门控
      (f) A_ij 边注意力矩阵 (因果推理探针)
    """
    tfr_raw = results['tfr_raw']
    tfr_enhanced = results['tfr_enhanced']
    sigma_sq = results['sigma_sq']
    C_i = results['C_i']
    gate = results['gate']
    A_ij = results['A_ij'].mean(axis=0)  # [M, T_if] 多头平均
    ridge_freq = results['ridge_freq']
    freqs = results['freqs']
    t_frames = results['t_frames']

    K = C_i.shape[0]
    M = A_ij.shape[0]
    t_if = t_frames[1:-1]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # (a) Raw STFT
    ax = axes[0, 0]
    db_raw = 10 * np.log10(tfr_raw + 1e-12)
    im = ax.pcolormesh(t_frames, freqs, db_raw, shading='gouraud',
                       cmap='jet', vmin=-30, vmax=10)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title('(a) Raw STFT (No Squeeze)')
    plt.colorbar(im, ax=ax, label='dB')

    # (b) SAST TFR
    ax = axes[0, 1]
    db_enh = 10 * np.log10(tfr_enhanced + 1e-12)
    im2 = ax.pcolormesh(t_frames, freqs, db_enh, shading='gouraud',
                        cmap='jet', vmin=-30, vmax=10)
    ax.set_ylim(0, freq_max)
    for k in range(K):
        ax.plot(t_if, ridge_freq[k], lw=0.5, alpha=0.4, color='white')
    ax.set_xlabel('Time [s]')
    ax.set_title('(b) SAST Enhanced TFR + Tracked Ridges')
    plt.colorbar(im2, ax=ax, label='dB')

    # (c) σ_sq 逐 bin 带宽
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
                 'gate→1: Matches known prototype  |  gate→0: Anonymous')
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    # (f) A_ij 边注意力矩阵
    ax = axes[1, 2]
    im4 = ax.imshow(A_ij, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Edge Index (i→j)')
    edge_count = K * (K - 1)
    ax.set_title(f'(f) Edge Attention A_ij (Diagnostic Probe)\n'
                 f'{M} edges ({K} nodes × {K-1} directed)\n'
                 f'Bright = strong message passing')
    plt.colorbar(im4, ax=ax, label='A_ij')

    plt.suptitle('SAST Inference — Full Diagnostic Panel\n'
                 'Anonymous Graph + Physics Prototype Memory → Compressibility Token',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def export_features(model, X, device='cuda'):
    """
    批量导出 SAST 增强 TFR 特征, 用于下游分类/聚类任务。

    Args:
        model: SAST 模型
        X:     [N, T] numpy 信号
        device: 计算设备

    Returns:
        features: [N, F_bins] 时间池化频域特征
    """
    model = model.to(device)
    features = []

    for i in range(len(X)):
        x = torch.from_numpy(X[i]).float().unsqueeze(0).to(device)
        feat = model.get_freq_features(x)  # [1, F_bins]
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)  # [N, F_bins]


# ============================================================
# 4. CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='SAST 推理 + 可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 增强 TFR 三面板图
  python infer_sast.py -c sast_v2_model.pt -d 5_dataset.npz --class 1

  # 全诊断面板
  python infer_sast.py -c sast_v2_model.pt -d 5_dataset.npz --class 1 --mode full

  # 批量导出特征
  python infer_sast.py -c sast_v2_model.pt -d 5_dataset.npz --mode batch -o features.npy
        """,
    )
    parser.add_argument('-c', '--checkpoint', required=True,
                        help='SAST checkpoint 路径 (.pt)')
    parser.add_argument('-d', '--data', default='5_dataset.npz',
                        help='数据集路径')
    parser.add_argument('--class', dest='target_class', type=int, default=1,
                        help='目标类别 (默认 1)')
    parser.add_argument('--sample', type=int, default=None,
                        help='指定样本索引 (默认: 该类能量最高者)')
    parser.add_argument('-o', '--output', default=None,
                        help='输出文件路径')
    parser.add_argument('--mode', choices=['tfr', 'full', 'batch'],
                        default='tfr',
                        help='输出模式: tfr=三面板TFR, full=六面板诊断, batch=批量特征')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--freq-max', type=float, default=200,
                        help='频率上限 (Hz)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 加载模型 ──
    model = load_sast(args.checkpoint, device=device)

    # ── 加载数据 ──
    data = np.load(args.data, allow_pickle=True)
    X, y = data['train_X'], data['train_y']
    if X.ndim == 3:
        X = X[:, :, 0]

    if args.mode == 'batch':
        print(f"Exporting features for {len(X)} samples...")
        feats = export_features(model, X, device=device)
        out_path = args.output or 'sast_features.npy'
        np.save(out_path, feats)
        labels_path = out_path.replace('.npy', '_labels.npy')
        np.save(labels_path, y)
        print(f"  Features: {out_path}  shape={feats.shape}")
        print(f"  Labels:   {labels_path}")
        return

    # ── 选样本 ──
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

    # ── 推理 ──
    print("Running SAST inference...")
    results = infer_sast_full(model, signal)

    # ── 绘图 ──
    out_dir = Path(args.output).parent if args.output else Path('.')
    out_dir = Path(out_dir)

    if args.mode == 'full':
        out_path = args.output or f'sast_full_class{y[sample_idx]}_sample{sample_idx}.png'
        plot_full_diagnostics(results, str(out_path), freq_max=args.freq_max)
    else:
        out_path = args.output or f'sast_tfr_class{y[sample_idx]}_sample{sample_idx}.png'
        plot_enhanced_tfr(results, str(out_path), freq_max=args.freq_max)

    # ── 打印摘要 ──
    C_i = results['C_i']
    gate = results['gate']
    ridge_freq = results['ridge_freq']
    sigma_i = results.get('sigma_i',
                          np.full_like(C_i, np.nan))

    K = C_i.shape[0]
    print(f"\n{'='*60}")
    print(f"SAST Inference Summary — Class {y[sample_idx]}, Sample #{sample_idx}")
    print(f"{'='*60}")
    print(f"{'Ridge':>6s}  {'Freq(Hz)':>9s}  {'C_i':>6s}  {'gate':>6s}  {'σ_i':>6s}  Interpretation")
    print(f"{'-'*60}")
    for k in range(K):
        f_mean = ridge_freq[k].mean()
        c_mean = C_i[k].mean()
        g_mean = gate[k].mean()
        s_mean = sigma_i[k].mean() if not np.isnan(sigma_i[k].mean()) else 0

        if c_mean > 0.7 and g_mean > 0.5:
            interp = '🔵 Known component — AGGRESSIVE squeeze'
        elif c_mean > 0.5:
            interp = '🟢 Trusted ridge — moderate squeeze'
        elif g_mean > 0.5:
            interp = '🟡 Matched prototype, low confidence'
        else:
            interp = '⚪ Anonymous ridge — SOFT squeeze (preserve bandwidth)'

        print(f"  R{k:>3d}  {f_mean:>8.1f} Hz  {c_mean:>5.3f}  {g_mean:>5.3f}  {s_mean:>5.1f}  {interp}")

    print(f"\nSAST adaptivity check:")
    c_spread = C_i.max(axis=1).mean() - C_i.min(axis=1).mean()
    print(f"  C_i spread (max-min across ridges): {c_spread:.3f}")
    print(f"  → {'✅ SAST is differentiating ridges' if c_spread > 0.2 else '⚠️  SAST treats all ridges similarly (need more training?)'}")


if __name__ == '__main__':
    main()
