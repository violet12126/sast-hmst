"""
SAST 推理 vs 原 MSST 时频图对比
================================
加载训练好的 SAST checkpoint, 对真实样本推理, 与原始 MSST (models/tfr.py msst,
MATLAB MSST_Y_new 直译) 并排对比时频图, 并附 Rényi 熵浓度度量。

输出两张图:
  图1 时频对比 (1×2):
    (a) 原 MSST           - 独立 msst() 硬挤压同步压缩 (基准)
    (b) SAST Enhanced     - 训练模型软高斯重排 + w_i 自适应挤压
  图2 IF 分析 (2×1):
    上: 各物理节点瞬时频率 IF(t)
    下: w_i(t) IF 可信度曲线

用法:
  python scripts/plot/plot_sast_vs_msst.py
  python scripts/plot/plot_sast_vs_msst.py --class 3
  python scripts/plot/plot_sast_vs_msst.py --sample 42
  python scripts/plot/plot_sast_vs_msst.py --freq-max 200 --output cmp.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sast_utils import load_checkpoint, infer_sast
from models.tfr import msst, compute_renyi
from models.sast_graph import NODE_NAMES

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 8
CMAP = 'jet'

CLASS_NAMES = {
    0: 'Class 0: No-load',
    1: 'Class 1: Low load',
    2: 'Class 2: Mid load',
    3: 'Class 3: High load',
    4: 'Class 4: Pumping',
}


def _db(tfr, ref=None):
    """转 dB, 可选统一参考 ref (用于跨面板同色标)."""
    eps = 1e-12
    if ref is None:
        ref = tfr.max()
    return 10 * np.log10(tfr / max(ref, eps) + eps)


def select_sample(X, y, target_class=None, sample_idx=None):
    """选样本: 指定 idx > 指定类内最高能量 > 全局最高能量."""
    if sample_idx is not None:
        return X[sample_idx], int(y[sample_idx]), sample_idx
    if target_class is not None:
        idx_c = np.where(y == target_class)[0]
        if len(idx_c) == 0:
            raise ValueError(f"class {target_class} 无样本, 可用: {np.unique(y)}")
        energies = np.sum(X[idx_c] ** 2, axis=1)
        best = idx_c[np.argmax(energies)]
    else:
        energies = np.sum(X ** 2, axis=1)
        best = int(np.argmax(energies))
    return X[best], int(y[best]), best


def main():
    p = argparse.ArgumentParser(description='SAST inference vs original MSST TFR comparison')
    p.add_argument('-c', '--checkpoint', default='sast_checkpoints/sast_v3_model.pt')
    p.add_argument('-d', '--data', default='5_dataset.npz')
    p.add_argument('--class', dest='target_class', type=int, default=None)
    p.add_argument('--sample', type=int, default=None)
    p.add_argument('--freq-max', type=float, default=500.0)
    p.add_argument('--msst-num', type=int, default=2,
                   help='原 MSST 挤压迭代次数 (默认 2)')
    p.add_argument('-o', '--output', default=None)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device_str = args.device if __import__('torch').cuda.is_available() else 'cpu'
    import torch
    device = torch.device(device_str)
    print(f"Device: {device}")

    # ── 1. 加载模型 ──
    model, freq_encoder, meta = load_checkpoint(args.checkpoint, device)
    cfg = meta['config']
    fs = cfg.fs
    msst_num = cfg.msst_num
    print(f"  msst_num={msst_num}, fs={fs}, F_bins={meta['F_bins']}")

    # ── 2. 加载数据 + 选样本 ──
    data = np.load(args.data, allow_pickle=True)
    X, y = data['train_X'], data['train_y']
    if X.ndim == 3:
        X = X[:, :, 0]
    y = y.ravel().astype(np.int64)
    X = X[:, :cfg.max_len]

    signal, cls, sidx = select_sample(X, y, args.target_class, args.sample)
    signal = signal.astype(np.float64)
    N = len(signal)
    t_axis = np.arange(N) / fs
    cls_name = CLASS_NAMES.get(cls, f'Class {cls}')
    print(f"Sample: class={cls} ({cls_name}), idx={sidx}, T={N} ({N/fs:.2f}s)")

    # ── 3. SAST 推理 ──
    print("Running SAST inference ...")
    r = infer_sast(model, signal, return_all=True)
    tfr_raw = r['tfr_raw']          # [F, T] |STFT|
    tfr_enh = r['tfr_enhanced']     # [F, T] SAST 增强
    tfr_msst_int = r.get('tfr_msst')  # [F, T] 模型内硬挤压 (sanity)
    w_i = r['w_i']                  # [N_phys, T]
    node_if = r['node_if']          # [N_phys, T]
    freqs_m = r['freqs']            # [F]
    t_axis_m = r['t_axis']          # [T]
    print(f"  tfr_enhanced {tfr_enh.shape}, w_i {w_i.shape}")

    # ── 4. 原 MSST (独立 MATLAB 直译实现, num=2) ──
    print(f"Running original MSST (num={args.msst_num}) ...")
    r_msst = msst(signal, fs, hlength=None, num=args.msst_num, save_trajectory=False)
    tfr_msst = r_msst['MSST']       # [F, T] 原始 MSST 幅度
    freqs_ms = r_msst['freqs']
    t_axis_ms = r_msst['t']
    print(f"  msst {tfr_msst.shape}, freqs [{freqs_ms[0]:.1f}, {freqs_ms[-1]:.1f}] Hz")

    # ── 5. Rényi 熵 (越低越集中) ──
    fmask_m = freqs_m <= args.freq_max
    fmask_ms = freqs_ms <= args.freq_max
    re_raw = compute_renyi(tfr_raw[fmask_m, :])
    re_msst = compute_renyi(tfr_msst[fmask_ms, :])
    re_sast = compute_renyi(tfr_enh[fmask_m, :])
    print(f"\n  Rényi α=3 (0-{args.freq_max:.0f} Hz, 越低越集中):")
    print(f"    Raw STFT : {re_raw:6.3f}")
    print(f"    原 MSST  : {re_msst:6.3f}  (Δ vs STFT = {re_msst - re_raw:+.3f})")
    print(f"    SAST     : {re_sast:6.3f}  (Δ vs MSST = {re_sast - re_msst:+.3f})")

    # ═════════════════════════════════════════════════════════════
    # 6. 图1: 时频对比 (原 MSST vs SAST), 1×2, 无 STFT 无 colorbar
    # ═════════════════════════════════════════════════════════════
    fig, (ax_ms, ax_sa) = plt.subplots(1, 2, figsize=(14, 6))

    # 各面板线性显示, vmax=0.5*max (隐去背景噪声, 同 plot_msst_tfr.py)
    def _panel(ax, tfr, t_ax, f_ax, fmask, title):
        sub = tfr[fmask, :]
        vmax = sub.max() * 0.5
        ax.pcolormesh(t_ax, f_ax[fmask], sub, shading='gouraud',
                      cmap=CMAP, vmin=0, vmax=vmax)
        ax.set_ylim(0, args.freq_max)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(title, fontsize=10, fontweight='bold')

    _panel(ax_ms, tfr_msst, t_axis_ms, freqs_ms, fmask_ms,
           f'(a) 原 MSST (硬挤压, num={args.msst_num})\nRényi={re_msst:.2f}')
    _panel(ax_sa, tfr_enh, t_axis_m, freqs_m, fmask_m,
           f'(b) SAST Enhanced (软高斯重排 + w_i 自适应)\nRényi={re_sast:.2f}')

    plt.suptitle(
        f'SAST vs 原 MSST 时频图对比  |  {cls_name}  idx={sidx}  '
        f'fs={fs}Hz T={N/fs:.1f}s\n'
        f'Rényi: MSST={re_msst:.2f} -> SAST={re_sast:.2f} '
        f'(Δ={re_sast - re_msst:+.2f})',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out = args.output or f'sast_vs_msst_c{cls}_s{sidx}.png'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=170, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {out}")

    # ═════════════════════════════════════════════════════════════
    # 7. 图2: IF 分析 (节点瞬时频率 + w_i 可信度), 独立一张图
    # ═════════════════════════════════════════════════════════════
    fig_if, (ax_if, ax_w) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    Nn = w_i.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(Nn, 3)))
    names = NODE_NAMES[1:] if Nn <= 3 else NODE_NAMES

    for n in range(Nn):
        nm = names[n] if n < len(names) else f'N{n}'
        ax_if.plot(t_axis_m, node_if[n], lw=1.2, alpha=0.85,
                   color=colors[n], label=f'{nm} (w={w_i[n].mean():.2f})')
        ax_w.plot(t_axis_m, w_i[n], color=colors[n], lw=1.3, label=nm)

    ax_if.set_ylim(bottom=0)
    ax_if.set_ylabel('Instantaneous Frequency [Hz]')
    ax_if.set_title('各物理节点瞬时频率 IF(t)', fontsize=10, fontweight='bold')
    ax_if.legend(fontsize=7, loc='upper right')
    ax_if.grid(alpha=0.3)

    ax_w.axhline(0.5, color='gray', ls='--', lw=0.5)
    ax_w.set_ylim(-0.05, 1.05)
    ax_w.set_xlabel('Time [s]')
    ax_w.set_ylabel('w_i')
    ax_w.set_title('IF Trust w_i(t)   w_i->1: 信任并强挤 | w_i->0: 保守软挤',
                   fontsize=10, fontweight='bold')
    ax_w.legend(fontsize=7, loc='upper right')
    ax_w.grid(alpha=0.3)

    plt.suptitle(f'IF 分析  |  {cls_name}  idx={sidx}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_if = Path(out).with_name(Path(out).stem + '_if.png')
    plt.savefig(out_if, dpi=170, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {out_if}")


if __name__ == '__main__':
    main()
