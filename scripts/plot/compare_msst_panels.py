"""
MSST 面板对齐检查: infer_sast 的 (a) 面板 vs hmst_figures/msst_comparison 的 MSST
=================================================================================
同一个样本、同一条信号, 只改「算法配置」与「显示方式」, 逐项拆开看差异来自哪里:

  算法配置
    R1  models.tfr.msst(x, hlength=320, num=1)   ← plot_msst_tfr.py 用的就是这个
    R4  models.tfr.msst(x, hlength=320, num=4)   只换 IF 迭代层数
    S   SAST.forward 内部 tfr_msst              hlength=round(N/8)=250, num=4

  显示方式
    linear (vmax = 0.5·max, 与 plot_msst_tfr.py 一致)  |  dB (峰值下 25 dB, 与 sast_viz 一致)
    频率窗 0-500 Hz (对比图)  |  0-200 Hz (infer_sast 默认)

输出: 2×3 面板图 + 控制台数值 (Pearson r / 相对误差 / 能量占比)

用法:
  python scripts/plot/compare_msst_panels.py --class 1 --idx 4876
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from models.tfr import msst as msst_np           # noqa: E402


def squeeze_hard_ref(tfr_mag, omega_final):
    """硬挤压 (与 models/tfr.py msst Step4 同构), 用于 num 层之间的对照。"""
    F, T = omega_final.shape
    out = np.zeros_like(tfr_mag, dtype=np.float64)
    valid = omega_final >= 1
    k = np.clip(omega_final - 1, 0, F - 1)
    np.add.at(out, (k[valid], np.broadcast_to(np.arange(T), (F, T))[valid]),
              tfr_mag[valid])
    return out


def pearson(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def renyi2d(mag):
    p = mag.astype(np.float64) ** 2
    p = p / (p.sum() + 1e-12)
    return float(-np.log((p ** 2).sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--class', dest='cls', type=int, default=1)
    ap.add_argument('--idx', type=int, default=None)
    ap.add_argument('--ckpt', default='sast_checkpoints/retrain_e15/sast_v3_e015.pt')
    ap.add_argument('--out', default='results/msst_check/msst_infer_vs_ref.png')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y = d['train_X'], d['train_y'].ravel()
    if X.ndim == 3:
        X = X[:, :, 0]
    idx = args.idx if args.idx is not None else int(np.where(y == args.cls)[0][0])
    x = X[idx][:2000].astype(np.float64)
    FS = 1000
    print(f'sample: class={int(y[idx])} idx={idx}  N={len(x)}')

    # ── 参考侧 (numpy, plot_msst_tfr.py 的配置) ──
    r1 = msst_np(x, FS, hlength=320, num=1, save_trajectory=True)
    r4 = msst_np(x, FS, hlength=320, num=4, save_trajectory=True)
    mag_r1 = np.abs(r1['STFT'])
    freqs = r1['freqs']
    m_r1 = r1['MSST']                                    # num=1 硬挤压
    m_r4 = squeeze_hard_ref(mag_r1, np.asarray(r4['omegas'][-1]) + 1)
    # 说明: msst() 返回的 omegas 已是 0-based? 保守起见两者都试, 取与官方 MSST 相关性高的
    m_r4b = squeeze_hard_ref(mag_r1, np.asarray(r4['omegas'][-1]))
    if pearson(m_r4, m_r4b) < 0.999:
        print('  [warn] omegas 索引约定: 取 b 版本')
    print(f'  ref num=1: shape={m_r1.shape}  freqs {freqs[0]:.1f}-{freqs[-1]:.1f} Hz')
    print(f'  ref num=4: shape={m_r4.shape}')

    # ── SAST 内部 tfr_msst ──
    import torch
    from sast_model import load_checkpoint
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, _ = load_checkpoint(args.ckpt, dev)
    model.eval()
    with torch.no_grad():
        res = model(torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(dev))
    m_s = res['tfr_msst'][0].float().cpu().numpy()
    f_s = res['freqs'].float().cpu().numpy()
    print(f'  SAST internal: shape={m_s.shape}  freqs {f_s[0]:.1f}-{f_s[-1]:.1f} Hz'
          f'  msst_hlength={model.node_extractor.msst_hlength}  num={model.node_extractor.msst_num}')

    # ── 数值对齐 (同形状才比) ──
    print('\n--- 数值对比 (0-500 Hz 全带) ---')
    pairs = [('ref num=1 vs SAST', m_r1, m_s),
             ('ref num=4 vs SAST', m_r4, m_s),
             ('ref num=1 vs ref num=4', m_r1, m_r4)]
    for name, a, b in pairs:
        if a.shape != b.shape:
            print(f'{name}: shape 不一致 {a.shape} vs {b.shape}, 跳过')
            continue
        num = np.abs(a - b).sum()
        den = np.abs(a).sum() + 1e-12
        print(f'{name}:  r={pearson(a, b):.4f}   L1 相对误差={num/den*100:5.2f}%   '
              f'Rényi2D {renyi2d(a):.3f} vs {renyi2d(b):.3f}')

    # 50 Hz / 100 Hz 附近能量占比 (定位"像不像")
    def band_frac(m, f, lo, hi):
        sel = (f >= lo) & (f < hi)
        return float((m[sel] ** 2).sum() / ((m ** 2).sum() + 1e-12))
    print('\n--- 关键带能量占比 (0-500 Hz) ---')
    for lo, hi in [(2, 18), (42, 48.5), (48.5, 53), (90, 105), (53, 90)]:
        print(f'  [{lo:5.1f},{hi:5.1f}) Hz: ref num=1 {100*band_frac(m_r1, freqs, lo, hi):5.2f}%'
              f'   ref num=4 {100*band_frac(m_r4, freqs, lo, hi):5.2f}%'
              f'   SAST {100*band_frac(m_s, f_s, lo, hi):5.2f}%')

    # ── 图: 上排参考侧, 下排 SAST 内部; 列 = 显示方式 ──
    t = np.arange(x.size) / FS
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    def panel_linear(ax, m, f, fmax, title):
        ms = f <= fmax
        vmax = m[ms].max() * 0.5          # 与 plot_msst_tfr.py 完全一致的显示口径
        ax.pcolormesh(t, f[ms], m[ms], shading='gouraud', cmap='jet', vmax=vmax)
        ax.set_ylim(0, fmax); ax.set_xlim(0, t[-1])
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Freq (Hz)')

    def panel_db(ax, m, f, fmax, title):
        ms = f <= fmax
        db = 10 * np.log10(np.maximum(m[ms], 1e-6))
        vmax = db.max()
        ax.pcolormesh(t, f[ms], db, shading='gouraud', cmap='jet',
                      vmin=vmax - 25.0, vmax=vmax)
        ax.set_ylim(0, fmax); ax.set_xlim(0, t[-1])
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Freq (Hz)')

    panel_linear(axes[0, 0], m_r1, freqs, 500,
                 'ref: msst(hlength=320, num=1) | linear vmax=0.5max | 0-500 Hz\n'
                 '= hmst_figures/msst_comparison 的显示口径')
    panel_linear(axes[0, 1], m_r1, freqs, 200,
                 'ref 同上, 但频率窗收到 0-200 Hz')
    panel_db(axes[0, 2], m_r1, freqs, 200,
             'ref 同上, 但换成 dB (峰值下 25 dB)')
    panel_linear(axes[1, 0], m_s, f_s, 500,
                 f'SAST 内部 tfr_msst (hlength={model.node_extractor.msst_hlength}, num={model.node_extractor.msst_num})'
                 ' | linear | 0-500 Hz')
    panel_linear(axes[1, 1], m_s, f_s, 200,
                 'SAST 内部 | linear | 0-200 Hz')
    panel_db(axes[1, 2], m_s, f_s, 200,
             'SAST 内部 | dB 25 | 0-200 Hz  ← infer_out_*/panel (a) 实际画面')

    fig.suptitle(f'MSST panel alignment — class {int(y[idx])} idx={idx}  '
                 f'(same signal; columns = display, rows = algorithm config)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'\nplot -> {out}')


if __name__ == '__main__':
    main()
