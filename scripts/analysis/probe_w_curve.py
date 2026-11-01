"""
w_i(t) 时序体检 —— 重训后逐物理节点看挤压宽度是否"钉在小范围里"
==================================================================
判据来自 models/sast_nodes.PUMP_TURBINE_REGIONS[].w_range_* (3 节点, 单一真源):
    LOW_FREQ [0.05, 0.35]   BPF [0.80, 0.98]   2xBPF [0.80, 0.98]
下界在分量缺席时由 presence 门控松弛到 w_low_min = 0.05 (见 w_range_loss)。

本脚本不训练、不改模型, 只跑推理 + 统计 + 画图。

逐 (类别, 节点) 指标 (先在样本内沿时间算, 再跨样本平均):
  mean      管内时间均值
  std_t     管内时间标准差              —— 越小越"平"
  ptp_t     管内时间极差 max-min        —— 直接对应"是否钉在小范围"
  max_dw    相邻帧最大跳变 |Δw|         —— 逐帧毛刺
  p99_dw    |Δw| 的 99 分位
  max_dw50  |w(t+50ms)-w(t)| 最大值     —— 2 s 图上肉眼可见的"台阶"高度
  jump_rate |Δw|       > 0.02 的帧占比  —— 毛刺口径
  step_rate |Δ_50ms w| > 0.05 的帧占比  —— 台阶口径
  out_rate  w ∉ [w_low−ε, w_high+ε] 的帧占比 (ε=RANGE_TOL)。**必须留容差**:
            w_range_loss 的二次铰链最优解就落在边界上, 收敛后 w_i 会精确停在
            w_high (如 0.98000…1), 严格 `>` 会把它记成"越界"并报出 100%,
            掩盖真正逃出带外的样本
  below_floor  w < 0.05+eps 的帧占比 (靠 presence 松弛活着)
  peak_at / trough_at  w 取到最大/最小的相对时刻 (0→1) —— 判定是否窗口边缘效应

输出 (默认 results/w_curve/):
  w_curve_stats.csv          逐 (class, node) 汇总
  w_curve_per_sample.csv     逐样本原始指标 (可复查极值来自哪个样本)
  w_curve_class{c}.png       每类一张: 逐节点子图, 抽样样本曲线 + 类均值
  w_curve_contact.png        5 类 × 逐节点 均值曲线网格

用法:
  python scripts/analysis/probe_w_curve.py \
      --checkpoint sast_checkpoints/retrain_e15/sast_v3_e015.pt \
      --per-class 8 --outdir results/w_curve
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from sast_model import load_checkpoint                       # noqa: E402
from models.sast_nodes import PUMP_TURBINE_REGIONS           # noqa: E402

NODE_NAMES = [r.name for r in PUMP_TURBINE_REGIONS]
N_NODE = len(NODE_NAMES)
W_LOW = np.array([r.w_range_low for r in PUMP_TURBINE_REGIONS], dtype=np.float64)
W_HIGH = np.array([r.w_range_high for r in PUMP_TURBINE_REGIONS], dtype=np.float64)
ABSENT_FLOOR = 0.05          # w_low_min: 分量缺席时下界松弛到的值
CLASS_NAMES = ['No-load', 'Low load', 'Mid load', 'High load', 'Pumping']
NODE_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

JUMP_TOL = 0.02              # 逐帧"毛刺"阈值
STEP_TOL = 0.05              # 50 ms 粗尺度"台阶"阈值
COARSE_HOP = 50              # 50 帧 = 50 ms (fs=1000, hop=1)
RANGE_TOL = 1e-3             # w_range 越界判定的边界容差 (见 out_rate 说明)

METRIC_KEYS = ('mean', 'std_t', 'ptp_t', 'max_dw', 'p99_dw', 'max_dw50',
               'jump_rate', 'step_rate', 'out_rate', 'below_floor',
               'peak_at', 'trough_at')


def sample_metrics(w: np.ndarray):
    """w: [N_phys, T] -> {metric: [N_phys]}"""
    N, T = w.shape
    dw = np.abs(np.diff(w, axis=1))                      # [N, T-1]
    dc = np.abs(w[:, COARSE_HOP:] - w[:, :-COARSE_HOP])  # 50 ms 差分
    out = {k: [] for k in METRIC_KEYS}
    for n in range(N):
        ww = w[n]
        out['mean'].append(float(ww.mean()))
        out['std_t'].append(float(ww.std()))
        out['ptp_t'].append(float(ww.max() - ww.min()))
        out['max_dw'].append(float(dw[n].max()))
        out['p99_dw'].append(float(np.quantile(dw[n], 0.99)))
        out['max_dw50'].append(float(dc[n].max()))
        out['jump_rate'].append(float((dw[n] > JUMP_TOL).mean()))
        out['step_rate'].append(float((dc[n] > STEP_TOL).mean()))
        out['out_rate'].append(
            float(((ww < W_LOW[n] - RANGE_TOL)
                   | (ww > W_HIGH[n] + RANGE_TOL)).mean()))
        out['below_floor'].append(float((ww < ABSENT_FLOOR + 1e-6).mean()))
        out['peak_at'].append(float(ww.argmax() / max(1, T - 1)))
        out['trough_at'].append(float(ww.argmin() / max(1, T - 1)))
    return out


def main():
    ap = argparse.ArgumentParser(description='w_i(t) 时序体检')
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--checkpoint', default='sast_checkpoints/retrain_e15/sast_v3_e015.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--per-class', type=int, default=8)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--cut', type=int, default=2000)
    ap.add_argument('--outdir', default='results/w_curve')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model, _, meta = load_checkpoint(args.checkpoint, dev)
    model.eval()
    print(f'checkpoint: {args.checkpoint}')
    print(f'device: {dev}  fs={meta["config"].fs} Hz  per_class={args.per_class}')

    d = np.load(args.data, allow_pickle=True)
    X, y = d['train_X'], d['train_y'].ravel().astype(np.int64)
    if X.ndim == 3:
        X = X[:, :, 0]
    X = X[:, :args.cut]
    classes = sorted(np.unique(y).tolist())

    # ── 推理 ──
    curves, t_axis = {}, None
    for c in classes:
        cidx = np.where(y == c)[0]
        # 等间隔取样: 覆盖整段录制, 而不是只看能量最高的那一小撮
        chosen = cidx[np.linspace(0, len(cidx) - 1,
                                  min(args.per_class, len(cidx))).astype(int)]
        ws = []
        for i in range(0, len(chosen), args.batch):
            xb = torch.from_numpy(
                X[chosen[i:i + args.batch]].astype(np.float32)).to(dev)
            with torch.no_grad():
                res = model(xb, return_all=False, training=False)
            ws.append(res['w_i'].float().cpu().numpy())
            if t_axis is None:
                t_axis = res['t_axis'].float().cpu().numpy()
        curves[c] = np.concatenate(ws, axis=0)            # [n, N, T]
        print(f'  class {c} ({CLASS_NAMES[c]}): n={curves[c].shape[0]}  '
              f'S={curves[c].shape[1]}  T={curves[c].shape[2]}')

    # ── 统计 ──
    per_sample_rows, agg_rows = [], []
    for c in classes:
        w = curves[c]
        for s in range(w.shape[0]):
            m = sample_metrics(w[s])
            for n in range(N_NODE):
                per_sample_rows.append(
                    [c, CLASS_NAMES[c], s, NODE_NAMES[n]]
                    + [f'{m[k][n]:.5f}' for k in METRIC_KEYS])
        # 类内先把样本拼起来(等价于"以样本为独立单位取平均")
        ms = [sample_metrics(w[s]) for s in range(w.shape[0])]
        for n in range(N_NODE):
            row = {'class': c, 'class_name': CLASS_NAMES[c], 'node': NODE_NAMES[n],
                   'w_low': float(W_LOW[n]), 'w_high': float(W_HIGH[n])}
            for k in METRIC_KEYS:
                v = np.array([mm[k][n] for mm in ms])
                # 计数类指标取均值; 极值类指标取最坏
                row[k] = float(v.max() if k.startswith('max_') else v.mean())
            agg_rows.append(row)

    # ── 控制台表 ──
    hdr = (f'{"class":<10s} {"node":<10s} {"mean":>6s} {"std_t":>6s} {"ptp_t":>6s} '
           f'{"max|dw|":>8s} {"max|d50|":>8s} {"jump%":>6s} {"step%":>6s} '
           f'{"out%":>6s} {"floor%":>7s} {"w_range":>12s}')
    print('\n' + '=' * len(hdr))
    print('w_i(t) 时序稳定性 (指标先在样本内沿时间算, 再跨样本平均/取最坏)')
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))
    for r in agg_rows:
        print(f'{r["class_name"]:<10s} {r["node"]:<10s} {r["mean"]:6.3f} '
              f'{r["std_t"]:6.4f} {r["ptp_t"]:6.4f} {r["max_dw"]:8.4f} '
              f'{r["max_dw50"]:8.4f} {100 * r["jump_rate"]:5.1f}% '
              f'{100 * r["step_rate"]:5.1f}% {100 * r["out_rate"]:5.1f}% '
              f'{100 * r["below_floor"]:6.1f}%   [{r["w_low"]:.2f},{r["w_high"]:.2f}]')

    # 跨类汇总: 每个节点的"最坏情况" —— 论文里该报这个
    hdr2 = (f'{"ALL":<10s} {"node":<10s} {"mean":>6s} {"std_t":>6s} {"ptp_t":>6s} '
            f'{"max|dw|":>8s} {"max|d50|":>8s} {"jump%":>6s} {"step%":>6s} '
            f'{"out%":>6s}')
    print('\n' + '-' * len(hdr2))
    print(hdr2)
    print('-' * len(hdr2))
    for n, node in enumerate(NODE_NAMES):
        rn = [r for r in agg_rows if r['node'] == node]
        print(f'{"ALL":<10s} {node:<10s} '
              f'{np.mean([r["mean"] for r in rn]):6.3f} '
              f'{np.max([r["std_t"] for r in rn]):6.4f} '
              f'{np.max([r["ptp_t"] for r in rn]):6.4f} '
              f'{np.max([r["max_dw"] for r in rn]):8.4f} '
              f'{np.max([r["max_dw50"] for r in rn]):8.4f} '
              f'{100 * np.max([r["jump_rate"] for r in rn]):5.1f}% '
              f'{100 * np.max([r["step_rate"] for r in rn]):5.1f}% '
              f'{100 * np.max([r["out_rate"] for r in rn]):5.1f}%')

    # ── CSV ──
    with open(outdir / 'w_curve_stats.csv', 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.writer(f)
        wr.writerow(['class', 'class_name', 'node'] + list(METRIC_KEYS)
                    + ['w_low', 'w_high'])
        for r in agg_rows:
            wr.writerow([r['class'], r['class_name'], r['node']]
                        + [f'{r[k]:.5f}' for k in METRIC_KEYS]
                        + [f'{r["w_low"]:.2f}', f'{r["w_high"]:.2f}'])
    with open(outdir / 'w_curve_per_sample.csv', 'w', newline='',
              encoding='utf-8-sig') as f:
        wr = csv.writer(f)
        wr.writerow(['class', 'class_name', 'sample', 'node'] + list(METRIC_KEYS))
        wr.writerows(per_sample_rows)
    print(f'\nCSV -> {outdir / "w_curve_stats.csv"}')
    print(f'CSV -> {outdir / "w_curve_per_sample.csv"}')

    # ── 画图 ──
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.unicode_minus'] = False

    def draw_curves(ax, w2d, n, title):
        lo, hi = W_LOW[n], W_HIGH[n]
        ax.axhspan(lo, hi, color='#2ca02c', alpha=0.07, label='w_range')
        ax.axhline(ABSENT_FLOOR, color='k', lw=0.6, ls=':', alpha=0.6,
                   label='absent floor 0.05')
        for s in range(w2d.shape[0]):
            ax.plot(t_axis, w2d[s], lw=0.6, alpha=0.35, color=NODE_COLORS[n])
        ax.plot(t_axis, w2d.mean(axis=0), lw=1.6, color=NODE_COLORS[n],
                label='mean over samples')
        ax.set_ylim(0, 1.02)
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylabel('w_i')
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_title(title, fontsize=8)

    # 每类一张 (节点数随 PUMP_TURBINE_REGIONS 变化: 现为 3)
    n_cols = min(3, N_NODE)
    n_rows = int(np.ceil(N_NODE / n_cols))
    for c in classes:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 2.8 * n_rows),
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes.ravel()
        w = curves[c]
        rows_c = [r for r in agg_rows if r['class'] == c]
        for n in range(N_NODE):
            r = rows_c[n]
            draw_curves(axes[n], w[:, n, :], n,
                        f'{NODE_NAMES[n]}  mean={r["mean"]:.3f}  '
                        f'std_t={r["std_t"]:.4f}  ptp={r["ptp_t"]:.4f}  '
                        f'max|dw50|={r["max_dw50"]:.3f}')
            axes[n].legend(fontsize=6, loc='upper right', ncol=3)
        for ax in axes[N_NODE:]:
            ax.axis('off')
        for ax in axes[n_cols * (n_rows - 1):]:
            ax.set_xlabel('time (s)')
        fig.suptitle(f'w_i(t) stability — {CLASS_NAMES[c]} (class {c}), '
                     f'n={w.shape[0]} samples, ckpt={Path(args.checkpoint).stem}',
                     fontsize=11, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = outdir / f'w_curve_class{c}.png'
        fig.savefig(out, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f'  plot -> {out}')

    # 5 类 × 逐节点 均值曲线网格
    fig, axes = plt.subplots(len(classes), N_NODE,
                             figsize=(13, 1.9 * len(classes)),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, c in enumerate(classes):
        w = curves[c]
        for n in range(N_NODE):
            ax = axes[i, n]
            ax.axhspan(W_LOW[n], W_HIGH[n], color='#2ca02c', alpha=0.07)
            ax.axhline(ABSENT_FLOOR, color='k', lw=0.5, ls=':', alpha=0.5)
            ax.plot(t_axis, w[:, n, :].mean(axis=0), lw=1.3, color=NODE_COLORS[n])
            r = [x for x in agg_rows if x['class'] == c and x['node'] == NODE_NAMES[n]][0]
            ax.text(0.02, 0.96,
                    f'mean={r["mean"]:.3f}\nptp={r["ptp_t"]:.3f}\n'
                    f'max|d50|={r["max_dw50"]:.3f}',
                    transform=ax.transAxes, fontsize=6, va='top', ha='left')
            ax.set_ylim(0, 1.02)
            ax.set_xlim(t_axis[0], t_axis[-1])
            ax.grid(alpha=0.2, lw=0.4)
            if i == 0:
                ax.set_title(f'{NODE_NAMES[n]}\n'
                             f'range [{W_LOW[n]:.2f},{W_HIGH[n]:.2f}]', fontsize=8)
            if n == 0:
                ax.set_ylabel(f'{CLASS_NAMES[c]}\nw_i', fontsize=7)
            if i == len(classes) - 1:
                ax.set_xlabel('time (s)')
    fig.suptitle(f'w_i(t) per class x node (mean over samples) — '
                 f'{Path(args.checkpoint).stem}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = outdir / 'w_curve_contact.png'
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'  plot -> {out}')


if __name__ == '__main__':
    main()
