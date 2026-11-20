"""SAST v4 逐类诊断图 (5 类 × N 张对照)
=====================================

版式沿用 ``plot_retrained_diagnostics.py`` (2×4):

    行 1:  (a) 双窗底图 S_base | (b) 长窗重排 R_long | (c) WSST1 | (d) SAST v4 T
    行 2:  (e) w_i(t) 三节点 | (f) FFT 频谱 + 节点 IF

三个时频面板全部走 ``tfr_render.tfr_to_rgb`` (jet / vmin=0 / vmax=逐图峰值 /
0-200 Hz / 与 CNN 输入同口径)。

FFT 面板用 ``sast_viz._plot_fft_panel(annotate=True)``: **水平短线**标在
谱线的实际幅值高度上, 上方标频率值 —— 不用竖线, 竖线会横跨整个坐标区、
盖住谱线本身, 看不出"这条线有多高"。

用法
----
  python -m scripts.plot.plot_v4_diagnostics                    # 每类 5 张
  python -m scripts.plot.plot_v4_diagnostics --per-class 5
  python -m scripts.plot.plot_v4_diagnostics --samples 0:2431,0:309
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch                                                  # noqa: E402

import matplotlib                                             # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib.colors import Normalize                       # noqa: E402

from models.sast_v4 import SASTv4                             # noqa: E402
from models.sast_nodes import band_centroid_if                # noqa: E402
from scripts.data.build_cnn_tfr_dataset import (              # noqa: E402
    CLASS_NAMES, N_CLASS, build_split, load_pools,
)
from scripts.plot.tfr_render import F_MAX, tfr_to_rgb         # noqa: E402
from sast_viz import _plot_fft_panel, _plot_w_i_panel         # noqa: E402

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FS = 1000.0


def add_colorbar(fig, ax, peak):
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, max(peak, 1e-12)), cmap='jet'),
        ax=ax, fraction=0.046, pad=0.01)
    cb.ax.tick_params(labelsize=6)
    cb.set_label('|TFR|', fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / '5_dataset.npz')
    ap.add_argument('--ckpt', type=Path,
                    default=ROOT / 'sast_checkpoints' / 'v4_gat2' / 'best.pt')
    ap.add_argument('--out', type=Path, default=ROOT / 'infer_out_v4_fix')
    ap.add_argument('--per-class', type=int, default=5)
    ap.add_argument('--samples', default=None,
                    help='显式样本 "c:i,c:i"; 缺省每类沿训练池等距取 per_class 个')
    ap.add_argument('--fmax', type=float, default=F_MAX)
    ap.add_argument('--rows', type=int, default=0,
                    help='渲染行数; 0 = 2*fmax+1 (行距恰为 STFT 的 0.5 Hz). '
                         '2026-11-14: 原为 max(256,2*fmax)=500 行 → 0.5010 '
                         'Hz/行, 与 0.5 Hz 栅格错位, R_long 的 1-bin 脊线被'
                         '线性插值削到 62%, T 的全图最大值因此翻到低频带')
    ap.add_argument('--beta-lo', type=float, default=None,
                    help='β≡0 区的上界 / 斜坡起点 [Hz]; 缺省取模块常量')
    ap.add_argument('--beta-hi', type=float, default=None,
                    help='β 斜坡终点 [Hz] (其上 β=w_node); 缺省取模块常量')
    ap.add_argument('--no_absence_gate', action='store_true',
                    help='按去门控配置构建模型 (加载 v4_smoke 类权重时必须)')
    ap.add_argument('--no_delta_omega', action='store_true',
                    help='不建 Δω 头 (加载去 Δω 的权重时必须)')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, Xte, yte = load_pools(args.data)
    y = np.asarray(y)
    splits = build_split(y, np.asarray(yte))

    model = SASTv4(use_gat=True, low_beta_max_hz=args.beta_lo,
                   beta_ramp_hi_hz=args.beta_hi,
                   gat_kwargs=dict(
                       use_absence_gate=not args.no_absence_gate,
                       use_delta_omega=not args.no_delta_omega),
                   ).to(device).eval()
    print(f'β 斜坡: [{model.low_beta_max_hz:g}, {model.beta_ramp_hi_hz:g}) Hz  '
          f'| LOW_FREQ 区域: [{model.regions[0].f_min:g}, '
          f'{model.regions[0].f_max:g}) Hz')
    loaded = False
    if args.ckpt and Path(args.ckpt).exists():
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck.get('state_dict', ck))
        loaded = True
    print(f'SAST v4 权重={args.ckpt}  已加载={loaded}')

    if args.samples:
        cases = [(int(a), int(b)) for a, b in
                 (t.split(':') for t in args.samples.split(',') if t.strip())]
    else:
        cases = []
        for c in range(N_CLASS):
            pool = [i for cc, i in splits['train'] if cc == c]
            if not pool:
                continue
            pick = np.linspace(0, len(pool) - 1, args.per_class).astype(int)
            cases += [(c, pool[k]) for k in pick]

    rows = args.rows if args.rows > 0 else 2 * int(round(args.fmax)) + 1

    print('加载 WSST1 ...')
    from scripts.plot.tfr_common import method_wsst1

    for c, idx in cases:
        x = np.asarray(X[idx], np.float32)
        if x.ndim > 1:
            x = x[:, 0]
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            T, d = model(xt, return_detail=True)
        T = T[0].cpu().numpy()
        S_base = d['S_base'][0].cpu().numpy()
        R_long = d['R_long'][0].cpu().numpy()
        w_i = d['w_node'][0].cpu().numpy()                  # [3, T]
        freqs = d['freqs'].cpu().numpy()
        t_axis = np.arange(T.shape[1]) / FS
        node_if = band_centroid_if(torch.from_numpy(T)[None].to(device),
                                   d['freqs'], model.regions
                                   )[0].mean(-1).cpu().numpy()

        try:
            wsst = method_wsst1(x.astype(np.float64))
        except Exception as exc:                            # ssqueezepy 缺失时降级
            print(f'  WSST 失败 ({exc}), 该面板留空')
            wsst = None

        _kw = dict(fmax=args.fmax, n_rows=rows, height=rows)
        p_stft, peak_stft = tfr_to_rgb(S_base, freqs, t_axis,
                                       return_peak=True, **_kw)
        p_rawl, peak_rawl = tfr_to_rgb(R_long, freqs, t_axis,
                                       return_peak=True, **_kw)
        p_sast, peak_sast = tfr_to_rgb(T, freqs, t_axis,
                                       return_peak=True, **_kw)
        if wsst is not None:
            p_wsst, peak_wsst = tfr_to_rgb(wsst['mag'], wsst['freqs'],
                                           wsst['t'], return_peak=True, **_kw)
        else:
            p_wsst, peak_wsst = np.zeros((rows, 8, 3), np.uint8), float('nan')

        names = [r.name for r in model.regions]
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 3)))

        fig = plt.figure(figsize=(28, 12))
        gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 0.9],
                              hspace=0.30, wspace=0.18)
        panels = [
            (fig.add_subplot(gs[0, 0]), p_stft, peak_stft, '(a) S_base 双窗底图'),
            (fig.add_subplot(gs[0, 1]), p_rawl, peak_rawl, '(b) R_long 长窗纯重排'),
            (fig.add_subplot(gs[0, 2]), p_wsst, peak_wsst, '(c) WSST1'),
            (fig.add_subplot(gs[0, 3]), p_sast, peak_sast, '(d) SAST v4  T'),
        ]
        for ax, rgb, peak, title in panels:
            # interpolation='nearest': 面板高约 485 px, 而 rows=501 (0.5 Hz/行),
            # 默认的 antialiased 会把 1 bin 宽的脊线与相邻行平均掉, 峰值再被
            # 削掉一半 (实测 c2_p03711 的 100 Hz 从深红掉成黄绿)。
            ax.imshow(rgb, extent=[float(t_axis[0]), float(t_axis[-1]),
                                   0, args.fmax],
                      aspect='auto', origin='upper', interpolation='nearest')
            ax.set_ylim(0, args.fmax)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title(f'{title}\n(vmax = peak = {peak:.3e})', fontsize=10)
            add_colorbar(fig, ax, peak)

        _plot_w_i_panel(fig.add_subplot(gs[1, :3]), w_i, t_axis, colors,
                        names, label='(e)')
        _plot_fft_panel(fig.add_subplot(gs[1, 3]), x, FS, args.fmax,
                        node_if, w_i, colors, names, label='(f)')

        fig.suptitle(f'class {c} — {CLASS_NAMES[c]}  (train pool #{idx})  '
                     f'β ramp [{model.low_beta_max_hz:g}, '
                     f'{model.beta_ramp_hi_hz:g}) Hz\n'
                     f'jet, vmin=0, vmax=per-image peak, 0–{args.fmax:g} Hz',
                     fontsize=12)
        out = args.out / f'sast_tfr_class{c}_p{idx:05d}{args.tag}.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {out}')


if __name__ == '__main__':
    main()
