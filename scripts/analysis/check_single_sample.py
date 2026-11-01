"""
单样本前向检查 —— BPF 带的 node_if 落点、时频图锐度与前向开销
=============================================================
本脚本只做推理与统计, 不训练、不改模型参数。

回答两个判据:
  (1) BPF 带的 node_if 落在 [40, 53) 内 (合并带后带内实际有 41 Hz 流动峰
      与 50 Hz 锁定线两条脊线, node_if 是能量加权值, 落在两者之间属正常)
  (2) 推理图中 41 / 50 Hz 两条脊线均清晰锐利, 无粗黄带与垂直毛刺

⚠ 3 节点合并 (B4 撤回) 后, 原来"BPF_LINE 是否被拉回 50 Hz"这一问已失去
  对象 —— 带内不再只装锁频线。本脚本保留同名指标 (带内 H2、分段重心),
  只是把比较对象从"漂移带 vs 线带"改成"合并带 + 每条脊线"。

三段:
  A. IF 估计扫描 (不依赖 checkpoint) —— hl ∈ {320,512,1024} × num ∈ {1,4}
     对每类各取 1 个样本, 报 BPF 的 node_if 分布、越界帧比例。
  B. 单样本前向 —— 用当前 checkpoint, 报 3 个物理节点的 node_if / w_i。
     ⚠ checkpoint 必须与 3 节点拓扑同源; 4 节点期的 ckpt 会因原型维度
       (5x6 vs 5x4) 与边数 (11 vs 10) 不一致而加载失败。
  C. 预览图 —— 三行: 原始 STFT | 当前 ckpt 的 SAST | w 落到 C_prior 后的 SAST。
     第三行是"重训收敛后应该长什么样"的事前预览 (w_override 诊断通道)。

用法:
  python scripts/analysis/check_single_sample.py --class 1
  python scripts/analysis/check_single_sample.py --class 3 --hl 1024 --num 1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from models.sast_nodes import (MSSTNodeExtractor, PUMP_TURBINE_REGIONS,   # noqa: E402
                               IF_LANDING_TOL_HZ)
from sast_model import load_checkpoint                                     # noqa: E402
from sast_config import SastConfig                                          # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

FS = 1000
CUT = 2000
CLASS_NAMES = ['No-load', 'Low load', 'Mid load', 'High load', 'Pumping']
NODE_NAMES = [r.name for r in PUMP_TURBINE_REGIONS]
IDX_BPF = [r.name for r in PUMP_TURBINE_REGIONS].index('BPF')
REG_BPF = PUMP_TURBINE_REGIONS[IDX_BPF]


# ── 指标 (与训练侧同口径的简化实现) ──
def renyi2d(tfr, freqs, lo, hi):
    """带内 Rényi-2D 熵 (越低越集中)。tfr: [F,T] 幅值。"""
    m = (freqs >= lo) & (freqs < hi)
    p = tfr[m, :].astype(np.float64) ** 2
    p = p / (p.sum() + 1e-12)
    return float(-np.log((p ** 2).sum() + 1e-12))


def band_centroid(tfr, freqs, lo, hi):
    """逐帧能量加权重心 (Hz) -> [T]; 用于看脊线是否随时间起伏。"""
    m = (freqs >= lo) & (freqs < hi)
    e = tfr[m, :].astype(np.float64) ** 2
    tot = e.sum(axis=0) + 1e-12
    return (freqs[m][:, None] * e).sum(axis=0) / tot


def load_sample(path, cls):
    d = np.load(path, allow_pickle=True)
    X = d['train_X']
    if X.ndim == 3:
        X = X[:, :, 0]
    y = d['train_y'].ravel()
    idx = int(np.where(y == cls)[0][0])
    return X[idx][:CUT].astype(np.float64), idx


# ── 阶段 A: IF 估计扫描 ──
def phase_a(X, y, hls, nums, n_per_class=1):
    print('=' * 78)
    print('A. IF 估计扫描 — BPF node_if (带 [%.1f, %.1f) Hz)'
          % (REG_BPF.f_min, REG_BPF.f_max))
    print('=' * 78)
    lo_tol = REG_BPF.f_min - IF_LANDING_TOL_HZ
    hi_tol = REG_BPF.f_max + IF_LANDING_TOL_HZ
    print(f'{"hl":>5s} {"num":>4s} | ' +
          ' | '.join(f'{n:^22s}' for n in ['class  median  min  max  out%']))
    print('-' * 78)
    for hl in hls:
        for num in nums:
            ex = MSSTNodeExtractor(fs=FS, msst_hlength=hl, msst_num=num)
            cells = []
            for c in sorted(np.unique(y).tolist()):
                ci = np.where(y == c)[0][0]
                nodes = ex(X[ci])
                v = nodes.if_hz[IDX_BPF]
                out = float(((v < lo_tol) | (v > hi_tol)).mean())
                cells.append(f'{c}  {np.median(v):6.2f} {v.min():5.2f} '
                             f'{v.max():5.2f} {100*out:5.1f}%')
            print(f'{hl:5d} {num:4d} | ' + ' | '.join(cells))


# ── 阶段 B/C 共用的小工具 ──
def _to_np(t):
    return t.detach().float().cpu().numpy()


def summarize(res, label):
    freqs = _to_np(res['freqs'])
    tfr_raw = _to_np(res['tfr_raw'][0])
    tfr_enh = _to_np(res['tfr_enhanced'][0])
    node_if = _to_np(res['node_if'][0])
    w_i = _to_np(res['w_i'][0])
    print(f'\n[{label}]')
    print(f'{"node":<11s} {"IF med":>8s} {"IF min":>8s} {"IF max":>8s} '
          f'{"w mean":>8s} {"w ptp":>7s}')
    for n, name in enumerate(NODE_NAMES):
        print(f'{name:<11s} {np.median(node_if[n]):8.2f} {node_if[n].min():8.2f} '
              f'{node_if[n].max():8.2f} {w_i[n].mean():8.3f} '
              f'{w_i[n].max() - w_i[n].min():7.3f}')
    # 带外是否还被挤压 (最高节点上界之外应逐位等于原始 STFT)
    f_top = max(r.f_max for r in PUMP_TURBINE_REGIONS)
    out_band = freqs > f_top
    if out_band.any():
        d = np.abs(tfr_enh[out_band] - tfr_raw[out_band]).max()
        print(f'  >{f_top:.0f} Hz 与原始 STFT 的最大逐点差: {d:.3e} '
              f'({"未挤压 ✓" if d < 1e-6 else "仍在挤压 ✗"})')
    for reg in PUMP_TURBINE_REGIONS:
        print(f'  {reg.name:<10s} H2(raw)={renyi2d(tfr_raw, freqs, reg.f_min, reg.f_max):.4f}'
              f'  H2(enh)={renyi2d(tfr_enh, freqs, reg.f_min, reg.f_max):.4f}')
    # BPF 带内的两条脊线: 41-44 Hz 流动分量 vs 49-52 Hz 锁定线,
    # 逐帧重心的时间统计 (重心飘动 = 脊线不稳)。
    for lo, hi, tag in ((40.0, 46.0, '41-44 Hz 流动分量'),
                        (49.0, 52.0, '50 Hz 锁频线  ')):
        tr = band_centroid(tfr_enh, freqs, lo, hi)
        print(f'  {tag}: 重心均值={tr.mean():.2f} Hz  std_t={tr.std():.3f} Hz')
    return freqs, tfr_raw, tfr_enh


def panel(ax, tfr, freqs, t, f_lo, f_hi, title, scale='db'):
    m = (freqs >= f_lo) & (freqs <= f_hi)
    if scale == 'db':
        d = 10 * np.log10(np.maximum(tfr[m], 1e-6))
        vm = d.max()
        ax.pcolormesh(t, freqs[m], d, shading='gouraud', cmap='jet',
                      vmin=vm - 25.0, vmax=vm)
    else:
        d = tfr[m]
        ax.pcolormesh(t, freqs[m], d, shading='gouraud', cmap='jet',
                      vmin=0.0, vmax=max(d.max() * 0.5, 1e-12))
    ax.set_ylim(f_lo, f_hi)
    ax.set_xlim(t[0], t[-1])
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('Time [s]')


def main():
    ap = argparse.ArgumentParser(description='单样本前向检查')
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--checkpoint', default='sast_checkpoints/retrain_e15/sast_v3_e015.pt')
    ap.add_argument('--class', dest='cls', type=int, default=1)
    _cfg = SastConfig()
    ap.add_argument('--hl', type=int, default=_cfg.msst_hlength,
                    help='阶段 B/C 使用的窗长 (默认取 SastConfig)')
    ap.add_argument('--num', type=int, default=_cfg.msst_num,
                    help='阶段 B/C 使用的迭代层数 (默认取 SastConfig)')
    ap.add_argument('--out', default='results/msst_check/single_sample_check.png')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y = d['train_X'], d['train_y'].ravel()
    if X.ndim == 3:
        X = X[:, :, 0]
    X = X[:, :CUT].astype(np.float32)

    # ── A ──
    phase_a(X, y, hls=[320, 512, 1024], nums=[1, 4])

    # ── B ──
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, _ = load_checkpoint(args.checkpoint, dev)
    # checkpoint 里存的是旧口径 (num=4); 这里覆盖成新口径以验证改动
    model.msst_num = args.num
    model.node_extractor.msst_num = args.num
    model.node_extractor.msst_hlength = args.hl

    idx = int(np.where(y == args.cls)[0][0])
    x_np = X[idx]
    xb = torch.from_numpy(x_np).unsqueeze(0).to(dev)
    print(f'\nsample: class={args.cls} ({CLASS_NAMES[args.cls]}) idx={idx} '
          f'N={len(x_np)}  hl={args.hl} num={args.num}')
    with torch.no_grad():
        res = model(xb, training=False)

    print('\n' + '=' * 78)
    print('B. 单样本前向 (ckpt 的 GAT 权重仍是旧分类下训练的, w 仅供参考)')
    print('=' * 78)
    summarize(res, 'current ckpt')

    # ── C ──
    # 预览: 把 w_i 钉到各节点 C_prior (= w_center_loss 想拉到的平衡点),
    # 代表"重训收敛后的预期形态"。3 节点: 0.20 / 0.90 / 0.95。
    T = res['w_i'].shape[-1]
    w_prev = torch.tensor([r.C_prior for r in PUMP_TURBINE_REGIONS],
                          device=dev, dtype=torch.float32)
    w_prev = w_prev.view(1, -1, 1).expand(1, -1, T)
    with torch.no_grad():
        res_prev = model(xb, training=False, w_override=w_prev)

    print('\n' + '=' * 78)
    print('C. 预览: w_i -> C_prior ' +
          str({r.name: r.C_prior for r in PUMP_TURBINE_REGIONS}))
    print('=' * 78)
    freqs, tfr_raw, _ = summarize(res_prev, 'preview w=C_prior')

    t = _to_np(res['t_axis'])
    tfr_cur = _to_np(res['tfr_enhanced'][0])
    tfr_prev = _to_np(res_prev['tfr_enhanced'][0])

    fig, axes = plt.subplots(3, 3, figsize=(17, 11))
    rows = [('(a) raw STFT', tfr_raw),
            ('(b) SAST — current ckpt', tfr_cur),
            ('(c) SAST — preview w=C_prior', tfr_prev)]
    for i, (name, m) in enumerate(rows):
        panel(axes[i, 0], m, freqs, t, 0, 120, f'{name} | dB | 0-120 Hz', 'db')
        panel(axes[i, 1], m, freqs, t, 0, 120, f'{name} | linear | 0-120 Hz', 'lin')
        panel(axes[i, 2], m, freqs, t, 40, 56, f'{name} | dB | 40-56 Hz zoom', 'db')
    fig.suptitle(f'single-sample check — class {args.cls} '
                 f'({CLASS_NAMES[args.cls]}) idx={idx}  hl={args.hl} num={args.num}',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'\nplot -> {out}')


if __name__ == '__main__':
    main()
