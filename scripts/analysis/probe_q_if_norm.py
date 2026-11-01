"""
Q_IF 对照探针 v2 (只读): 噪声地板归一化 + 脊线核心 bin
=====================================================
第一版 (probe_q_if.py) 的事前判据 1/3 通过, 两个失败点诊断清楚:

  (1) LOW_FREQ 的阈值 0.30 低于指标自身的地板 —— 纯白噪声的 Q_raw ≈ 0.45
      (q_v 在邻 bin 随机游走时期望 exp(-1.4/2) ≈ 0.5)。
  (3) BPF 的 Q 随负荷**下降** (空转 0.61 → 高负荷 0.35), 与"锁频线随负荷增强"
      的预期相反。机理是带内 41 Hz 流动峰与 50 Hz 锁频线共存, 两簇脊线之间
      的 ω̂ 逐 bin 跳变拉低 q_v —— 即指标度量的是"带内谱纯度"而非"脊线 IF 质量"。

本探针只改两处, 不改 q_v / q_t 公式, 不改数据/ω̂ (仍用 msst_torch num=1,
hlength=1024 的 omega_final 与未增强 STFT):

  A. 噪声地板归一化
       Q_norm = clamp((Q_raw - q_noise) / (1 - q_noise), 0, 1)
     q_noise 由**同口径**白噪声标定, 且按 (估计口径 × 节点) 逐个标定 ——
     地板本身随"是否只取核心 bin"而变, 用单一常数会混入口径差。

  B. 脊线核心 bin
     每帧每带只统计能量 ≥ τ·max_{f∈Band} E(f,t) 的格点 (τ = 0.25),
     以及"每帧带内 top-3 能量 bin"两种取法, 排除两簇脊线交界分水岭的跳变。

同时打印**含 presence 门控**与**不含门控**两组值 (Q_norm 先归一化再乘门控)。

输出 results/band_census/q_if_norm_probe.csv, 不覆盖已跟踪文件。
用法: python scripts/analysis/probe_q_if_norm.py --per-class 10
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent.parent
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_HERE))

from sast_data import load_dataset                                  # noqa: E402
from models.msst_torch import msst_torch                            # noqa: E402
from models.sast_nodes import PUMP_TURBINE_REGIONS, region_weights  # noqa: E402
from models.sast_losses import node_presence                        # noqa: E402
from probe_q_if import (CLASS_NAMES, DT_FRAMES, FS, NODE_NAMES,     # noqa: E402
                        q_terms)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

TAU = 0.25
TOPK = 3
PRESENCE_NORM = 0.20
EPS = 1e-12
N_NOISE = 8
VARIANTS = ('full', 'tau25', 'top3')


def band_scores(E, q, W, core_mode):
    """逐节点逐帧 (Q_raw) —— 只在 core 选中的格点上做能量加权平均。

    E: [F, T'] 能量 |S|^2 (已切到有效帧段)
    q: [F, T'] q_v * q_t
    W: [N, F] 带权重
    core_mode: 'full' | 'tau25' | 'top3'

    Returns [N, T']
    """
    N, T = W.shape[0], E.shape[1]
    out = torch.zeros(N, T, device=E.device, dtype=E.dtype)
    for n in range(N):
        b = W[n] > 0
        Eb, qb, wb = E[b], q[b], W[n][b].view(-1, 1)      # [Fn, T']
        if core_mode == 'full':
            core = torch.ones_like(Eb, dtype=torch.bool)
        elif core_mode == 'tau25':
            pmax = Eb.max(dim=0).values.clamp(min=EPS)
            core = Eb >= TAU * pmax
        elif core_mode == 'top3':
            k = min(TOPK, Eb.shape[0])
            core = torch.zeros_like(Eb, dtype=torch.bool)
            core.scatter_(0, Eb.topk(k, dim=0).indices, True)
        else:
            raise ValueError(core_mode)
        den = (Eb * wb * core).sum(dim=0) + EPS
        out[n] = (Eb * wb * core * qb).sum(dim=0) / den
    return out


def frame_scores(S, omega, freqs, device):
    """一条信号 -> (Q_raw per variant [N,T'], presence [N,T'], gate [N,T'])."""
    mag = S.abs()
    E = mag ** 2
    q, _, _ = q_terms(omega, E)
    sl = slice(DT_FRAMES, omega.shape[1] - DT_FRAMES)
    W = region_weights(freqs, PUMP_TURBINE_REGIONS)
    pres = node_presence(mag.unsqueeze(0), freqs, PUMP_TURBINE_REGIONS,
                         frac=0.1)[0]
    gate = (pres[:, sl] / PRESENCE_NORM).clamp(0.0, 1.0)
    scores = {v: band_scores(E[:, sl], q[:, sl], W, v) for v in VARIANTS}
    return scores, pres[:, sl], gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--hlength', type=int, default=1024)
    ap.add_argument('--num', type=int, default=1)
    ap.add_argument('--out', default='results/band_census/q_if_norm_probe.csv')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    freqs = torch.arange(1000, device=device, dtype=torch.float32) / 2000 * FS

    def run_one(x_np):
        x = torch.from_numpy(np.asarray(x_np, dtype=np.float64)).to(device)
        r = msst_torch(x, FS, hlength=args.hlength, num=args.num,
                       save_trajectory=False, skip_squeeze=True)
        return frame_scores(r['STFT'], r['omega_final'], freqs, device)

    t0 = time.time()

    # ── A. 同口径白噪声标定地板 (按 口径 × 节点) ──
    rng = np.random.RandomState(1234)
    floors = {v: np.zeros(len(NODE_NAMES)) for v in VARIANTS}
    for _ in range(N_NOISE):
        sc, _, _ = run_one(rng.randn(2000))
        for v in VARIANTS:
            floors[v] += sc[v].mean(dim=1).cpu().numpy()
    floors = {v: floors[v] / N_NOISE for v in VARIANTS}

    print(f'[calib] 白噪声地板 q_noise (逐口径 × 节点), device={dev}')
    hd = f'{"口径":<8}' + ''.join(f'{nm:>12}' for nm in NODE_NAMES)
    print(hd)
    for v in VARIANTS:
        print(f'{v:<8}' + ''.join(f'{floors[v][n]:>12.3f}'
                                  for n in range(len(NODE_NAMES))))
    print('  注: Q_norm 用**匹配口径**的节点地板归一, 而非全局常数 0.45。')

    # ── B. 验证集 5 工况抽样 ──
    ds = load_dataset(args.data, max_samples=None, val_split=0.15, seed=42)
    X, y = ds['val_X'], ds['val_y']
    rng = np.random.RandomState(0)
    picks = []
    for c in range(len(CLASS_NAMES)):
        idx = np.where(y == c)[0]
        sel = rng.choice(idx, size=min(args.per_class, len(idx)), replace=False)
        picks.extend((int(i), c) for i in sel)
    print(f'\n[data] val = {len(X)} 条, 每类 {args.per_class} 条 '
          f'(共 {len(picks)} 条)')

    rows = []
    for i, c in picks:
        sc, pres, gate = run_one(X[i])
        rec = dict(cls=c, cls_name=CLASS_NAMES[c])
        for n, nm in enumerate(NODE_NAMES):
            for v in VARIANTS:
                raw = sc[v][n]
                fl_n = torch.as_tensor(floors[v][n], device=device,
                                       dtype=torch.float32)
                qn = ((raw - fl_n) / (1.0 - fl_n)).clamp(0.0, 1.0)
                rec[f'{v}_raw_{nm}'] = raw.mean().item()
                rec[f'{v}_ung_{nm}'] = qn.mean().item()
                rec[f'{v}_gat_{nm}'] = (qn * gate[n]).mean().item()
            rec[f'pres_{nm}'] = pres[n].mean().item()
        rows.append(rec)

    # ── 汇总 ──
    def agg(col, c, nm):
        v = [r[f'{col}_{nm}'] for r in rows if r['cls'] == c]
        return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)

    print(f'\n耗时 {time.time() - t0:.1f}s\n')
    print('Q_norm 逐工况均值  (UNG = 不含 presence 门控, GAT = 含门控)')
    for v in VARIANTS:
        hd = f'{"工况":<7}'
        for nm in NODE_NAMES:
            hd += f'{nm + " UNG":>18}{"GAT":>8}'
        print(f'\n[{v}]')
        print(hd)
        print('-' * len(hd))
        for c, cn in enumerate(CLASS_NAMES):
            if not any(r['cls'] == c for r in rows):
                continue
            line = f'{cn:<7}'
            for nm in NODE_NAMES:
                m, s = agg(f'{v}_ung', c, nm)
                g, _ = agg(f'{v}_gat', c, nm)
                line += f'{m:>12.3f}±{s:.3f}{g:>8.3f}'
            print(line)

    # ── 判据 ──
    def check(v):
        lo = [agg(f'{v}_ung', c, 'LOW_FREQ')[0] for c in range(5)]
        bpf = [agg(f'{v}_ung', c, 'BPF')[0] for c in range(5)]
        idle2 = agg(f'{v}_ung', 0, '2xBPF')[0]
        load2 = [agg(f'{v}_ung', c, '2xBPF')[0] for c in (1, 2, 3)]
        c1 = np.nanmax(lo) < 0.30
        c2 = idle2 < 0.25 and min(load2) > 0.75
        c3 = all(0.50 <= x <= 0.85 for x in bpf) and \
            (np.nanmax(bpf) - np.nanmin(bpf)) > 0.05
        order = idle2 < np.nanmean(lo) < np.nanmean(bpf) < np.nanmean(load2)
        return c1, c2, c3, order, lo, bpf, idle2, load2

    print('\n事前判据 (对 UNG 口径; 门控是另加的乘子, 不参与判据)')
    for v in VARIANTS:
        c1, c2, c3, order, lo, bpf, idle2, load2 = check(v)
        print(f'\n[{v}]  (1){"PASS" if c1 else "FAIL"} '
              f'(2){"PASS" if c2 else "FAIL"} '
              f'(3){"PASS" if c3 else "FAIL"}  '
              f'排序{"自洽" if order else "不自洽"}')
        print(f'   LOW_FREQ = {"/".join(f"{x:.3f}" for x in lo)}  '
              f'(max {np.nanmax(lo):.3f})')
        print(f'   BPF      = {"/".join(f"{x:.3f}" for x in bpf)}')
        print(f'   2xBPF 空转/加载 = {idle2:.3f} / '
              f'{"/".join(f"{x:.3f}" for x in load2)}')
        print(f'   物理排序 空转2xBPF < LOW_FREQ < BPF < 加载2xBPF : '
              f'{"成立" if order else "不成立"} '
              f'({idle2:.2f} / {np.nanmean(lo):.2f} / '
              f'{np.nanmean(bpf):.2f} / {np.nanmean(load2):.2f})')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with out.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)
    print(f'\n[out] {out}  ({len(rows)} 行)')


if __name__ == '__main__':
    main()