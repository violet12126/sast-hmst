"""
算子级 IF 质量指标 Q_IF 体检 (只读, 不改模型)
=============================================
检验一个问题: 目前的损失 (RE_2D 等) 对「时频排列的帧间一致性」是盲的,
那么**原始 SST 相位驻点理论**能不能给出一个可算的、区分度足够的算子级
IF 质量分 Q_IF(n, t) ∈ [0, 1]?

Q_IF 的定义 (照抄讨论稿, 不在此处改口径)
---------------------------------------
频域局部主瓣共识度 (三 bin 二阶差分, 避开双峰间距):
    q_vert(f, t) = exp( -(w[f+1,t] - w[f-1,t])^2 / (2 * 1.0^2) )
时域局部平稳度 (步长 dt = 20 帧 = 20 ms):
    q_temp(f, t) = exp( -(w[f,t+dt] - w[f,t-dt])^2 / (2 * 1.5^2) )
带内能量加权 + 存在度门控:
    Q_IF(n, t) = clamp(presence_n(t) / 0.20, 0, 1)
                 * sum_f |S(f,t)|^2 * q_vert * q_temp
                 / (sum_f |S(f,t)|^2 + eps)

其中 w = omega_final (MSST 一阶 IF, 1-indexed bin, 0 = invalid),
S = 未增强的 STFT (与模型 tfr_raw 同源), 求和只在该节点频带的 region_weights 上。
无效 bin (w = 0, 或邻点越界) 的 q 记 0 —— 相位估计失败的 bin 不贡献置信度。

**事前判据 (先写死, 避免事后挑数字)**
  (1) LOW_FREQ (宽带湍流) 的 Q_IF 最低: 均值 < 0.30
  (2) 空转 2xBPF (分量缺席, 纯底噪) < 0.25, 且 加载 (低/中/高负荷) 2xBPF > 0.75
  (3) BPF 在 5 个工况上落在 0.50 ~ 0.85, 且随负荷呈连续梯度 (不是二值跳变)

附带诊断列 (用于解释判据为何成/不成):
  qv / qt        带内能量加权的 q_vert / q_temp 均值 (不含门控)
  valid          带内 q 有效的 (bin, 帧) 占比
  presence       节点的平均存在度 (node_presence, frac = 0.1)
  Q_raw          不含门控的 Q_IF (即求和无门控); Q_IF = gate * Q_raw
  Q_p4           权重改用 |S|^4 的同口径结果 (与 §6.1 的"能量平方压制噪声"对照)

输出 results/band_census/q_if_probe.csv (逐样本), 不覆盖任何已跟踪文件。

用法:
  python scripts/analysis/probe_q_if.py --per-class 10
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ))

from sast_data import load_dataset                                  # noqa: E402
from models.msst_torch import msst_torch                            # noqa: E402
from models.sast_nodes import PUMP_TURBINE_REGIONS, region_weights  # noqa: E402
from models.sast_losses import node_presence                        # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CLASS_NAMES = ['空转', '低负荷', '中负荷', '高负荷', '抽水']
FS = 1000.0
NODE_NAMES = [r.name for r in PUMP_TURBINE_REGIONS]
DT_FRAMES = 20
SIGMA_V = 1.0
SIGMA_T = 1.5
PRESENCE_NORM = 0.20
EPS = 1e-12


def q_terms(omega: torch.Tensor, energy: torch.Tensor):
    """返回 (q, q_v, q_t) —— 均为 [F, T]; 无效 bin 的 q 记 0。

    omega:  [F, T] int32 (1-indexed, 0 = invalid)
    energy: [F, T] float32 |S|^2 (只用来算诊断列, 不参与 q 本身)
    """
    F, T = omega.shape
    valid = omega > 0

    # ── 频域: (w[f+1] - w[f-1]) / sqrt(2) ──
    q_v = torch.zeros_like(energy)
    dv = (omega[2:] - omega[:-2]).float()
    ok_v = valid[2:] & valid[1:-1] & valid[:-2]
    q_v[1:F - 1] = torch.exp(-dv ** 2 / (2.0 * SIGMA_V ** 2)) * ok_v

    # ── 时域: (w[t+dt] - w[t-dt]) / sqrt(2) ──
    q_t = torch.zeros_like(energy)
    if T > 2 * DT_FRAMES:
        dw = (omega[:, 2 * DT_FRAMES:] - omega[:, :-2 * DT_FRAMES]).float()
        ok_t = valid[:, 2 * DT_FRAMES:] & valid[:, :-2 * DT_FRAMES]
        q_t[:, DT_FRAMES:T - DT_FRAMES] = \
            torch.exp(-dw ** 2 / (2.0 * SIGMA_T ** 2)) * ok_t
    return q_v * q_t, q_v, q_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--hlength', type=int, default=1024)
    ap.add_argument('--num', type=int, default=1)
    ap.add_argument('--out', default='results/band_census/q_if_probe.csv')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(dev)

    # ── 与训练同口径的验证集切分 (val_split=0.15, seed=42) ──
    ds = load_dataset(args.data, max_samples=None, val_split=0.15, seed=42)
    X, y = ds['val_X'], ds['val_y']
    print(f'[data] val = {len(X)} 条, 每类取 {args.per_class} 条; '
          f'device = {dev}; hlength = {args.hlength}, num = {args.num}')

    rng = np.random.RandomState(0)
    picks = []
    for c in range(len(CLASS_NAMES)):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        sel = rng.choice(idx, size=min(args.per_class, len(idx)), replace=False)
        picks.extend((int(i), c) for i in sel)

    freqs = torch.arange(1000, device=device, dtype=torch.float32) / 2000 * FS
    W = region_weights(freqs, PUMP_TURBINE_REGIONS)          # [N, F]

    rows = []
    t0 = time.time()
    for k, (i, c) in enumerate(picks):
        x = torch.from_numpy(np.asarray(X[i], dtype=np.float64)).to(device)
        r = msst_torch(x, FS, hlength=args.hlength, num=args.num,
                       save_trajectory=False, skip_squeeze=True)
        S = r['STFT']                                        # [F, T] complex64
        omega = r['omega_final']                             # [F, T] int32
        mag = S.abs()
        E = mag ** 2
        E4 = E ** 2

        q, q_v, q_t = q_terms(omega, E)                      # [F, T]
        sl = slice(DT_FRAMES, omega.shape[1] - DT_FRAMES)

        pres = node_presence(mag.unsqueeze(0), freqs, PUMP_TURBINE_REGIONS,
                             frac=0.1)[0]                    # [N, T]
        gate = (pres[:, sl] / PRESENCE_NORM).clamp(0.0, 1.0)  # [N, T-dt]

        qs = q[:, sl]
        qv_s = q_v[:, sl]
        qt_s = q_t[:, sl]
        Es = E[:, sl]
        E4s = E4[:, sl]
        valid_any = (qs > 0).float()

        for n, name in enumerate(NODE_NAMES):
            w = W[n].view(-1, 1)                              # [F,1]
            band = W[n] > 0                                   # [F]
            den = (Es * w).sum(dim=0) + EPS
            num = (Es * w * qs).sum(dim=0)
            num4 = (E4s * w * qs).sum(dim=0)
            den4 = (E4s * w).sum(dim=0) + EPS
            q_raw = num / den
            q_p4 = num4 / den4
            q_if = gate[n] * q_raw
            qv_w = (Es * w * qv_s).sum(dim=0) / den
            qt_w = (Es * w * qt_s).sum(dim=0) / den
            vb = valid_any[band].mean() if band.any() else torch.tensor(0.0)

            rows.append(dict(
                cls=c, cls_name=CLASS_NAMES[c], node=name,
                q_if=q_if.mean().item(), q_raw=q_raw.mean().item(),
                q_p4=q_p4.mean().item(),
                qv=qv_w.mean().item(), qt=qt_w.mean().item(),
                valid=vb.item(),
                presence=pres[n].mean().item(),
                gate=gate[n].mean().item(),
            ))
        if (k + 1) % 10 == 0 or k == 0:
            print(f'  [{k + 1}/{len(picks)}] {CLASS_NAMES[c]} '
                  f'({time.time() - t0:.1f}s)')

    # ── 汇总 ──
    def agg(key, c, n):
        v = [r[key] for r in rows if r['cls'] == c and r['node'] == n]
        return (np.mean(v), np.std(v)) if v else (float('nan'), float('nan'))

    print(f'\n耗时 {time.time() - t0:.1f}s, 有效样本 {len(picks)} 条\n')
    print('Q_IF(n, t) 逐工况均值  (mean ± std over samples)')
    hdr = f'{"工况":<8}' + ''.join(f'{nm:>20}' for nm in NODE_NAMES)
    print(hdr)
    print('-' * len(hdr))
    for c, cn in enumerate(CLASS_NAMES):
        if not any(r['cls'] == c for r in rows):
            continue
        line = f'{cn:<8}'
        for nm in NODE_NAMES:
            m, s = agg('q_if', c, nm)
            line += f'{m:>12.3f} ±{s:>5.3f}'
        print(line)

    print('\n诊断: Q_raw / qv / qt / valid / presence / Q_p4')
    hdr2 = f'{"工况":<8}{"节点":<12}' + ''.join(
        f'{h:>10}' for h in ('Q_raw', 'qv', 'qt', 'valid', 'pres', 'Q_p4'))
    print(hdr2)
    print('-' * len(hdr2))
    for c, cn in enumerate(CLASS_NAMES):
        if not any(r['cls'] == c for r in rows):
            continue
        for nm in NODE_NAMES:
            vals = [f'{agg(k, c, nm)[0]:>10.3f}' for k in
                    ('q_raw', 'qv', 'qt', 'valid', 'presence', 'q_p4')]
            print(f'{cn:<8}{nm:<12}' + ''.join(vals))

    # ── 判据裁定 ──
    print('\n事前判据:')
    ok = []

    lo_q = agg('q_if', 0, 'LOW_FREQ')[0]
    vals_lo = [agg('q_if', c, 'LOW_FREQ')[0] for c in range(len(CLASS_NAMES))
               if any(r['cls'] == c for r in rows)]
    c1 = np.nanmax(vals_lo) < 0.30
    ok.append(c1)
    print(f'  (1) LOW_FREQ 全部工况 < 0.30 : {"PASS" if c1 else "FAIL"}  '
          f'(max = {np.nanmax(vals_lo):.3f}, 空转 = {lo_q:.3f})')

    idle_2x = agg('q_if', 0, '2xBPF')[0]
    load_2x = [agg('q_if', c, '2xBPF')[0] for c in (1, 2, 3)
               if any(r['cls'] == c for r in rows)]
    c2 = idle_2x < 0.25 and (len(load_2x) > 0 and min(load_2x) > 0.75)
    ok.append(c2)
    print(f'  (2) 空转 2xBPF < 0.25 且加载 > 0.75 : '
          f'{"PASS" if c2 else "FAIL"}  '
          f'(空转 = {idle_2x:.3f}, 加载 = '
          f'{"/".join(f"{v:.3f}" for v in load_2x)})')

    bpf = [agg('q_if', c, 'BPF')[0] for c in range(len(CLASS_NAMES))
           if any(r['cls'] == c for r in rows)]
    in_band = all(0.50 <= v <= 0.85 for v in bpf)
    spread = np.nanmax(bpf) - np.nanmin(bpf)
    c3 = in_band and spread > 0.05
    ok.append(c3)
    print(f'  (3) BPF 落在 0.50~0.85 且梯度 > 0.05 : '
          f'{"PASS" if c3 else "FAIL"}  '
          f'(BPF = {"/".join(f"{v:.3f}" for v in bpf)}, 跨度 = {spread:.3f})')

    print(f'\n裁定: {sum(ok)}/3 通过'
          + ('  -> Q_IF 可作为统一自监督损失的驱动信号'
             if all(ok) else '  -> 判据未全过, 不据此改损失'))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f'[out] {out}  ({len(rows)} 行)')


if __name__ == '__main__':
    main()