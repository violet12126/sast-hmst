"""目标带宽的修正估计量 (等半径 + 存在度门控 + 窗底扣除 + 帧内/帧间分解)。

WHY 需要修正: 朴素做法 (整框内积分 omega_0 的加权 std) 有三个结构性偏差——

  [B1] 框宽偏差: 实测 raw std ≈ B_n/sqrt(12)，即主要在测**搜索框宽度**而非信号。
       LOW_FREQ 16 Hz 框 -> 4.619; BPF_DRIFT 6.5 Hz 框 -> 1.876; BPF_LINE 4.5 -> 1.299。
       实测值几乎逐项吻合 (4.345/1.890/1.212)，是均匀分布的签名。
  [B2] 缺席帧稀释: 空转时 2xBPF 分量消失 (presence 0.074)，整框平底噪 -> 单帧 std 逼近
       B_n/sqrt(12)，把全样本均值抬高。
  [B3] 相位法对阻尼/调幅展宽**原理性失明**: x(t)=exp(-a t)exp(i2pi f0 t) 的相位导数
       在所有 bin 上恒为 f0，故任何 omega_0 指标都测不出包络展宽。

修正:
  - **等半径局部窗**: 以每帧峰值频率为中心取 |f - f_peak| <= DF，**所有节点同一 DF**，
    消除框宽偏差 (B1)。
  - **存在度门控**: presence < 0.15 的帧不计 (B2)。
  - **窗底扣除**: 用同一 DF 下的合成纯音测得的 sigma_win，按方差可加性扣除。
  - **全方差分解** (B3 的对策之一): bw^2 = E_tau[sigma_intra^2] + Var_tau[mu]
       - 第一项抓 LOW_FREQ 的**帧内**瞬时多模态弥散
       - 第二项抓 BPF_DRIFT 的**帧间**中心频率游走
  - 同时给出**幅值法** (raw 谱在等半径窗内的加权 std - 窗底)，它对阻尼展宽敏感 (B3)。

判据 (事前写定): 修正后排序应恢复为
    LOW_FREQ > BPF_DRIFT > BPF_LINE ≈ 2xBPF
只读, 不改源码。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ))

from models.tfr import msst                                          # noqa: E402
from models.sast_nodes import (MSSTNodeExtractor, PUMP_TURBINE_REGIONS,  # noqa: E402
                               region_weights)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CLASS_NAMES = ['空转', '低负荷', '中负荷', '高负荷', '抽水']
FS, N_SIG = 1000.0, 2000


def presence_frame(col, w, frac=0.1):
    """单帧存在度 (与 models.sast_losses.node_presence 同式)。col=[Fn], w=[Fn]。"""
    e = np.maximum(col, 0.0)
    tot = e.sum()
    if tot <= 0:
        return 0.0
    kk = max(1, int(np.ceil(frac * e.size)))
    p = float(np.sort(w)[::-1][:kk].sum() / max(1e-12, w.sum()))
    c = float(np.sort(e)[::-1][:kk].sum() / tot)
    return float(np.clip((c - p) / max(1e-8, 1 - p), 0.0, 1.0))


def node_estimate(E, om0, freqs, reg, df_win, pres_thr):
    """返回 (intra, inter, total, amp) 四项 (Hz) 与有效帧数。

    E:    [F, T] 原始功率谱 |STFT|^2
    om0:  [F, T] 第 0 层 omega (1-indexed bin)
    """
    w = region_weights(freqs, [reg])[0].numpy()
    mk = w > 0.0
    fb = freqs.numpy()[mk]
    Wb = w[mk]
    Eb = (E.numpy()[mk, :] * Wb[:, None])                  # [Fn, T]
    ob = (om0.numpy()[mk, :].astype(np.float64) - 1.0) * df_win_global

    intra2, mus = [], []
    amp2 = []
    for t in range(Eb.shape[1]):
        col = Eb[:, t]
        if col.sum() <= 0:
            continue
        if presence_frame(col, Wb) < pres_thr:
            continue
        i_pk = int(np.argmax(col))
        f_pk = fb[i_pk]
        sel = np.abs(fb - f_pk) <= df_win
        if sel.sum() < 2:
            continue
        wt = col[sel]
        ws = wt.sum()
        if ws <= 0:
            continue
        o = ob[sel, t]
        mu = float((wt * o).sum() / ws)
        intra2.append(float((wt * (o - mu) ** 2).sum() / ws))
        mus.append(mu)
        # 幅值法: 同一等半径窗内, 原始谱的能量加权频率 std
        ff = fb[sel]
        amp2.append(float((wt * (ff - (wt * ff).sum() / ws) ** 2).sum() / ws))
    if len(mus) < 2:
        return None
    intra2 = np.array(intra2)
    amp2 = np.array(amp2)
    mus = np.array(mus)
    inter2 = float(np.var(mus))
    return (float(np.sqrt(intra2.mean())), float(np.sqrt(inter2)),
            float(np.sqrt(intra2.mean() + inter2)), float(np.sqrt(amp2.mean())),
            len(mus))


df_win_global = 0.5      # 频率分辨率, 由 main 覆盖


def main():
    ap = argparse.ArgumentParser(description='目标带宽修正估计量 (只读)')
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--per-class', type=int, default=4)
    ap.add_argument('--df_win', type=float, default=2.5,
                    help='等半径局部窗半宽 (Hz), 所有节点统一')
    ap.add_argument('--presence', type=float, default=0.15,
                    help='存在度门控阈值')
    args = ap.parse_args()

    global df_win_global
    ex = MSSTNodeExtractor(fs=FS, msst_num=4)
    freqs_t = torch.from_numpy(ex(np.zeros(N_SIG)).freqs).float()
    df = float(freqs_t[1] - freqs_t[0])
    df_win_global = df
    print(f'freqs {len(freqs_t)} bin, df={df} Hz;  等半径窗 ±{args.df_win} Hz '
          f'(约 ±{int(round(args.df_win / df))} bin);   presence > {args.presence}')
    print(f'节点框宽: ' + '  '.join(
        f'{r.name} {r.f_max - r.f_min:.1f}Hz' for r in PUMP_TURBINE_REGIONS))
    print(f'(框宽/sqrt(12): ' + '  '.join(
        f'{r.name} {(r.f_max - r.f_min) / np.sqrt(12):.3f}' for r in PUMP_TURBINE_REGIONS)
        + ')  <- 朴素做法的系统性偏差来源\n')

    # 窗底: 用同一 ±DF 下的合成纯音测 (50 Hz / 100 Hz 应一致, 证明等半径消除了框宽偏差)
    t = np.arange(N_SIG) / FS
    print('窗底标定 (合成纯音, 等半径窗内):')
    floors = {}
    for f0 in (50.0, 100.0):
        r = msst(np.sin(2 * np.pi * f0 * t), FS, hlength=None, num=4,
                 save_trajectory=True)
        E = torch.from_numpy(np.abs(r['STFT']) ** 2).float()
        om0 = np.asarray(r['omegas'][0])
        for reg in PUMP_TURBINE_REGIONS:
            w = region_weights(freqs_t, [reg])[0].numpy()
            if w[(np.abs(freqs_t.numpy() - f0) <= args.df_win)].sum() == 0:
                continue
            est = node_estimate(E, torch.from_numpy(om0), freqs_t, reg,
                                args.df_win, 0.0)
            if est:
                floors.setdefault(reg.name, []).append(est)
    for nm, v in floors.items():
        a = np.array(v)
        print(f'  {nm:<11s} intra={a[:,0].mean():.4f} inter={a[:,1].mean():.4f} '
              f'total={a[:,2].mean():.4f} amp={a[:,3].mean():.4f} Hz')

    # ── 真实数据 ──
    d = np.load(args.data, allow_pickle=True)
    X, y = d['train_X'], d['train_y'].ravel().astype(np.int64)
    if X.ndim == 3:
        X = X[:, :, 0]
    rng = np.random.RandomState(0)
    sel = {c: rng.choice(np.where(y == c)[0], args.per_class, replace=False)
           for c in sorted(np.unique(y).tolist())}

    acc = {r.name: [] for r in PUMP_TURBINE_REGIONS}
    print(f'\n真实数据 ({sum(len(v) for v in sel.values())} 条):')
    for c in sorted(sel):
        for i in sel[c]:
            r = msst(X[i].astype(np.float64), FS, hlength=None, num=4,
                     save_trajectory=True)
            E = torch.from_numpy(np.abs(r['STFT']) ** 2).float()
            om0 = torch.from_numpy(np.asarray(r['omegas'][0])).long()
            for reg in PUMP_TURBINE_REGIONS:
                est = node_estimate(E, om0, freqs_t, reg, args.df_win,
                                    args.presence)
                if est:
                    acc[reg.name].append(est)

    print(f'\n  {"node":<11s} {"类型":>4s} {"intra":>8s} {"inter":>8s} '
          f'{"total":>8s} {"amp":>8s} {"帧数":>6s}')
    order = []
    for reg in PUMP_TURBINE_REGIONS:
        if not acc[reg.name]:
            continue
        a = np.array(acc[reg.name])
        soft = reg.f_type in ('HYDRAULIC', 'BLADE_DRIFT')
        print(f'  {reg.name:<11s} {"软" if soft else "硬":>4s} {a[:,0].mean():>8.3f} '
              f'{a[:,1].mean():>8.3f} {a[:,2].mean():>8.3f} {a[:,3].mean():>8.3f} '
              f'{a[:,4].mean():>6.0f}')
        order.append((reg.name, a[:,2].mean()))

    print(f'\n  判据: total 应满足  LOW_FREQ > BPF_DRIFT > BPF_LINE ≈ 2xBPF')
    got = ' > '.join(f'{n}({v:.2f})' for n, v in sorted(order, key=lambda z: -z[1]))
    print(f'  实测排序: {got}')
    d_ = dict(order)
    ok = (d_.get('LOW_FREQ', 0) > d_.get('BPF_DRIFT', 0) >
          max(d_.get('BPF_LINE', 0), d_.get('2xBPF', 0)))
    print(f'  {"排序恢复 OK" if ok else "排序仍未恢复"}')


if __name__ == '__main__':
    main()
