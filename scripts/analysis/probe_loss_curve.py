"""只读探针：验证"无边界依赖的物理自洽监督损失" J(w) 的 U 型特性。

不修改任何模型参数、不写任何文件。做法是固定一个真实样本的 STFT 与 IF 估计
（模型的 w_override 入口，见 SAST.forward 的 docstring），只把某个物理节点的
w 从 0.05 扫到 0.98，前向算出增强谱，再在**该节点频带内**计算三项损失：

  L_ent(w)     = H_2( S̃_band(w) )                      —— 拉力：去窗泄漏
  L_deconv(w)  = || Norm(K_win *_f S̃) − Norm(K_win *_f S_raw) ||²
                                                        —— 斥力 1：窗主瓣尺度包络自洽
  L_flicker(w) = 1 − cos( S̃(t), S̃(t+Δt) )              —— 斥力 2：防底噪帧间闪烁

K_win 取 Hann(hlength=1024) 频域主瓣半宽的等效高斯（σ_win = 1.44 bin）；
样本网格为 F=1000 bins / 500 Hz ⇒ 1 bin = 0.5 Hz ⇒ σ_win ≈ 0.72 Hz。

四个场景：
  S1  LOW_FREQ 软带（低负荷，宽带水力湍流）      期望 w* < 0.25
  S2  BPF 硬带加载（中负荷，41 Hz + 50 Hz 双脊线） 期望 w* > 0.85
  S3  2xBPF 带负荷（高负荷，强锁频线）            期望 w* > 0.85
  S4  2xBPF 空转（分量缺席，纯底噪）              期望 w* < 0.30

用法:
    conda run -n my_work python scripts/analysis/probe_loss_curve.py \
        -c sast_checkpoints/gate3node/sast_v3_e020.pt -d 5_dataset.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sast_model import load_checkpoint          # noqa: E402
from models.sast_nodes import PUMP_TURBINE_REGIONS  # noqa: E402

SIGMA_WIN = 1.44        # bin —— Hann(1024) 主瓣半宽的等效高斯 σ
SIGMA_MIN, SIGMA_MAX = 0.20, 3.0


# ══════════════════════════════════════════════════════════
# 三项损失（全部在单节点频带内计算）
# ══════════════════════════════════════════════════════════
def renyi2_band(s_band, eps=1e-12):
    """Rényi-2 熵 H_2 = −log Σ_f p²，逐帧归一化后取时间平均。"""
    e = s_band.clamp_min(0.0)
    p = e / (e.sum(dim=0, keepdim=True) + eps)
    return float(-torch.log((p ** 2).sum(dim=0) + eps).mean())


def _gauss_kernel(sigma, device, radius=None):
    r = radius if radius is not None else max(3, int(np.ceil(3 * sigma)))
    k = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    g = torch.exp(-k ** 2 / (2.0 * sigma ** 2))
    return g / g.sum()


def _conv_freq(s_band, k):
    """沿频率轴（dim 0）做一维卷积，replicate 边界。"""
    pad = k.numel() // 2
    x = s_band.transpose(0, 1).unsqueeze(1)                 # [T,1,Fb]
    y = F.conv1d(F.pad(x, (pad, pad), mode='replicate'),
                 k.view(1, 1, -1))
    return y.squeeze(1).transpose(0, 1)                     # [Fb,T]


def deconv_residual(s_hat, s_raw, k, eps=1e-12):
    """窗主瓣尺度包络自洽：两次同样卷积后的逐帧余弦距离。"""
    a = _conv_freq(s_hat, k)
    b = _conv_freq(s_raw, k)
    a = a / (a.norm(dim=0, keepdim=True) + eps)
    b = b / (b.norm(dim=0, keepdim=True) + eps)
    return float(((a - b) ** 2).sum(dim=0).mean())


def flicker(s_hat, dt, eps=1e-12):
    """帧间闪烁：相隔 dt 帧的余弦相似度的补。"""
    a, b = s_hat[:, :-dt], s_hat[:, dt:]
    num = (a * b).sum(dim=0)
    den = a.norm(dim=0) * b.norm(dim=0) + eps
    return float((1.0 - num / den).mean())


# ══════════════════════════════════════════════════════════
# 数据与样本选择
# ══════════════════════════════════════════════════════════
def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    X = d['train_X']
    if X.ndim == 3:
        X = X[:, :, 0]
    return X, d['train_y'].ravel().astype(np.int64)


def _rank_by_band_energy(X, y, cls, region):
    idx = np.where(y == cls)[0]
    if len(idx) == 0:
        raise SystemExit(f'class {cls} 不存在')
    spec = np.abs(np.fft.rfft(X[idx], axis=1)) ** 2          # [n, F_fft]
    fft_f = np.fft.rfftfreq(X.shape[1], 1.0 / 1000.0)
    mm = (fft_f >= region.f_min) & (fft_f < region.f_max)
    order = np.argsort(-spec[:, mm].sum(axis=1))
    return [(X[idx[j]], int(idx[j])) for j in order]


def pick_sample(X, y, cls, region):
    """该类中，该节点频带内能量最大的样本。"""
    sig, idx = _rank_by_band_energy(X, y, cls, region)[0]
    return sig, idx


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════
def sweep(model, sig, node, w_grid, freqs, region, dt, kernel):
    device = next(model.parameters()).device
    x = torch.from_numpy(sig).float().view(1, -1).to(device)

    with torch.no_grad():
        base = model(x, training=False)
    w0 = base['w_i'].clone()                                 # [1,N,T]
    tfr_raw = base['tfr_raw'][0]                             # [F,T]
    m = (freqs >= region.f_min) & (freqs < region.f_max)
    raw_band = tfr_raw[m]

    rows = []
    for w in w_grid:
        w_ov = w0.clone()
        w_ov[0, node, :] = float(w)
        with torch.no_grad():
            out = model(x, training=False, w_override=w_ov)
        s_band = out['tfr_enhanced'][0][m]
        rows.append(dict(
            w=float(w),
            sigma=SIGMA_MIN + (1.0 - float(w)) * (SIGMA_MAX - SIGMA_MIN),
            ent=renyi2_band(s_band),
            dec=deconv_residual(s_band, raw_band, kernel),
            flk=flicker(s_band, dt),
        ))
    return rows, w0[0, node].mean().item()


def endpoint_deltas(model, sig, node, w_lo, w_hi, m, kernel, dt):
    """只跑两端点，返回 (ΔH2, Δdeconv, Δflicker)，方向均为「w 由低到高」。

    ΔJ = ΔH2 − λ_dec·Δdeconv − λ_flk·Δflicker；ΔJ>0 表示右端更优。
    """
    device = next(model.parameters()).device
    x = torch.from_numpy(sig).float().view(1, -1).to(device)
    with torch.no_grad():
        base = model(x, training=False)
    w0 = base['w_i'].clone()
    raw = base['tfr_raw'][0][m]

    vals = {}
    for tag, wv in (('lo', w_lo), ('hi', w_hi)):
        wo = w0.clone()
        wo[0, node, :] = float(wv)
        with torch.no_grad():
            o = model(x, training=False, w_override=wo)
        s = o['tfr_enhanced'][0][m]
        vals[tag] = (renyi2_band(s), deconv_residual(s, raw, kernel),
                     flicker(s, dt))
    return (vals['lo'][0] - vals['hi'][0],
            vals['hi'][1] - vals['lo'][1],
            vals['hi'][2] - vals['lo'][2])


def spark(vals, chars=' .:-=+*#@'):
    """20 点的横向 sparkline：越低越暗，便于一眼看出 U 型。"""
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0
    return ''.join(chars[int((v - lo) / span * (len(chars) - 1))] for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--checkpoint', required=True)
    ap.add_argument('-d', '--data', required=True)
    ap.add_argument('--n', type=int, default=20, help='扫描点数')
    ap.add_argument('--w-min', type=float, default=0.05)
    ap.add_argument('--w-max', type=float, default=0.98)
    ap.add_argument('--dt', type=int, default=20, help='闪烁惩罚的帧间隔')
    ap.add_argument('--samples', type=int, default=1,
                    help='>1 时追加多样本端点稳健性检验')
    ap.add_argument('--lambda-ent', type=float, default=1.0)
    ap.add_argument('--lambda-reconv', type=float, default=1.0)
    ap.add_argument('--lambda-flicker', type=float, default=1.0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    model, _, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    X, y = load_dataset(args.data)
    w_grid = np.linspace(args.w_min, args.w_max, args.n)

    device_ = next(model.parameters()).device
    kernel = _gauss_kernel(SIGMA_WIN, device_)

    # (标题, class, 节点号)
    scenarios = [
        ('S1 LOW_FREQ 软带 / 低负荷',     1, 0),
        ('S2 BPF 硬带 / 中负荷',          2, 1),
        ('S3 2xBPF 带负荷 / 高负荷',      3, 2),
        ('S4 2xBPF 空转 / 分量缺席',      0, 2),
    ]

    print('=' * 96)
    print(f'checkpoint : {args.checkpoint}')
    print(f'data       : {args.data}   (X={X.shape})')
    print(f'w 网格     : {args.w_min} → {args.w_max}, {args.n} 点   '
          f'σ(w)= {SIGMA_MIN}+(1−w)·{SIGMA_MAX - SIGMA_MIN:.2f}')
    print(f'K_win      : σ_win = {SIGMA_WIN} bin (Hann-1024 主瓣等效)   '
          f'闪烁 Δt = {args.dt} 帧')
    print('=' * 96)

    verdict = []
    store = []          # (title, ent, dec, flk, 'low'|'high') —— 供 λ 联合扫描
    for title, cls, node in scenarios:
        region = PUMP_TURBINE_REGIONS[node]
        sig, idx = pick_sample(X, y, cls, region)
        with torch.no_grad():
            fr = model(torch.from_numpy(sig).float().view(1, -1).to(device),
                       training=False)['freqs']
        rows, w_model = sweep(model, sig, node, w_grid, fr, region, args.dt, kernel)

        ent = np.array([r['ent'] for r in rows])
        dec = np.array([r['dec'] for r in rows])
        flk = np.array([r['flk'] for r in rows])

        def norm(a):
            d = a.max() - a.min()
            return (a - a.min()) / (d if d > 1e-12 else 1.0)

        J = (args.lambda_ent * norm(ent)
             + args.lambda_reconv * norm(dec)
             + args.lambda_flicker * norm(flk))
        i_best = int(np.argmin(J))
        w_star = float(w_grid[i_best])

        m = (fr >= region.f_min) & (fr < region.f_max)
        print(f'\n{"─" * 96}')
        print(f'{title}')
        print(f'  样本 index={idx}  节点={region.name} [{region.f_min},{region.f_max}) Hz '
              f'= {int(m.sum())} bins   模型当前 w̄={w_model:.3f}')
        print(f'  {"w":>6} {"σ(bin)":>8} {"H2":>9} {"deconv":>10} {"flicker":>9} '
              f'{"J(norm)":>9}')
        for r, j in zip(rows, J):
            mark = '  ← w*' if abs(r['w'] - w_star) < 1e-9 else ''
            print(f'  {r["w"]:6.3f} {r["sigma"]:8.3f} {r["ent"]:9.4f} '
                  f'{r["dec"]:10.5f} {r["flk"]:9.5f} {j:9.4f}{mark}')

        print(f'\n  w:       {w_grid[0]:.2f}{"":15}…{"":12}{w_grid[-1]:.2f}')
        for lab, arr, arg in (('H2     ', ent, np.argmin(ent)),
                              ('deconv ', dec, np.argmin(dec)),
                              ('flicker', flk, np.argmin(flk)),
                              ('J      ', J, i_best)):
            print(f'  {lab} |{spark(list(arr))}|  argmin w = {w_grid[arg]:.3f}')

        # ── 权重敏感性：J_a = a·norm(H2) + (1−a)·norm(deconv)（闪烁项量级小，略） ──
        ne, nd = norm(ent), norm(dec)
        alphas = np.linspace(0.0, 1.0, 101)
        arg_a = [float(w_grid[int(np.argmin(a * ne + (1 - a) * nd))])
                 for a in alphas]
        inner = [(a, wv) for a, wv in zip(alphas, arg_a)
                 if args.w_min < wv < args.w_max]
        if inner:
            print(f'\n  权重扫描: a∈[{inner[0][0]:.2f},{inner[-1][0]:.2f}] 时存在内部极小 '
                  f'(w* {min(w for _, w in inner):.2f}–{max(w for _, w in inner):.2f})')
        else:
            print('\n  权重扫描: **任意** a∈[0,1] 下极小都落在端点 —— 无 U 型')

        verdict.append((title, w_star, float(ent[0] - ent[-1]),
                        float(dec[-1] - dec[0]), float(flk[-1] - flk[0]),
                        bool(inner)))
        store.append((title, ent, dec, flk,
                      'low' if title.startswith(('S1', 'S4')) else 'high'))

    # ══════════════════════════════════════════════════════════════════
    # λ 联合扫描（原始量纲，不做逐样本归一化 —— 训练时 λ 是全局常数）
    #   J_raw(w) = λ_ent·H2 + λ_dec·deconv + λ_flk·flicker,  λ_ent ≡ 1
    # 找一个 (λ_dec, λ_flk) 使 4 条判据同时成立；并给出最差裕度。
    # ══════════════════════════════════════════════════════════════════
    print('\n' + '=' * 96)
    print('λ 联合扫描（原始量纲, λ_ent=1）：J = H2 + λ_dec·deconv + λ_flk·flicker')
    print('=' * 96)

    def pass_all(ld, lf):
        argmins, gaps = [], []
        for title, ent, dec, flk, need in store:
            J = ent + ld * dec + lf * flk
            i = int(np.argmin(J))
            wv = float(w_grid[i])
            ok = wv < 0.25 if need == 'low' else wv > 0.85
            lo = J[w_grid <= 0.5].min()
            hi = J[w_grid > 0.5].min()
            star, alt = (lo, hi) if need == 'low' else (hi, lo)
            argmins.append(wv)
            gaps.append((alt - star) / max(abs(star), 1e-12))
            if not ok:
                return None
        return argmins, min(gaps)

    grid = [(ld, lf) for ld in np.linspace(0.0, 15.0, 61)
            for lf in np.linspace(0.0, 20.0, 41)]
    feas = [(ld, lf, pass_all(ld, lf)) for ld, lf in grid]
    feas = [(ld, lf, r) for ld, lf, r in feas if r is not None]

    best_ld, best_lf = 5.5, 0.5                 # 默认按单样本扫描的结果
    if not feas:
        print('  无任何 (λ_dec, λ_flk) 使 4 条判据同时成立')
    else:
        best = max(feas, key=lambda t: t[2][1])
        best_ld, best_lf = float(best[0]), float(best[1])
        ls = [t[0] for t in feas]
        fs = [t[1] for t in feas]
        print(f'  可行域: λ_dec ∈ [{min(ls):.2f}, {max(ls):.2f}]   '
              f'λ_flk ∈ [{min(fs):.2f}, {max(fs):.2f}]   '
              f'({len(feas)}/{len(grid)} 个网格点)')
        ld, lf, (argmins, gap) = best
        print(f'  最稳健点 λ_dec={ld:.2f}  λ_flk={lf:.2f}   '
              f'最差裕度 = {gap * 100:.1f}%')
        for (title, *_), wv in zip(store, argmins):
            print(f'    {title:<30}  w* = {wv:.3f}')
        print('  该权重下的 J_raw 曲线（末行标注最小值位置）:')
        for title, ent, dec, flk, _ in store:
            J = ent + ld * dec + lf * flk
            print(f'    {title:<30} |{spark(list(J))}|  '
                  f'argmin w = {w_grid[int(np.argmin(J))]:.3f}')

    # ══════════════════════════════════════════════════════════════════
    # 多样本稳健性：只跑两端点，检验 ΔJ 的符号是否跨样本稳定
    # ══════════════════════════════════════════════════════════════════
    if args.samples > 1:
        print('\n' + '=' * 96)
        print(f'多样本端点检验  λ_dec={best_ld:.2f}  λ_flk={best_lf:.2f}  '
              f'每场景 {args.samples} 个样本（按带内能量排序）')
        print('  ΔJ = ΔH2 − λ_dec·Δdeconv − λ_flk·Δflicker；'
              'ΔJ>0 倾向窄核(w→高)，ΔJ<0 倾向宽核(w→低)')
        print('=' * 96)
        print(f'{"场景":<30}{"需":>6}{"符号正确":>10}{"ΔJ 均值":>12}'
              f'{"ΔJ std":>10}{"最小|ΔJ|":>11}')
        for title, cls, node in scenarios:
            region = PUMP_TURBINE_REGIONS[node]
            need = 'high' if title.startswith(('S2', 'S3')) else 'low'
            cand = _rank_by_band_energy(X, y, cls, region)[:args.samples]
            with torch.no_grad():
                fr = model(torch.from_numpy(cand[0][0]).float().view(1, -1)
                           .to(device), training=False)['freqs']
            mm = (fr >= region.f_min) & (fr < region.f_max)
            dJ = []
            for sig, _ in cand:
                dH, dD, dF = endpoint_deltas(
                    model, sig, node, args.w_min, args.w_max, mm,
                    kernel, args.dt)
                dJ.append(dH - best_ld * dD - best_lf * dF)
            dJ = np.array(dJ)
            ok = (dJ > 0) if need == 'high' else (dJ < 0)
            print(f'{title:<30}{need:>6}{int(ok.sum()):>7}/{len(dJ):<3}'
                  f'{dJ.mean():>12.4f}{dJ.std():>10.4f}'
                  f'{np.abs(dJ).min():>11.4f}')

    print('\n' + '=' * 96)
    print(f'{"场景":<30}{"w*":>8}{"ΔH2(左−右)":>14}{"Δdeconv":>12}'
          f'{"Δflicker":>12}{"内部极小":>10}')
    for t, w, dh, dd, df, ok in verdict:
        print(f'{t:<30}{w:>8.3f}{dh:>14.4f}{dd:>12.5f}{df:>12.5f}'
              f'{"有" if ok else "无":>10}')

    print('\n判据对照:')
    checks = [
        ('S1 w* < 0.25', verdict[0][1] < 0.25),
        ('S2 w* > 0.85', verdict[1][1] > 0.85),
        ('S3 w* > 0.85', verdict[2][1] > 0.85),
        ('S4 w* < 0.30', verdict[3][1] < 0.30),
    ]
    for name, ok in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    print('=' * 96)


if __name__ == '__main__':
    main()