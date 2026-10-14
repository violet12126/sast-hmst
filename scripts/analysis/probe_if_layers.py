"""
MSST IF 迭代层 (omegas[:, k]) 的质量普查 — 补 papers/SAST_讲解文档.md §5.1
==========================================================================
背景: §18.0 断言「MSST 的 lookup 迭代对所有 bin 无害, 因此所有 bin 统一用
omega_5, 不需要为不同 bin 选不同的 IF 层」, 但该断言从未被实验验证 ——
讲解文档 §5.1「MSST IF 迭代次数对不同分量的影响」一直是空壳。

事实澄清 (代码级):
  omegas 是一个长度 = num 的**列表**, omegas[k] 是同一个一阶 IF 估计的固定点
  迭代层:  omega_{k+1}[eta,b] = omega_k[omega_k[eta,b], b]  (models/tfr.py:595)
  **不是** Taylor 相位导数阶 (那是另一个函数 sst2_stft)。
  生产路径只用了 omegas[:, -1] (omega_final), 其余层算了就丢。

本脚本回答两件事:
  [1] 层间收敛性 —— omegas[k] 与 omegas[k-1] 到底差多少? 若某层之后改变
      bin 占比 < 1%, 则「层」这一维不存在可择余地。
  [2] 逐分量、逐层、逐 sigma 的 TFR 质量 —— 用 omegas[:, k] 作落点做单轮
      重排, 量四个指标。这决定是否值得实现「自适应阶数」(β 软融合)。

**事前判据 (写死在脚本里, 避免事后挑数字)**:
  若存在 k* 使得 窄带 (BPF_LINE / 2xBPF) 上的 Rényi-2D 熵 / 集中度 / 锐度
  **显著优于** k = num-1, 且在宽带 (LOW_FREQ / BPF_DRIFT) 上不劣化 (展宽不
  被压掉), 则「自适应阶数」有据可依; 否则维持 §18.0 的结论, 只修文档。

输出 (results/band_census/, 不覆盖任何已跟踪文件):
  if_layer_quality.csv     逐 类×节点×层×sigma 的四项指标
  if_layer_convergence.csv 逐层间的改变 bin 占比与 |Δ|

用法:
  python scripts/analysis/probe_if_layers.py --per-class 6 --sigmas 0.5,1.0,2.0
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ))

from models.tfr import msst                                    # noqa: E402
from models.sast_nodes import (PUMP_TURBINE_REGIONS,           # noqa: E402
                               region_weights, SOFT_F_TYPES)
from models.sast import SparseGaussianReassigner               # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CLASS_NAMES = ['空转', '低负荷', '中负荷', '高负荷', '抽水']
NUM = 4          # msst num -> len(omegas) == num


# ── 指标 (与训练侧口径一致: renyi_2d_loss 用 alpha=2 + region_weights 加权) ──

def _band_weighted(tfr: torch.Tensor, freqs: torch.Tensor, region) -> torch.Tensor:
    """[1,F,T] -> [F_r,T] 带内加权幅值 (与 renyi_2d_loss 同口径)。"""
    w = region_weights(freqs, [region])[0]
    m = w > 0.0
    return tfr[0][m, :] * w[m].unsqueeze(1)


def renyi_2d(band: torch.Tensor, alpha: int = 2, eps: float = 1e-8) -> float:
    """Rényi-2D 熵 (越低越集中)。与 models.sast_losses.renyi_2d_loss 同式。"""
    p = band.clamp(min=eps).flatten()
    p = p / p.sum().clamp(min=eps)
    return float(torch.log((p ** alpha).sum().clamp(min=eps)) / (1.0 - alpha))


def concentration(band: torch.Tensor, frac: float = 0.1) -> float:
    """带内能量最高的 k 个 bin (跨时间合并) 的能量占比。越高越集中。"""
    col = band.sum(dim=1)
    k = max(1, int(np.ceil(frac * col.numel())))
    tot = col.sum()
    if float(tot) <= 0:
        return 0.0
    return float(col.topk(k).values.sum() / tot)


def sharpness(band: torch.Tensor, eps: float = 1e-8) -> float:
    """逐帧 峰/带内均值 的均值 (无量纲, 越高脊线越突出)。"""
    mean_t = band.mean(dim=0).clamp(min=eps)
    return float((band.max(dim=0).values / mean_t).mean())


def spread_hz(band: torch.Tensor, fb: torch.Tensor, eps: float = 1e-12) -> float:
    """带内能量加权的频率标准差 (Hz)。越小越窄, 越大越展宽。

    band: [F_r, T]; fb: [F_r]。逐帧对频率求加权方差, 再对时间取均值。
    """
    wsum = band.sum(dim=0).clamp(min=eps)                    # [T]
    mu = (fb.unsqueeze(1) * band).sum(dim=0) / wsum          # [T]
    var = ((fb.unsqueeze(1) - mu.unsqueeze(0)) ** 2 *
           band).sum(dim=0) / wsum                           # [T]
    return float(torch.sqrt(var.clamp(min=eps)).mean())


def main():
    ap = argparse.ArgumentParser(description='MSST IF 层质量普查 (§5.1)')
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--per-class', type=int, default=6)
    ap.add_argument('--sigmas', default='0.5,1.0,2.0',
                    help='重排核宽 (bin), 逗号分隔')
    ap.add_argument('--outdir', default='results/band_census')
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dev = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else args.device
    sigmas = [float(s) for s in args.sigmas.split(',')]

    d = np.load(args.data, allow_pickle=True)
    X, y = d['train_X'], d['train_y'].ravel().astype(np.int64)
    if X.ndim == 3:
        X = X[:, :, 0]

    print(f'设备 {dev}; 层数 = {NUM} (msst num={NUM})')
    print(f'sigma 取值 {sigmas}')
    print(f'节点: ' + '  '.join(
        f'{r.name}[{r.f_min:.1f},{r.f_max:.1f})' for r in PUMP_TURBINE_REGIONS))
    print('判据 (事前写定): 若存在 k* 在窄带显著优于 k=NUM-1 且在宽带不劣化, '
          '则「自适应阶数」有据; 否则维持 §18.0, 只修文档。\n')

    rows, conv_rows = [], []
    rng = np.random.RandomState(0)
    # 无参算子, 建一次即可 (kernel/kernel_radius 与 sigma 无关)
    reassigner = SparseGaussianReassigner(sigma_min=0.5, sigma_max=15.0)

    for c in sorted(np.unique(y).tolist()):
        idx = np.where(y == c)[0]
        if len(idx) > args.per_class:
            idx = rng.choice(idx, args.per_class, replace=False)
        print(f'{"=" * 78}\n{CLASS_NAMES[c]}  (n={len(idx)})')

        # 累计: [layer][node] -> list of metrics
        acc = {k: {n: {'re': [], 'co': [], 'sh': [], 'sp': []}
                   for n in range(len(PUMP_TURBINE_REGIONS))}
               for k in range(NUM)}
        conv = {k: [] for k in range(1, NUM)}   # 层间改变 bin 占比

        for x in idx:
            r = msst(X[x].astype(np.float64), 1000.0, hlength=None, num=NUM,
                     save_trajectory=True)
            om = [torch.as_tensor(np.asarray(o)).long() for o in r['omegas']]
            freqs = torch.as_tensor(np.asarray(r['freqs'])).float()
            tfr_mag = torch.as_tensor(np.abs(np.asarray(r['STFT']))).float()
            Fb, T = tfr_mag.shape

            # ── [1] 层间收敛性 ──
            for k in range(1, NUM):
                valid = (om[k] >= 1) & (om[k] <= Fb) & (om[k - 1] >= 1)
                if int(valid.sum()) == 0:
                    continue
                chg = (om[k][valid] != om[k - 1][valid]).float().mean()
                conv[k].append(float(chg))

            # ── [2] 逐层重排 + 指标 ──
            tfr_mag_b = tfr_mag.unsqueeze(0).to(dev)
            freqs_d = freqs.to(dev)
            for k in range(NUM):
                # SparseGaussianReassigner 取 omegas[:, -1], 故把目标层铺满最后一维
                om_pick = om[k].unsqueeze(0).unsqueeze(0).repeat(1, NUM, 1, 1).to(dev)
                for sg in sigmas:
                    sigma = torch.full((1, Fb, T), sg, device=dev)
                    with torch.no_grad():
                        # w_bin=None -> 单轮完整重排 (本探针要的就是"单次重排
                        # 落在第 k 层 IF 上"的效果, 不带软门控)
                        enh = reassigner(tfr_mag_b, om_pick, sigma, freqs_d,
                                         ridge_factor=None, w_bin=None, M=NUM)
                    enh = enh[0].cpu()
                    for n, reg in enumerate(PUMP_TURBINE_REGIONS):
                        band = _band_weighted(enh.unsqueeze(0), freqs, reg)
                        w = region_weights(freqs, [reg])[0]
                        fb = freqs[w > 0.0]
                        acc[k][n]['re'].append(renyi_2d(band))
                        acc[k][n]['co'].append(concentration(band))
                        acc[k][n]['sh'].append(sharpness(band))
                        acc[k][n]['sp'].append(spread_hz(band, fb))
                        rows.append((c, CLASS_NAMES[c], reg.name, k, sg,
                                     f'{acc[k][n]["re"][-1]:.5f}',
                                     f'{acc[k][n]["co"][-1]:.5f}',
                                     f'{acc[k][n]["sh"][-1]:.5f}',
                                     f'{acc[k][n]["sp"][-1]:.5f}'))

        print(f'  层间改变 bin 占比: ' + '  '.join(
            f'{k}->{k+1}: {np.mean(conv[k]):.4%}' for k in sorted(conv)))

        # 逐节点打印: 每层的 (Rényi 越低越好 / 集中度越高越好 / 锐度越高越好)
        for n, reg in enumerate(PUMP_TURBINE_REGIONS):
            soft = ' (软带: 应保留展宽)' if reg.f_type in SOFT_F_TYPES else ''
            print(f'  {reg.name}{soft}')
            print(f'    {"k":>2s} {"Rényi↓":>9s} {"集中度↑":>9s} '
                  f'{"锐度↑":>9s} {"带内std(Hz)":>11s}')
            for k in range(NUM):
                f = lambda key: np.mean(acc[k][n][key]) \
                    if len(acc[k][n][key]) else float('nan')
                print(f'    {k:>2d} {f("re"):>9.4f} {f("co"):>9.4f} '
                      f'{f("sh"):>9.4f} {f("sp"):>11.4f}')

        for k in range(1, NUM):
            if conv[k]:
                conv_rows.append((c, CLASS_NAMES[c], k, k + 1,
                                  f'{np.mean(conv[k]):.6f}'))

    # ── 汇总: 每节点各指标的最佳层 ──
    print(f'\n{"=" * 78}\n汇总 (跨工况平均): 各节点在哪个 k 上最优')
    print(f'{"node":<11s} {"Rényi 最优 k":>13s} {"集中度 最优 k":>15s} '
          f'{"锐度 最优 k":>13s}')
    for n, reg in enumerate(PUMP_TURBINE_REGIONS):
        def best(key, fn):
            vals = [np.mean(agg[n][key]) if agg[n][key] else np.nan
                    for agg in [acc[k] for k in range(NUM)]]
            return int(np.nanargmin(vals)) if fn == 'min' else int(np.nanargmax(vals))
        print(f'{reg.name:<11s} {best("re", "min"):>13d} '
              f'{best("co", "max"):>15d} {best("sh", "max"):>13d}')

    # ── 输出 ──
    with open(outdir / 'if_layer_quality.csv', 'w', newline='',
              encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['class', 'class_name', 'node', 'if_layer', 'sigma',
                    'renyi2d', 'concentration', 'sharpness', 'band_spread_hz'])
        w.writerows(rows)
    with open(outdir / 'if_layer_convergence.csv', 'w', newline='',
              encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['class', 'class_name', 'from_layer', 'to_layer',
                    'changed_bin_frac'])
        w.writerows(conv_rows)
    print(f'\n输出: {outdir / "if_layer_quality.csv"}')
    print(f'      {outdir / "if_layer_convergence.csv"}')


if __name__ == '__main__':
    main()
