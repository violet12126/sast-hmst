"""
频带可压缩性分析 (支撑 Word 文档 §4.2)
========================================

要回答的问题: 数据里"真实频带"到底有多宽? 哪些频带应当被硬挤压, 哪些不该?

思路: 把"该不该挤压"从先验假设变成对数据的实测结论, 再检验 SAST 学到的
      sigma_i 与该实测结论是否一致 —— 形成"物理判据 -> 数据测量 -> 模型学对"闭环.

两个尺度无关的实测指标 (单样本取值 -> 类内平均):
  f_c      分量中心频率 (Hz)
  B6dB     -6 dB 带宽 (Hz)   —— 与窗函数极限带宽对比
  sigma_f  时频脊线瞬时频率的标准差 (Hz) —— 相位相干性 / 漂移程度

判据 (两条同时满足, 才判定为"应当硬挤压"):
  1) B6dB 接近窗函数极限带宽
     Hann-512 @ fs=1000 的 -6 dB 主瓣约 3 Hz. B6dB ≈ 3 Hz 说明谱线的
     "窄"是真实的, 挤压不制造信息; B6dB >> 3 Hz 说明带宽是信号固有的,
     硬挤压必然失真.
  2) sigma_f 小 (相位相干)
     同步压缩的无偏性依赖相位稳定; sigma_f 大表示分量在漂移, 挤压会把
     能量搬到错误的频率上.

输出 (默认 results/band_analysis/):
  welch_psd_by_class.png   5 类均值 PSD (± 25/75 分位带) + 检测峰标注
  peaks.csv                每类每个检测峰的 (f_c, level, B6dB, PMR, sigma_f, 归属)
  compressibility.csv      分量级汇总 + SAST 学到的 sigma/lambda/w_i 对照
  sigma_vs_bandwidth.png   B6dB vs SAST sigma 散点 (闭环验证)

用法:
  python scripts/analysis/analyze_band_compressibility.py \
      --checkpoint sast_checkpoints/sast_v3_model.pt --device cuda --per-class 8
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, stft as scipy_stft, find_peaks

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / 'scripts' / 'plot'))

from tfr_common import apply_plot_style, FMAX_DEFAULT       # noqa: E402

apply_plot_style()

# 控制台中文 (Windows GBK 控制台会把 UTF-8 中文打成乱码)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

FS = 1000.0
NPERSEG = 512          # 与 SAST 内部 STFT 窗长一致 -> 极限带宽可直接比较
NOVERLAP = 384
NFFT = 2048            # 零填充, 让 -6 dB 交点插值更细 (分辨率仍受 NPERSEG 限制)
FMAX = FMAX_DEFAULT    # 500 Hz

CLASS_NAMES = ['空转', '低负荷', '中负荷', '高负荷', '抽水']

# 物理分量定义 (频带为"候选区间", 实际中心频率由实测峰值决定)
COMPONENTS = [
    ('低频水力分量 (涡带/压力脉动)', 2.0, 25.0, 10.0),
    ('叶片通过频率 BPF', 42.0, 55.0, 8.0),
    ('二倍叶片通过频率 2×BPF', 90.0, 105.0, 10.0),
]
OTHER_LABEL = '其他/高频'
# 高频宽带区 (无显著峰, 用于"宽带噪声不可挤压"的对照)
BROADBAND = (150.0, 500.0)

# 窗函数极限带宽参考: Hann 窗主瓣宽 4 bin, -6 dB 交点约 1.87 bin
BIN_HZ = FS / NPERSEG                      # 1.95 Hz
WINDOW_6DB_HZ = 1.87 * BIN_HZ              # ≈ 3.65 Hz (纯音经该窗后的 -6 dB 宽度)

# 判据阈值
NARROW_FACTOR = 1.2                        # B6dB <= 1.2 x 窗极限 视为"窄"
STEADY_SIGMA_F = 1.5                       # σ_f <= 1.5 Hz 视为"相位稳定"

# SAST 频率网格 (freqs 0-500 Hz 共 1000 点) -> 1 bin = 0.5 Hz
SAST_BIN_HZ = 0.5


# ═══════════════════════════════════════════════════════════════
# 指标
# ═══════════════════════════════════════════════════════════════

def _crossing(x1, y1, x2, y2, yt):
    """线性插值求 y = yt 处的 x."""
    if y2 == y1:
        return x1
    return x1 + (yt - y1) * (x2 - x1) / (y2 - y1)


def bandwidth_6db(f, p_db, i_pk):
    """-6 dB 带宽 (Hz). 未在边界前收敛则返回 nan (表示看不到底, 宽带)."""
    thr = p_db[i_pk] - 6.0
    n = len(f)

    i = i_pk
    while i > 0 and p_db[i] > thr:
        i -= 1
    if p_db[i] > thr:                       # 一路到 DC 都没降下来
        return float('nan')
    f_lo = _crossing(f[i], p_db[i], f[i + 1], p_db[i + 1], thr)

    j = i_pk
    while j < n - 1 and p_db[j] > thr:
        j += 1
    if p_db[j] > thr:                       # 一路到 Nyquist 都没降下来
        return float('nan')
    f_hi = _crossing(f[j - 1], p_db[j - 1], f[j], p_db[j], thr)

    return float(f_hi - f_lo)


def pmr_db(f, p_db, i_pk, half_hz=8.0):
    """峰-中值比 (dB): 峰功率 / 邻域中值功率. 纯音 >> 0, 宽带 ≈ 0."""
    m = (f >= f[i_pk] - half_hz) & (f <= f[i_pk] + half_hz)
    if m.sum() < 5:
        return float('nan')
    return float(p_db[i_pk] - np.median(p_db[m]))


def if_jitter_all(xs, f_cs, halfs, fs=FS, nperseg=256, fmax=FMAX):
    """
    各分量时频脊线瞬时频率的标准差 (Hz), 逐样本算再取类内平均.

    逐样本先算再平均 (而不是先把 TFR 平均掉), 否则样本间的漂移会被抹平.
    窗长取 256 (而非 PSD 用的 512) 是为了拿到足够的帧数来估抖动:
    512 点窗在 2 s 记录上只有 12 帧, 估不准.
    """
    out = [[] for _ in f_cs]
    for x in xs:
        f, _, Z = scipy_stft(np.asarray(x, dtype=np.float64), fs=fs,
                             nperseg=nperseg, noverlap=nperseg * 3 // 4,
                             boundary=None)
        mag_all = np.abs(Z)
        for k, (fc, hw) in enumerate(zip(f_cs, halfs)):
            m = (f >= fc - hw) & (f <= fc + hw) & (f <= fmax)
            if m.sum() < 3:
                continue
            out[k].append(float(np.std(f[m][np.argmax(mag_all[m], axis=0)])))
    return [float(np.mean(s)) if s else float('nan') for s in out]


def _db(p, ref=None):
    ref = p.max() if ref is None else ref
    return 10.0 * np.log10(np.maximum(p, 1e-30) / max(ref, 1e-30))


def class_mean_psd(xs, fs=FS, fmax=FMAX):
    """类内平均 PSD (仅用于画图; 注意它含工况漂移, 不能用来测固有带宽)."""
    acc = None
    for x in xs:
        f, p = welch(np.asarray(x, dtype=np.float64), fs=fs,
                     nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT)
        acc = p if acc is None else acc + p
    acc = acc / len(xs) if acc is not None else 0.0
    m = f <= fmax
    return f[m], acc[m]


def eff_bandwidth(f, p, lo, hi):
    """带内有效带宽 (谱熵法): exp(H)·Δf. 与 §5.3 挤压权衡图同口径."""
    m = (f >= lo) & (f <= hi)
    if m.sum() < 3:
        return float('nan')
    q = np.asarray(p, dtype=np.float64)[m]
    s = q.sum()
    if s <= 0:
        return float('nan')
    q = q / s
    h = -float(np.sum(q * np.log(np.maximum(q, 1e-30))))
    return float(np.exp(h) * (f[1] - f[0]))


def measure_peaks_per_sample(xs, fs=FS, fmax=FMAX):
    """
    逐样本测量各分量的 (f_c, B6dB, PMR).

    逐样本测再统计 —— 若先在类内把 PSD 平均掉, 各样本中心频率的差异
    (转速/负荷漂移) 会被摊进峰宽, 测到的就不是分量的固有带宽了.

    Returns:
        f, [(f_c, b6db, pmr) per sample per component]  (len = n_comp + 1, 末位高频宽带)
    """
    n_comp = len(COMPONENTS)
    per_comp = [[] for _ in range(n_comp)]
    bb_flat, bb_peak = [], []
    f_ref = None
    for x in xs:
        f, p = welch(np.asarray(x, dtype=np.float64), fs=fs,
                     nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT)
        m = f <= fmax
        f, p = f[m], p[m]
        f_ref = f
        p_db = _db(p)
        for k, (_, lo, hi, _) in enumerate(COMPONENTS):
            mm = (f >= lo) & (f <= hi)
            if not mm.any():
                continue
            i_pk = int(np.where(mm)[0][int(np.argmax(p_db[mm]))])
            per_comp[k].append((float(f[i_pk]),
                                bandwidth_6db(f, p_db, i_pk),
                                pmr_db(f, p_db, i_pk),
                                eff_bandwidth(f, p, lo, hi)))
        mb = (f >= BROADBAND[0]) & (f <= BROADBAND[1])
        if mb.any():
            pb = 10 ** (p_db[mb] / 10)
            bb_flat.append(float(pb.mean() / (pb.max() + 1e-30)))
            bb_peak.append(float(p_db[mb].max()))
    return f_ref, per_comp, (bb_flat, bb_peak)


def summarize(per_comp_sample):
    """逐样本测量结果 -> 各指标的类内中位数 / 离散度."""
    f_c = [s[0] for s in per_comp_sample]
    b6 = np.array([s[1] for s in per_comp_sample], dtype=float)
    pmr = np.array([s[2] for s in per_comp_sample], dtype=float)
    beff = np.array([s[3] for s in per_comp_sample], dtype=float)
    if not f_c:
        return dict(f_c=np.nan, f_c_std=np.nan, b6=np.nan, b6_lo=np.nan,
                    b6_hi=np.nan, pmr=np.nan, beff=np.nan, beff_lo=np.nan,
                    beff_hi=np.nan, n=0)
    return dict(
        f_c=float(np.median(f_c)),
        f_c_std=float(np.std(f_c)),                 # 类内中心频率漂移
        b6=float(np.nanmedian(b6)),
        b6_lo=float(np.nanpercentile(b6, 25)),
        b6_hi=float(np.nanpercentile(b6, 75)),
        pmr=float(np.nanmedian(pmr)),
        beff=float(np.nanmedian(beff)),
        beff_lo=float(np.nanpercentile(beff, 25)),
        beff_hi=float(np.nanpercentile(beff, 75)),
        n=int(np.sum(~np.isnan(b6))),
    )


def top_peaks(f, p_db, fmax=FMAX, n=6, prominence=6.0, min_dist_hz=6.0):
    """类均值谱上的显著峰 (诊断用)."""
    df = f[1] - f[0]
    dist = max(1, int(round(min_dist_hz / df)))
    pk, _ = find_peaks(p_db, prominence=prominence, distance=dist)
    if len(pk) == 0:
        pk = np.array([int(np.argmax(p_db))])
    pk = pk[np.argsort(p_db[pk])[::-1]][:n]
    return [(float(f[i]), float(p_db[i])) for i in pk]


def assign_component(f_c):
    """按频带把实测峰归到物理分量."""
    for name, lo, hi, _ in COMPONENTS:
        if lo <= f_c <= hi:
            return name
    return OTHER_LABEL


# ═══════════════════════════════════════════════════════════════
# SAST 学到的挤压强度
# ═══════════════════════════════════════════════════════════════

def sast_sigma_by_class(checkpoint, device, X, y, per_class, data_path):
    """每类每物理节点的 (IF, w_i, sigma_bin, sigma_hz, lambda)."""
    if not checkpoint:
        return None
    import torch
    from sast_model import load_checkpoint
    from sast_config import SastConfig

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model, _, meta = load_checkpoint(checkpoint, dev)
    cfg = SastConfig.from_dict(meta.get('args', {}) if isinstance(meta, dict) else {})

    out = {}
    for c in sorted(np.unique(y).tolist()):
        cidx = np.where(y == c)[0]
        en = (X[cidx] ** 2).sum(axis=1)
        chosen = cidx[np.argsort(en)[::-1][:per_class]]

        w_list, if_list = [], []
        for idx in chosen:
            xt = torch.from_numpy(X[idx].astype(np.float32)).unsqueeze(0).to(dev)
            with torch.no_grad():
                res = model(xt, return_all=True, training=False)
            w_list.append(res['w_i'][0].cpu().numpy())
            if_list.append(res['node_if'][0].cpu().numpy())
        w_mean = np.stack(w_list).mean(axis=(0, 2))     # [3]
        if_mean = np.stack(if_list).mean(axis=(0, 2))   # [3]
        sigma_bin = cfg.sigma_min + (1 - w_mean) * (cfg.sigma_max - cfg.sigma_min)
        out[c] = dict(
            if_hz=if_mean,
            w=w_mean,
            sigma_bin=sigma_bin,
            sigma_hz=sigma_bin * SAST_BIN_HZ,
            lam=np.round(w_mean * cfg.n_sqz_max).astype(int),
        )
    return out


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='频带可压缩性分析')
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--checkpoint', default='sast_checkpoints/sast_v3_model.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--per-class', type=int, default=8, help='SAST 逐类样本数')
    ap.add_argument('--psd-samples', type=int, default=300, help='PSD 逐类样本数')
    ap.add_argument('--outdir', default='results/band_analysis')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    X, y = d['train_X'], d['train_y'].ravel().astype(np.int64)
    if X.ndim == 3:
        X = X[:, :, 0]
    classes = sorted(np.unique(y).tolist())

    print(f'窗函数极限带宽参考: Hann-{NPERSEG} @ {FS:.0f} Hz -> '
          f'Δf={BIN_HZ:.2f} Hz, -6 dB 主瓣 ≈ {WINDOW_6DB_HZ:.2f} Hz')

    peak_rows = []       # 明细
    comp_rows = []       # 与 SAST 的对照
    curves = {}          # class -> (f, mean_db, p25_db, p75_db)
    sast = sast_sigma_by_class(args.checkpoint, args.device, X, y,
                               args.per_class, args.data)

    for c in classes:
        cidx = np.where(y == c)[0]
        if len(cidx) > args.psd_samples:
            rng = np.random.RandomState(42)
            cidx = rng.choice(cidx, args.psd_samples, replace=False)
        xs = X[cidx].astype(np.float64)

        f, pxx = class_mean_psd(xs)
        p_db = _db(pxx)
        curves[c] = (f, pxx, p_db)

        # 逐样本测固有带宽 / 中心频率 / PMR
        _, per_comp, (bb_flat, bb_peak) = measure_peaks_per_sample(xs)
        stats = [summarize(s) for s in per_comp]
        # σ_f 用各分量实测中心频率的中值作为跟踪中心
        sfs = if_jitter_all(xs, [st['f_c'] for st in stats],
                            [cmp[3] for cmp in COMPONENTS])

        print(f'\n{"=" * 78}')
        print(f'class {c} ({CLASS_NAMES[c]})  n_psd={len(xs)}')
        print(f'类均值谱显著峰(诊断): '
              + ', '.join(f'{fp:.1f}Hz/{lp:.1f}dB'
                          for fp, lp in top_peaks(f, p_db)))
        print(f'{"分量":<28s} {"f_c(Hz)":>8s} {"Δf_c(Hz)":>9s} {"B6dB(Hz)":>9s} '
              f'{"B_eff(Hz)":>10s} {"PMR(dB)":>8s} {"σ_f(Hz)":>8s}  判定')
        print('-' * 96)

        for k, (name, lo, hi, if_half) in enumerate(COMPONENTS):
            st = stats[k]
            sf = sfs[k]
            # 判定: 固有带宽接近窗极限(窄) 且 相位稳定(σ_f 小)
            narrow = (not np.isnan(st['b6'])) and st['b6'] <= NARROW_FACTOR * WINDOW_6DB_HZ
            steady = (not np.isnan(sf)) and sf <= STEADY_SIGMA_F
            verdict = '可硬挤压' if (narrow and steady) else '不应硬挤压'

            peak_rows.append((c, CLASS_NAMES[c], name, st['f_c'], st['f_c_std'],
                              st['b6'], st['b6_lo'], st['b6_hi'],
                              st['beff'], st['beff_lo'], st['beff_hi'],
                              st['pmr'], sf, verdict))
            print(f'{name:<28s} {st["f_c"]:8.2f} {st["f_c_std"]:9.2f} '
                  f'{st["b6"]:9.2f} {st["beff"]:10.2f} {st["pmr"]:8.2f} '
                  f'{sf:8.2f}  {verdict}')

        # 高频宽带对照: 无显著峰 + 谱平坦
        if bb_flat:
            flat = float(np.mean(bb_flat))
            print(f'{OTHER_LABEL + " " + str(BROADBAND):<28s} '
                  f'{"—":>8s} {"—":>9s} {"—":>9s} {flat:8.3f} {"—":>8s}  '
                  f'不应硬挤压 (宽带噪声)')
            peak_rows.append((c, CLASS_NAMES[c], OTHER_LABEL + ' 150-500Hz', np.nan,
                              np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                              np.nan, flat, np.nan,
                              '不应硬挤压 (宽带噪声)'))

        # ── 与 SAST 对照 ──
        if sast is not None and c in sast:
            node_names = ['LOW_FREQ', 'BPF', '2xBPF']
            for n in range(3):
                comp = COMPONENTS[n][0]
                hit = [r for r in peak_rows
                       if r[0] == c and r[2] == comp]
                b6 = hit[0][5] if hit else float('nan')
                comp_rows.append((
                    c, CLASS_NAMES[c], node_names[n], comp, b6,
                    hit[0][8] if hit else np.nan,          # B_eff
                    hit[0][4] if hit else np.nan,          # f_c 跨样本标准差
                    hit[0][11] if hit else np.nan,         # PMR
                    sfs[n] if n < len(sfs) else np.nan,    # σ_f
                    float(sast[c]['if_hz'][n]), float(sast[c]['w'][n]),
                    float(sast[c]['sigma_bin'][n]), float(sast[c]['sigma_hz'][n]),
                    int(sast[c]['lam'][n]),
                ))

    # ── CSV ──
    with open(outdir / 'peaks.csv', 'w', newline='', encoding='utf-8-sig') as fp:
        w = csv.writer(fp)
        w.writerow(['class', 'class_name', 'component', 'f_center_hz',
                    'f_center_std_hz', 'b6db_hz', 'b6db_p25', 'b6db_p75',
                    'beff_hz', 'beff_p25', 'beff_p75',
                    'pmr_db', 'sigma_f_hz', 'verdict'])
        w.writerows(peak_rows)

    if comp_rows:
        with open(outdir / 'compressibility.csv', 'w', newline='',
                  encoding='utf-8-sig') as fp:
            w = csv.writer(fp)
            w.writerow(['class', 'class_name', 'node', 'component', 'b6db_hz',
                        'beff_hz', 'f_center_std_hz', 'pmr_db', 'sigma_f_hz',
                        'sast_if_hz', 'sast_w', 'sast_sigma_bin',
                        'sast_sigma_hz', 'sast_lambda'])
            w.writerows(comp_rows)

    # ── 图 1: 各类均值 PSD ──
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), squeeze=False)
    for ax, c in zip(axes[0], classes):
        f, pxx, p_db = curves[c]
        ax.plot(f, p_db, lw=0.9, color='#4C72B0')
        for lo, hi, lbl in [(2, 25, 'LOW_FREQ'), (42, 55, 'BPF'), (90, 105, '2×BPF')]:
            ax.axvspan(lo, hi, color='#C44E52', alpha=0.10)
            ax.axvline(lo, color='#C44E52', lw=0.5, ls='--', alpha=0.6)
            ax.axvline(hi, color='#C44E52', lw=0.5, ls='--', alpha=0.6)
        ax.axhline(-6.0, color='gray', lw=0.8, ls=':')
        ax.text(0.98, 0.04, '-6 dB', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=6, color='gray')
        for r in [r for r in peak_rows if r[0] == c and not np.isnan(r[3])]:
            ax.annotate(f'{r[3]:.1f} Hz\nB₆dB={r[5]:.1f} Hz',
                        xy=(r[3], _db(pxx)[int(np.argmin(np.abs(f - r[3])))]),
                        xytext=(4, 6), textcoords='offset points', fontsize=5.5,
                        color='#8C2D2F')
        ax.set_title(f'class {c} ({CLASS_NAMES[c]})', fontsize=9, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_xlim(0, FMAX)
        ax.set_ylim(-60, 3)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel('PSD (dB, 类内峰值归一)')
    fig.suptitle('各工况均值 PSD 与实测分量带宽 '
                 f'(Welch, nperseg={NPERSEG}; 窗极限 -6 dB ≈ {WINDOW_6DB_HZ:.1f} Hz)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / 'welch_psd_by_class.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 相关性小结: 三个候选判据分别与 SAST σ 比 ──
    def _col(i):
        return np.array([r[i] for r in comp_rows], dtype=float)

    if comp_rows:
        from scipy.stats import spearmanr
        sig = _col(12)                                   # sast_sigma_hz
        b6, beff, sf_, pmr_ = _col(4), _col(5), _col(8), _col(7)
        CANDS = [('B₆dB (峰带宽)', b6), ('B_eff (带内有效带宽)', beff),
                 ('σ_f (相位漂移)', sf_), ('PMR (谱纯度)', pmr_)]

        print('\n闭环验证 (n=%d):' % len(comp_rows))
        for nm, x in CANDS:
            m = ~np.isnan(x) & ~np.isnan(sig)
            if m.sum() >= 3:
                pe = float(np.corrcoef(x[m], sig[m])[0, 1])
                sp = float(spearmanr(x[m], sig[m]).statistic)
                print(f'  {nm:<22s} Pearson r = {pe:+.3f}   Spearman ρ = {sp:+.3f}')

        # 分组均值 (两级结构: 低频 vs 谐波). 组内仅 5 点, 不追组内单调性.
        print('\n分组均值:')
        print(f'  {"node":<10s} {"B₆dB":>8s} {"B_eff":>8s} {"σ_f":>7s} '
              f'{"PMR":>7s} {"SAST σ":>8s} {"λ":>4s}')
        for node in ['LOW_FREQ', 'BPF', '2xBPF']:
            sel = [r for r in comp_rows if r[2] == node]
            if not sel:
                continue
            gm = lambda i: float(np.nanmean([r[i] for r in sel]))
            print(f'  {node:<10s} {gm(4):>8.2f} {gm(5):>8.2f} {gm(8):>7.2f} '
                  f'{gm(7):>7.2f} {gm(12):>8.2f} {gm(13):>4.1f}')

        # 稳健性: 剔除"空转时 2×BPF 分量实际不存在"的离群点
        keep = ~np.isnan(b6) & ~np.isnan(sig) & (b6 < 15.0)
        if keep.sum() >= 3:
            print(f'\n  稳健性 (剔除 空转/2×BPF, PMR≈2.9 dB 说明该工况下分量不存在, '
                  f'n={int(keep.sum())}):')
            for nm, x in CANDS:
                pe = float(np.corrcoef(x[keep], sig[keep])[0, 1])
                sp = float(spearmanr(x[keep], sig[keep]).statistic)
                print(f'    {nm:<22s} Pearson r = {pe:+.3f}   Spearman ρ = {sp:+.3f}')

    # ── 图 2: 两个候选判据 vs SAST 学到的 σ ──
    if comp_rows:
        colors = {0: '#4C72B0', 1: '#DD8452', 2: '#55A868', 3: '#8172B3', 4: '#937860'}
        panels = [
            (5, '实测带内有效带宽 B_eff (Hz)',
             '带宽几乎不区分谐波与低频 → 不是判据'),
            (8, '实测相位漂移 σ_f (Hz)',
             '漂移越大, SAST 给的 σ 越大 (越不挤压)'),
            (7, '实测谱纯度 PMR (dB)',
             '越纯, SAST 给的 σ 越小 (越挤压)'),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), squeeze=False)
        for ax, (col, xlabel, subtitle) in zip(axes[0], panels):
            for node, mk in [('LOW_FREQ', 'o'), ('BPF', 's'), ('2xBPF', '^')]:
                for c in classes:
                    pts = [r for r in comp_rows if r[2] == node and r[0] == c]
                    if pts and not np.isnan(pts[0][col]):
                        ax.scatter([p[col] for p in pts], [p[12] for p in pts],
                                   marker=mk, s=60, color=colors[c], alpha=0.85,
                                   edgecolor='k', linewidth=0.4,
                                   label=f'{node} / class {c}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('SAST 学到的核宽 σ (Hz)')
            ax.set_xlim(0, None)
            ax.set_ylim(0, None)
            x1 = ax.get_xlim()[1]
            for s_bin, lbl in [(0.5, 'σ=0.5 bin (0.25 Hz)'),
                               (2.0, 'σ=2 bin (1.0 Hz)'), (3.0, 'σ=3 bin (1.5 Hz)')]:
                ax.axhline(s_bin * SAST_BIN_HZ, color='gray', lw=0.7, ls=':')
                ax.text(x1, s_bin * SAST_BIN_HZ, ' ' + lbl, fontsize=5.5,
                        va='bottom', ha='right', color='gray')
            xv = _col(col)
            m = ~np.isnan(xv) & ~np.isnan(_col(12))
            rv = float(np.corrcoef(xv[m], _col(12)[m])[0, 1]) if m.sum() >= 3 else np.nan
            ax.set_title(f'{subtitle}\nPearson r = {rv:+.3f}  (n={int(m.sum())})',
                         fontsize=9, fontweight='bold')
            ax.grid(alpha=0.25)
        axes[0][0].legend(fontsize=5.5, ncol=2, loc='upper left')
        fig.suptitle('闭环验证: SAST 学到的挤压强度 σ 与数据实测特征的关系',
                     fontsize=11, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(outdir / 'sigma_vs_bandwidth.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f'\n输出目录: {outdir}')


if __name__ == '__main__':
    main()