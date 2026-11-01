"""
全频段谱峰普查 (支撑 §4.2 的"还有没有别的特征频带"与 100 Hz 归属判定)
========================================================================

要回答的问题:
  1) 100.1 Hz 到底是不是 2×BPF? (与理论转速锁定梳齿对比)
  2) 除了 SAST 现有的 4 个节点, 0-500 Hz 内还有没有别的稳定特征频带?
  3) 每个选出的频带是"转速锁定结构分量"还是"工况相关水力分量"?

方法:
  A. 候选峰: 在 5 类类均值 PSD 上取并集, prominence>=6 dB, 0-500 Hz
  B. 逐样本锁定: 对每个候选频率 f0, 在每个样本的谱上取 [f0-4, f0+4] Hz 内
     局部极大 -> 得到逐样本峰频 f_s 与峰级 L_s. 由此得到
       df_sample = std(f_s)     跨样本频率漂移 (Hz)  —— 小 => 转速锁定
       df_class  = std(类均值峰频)                   跨工况频率漂移 (Hz)
  C. 理论梳齿对比: f_rot 由额定转速 333.3 rpm 得 5.5556 Hz (9 对极 @50Hz 电网,
     同步机组转速被电网锁定, 与负荷无关). 结构与动静干涉分量的候选频率:
       k*Z_r*f_rot  (Z_r=9)  -> 50, 100, 150, 200, 250, ...
       m*Z_g*f_rot  (Z_g=20) -> 111.1, 222.2, 333.3, 444.4
       (Z_g-Z_r)*f_rot       -> 61.1
       (Z_g+Z_r)*f_rot       -> 161.1
       f_rot=5.56, 2*f_rot=11.11 (轴频/二倍轴频)
  D. 判别力: 对每个候选频带取带内能量, 单独做 5 类 silhouette / 1-NN,
     与 SAST 的 4 个节点对比, 看是否值得加节点.

  ⚠ 已删除的假设: 曾把空转主峰归因于"45.65% 额定转速 -> f_rot=2.536 Hz,
    Z_r·f_rot=22.82 Hz"。该假设已被实测**证伪**:
      - 空转主峰实测在 45.90-46.39 Hz (不是 22.82 Hz, 也不是 50 Hz);
      - comb_check.csv 独立印证: 22.8227 Hz 的最近实测峰是 15.137 Hz
        (偏移 -7.686 Hz, 落空), 而 2×Z_r·f_rot 预测的 45.645 Hz 与实测
        45.898 Hz 相符 (偏移 +0.253 Hz) —— 即该假设的"倍频"预测命中、
        "基频"预测落空, 说明 45.9 Hz 不是 22.8 Hz 的二次谐波。
      - 结论: 该分量是**工况漂移分量** (频率由流动不稳定性决定), 不是
        转速锁定分量。详见 docs §14.4。

输出 (默认 results/band_census/):
  peak_census.csv       每个候选峰 × 每类: f_c, level, df, 是否为候选
  band_discriminability.csv  每个候选带 (以及 SAST 节点/带宽) 的判别力
  comb_check.csv        理论梳齿 vs 实测最近峰
  psd_candidates.png    5 类类均值 PSD + 候选峰 + 梳齿线
  peak_stability.png    候选峰: 跨样本漂移 vs 峰级, 分类着色

用法:
  python scripts/analysis/check_spectral_bands.py --samples 300
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.spatial.distance import cdist

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / 'scripts' / 'plot'))

from tfr_common import apply_plot_style       # noqa: E402

apply_plot_style()

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ── 采集/估计参数 (与 analyze_band_compressibility 一致) ──
FS = 1000.0
NPERSEG = 512
NOVERLAP = 384
NFFT = 2048
FMAX = 500.0
BIN_HZ = FS / NPERSEG                    # 1.953 Hz 真实频率分辨率
WIN_6DB = 1.87 * BIN_HZ                  # Hann-512 -6 dB 主瓣 ≈ 3.65 Hz

CLASS_NAMES = ['空转', '低负荷', '中负荷', '高负荷', '抽水']

# ── 机组参数 (用户提供) ──
N_RATED = 333.3          # rpm
Z_R = 9                  # 转轮叶片数
Z_G = 20                 # 活动导叶数
F_GRID = 50.0
F_ROT = N_RATED / 60.0                 # 5.5556 Hz — 电网锁定
# F_ROT_IDLE (0.4565·F_ROT ≈ 2.536 Hz, "空转 45.65% 转速") 已删除 —— 该假设
# 被实测证伪, 见文件头 ⚠ 段。

# ── SAST 现有节点 ──
# 从单一真源导出, 不再硬编码: B4 的拆分撤回后是 3 节点 (42/48.5 那条内部边
# 已不存在), 硬编码副本会静默过期并把普查结论挂在错误的带上。
from models.sast_nodes import PUMP_TURBINE_REGIONS as _REGIONS  # noqa: E402
SAST_NODES = [(r.name, r.f_min, r.f_max) for r in _REGIONS]

LOCK_HALF = 4.0          # 逐样本峰锁定搜索半径 (Hz)
PROM_DB = 6.0            # 类均值谱峰显著性门限
MIN_DIST_HZ = 5.0


def db(p, ref=None):
    ref = p.max() if ref is None else ref
    return 10.0 * np.log10(np.maximum(p, 1e-30) / max(ref, 1e-30))


def per_sample_psd_matrix(xs):
    """[S, F] PSD 矩阵 + 频率轴 (0-FMAX)."""
    mats = []
    for x in xs:
        f, p = welch(np.asarray(x, dtype=np.float64), fs=FS,
                     nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT)
        mats.append(p)
    P = np.stack(mats)
    m = f <= FMAX
    return f[m], P[:, m]


def lock_peak(f, P, f0, half=LOCK_HALF):
    """
    在每条样本谱的 [f0-half, f0+half] 内取局部极大.
    返回 (f_s[S], level_db_s[S] rel. 该样本谱峰值, peak_hz, level_db_median)
    """
    m = (f >= f0 - half) & (f <= f0 + half)
    if m.sum() < 3:
        return None
    idx = np.where(m)[0]
    sub = P[:, idx]                                     # [S, k]
    j = np.argmax(sub, axis=1)
    f_s = f[idx][j]
    # 以"该样本谱内 +-half 内峰值"为参考的绝对级 (dB re 每样本谱最大)
    ref = P.max(axis=1, keepdims=True)
    lv = 10.0 * np.log10(np.maximum(sub[np.arange(P.shape[0]), j], 1e-30)
                         / np.maximum(ref.ravel(), 1e-30))
    return f_s, lv, float(np.median(f_s)), float(np.median(lv))


def detect_peaks(f, p_db, n=40):
    df = f[1] - f[0]
    dist = max(1, int(round(MIN_DIST_HZ / df)))
    pk, prop = find_peaks(p_db, prominence=PROM_DB, distance=dist)
    if len(pk) == 0:
        return []
    order = np.argsort(p_db[pk])[::-1][:n]
    return [(float(f[pk[i]]), float(p_db[pk[i]]), float(prop['prominences'][i]))
            for i in order]


def silhouette_1nn(Z, y):
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    n = len(y)
    if n < 4:
        return float('nan'), float('nan')
    nc = min(5, n - 1, Z.shape[1])
    Zp = PCA(n_components=nc).fit_transform(Z) if Z.shape[1] > nc else Z
    try:
        sil = float(silhouette_score(Zp, y))
    except Exception:
        sil = float('nan')
    D = cdist(Zp, Zp)
    np.fill_diagonal(D, np.inf)
    knn = float((y[np.argmin(D, axis=1)] == y).mean())
    return sil, knn


def the_comb():
    """(label, freq_hz) 理论候选."""
    c = []
    for k in range(1, 10):
        c.append((f'{k}×Z_r·f_rot (k·BPF)', k * Z_R * F_ROT))
    for m in range(1, 5):
        c.append((f'{m}×Z_g·f_rot (导叶通过)', m * Z_G * F_ROT))
    c.append(('(Z_g−Z_r)·f_rot', (Z_G - Z_R) * F_ROT))
    c.append(('(Z_g+Z_r)·f_rot', (Z_G + Z_R) * F_ROT))
    c.append(('f_rot (轴频)', F_ROT))
    c.append(('2·f_rot', 2 * F_ROT))
    c.append(('3·f_rot', 3 * F_ROT))
    # (原"空转 Z_r·f_rot / 空转 2·Z_r·f_rot"已删除 —— 45.65% 转速假设被证伪。
    #  其 2×BPF 预测 45.645 Hz 虽与实测 45.898 接近, 但 1×BPF 预测 22.823 Hz
    #  落空 (最近实测峰 15.137), 说明 45.9 Hz 并非 22.8 Hz 的二次谐波。)
    return [(n, fq) for n, fq in c if fq <= FMAX]


def main():
    ap = argparse.ArgumentParser(description='全频段谱峰普查')
    ap.add_argument('--data', default='5_dataset.npz')
    ap.add_argument('--samples', type=int, default=300, help='每类样本数')
    ap.add_argument('--outdir', default='results/band_census')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    X, y = d['train_X'], d['train_y'].ravel().astype(np.int64)
    if X.ndim == 3:
        X = X[:, :, 0]
    classes = sorted(np.unique(y).tolist())

    print(f'f_rot = {F_ROT:.4f} Hz (额定 {N_RATED} rpm, 电网锁定)')
    print(f'BPF = Z_r·f_rot = {Z_R * F_ROT:.2f} Hz,  2×BPF = {2 * Z_R * F_ROT:.2f} Hz')
    print('空转主峰: 实测 45.90-46.39 Hz, 属工况漂移分量 '
          '(45.65% 转速假设已证伪, 见文件头)')
    print(f'频率分辨率 {BIN_HZ:.3f} Hz, Hann-{NPERSEG} -6 dB 主瓣 {WIN_6DB:.2f} Hz\n')

    # ── 逐类 PSD ──
    f = None
    Ps, mean_psd, mean_db = {}, {}, {}
    for c in classes:
        cidx = np.where(y == c)[0]
        if len(cidx) > args.samples:
            cidx = np.random.RandomState(42).choice(cidx, args.samples, replace=False)
        f, P = per_sample_psd_matrix(X[cidx].astype(np.float64))
        Ps[c] = P
        mean_psd[c] = P.mean(axis=0)
        mean_db[c] = db(mean_psd[c])
        print(f'class {c} ({CLASS_NAMES[c]})  n={len(cidx)}')

    # ── 候选峰: 各类均值谱峰的并集 ──
    cand = []
    for c in classes:
        for fp, lp, pr in detect_peaks(f, mean_db[c]):
            if any(abs(fp - q) <= MIN_DIST_HZ for q in cand):
                continue
            cand.append(fp)
    cand = sorted(cand)
    print(f'\n候选峰 ({len(cand)} 个): '
          + ', '.join(f'{q:.1f}' for q in cand))

    # ── 逐类逐样本锁定 ──
    rows = []          # (f0, class, f_sample_med, df_sample, level_db_med, n_present)
    for f0 in cand:
        for c in classes:
            r = lock_peak(f, Ps[c], f0)
            if r is None:
                continue
            f_s, lv, fm, lm = r
            # 只统计真正存在峰的样本 (级 > 该样本谱峰 -35 dB)
            pres = lv > -35.0
            rows.append((f0, c, fm, float(np.std(f_s[pres])) if pres.sum() > 3 else np.nan,
                         lm, float(pres.mean()),
                         float(np.median(lv[pres])) if pres.sum() else np.nan))

    with open(outdir / 'peak_census.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['f_candidate_hz', 'class', 'class_name', 'f_sample_med_hz',
                    'df_sample_std_hz', 'level_db_med', 'presence_ratio'])
        for r in rows:
            f0, c, fm, dfs, lm, pres, _lvp = r
            w.writerow([f'{f0:.2f}', c, CLASS_NAMES[c], f'{fm:.2f}',
                        f'{dfs:.3f}' if dfs == dfs else '', f'{lm:.2f}',
                        f'{pres:.2f}'])

    # ── 打印普查表 ──
    print(f'\n{"f0(Hz)":>8s} {"实测f(Hz)":>10s} {"跨样本σ":>9s} {"类间σ":>8s} '
          f'{"级(dB)":>8s} {"出现率":>7s}  归属')
    print('-' * 78)
    census = []
    for f0 in cand:
        sub = [r for r in rows if r[0] == f0]
        if not sub:
            continue
        fms = np.array([r[2] for r in sub])
        dfs = np.array([r[3] for r in sub])
        lvs = np.array([r[4] for r in sub])          # 类内中位级
        pr = np.array([r[5] for r in sub])           # 出现率
        f_med = float(np.median(fms))
        d_s = float(np.nanmedian(dfs))
        d_c = float(np.std(fms))
        lv = float(np.nanmedian(lvs))
        pres_r = float(np.nanmedian(pr))
        # 归属
        owner = '未归属'
        for nm, lo, hi in SAST_NODES:
            if lo <= f_med <= hi:
                owner = nm
                break
        else:
            if f_med > 150:
                owner = '高频宽带'
        census.append((f0, f_med, d_s, d_c, lv, pres_r, owner))
        print(f'{f0:8.2f} {f_med:10.2f} {d_s:9.3f} {d_c:8.3f} '
              f'{lv:8.2f} {pres_r:7.2f}  {owner}')

    # ── 理论梳齿对比 ──
    comb_rows = []
    print(f'\n{"理论分量":<26s} {"理论f(Hz)":>10s} {"最近实测峰":>10s} {"偏差(Hz)":>9s} '
          f'{"级(dB)":>8s}')
    print('-' * 74)
    for name, fq in the_comb():
        if len(cand) == 0:
            continue
        i = int(np.argmin([abs(q - fq) for q in cand]))
        f_near = cand[i]
        sub = [r for r in rows if r[0] == f_near]
        lv = float(np.nanmedian([r[5] for r in sub])) if sub else float('nan')
        comb_rows.append((name, fq, f_near, f_near - fq, lv))
        print(f'{name:<26s} {fq:10.2f} {f_near:10.2f} {f_near - fq:9.2f} {lv:8.2f}')

    with open(outdir / 'comb_check.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['component', 'theory_hz', 'nearest_peak_hz', 'offset_hz', 'level_db'])
        w.writerows(comb_rows)

    # ── 候选带的判别力 ──
    # 特征: 每个候选带 ±6 Hz 内能量 (逐样本)
    feats, tags = [], []
    for f0 in cand:
        m = (f >= f0 - 6.0) & (f <= f0 + 6.0)
        if m.sum() < 2:
            continue
        v = []
        for c in classes:
            v.append(np.log10(np.maximum(Ps[c][:, m].sum(axis=1), 1e-30)))
        feats.append(np.concatenate(v))
        tags.append(f'{f0:.1f} Hz')

    # SAST 3 节点的判别力 + 全带 + 150-500 宽带
    groups = []
    for nm, lo, hi in SAST_NODES:
        m = (f >= lo) & (f <= hi)
        v = [np.log10(np.maximum(Ps[c][:, m].sum(axis=1), 1e-30)) for c in classes]
        groups.append((nm, np.concatenate(v)))
    m = f > 150
    groups.append(('150-500 Hz 宽带',
                   np.concatenate([np.log10(np.maximum(Ps[c][:, m].sum(axis=1), 1e-30))
                                   for c in classes])))
    m = f <= FMAX
    groups.append(('全带 0-500 Hz',
                   np.concatenate([np.log10(np.maximum(Ps[c][:, m].sum(axis=1), 1e-30))
                                   for c in classes])))
    yv = np.concatenate([np.full(len(Ps[c]), c) for c in classes])

    disc_rows = []
    print(f'\n{"频带":<24s} {"silhouette":>11s} {"1-NN":>7s}  说明')
    print('-' * 62)
    for tag, v in list(zip(tags, feats)) + groups:
        Z = v.reshape(-1, 1)
        sil, knn = silhouette_1nn(Z, yv)
        note = '候选单带'
        if tag in [g[0] for g in groups]:
            note = 'SAST 节点' if tag in [n for n, _, _ in SAST_NODES] else '参照'
        disc_rows.append((tag, sil, knn, note))
        print(f'{tag:<24s} {sil:11.3f} {knn:7.3f}  {note}')

    with open(outdir / 'band_discriminability.csv', 'w', newline='',
              encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['band', 'silhouette', 'knn_accuracy', 'note'])
        w.writerows(disc_rows)

    # ── 图 1: 5 类类均值 PSD + 候选峰 + 梳齿 ──
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    for ax, c in zip(axes, classes):
        ax.plot(f, mean_db[c], lw=0.8, color='#2E75B6')
        for nm, lo, hi in SAST_NODES:
            ax.axvspan(lo, hi, color='#E06666', alpha=0.13)
        for f0 in cand:
            i = int(np.argmin(np.abs(f - f0)))
            if mean_db[c][i] > -20:
                ax.annotate(f'{f0:.1f}', (f0, mean_db[c][i]), fontsize=6,
                            rotation=90, color='#333333',
                            xytext=(0, 3), textcoords='offset points')
        for name, fq in the_comb():
            if fq <= FMAX:
                ax.axvline(fq, color='#999999', lw=0.5, ls=':', zorder=0)
        ax.set_ylabel(f'{CLASS_NAMES[c]}\n(dB)', fontsize=8)
        ax.set_ylim(-60, 3)
        ax.grid(alpha=0.25, lw=0.4)
    axes[-1].set_xlabel('频率 (Hz)')
    fig.suptitle('5 类类均值功率谱与谱峰普查\n'
                 '(红带 = SAST 现有 3 节点, 灰色点线 = 转速锁定理论梳齿)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(outdir / 'psd_candidates.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 图 2: 峰稳定性散点 ──
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap('tab10')
    for ci, c in enumerate(classes):
        sub = [r for r in rows if r[1] == c and r[3] == r[3]]
        if not sub:
            continue
        xs_ = [r[3] for r in sub]
        ys_ = [r[5] for r in sub]
        ax.scatter(xs_, ys_, s=34, color=cmap(ci), label=CLASS_NAMES[c],
                   alpha=0.85, edgecolors='white', linewidths=0.5)
    # 标注几个关键点
    for r in rows:
        if r[3] == r[3] and r[5] > -14 and r[3] < 0.6 and r[0] < 130:
            ax.annotate(f'{r[0]:.1f}Hz', (r[3], r[5]), fontsize=6,
                        xytext=(4, 3), textcoords='offset points')
    ax.set_xlabel('跨样本中心频率标准差 σ (Hz)  — 越小越"转速锁定"')
    ax.set_ylabel('峰级 (dB, re 各样本谱峰)')
    ax.set_title('候选峰稳定性: 转速锁定结构分量 vs 工况相关水力分量\n'
                 '(左下 = 稳定且强, 最值得作为物理节点)')
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / 'peak_stability.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\n输出目录: {outdir}')


if __name__ == '__main__':
    main()