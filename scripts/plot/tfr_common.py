"""
TFR 对比脚本共用工具 (统一实现 / 统一绘图 / 统一指标)
======================================================

所有时频图对比脚本共用此模块, 保证三件事:
  1. 算法实现统一: 官方 5 个对比方法集中定义于此 (canonical methods),
     参数统一 (canonical params), 不再各自为政.
  2. 绘图统一: 线性幅度 + 0-500 Hz + jet + 主导频率标注,
     与 scripts/plot/plot_msst_tfr.py 完全一致.
  3. 指标统一: Rényi 熵 (α=3) + 脊线能量占比.

canonical 参数 (与 models/sast.py 内部保持一致):
  - msst:   hlength = min(N, 512) = 512
  - wsst2:  mywav = 'morse' (MATLAB 'amor' 默认)

对比方法 (用户选定, 见 待补充实验清单 §2.2):
  SST1  一阶同步压缩  (ssqueezepy ssq_stft)
  WSST1 一阶小波同步压缩 (ssqueezepy ssq_cwt)
  WSST2 二阶小波同步压缩 (models.tfr.wsst2, morse)
  MSST1 一阶多重同步压缩 (models.tfr.msst, num=1)
  MSST4 四阶多重同步压缩 (models.tfr.msst, num=4)
"""
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

# 保证 scripts/plot/*.py 能 import models.tfr
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import ssqueezepy as ssq                                       # noqa: E402
from models.tfr import wsst2, msst, compute_renyi              # noqa: E402

# ── canonical 参数 (统一) ──
FS = 1000.0
HLENGTH = 512            # msst 窗长 (SAST 内部默认 min(N,512)=512)
MSST_NUM = 4             # msst 迭代次数
FMAX_DEFAULT = 500.0     # 显示/指标频率上限 (Hz)
TOTAL = 2.0              # 时间轴长度 (s)
CMAP = 'jet'


# ═══════════════════════════════════════════════════════════════
# 统一绘图风格 (中文字体)
# ═══════════════════════════════════════════════════════════════
# matplotlib 默认字体 (DejaVu Sans) 没有 CJK 字形, 图题/标签中的中文会渲染
# 成方框. 这里统一挂上系统中文字体的回退链, 供所有绘图脚本调用.
CJK_FONTS = [
    'Microsoft YaHei',     # 微软雅黑 (Windows 首选)
    'SimHei',              # 黑体
    'Noto Sans CJK SC',    # Linux 常见
    'Source Han Sans SC',  # 思源黑体
    'DejaVu Sans',         # 兜底 (纯 ASCII)
]


def apply_plot_style(font_size=8):
    """统一绘图风格: 中文字体回退 + 负号正常显示 + 字号."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = CJK_FONTS
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = font_size


# ═══════════════════════════════════════════════════════════════
# 数据加载 & 抽样
# ═══════════════════════════════════════════════════════════════

def load_samples(data_path, per_class=1, select='highest-energy'):
    """
    从 npz 加载, 每类抽 per_class 个样本.

    Args:
        data_path: npz 路径 (含 train_X / train_y)
        per_class: 每类样本数
        select:    'highest-energy' (能量最大, 时频最干净) | 'first'

    Returns:
        xs:      list of [T] float64
        ys:      list of int
        idxs:    list of int (原始样本索引)
        classes: sorted unique class list
    """
    d = np.load(data_path, allow_pickle=True)
    X = d['train_X']
    y = d['train_y'].ravel().astype(np.int64)
    if X.ndim == 3:
        X = X[:, :, 0]

    classes = sorted(np.unique(y).tolist())
    xs, ys, idxs = [], [], []
    for c in classes:
        cidx = np.where(y == c)[0]
        if select == 'highest-energy':
            en = (X[cidx] ** 2).sum(axis=1)
            order = np.argsort(en)[::-1]
            chosen = cidx[order[:per_class]]
        else:
            chosen = cidx[:per_class]
        for idx in chosen:
            xs.append(X[idx].astype(np.float64))
            ys.append(int(c))
            idxs.append(int(idx))
    return xs, ys, idxs, classes


# ═══════════════════════════════════════════════════════════════
# 挤压 / 指标 (统一实现)
# ═══════════════════════════════════════════════════════════════

def hard_squeeze_mag(tfr_complex, omega, neta=None, threshold=1e-4):
    """
    硬最近邻挤压 (与 models/tfr.py msst() Step4 完全一致, 向量化).

    Ts[k-1, b] += |tfr[eta, b]|   for |tfr| > threshold and 1 <= k <= neta

    Args:
        tfr_complex: [F, T] 复数 STFT
        omega:       [F, T] int, 1-indexed 目标 bin (0 = 无效)
        neta:        有效 bin 上界 (默认 F)
        threshold:   幅值阈值 (默认 1e-4, 同 msst)

    Returns:
        Ts: [F, T] 挤压后幅度谱
    """
    tfr = np.asarray(tfr_complex)
    omega = np.asarray(omega)
    F, T = tfr.shape
    neta = neta or F
    mag = np.abs(tfr)
    valid = (mag > threshold) & (omega >= 1) & (omega <= neta)
    idx = np.clip(omega - 1, 0, F - 1).astype(np.int64)
    cols = np.broadcast_to(np.arange(T, dtype=np.int64)[None, :], (F, T))
    Ts = np.zeros((F, T), dtype=np.float64)
    vals = np.where(valid, mag, 0.0)
    np.add.at(Ts, (idx, cols), vals)
    return Ts


def band_mask(freqs, fmax=FMAX_DEFAULT):
    """频率上限掩码 (跨方法统一频率范围, 保证 Rényi/脊线可比)."""
    mask = np.asarray(freqs) <= fmax
    if not mask.any():
        mask = np.ones_like(np.asarray(freqs), dtype=bool)
    return mask


# ═══════════════════════════════════════════════════════════════
# 频段分区 (与 models/sast_graph.py 的物理节点定义一致)
# ═══════════════════════════════════════════════════════════════
# 全局指标会被"本就不该挤压"的低频段稀释 —— SAST 故意不在低频挤压,
# 于是在全带 Rényi 上吃亏. 因此汇报时应按物理分量分区统计:
#   harm : BPF + 2xBPF, 应该挤压 -> 越集中越好
#   low  : LOW_FREQ 涡带, 不应挤压 -> 应保留带宽
REGIONS = {
    'full': (0.0, 500.0),
    'harm': (42.0, 105.0),
    'low': (2.0, 25.0),
}

REGION_LABELS = {
    'full': '全带 (0-500 Hz)',
    'harm': '谐波带 (BPF+2xBPF, 42-105 Hz)',
    'low': '低频带 (LOW_FREQ, 2-25 Hz)',
}


def region_mask(freqs, band):
    """频段掩码 (闭区间 [lo, hi], Hz)."""
    freqs = np.asarray(freqs).ravel()
    return (freqs >= band[0]) & (freqs <= band[1])


def renyi_entropy(mag):
    """Rényi 熵 (α=3), 越低越集中. 封装 models.tfr.compute_renyi."""
    return compute_renyi(np.asarray(mag), alpha=3)


def ridge_energy_ratio(mag, rel_thresh=0.05, pad=1):
    """
    脊线能量占比 (脊线清晰度指标, 越高越好).

    每帧: 取局部极大 (≥ rel_thresh × 帧峰值) 作为脊线, 统计其 ±pad bin 内
    能量占该帧总能量的比例, 再对所有帧取平均.
    """
    mag = np.asarray(mag, dtype=np.float64)
    if mag.ndim == 1:
        mag = mag[:, None]
    F, T = mag.shape
    colmax = mag.max(axis=0, keepdims=True) + 1e-12
    above = mag >= rel_thresh * colmax
    is_peak = np.zeros_like(mag, dtype=bool)
    if F >= 3:
        is_peak[1:-1, :] = (mag[1:-1, :] >= mag[:-2, :]) & (mag[1:-1, :] >= mag[2:, :])
    peaks = is_peak & above
    ridge = ndimage.maximum_filter(peaks, size=(2 * pad + 1, 1))
    num = (mag * ridge).sum()
    den = mag.sum() + 1e-12
    return float(num / den)


# ═══════════════════════════════════════════════════════════════
# canonical 对比方法 (用户选定 5 个)
#   每个 method_*(x) -> dict(mag, freqs, t)
# ═══════════════════════════════════════════════════════════════

def method_sst1(x):
    """一阶 SST (ssqueezepy ssq_stft)."""
    Tx, _, f, _ = ssq.ssq_stft(x, fs=FS)
    Tx = np.asarray(Tx)
    t = np.arange(Tx.shape[1]) / FS
    return dict(mag=np.abs(Tx), freqs=np.asarray(f).ravel(), t=t)


def method_wsst1(x):
    """一阶 WSST (ssqueezepy ssq_cwt)."""
    Tx, _, f, _ = ssq.ssq_cwt(x, fs=FS)
    Tx = np.asarray(Tx)
    t = np.arange(Tx.shape[1]) / FS
    return dict(mag=np.abs(Tx), freqs=np.asarray(f).ravel(), t=t)


def method_wsst2(x):
    """二阶 WSST (models.tfr.wsst2, morse)."""
    r = wsst2(x, FS, mywav='morse')
    t = np.arange(r['WSST2'].shape[1]) / FS
    return dict(mag=np.abs(r['WSST2']), freqs=r['freqs'], t=t)


def _msst(x, num):
    """多重同步压缩 (models.tfr.msst) 的 `${num}` 阶结果."""
    r = msst(x, FS, hlength=HLENGTH, num=MSST_NUM, save_trajectory=True)
    omega = r['omegas'][0] if num == 1 else r['omega_final']
    mag = hard_squeeze_mag(r['STFT'], omega, r['STFT'].shape[0])
    return dict(mag=mag, freqs=r['freqs'], t=r['t'])


def method_msst1(x):
    """一阶 MSST (num=1)."""
    return _msst(x, 1)


def method_msst4(x):
    """四阶 MSST (num=4)."""
    return _msst(x, 4)


# 对比方法注册表: 名称 -> 计算函数
CANONICAL_METHODS = [
    ('SST1',  method_sst1),
    ('WSST1', method_wsst1),
    ('WSST2', method_wsst2),
    ('MSST1', method_msst1),
    ('MSST4', method_msst4),
]


# ═══════════════════════════════════════════════════════════════
# 主导频率 & 绘图 (线性幅度, 与 plot_msst_tfr.py 一致)
# ═══════════════════════════════════════════════════════════════

def find_dominant_freqs(tfr, freq_axis, fmax=FMAX_DEFAULT, n_peaks=8,
                        prominence=0.03, min_dist=8):
    """从 TFR 时间平均谱中提取主导频率 (Hz)."""
    freq_axis = np.asarray(freq_axis)
    mask = freq_axis <= fmax
    f = freq_axis[mask]
    spec = np.mean(np.asarray(tfr)[mask, :], axis=1)
    spec = spec / (spec.max() + 1e-12)

    peaks, _ = find_peaks(spec, prominence=prominence, distance=min_dist)
    if len(peaks) == 0:
        peaks, _ = find_peaks(spec, prominence=prominence * 0.5,
                              distance=min_dist // 2)
    if len(peaks) == 0:
        peaks = [np.argmax(spec)]

    sort_idx = np.argsort(spec[peaks])[::-1]
    peaks = peaks[sort_idx][:n_peaks]
    return f[peaks]


def plot_panel(ax, mag, freqs, t, title, fmax=FMAX_DEFAULT,
               annotate_dominant=True):
    """在 ax 上绘制一张线性幅度 TFR 面板 (0-fmax Hz, vmax=0.5·peak, jet)."""
    mag = np.asarray(mag, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    if freqs[0] > freqs[-1]:  # 部分实现频率轴倒序
        freqs = freqs[::-1]
        mag = mag[::-1, :]

    mask = band_mask(freqs, fmax)
    vmax = mag[mask, :].max() * 0.5
    ax.pcolormesh(t, freqs[mask], mag[mask, :],
                  shading='gouraud', cmap=CMAP, vmax=vmax)

    if annotate_dominant:
        dom = find_dominant_freqs(mag, freqs, fmax)
        dom_text = '\n'.join([f'{f:6.1f} Hz' for f in dom[:6]])
        ax.text(0.985, 0.97, f'Dominant:\n{dom_text}',
                transform=ax.transAxes, fontsize=5.5, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.75, edgecolor='gray', lw=0.5),
                family='monospace')

    ax.set_ylim(0, fmax)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Freq (Hz)')
    ax.set_title(title, fontsize=8, fontweight='bold')


def save_tfr_png(mag, freqs, t, save_path, title,
                 fmax=FMAX_DEFAULT, dpi=200):
    """保存单张 TFR PNG (线性幅度样式)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_panel(ax, mag, freqs, t, title, fmax=fmax)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)