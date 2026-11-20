"""
统一 TFR 渲染器
================

**唯一真值源**: CNN 输入图与诊断/汇报图里的时频面板都从这里出像素。

渲染口径 (第 6 章 §6.2.2):
  875x656 像素、全幅 axes、jet、vmin = 0、vmax = 逐图峰值、
  频率轴 0-250 Hz 线性映射到 656 行、JPEG 质量 95。

对外两个入口:
  tfr_to_rgb()    -> (656, 875, 3) uint8, 就是 JPEG 解码后的像素
  save_tfr_jpg()  -> 落盘 (PIL, q=95)

诊断图请用 tfr_to_rgb() 的结果去 imshow, 不要在别处另写一套 imshow 参数 ——
一旦两处口径分叉, "汇报图好看但 CNN 看到的是另一张" 这类问题极难发现。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from scipy.interpolate import interp1d                 # noqa: E402

F_MAX = 250.0        # 2026-11 起由 200 改为 250: 656 行 => 0.381 Hz/行,
                     # 细于 STFT 的 0.5 Hz 分辨率, 不再丢单 bin 脊线
IMG_W, IMG_H = 875, 656
DPI = 100
JPEG_Q = 95
CMAP = 'jet'


def _to_rows(mag: np.ndarray, freqs: np.ndarray, fmax: float,
             n_rows: int) -> np.ndarray:
    """把任意频率网格重采样成 [n_rows, T] 的 0..fmax 均匀网格。"""
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    mag = np.asarray(mag, dtype=np.float32)
    order = np.argsort(freqs)
    f_s, m_s = freqs[order], mag[order]
    if f_s.size == n_rows and np.isclose(f_s[0], 0.0) and np.isclose(f_s[-1], fmax):
        return m_s
    grid = np.linspace(0.0, fmax, n_rows)
    return interp1d(f_s, m_s, axis=0, kind='linear',
                    bounds_error=False, fill_value=0.0)(grid)


def tfr_to_rgb(mag: np.ndarray, freqs: np.ndarray, t: np.ndarray,
               fmax: float = F_MAX, n_rows: int = IMG_H,
               width: int = IMG_W, height: int = IMG_H,
               vmax: float | None = None,
               vmax_ratio: float = 1.0,
               return_peak: bool = False):
    """渲染成 (height, width, 3) uint8 RGB。

    vmax:       显式色阶上界; None 时取 vmax_ratio * 数据峰值。
    vmax_ratio: 1.0 = 逐图峰值 (CNN 口径); 0.5 = 旧的汇报图口径。
    """
    data = _to_rows(mag, freqs, fmax, n_rows)
    t = np.asarray(t, dtype=np.float64).ravel()
    peak = float(data.max())
    if vmax is None:
        vmax = max(peak * vmax_ratio, 1e-12)

    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, aspect='auto', origin='lower',
              extent=[float(t[0]), float(t[-1]), 0.0, fmax],
              cmap=CMAP, vmin=0.0, vmax=vmax,
              interpolation='bilinear')
    ax.set_axis_off()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return (rgb, peak) if return_peak else rgb


def save_tfr_jpg(mag, freqs, t, out_path: Path, **kw) -> float:
    """渲染并落盘为 JPEG (q=95)。返回该图的数据峰值。"""
    from PIL import Image
    rgb, peak = tfr_to_rgb(mag, freqs, t, return_peak=True, **kw)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path, format='JPEG', quality=JPEG_Q)
    return peak