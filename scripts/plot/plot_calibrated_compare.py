"""tfr_calibrated 终审图: 原始 STFT / SAST 未标定 / SAST 标定后 三联对比。

每个数据集一张图, 每行一个类别, 三列共享**该行原始 STFT 的色标上限**
(vmax = max_{f >= 2 Hz} tfr_raw, 避开 0-2 Hz 近直流镜像区)。共享色标是
本图的核心: (b)(c) 与 (a) 用同一色标, 才能直接看出标定是否把幅度拉回
原始 STFT 的标尺。

用法:
  conda run -n my_work python scripts/plot/plot_calibrated_compare.py \
      --which both --outdir results/calibrated_check
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.learnable_regions import LearnableBandMask  # noqa: E402
from models.sast import SAST  # noqa: E402
from sast_config import SastConfig  # noqa: E402
from sast_inference import infer_sast  # noqa: E402
from sast_model import load_checkpoint  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 0.6

F_LO = 2.0          # 色标/峰值统计的下限, 避开近直流镜像区
CMAP = 'jet'        # 与 sast_viz / plot_method_compare 一致

RSI_CKPT = ROOT / 'sast_checkpoints/gate3node/sast_v3_e020.pt'
RSI_DATA = ROOT / '5_dataset.npz'
RSI_LABELS = ['R0 空转', 'R1 低负荷', 'R2 中负荷', 'R3 高负荷', 'R4 抽水']

SHAFT_CKPT = ROOT / 'sast_checkpoints/band_flattop_shaft10/sast_v3_e010.pt'
SHAFT_DATA = ROOT / 'normalized_data.npz'
SHAFT_LABELS = [f'C{i}' for i in range(6)]


def _load_data(path):
    d = np.load(path, allow_pickle=True)
    X = d['train_X']
    if X.ndim == 3:
        X = X[:, :, 0]
    return X, d['train_y'].ravel().astype(np.int64)


def _pick(X, y, cls):
    """每类取带内能量最大的样本 (展示有代表性的强信号)。"""
    idx = np.where(y == cls)[0]
    if len(idx) == 0:
        raise SystemExit(f'class {cls} not found')
    e = np.sum(X[idx].astype(np.float64) ** 2, axis=1)
    j = int(np.argmax(e))
    return X[idx[j]], int(idx[j])


def load_shaft_model(path, device):
    """band_* 检查点带 band_mask.*; 当前 SAST 无该模块, 先用 learnable
    平顶窗的**等效边界**快照成静态 FreqRegion 再建模型 (软窗形状不复现)。"""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    args = ckpt.get('args', {})
    mask = LearnableBandMask.from_preset(args.get('band_preset', 'shaft'))
    sd = {k[len('band_mask.'):]: v for k, v in ckpt['model_state_dict'].items()
          if k.startswith('band_mask.')}
    mask.load_state_dict(sd, strict=False)
    print('  ' + mask.describe())
    cfg = SastConfig.from_dict(args)
    model = SAST(
        fs=cfg.fs, freq_regions=mask.as_regions(),
        d_h=cfg.d_h, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max,
        msst_num=cfg.msst_num, msst_hlength=cfg.msst_hlength,
        d_cond=cfg.d_cond, f_type_embed_dim=cfg.f_type_embed_dim,
        ppn_temperature=cfg.ppn_temperature,
        prototype_temperature=cfg.prototype_temperature,
        dropout=cfg.dropout, n_sqz_max=cfg.n_sqz_max,
        gate_mode=cfg.gate_mode, train_ridge=cfg.train_ridge,
        gate_detach=cfg.gate_detach, sigma_detach=cfg.sigma_detach,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model


def _panel(ax, tfr, freqs, t_axis, vmax, freq_max, title, note=None):
    t0, t1 = float(t_axis[0]), float(t_axis[-1])
    im = ax.imshow(tfr, origin='lower', aspect='auto', cmap=CMAP,
                   extent=[t0, t1, float(freqs[0]), float(freqs[-1])],
                   vmin=0.0, vmax=vmax, interpolation='nearest')
    ax.set_ylim(0, freq_max)
    ax.set_xlim(t0, t1)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel('频率 (Hz)', fontsize=8)
    ax.tick_params(labelsize=7)
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=7,
                va='top', color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.45))
    return im


def render(tag, model, data_path, labels, out_path, freq_max=200.0,
           dpi=170):
    device = next(model.parameters()).device
    X, y = _load_data(data_path)
    n = len(labels)
    fig = plt.figure(figsize=(11.6, 2.75 * n))
    gs = fig.add_gridspec(n, 4, width_ratios=[1, 1, 1, 0.035],
                          wspace=0.22, hspace=0.42)
    print(f'\n{tag}: {"class":<10} {"raw峰":>8} {"enh/raw":>8} '
          f'{"cal/raw":>8} {"γ@峰":>7} {"峰频":>9}')
    for k, lab in enumerate(labels):
        sig, sidx = _pick(X, y, k)
        r = infer_sast(model, sig, return_all=True)
        f, t = r['freqs'], r['t_axis']
        m = f >= F_LO
        raw, enh, cal = r['tfr_raw'], r['tfr_enhanced'], r['tfr_calibrated']
        vmax = float(raw[m].max())
        # 峰值取 (f>=2 Hz) 内全 record 最强点
        fi, ti = np.unravel_index(np.argmax(np.where(m[:, None], raw, -np.inf)),
                                  raw.shape)
        pr = float(raw[fi, ti])
        pe = float(enh[fi, ti]) / max(pr, 1e-12)
        pc = float(cal[fi, ti]) / max(pr, 1e-12)
        g = float(enh[fi, ti]) / max(float(cal[fi, ti]), 1e-12)
        print(f'  {tag:<8} {lab:<10} {pr:>8.4f} {pe:>8.3f} {pc:>8.3f} '
              f'{g:>7.3f} {f[fi]:>7.1f} Hz')

        axes = [fig.add_subplot(gs[k, c]) for c in range(3)]
        t0 = f'({"abcdefgh"[k]}) {lab}'
        im = _panel(axes[0], raw, f, t, vmax, freq_max,
                    t0 + (' — 原始 STFT' if k == 0 else ''),
                    note=f'峰值 {pr:.4f}')
        _panel(axes[1], enh, f, t, vmax, freq_max,
               'SAST 增强（未标定）' if k == 0 else '',
               note=f'增强/raw = {pe:.2f}×')
        _panel(axes[2], cal, f, t, vmax, freq_max,
               'SAST 标定后' if k == 0 else '',
               note=f'标定/raw = {pc:.2f}×')
        for c in range(3):
            axes[c].set_xlabel('时间 (s)', fontsize=8)
            if c > 0:
                axes[c].set_yticklabels([])
        cax = fig.add_subplot(gs[k, 3])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(f'{tag}: 原始 STFT / SAST 未标定 / SAST 标定后（同色标，'
                 f'vmax 取该行 f≥{F_LO:g} Hz 的原始峰）', fontsize=11, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('written:', out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', choices=['rsi', 'shaft', 'both'], default='both')
    ap.add_argument('--outdir', default=str(ROOT / 'results/calibrated_check'))
    ap.add_argument('--freq-max', type=float, default=200.0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outdir = Path(args.outdir)

    if args.which in ('rsi', 'both'):
        model, _, _ = load_checkpoint(RSI_CKPT, device)
        render('RSI 五类', model, RSI_DATA, RSI_LABELS,
               outdir / 'rsi_5class_calibrated_compare.png', args.freq_max)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if args.which in ('shaft', 'both'):
        print('\n主轴检查点: 由 band_mask 快照重建物理区域')
        model = load_shaft_model(SHAFT_CKPT, device)
        render('主轴六类', model, SHAFT_DATA, SHAFT_LABELS,
               outdir / 'shaft_6class_calibrated_compare.png', args.freq_max)


if __name__ == '__main__':
    main()