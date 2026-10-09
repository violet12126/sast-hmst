"""
SAST Visualization
==================
All plotting functions + inference summary printer.
matplotlib.use('Agg') is isolated here so importing other modules
(sast_config, sast_data, sast_model, sast_inference) has no side effect.
"""
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch  # noqa: F401  (needed for @torch.no_grad in some callers)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from models.sast_graph import NODE_NAMES

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8

# TFR 面板颜色映射动态范围 (dB): 显示峰值以下该范围内的分量。
# 调大 -> 弱分量(低频基线)更清楚, 但背景噪声也更明显; 调小则反之。
# 收紧 sigma 后低频脊线峰值约 -15 dB, 25 dB 已足够保留它并抑制更深噪声。
_TFR_DB_RANGE = 25.0


def _plot_stft_panel(ax, tfr, t_axis, freqs, freq_max, title, add_colorbar=False):
    """Plot a single STFT/TFR panel. dB scale with a fixed dynamic range:
    shows real components (incl. the low-frequency baseline) while clipping
    the deep background noise floor."""
    db = 10 * np.log10(np.maximum(tfr, 1e-6))
    vmax_db = db.max()
    vmin = vmax_db - _TFR_DB_RANGE
    im = ax.pcolormesh(t_axis, freqs, db, shading='gouraud', cmap='jet',
                       vmin=vmin, vmax=vmax_db)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title(title)
    if add_colorbar:
        plt.colorbar(im, ax=ax, label='dB')
    return im


def _plot_sigma_panel(ax, sigma_sq, t_axis, freqs, freq_max):
    """Plot squeeze bandwidth panel."""
    im = ax.pcolormesh(t_axis, freqs, sigma_sq, shading='gouraud',
                       cmap='RdYlGn_r', vmin=0, vmax=15)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]')
    ax.set_title('(c) Squeeze Bandwidth σ_sq [bins]\nRed=Broad(Soft), Green=Narrow(Hard)')
    plt.colorbar(im, ax=ax, label='σ_sq')


def _plot_w_i_panel(ax, w_i, t_axis, colors, node_names):
    """Plot w_i(t) per physics node."""
    N = w_i.shape[0]
    for n in range(N):
        ax.plot(t_axis, w_i[n], color=colors[n], lw=1.2,
                label=node_names[n] if n < len(node_names) else f'N{n}')
    ax.axhline(y=0.5, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('w_i')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(d) IF Trust w_i(t)\nw_i->1: Trust & squeeze  |  w_i->0: Conservative')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(alpha=0.3)


def _plot_gate_panel(ax, gate_edge, t_axis):
    """Plot edge gates panel."""
    M = gate_edge.shape[0]
    for m in range(M):
        ax.plot(t_axis, gate_edge[m], lw=0.5, alpha=0.6)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('gate_edge')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'(e) Edge Ratio Gates\n{M} physics edges')
    ax.grid(alpha=0.3)


def _plot_attention_panel(ax, A_ij):
    """Plot attention matrix panel."""
    M = A_ij.shape[0]
    im = ax.imshow(A_ij, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Edge Index')
    ax.set_title(f'(f) Edge Attention A_ij\n{M} edges -> diagnostic probe')
    plt.colorbar(im, ax=ax, label='A_ij')


def _plot_enhanced_tfr_panel(ax, tfr_enhanced, t_axis, freqs, freq_max):
    """Plot SAST TFR. dB scale with a fixed dynamic range; no node IF overlays."""
    db_enh = 10 * np.log10(np.maximum(tfr_enhanced, 1e-6))
    vmax_db = db_enh.max()
    vmin = vmax_db - _TFR_DB_RANGE
    ax.pcolormesh(t_axis, freqs, db_enh, shading='gouraud', cmap='jet',
                  vmin=vmin, vmax=vmax_db)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel('Time [s]')
    ax.set_title('(b) SAST Enhanced TFR')


def _plot_fft_panel(ax, signal, fs, freq_max, node_if, w_i, colors, node_names):
    """Plot FFT spectrum with dominant peaks and node IF markers."""
    N = len(signal)
    fft_freqs = np.fft.rfftfreq(N, 1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(signal))
    mask = fft_freqs <= freq_max
    f_f = fft_freqs[mask]
    m_f = fft_mag[mask]

    # Normalize to [0, 1]
    m_max = m_f.max()
    if m_max > 0:
        m_f = m_f / m_max

    ax.plot(f_f, m_f, 'k-', lw=0.6, alpha=0.8, label='FFT')

    # Detect dominant peaks
    peaks, props = find_peaks(m_f, prominence=0.02, distance=8)
    if len(peaks) == 0:
        peaks, _ = find_peaks(m_f, prominence=0.01, distance=4)
    if len(peaks) > 0:
        sort_idx = np.argsort(m_f[peaks])[::-1]
        peaks = peaks[sort_idx][:8]
        for p in peaks:
            ax.axvline(f_f[p], color='red', lw=0.6, ls='--', alpha=0.4)

    # Node IF markers (vertical lines at node IF, weighted by w_i)
    N_nodes = node_if.shape[0]
    for n in range(N_nodes):
        if_mean = node_if[n].mean()
        w_mean = w_i[n].mean()
        ax.axvline(if_mean, color=colors[n], lw=1.5, ls='-', alpha=0.3 + 0.5 * w_mean)
        name = node_names[n] if n < len(node_names) else f'N{n}'
        ax.annotate(f'{name}\n{if_mean:.1f} Hz', xy=(if_mean, 0.95),
                    fontsize=5.5, ha='center', va='top', color=colors[n],
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

    ax.set_xlim(0, freq_max)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Norm. Magnitude')
    ax.set_title('(d) FFT Spectrum + Node IF\nDominant peaks (red) vs GAT-tracked IF (colored)')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=6)


def plot_tfr(results: Dict[str, np.ndarray],
             save_path: str,
             freq_max: float = 200,
             signal: Optional[np.ndarray] = None,
             fs: float = 1000.0,
             title: Optional[str] = None):
    """
    4-panel plot (2x2): Raw MSST | SAST TFR + Node IF | w_i(t) | FFT Spectrum
    """
    tfr_raw = results.get('tfr_msst', results['tfr_raw'])
    tfr_enhanced = results['tfr_enhanced']
    w_i = results['w_i']
    node_if = results['node_if']
    freqs = results['freqs']
    t_axis = results['t_axis']

    N = w_i.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(N, 3)))
    if N <= 3:
        node_names = NODE_NAMES[1:]  # skip OP
    else:
        node_names = NODE_NAMES

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    _plot_stft_panel(axes[0, 0], tfr_raw, t_axis, freqs, freq_max,
                     '(a) Raw MSST (Hard Squeeze)\nStandard synchrosqueeze, no w_i gating')
    _plot_enhanced_tfr_panel(axes[0, 1], tfr_enhanced, t_axis, freqs, freq_max)
    _plot_w_i_panel(axes[1, 0], w_i, t_axis, colors, node_names)
    if signal is not None:
        _plot_fft_panel(axes[1, 1], signal, fs, freq_max, node_if, w_i, colors, node_names)
    else:
        axes[1, 1].set_visible(False)

    suptitle = title or 'SAST: Structure-Aware Synchrosqueezing\nPhysics Graph + Ratio-Gated PPM -> Adaptive Squeeze'
    plt.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Plot] Saved: {save_path}')


def plot_full_diagnostics(results: Dict[str, np.ndarray],
                          save_path: str,
                          epoch: Optional[int] = None,
                          freq_max: float = 200):
    """
    6-panel full diagnostic plot.
    """
    tfr_raw = results.get('tfr_msst', results['tfr_raw'])
    tfr_enhanced = results['tfr_enhanced']
    sigma_sq = results['sigma_sq']
    w_i = results['w_i']
    gate_edge = results['gate_edge']
    A_ij = results['A_ij'].mean(axis=0)  # average over heads
    node_if = results['node_if']
    freqs = results['freqs']
    t_axis = results['t_axis']

    N = w_i.shape[0]
    M_edges = gate_edge.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(N, 3)))
    if N <= 3:
        node_names = NODE_NAMES[1:]
    else:
        node_names = NODE_NAMES

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    _plot_stft_panel(axes[0, 0], tfr_raw, t_axis, freqs, freq_max,
                     '(a) Raw MSST (Hard Squeeze)', add_colorbar=True)
    _plot_enhanced_tfr_panel(axes[0, 1], tfr_enhanced, t_axis, freqs, freq_max)
    _plot_sigma_panel(axes[0, 2], sigma_sq, t_axis, freqs, freq_max)
    _plot_w_i_panel(axes[1, 0], w_i, t_axis, colors, node_names)
    _plot_gate_panel(axes[1, 1], gate_edge, t_axis)
    _plot_attention_panel(axes[1, 2], A_ij)

    title = 'SAST v3'
    if epoch is not None:
        title += f' - Epoch {epoch}'
    title += '\nPhysics Graph + Ratio-Gated PPM -> IF Trust w_i -> Adaptive Squeeze'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Plot] Saved: {save_path}')


def print_inference_summary(results: Dict[str, np.ndarray],
                            C_prior: Optional[np.ndarray] = None,
                            class_label: Optional[int] = None,
                            sample_idx: Optional[int] = None,
                            pred_class: Optional[int] = None,
                            probs: Optional[np.ndarray] = None):
    """Print a formatted inference summary table."""
    w_i = results['w_i']
    node_if = results['node_if']
    gate_node = results.get('gate_node')

    N = w_i.shape[0]
    if N <= 3:
        node_names = NODE_NAMES[1:]
    else:
        node_names = NODE_NAMES

    print(f"\n{'='*70}")
    header = "SAST Inference Summary"
    if class_label is not None:
        header += f" - Class {class_label}"
    if sample_idx is not None:
        header += f", Sample #{sample_idx}"
    print(header)
    print(f"{'='*70}")

    if pred_class is not None and probs is not None:
        print(f"Predicted: class {pred_class}  |  "
              + "  ".join(f"P({c})={probs[c]:.3f}" for c in range(len(probs))))

    print(f"{'Node':<15s} {'IF(Hz)':>8s} {'C_prior':>7s} {'w_i':>7s} {'gate':>7s}  Interpretation")
    print(f"{'-'*70}")

    for n in range(N):
        f_mean = node_if[n].mean()
        w_mean = w_i[n].mean()
        g_mean = gate_node[n].mean() if gate_node is not None else 0.0
        c_prior_val = C_prior[n] if C_prior is not None else float('nan')

        if w_mean > 0.7:
            interp = 'TRUSTED - aggressive squeeze'
        elif w_mean > 0.5:
            interp = 'Moderate trust'
        elif g_mean > 0.5:
            interp = 'Matched prior, low w_i'
        else:
            interp = 'CONSERVATIVE - soft squeeze'

        name = node_names[n] if n < len(node_names) else f'N{n}'
        c_str = f"{c_prior_val:7.3f}" if not np.isnan(c_prior_val) else "     N/A"
        print(f"  {name:<15s} {f_mean:8.1f} {c_str} {w_mean:7.3f} {g_mean:7.3f}  {interp}")

    w_spread = w_i.mean(axis=1).ptp()
    print(f"\n  w_i spread (max-min across nodes): {w_spread:.3f}")
    if w_spread > 0.2:
        print("  -> SAST is actively differentiating frequency components")
    else:
        print("  -> w_i not yet differentiated (need more training?)")
