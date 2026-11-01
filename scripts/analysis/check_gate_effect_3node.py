"""对比 3node_e15 (无缺席门控) 与 gate3node (有缺席门控) 的脊线指标.

只读; 输出 results/band_census/gate_effect_3node.csv
用法: conda run -n my_work python scripts/analysis/check_gate_effect_3node.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sast_model import load_checkpoint                          # noqa: E402

CKPTS = {
    'no_gate': 'sast_checkpoints/3node_e15/sast_v3_e015.pt',
    'gate':    'sast_checkpoints/gate3node/sast_v3_e020.pt',
}
SAMPLES = {0: 30, 1: 4876, 2: 3251}
BANDS = {'BPF[40,53)': (40.0, 53.0), '2xBPF[90,105)': (90.0, 105.0)}


def band_stats(tfr, freqs, f0, f1):
    m = (freqs >= f0) & (freqs < f1)
    t = tfr[m]                                   # [Fb, T]
    fb = freqs[m]
    e = t.sum(axis=1)                            # 时间边缘谱
    k = int(np.argmax(e))
    centre = float((fb * e).sum() / max(e.sum(), 1e-12))
    sd = float(np.sqrt(((fb - centre) ** 2 * e).sum() / max(e.sum(), 1e-12)))
    return dict(peak_hz=float(fb[k]), peak_share=float(e[k] / max(e.sum(), 1e-12)),
                sd_hz=sd, energy=float(t.sum()))


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = np.load('5_dataset.npz', allow_pickle=True)
    X, y = d['train_X'], d['train_y']
    rows = ['model,class,band,metric,raw,enh,ratio']
    for tag, path in CKPTS.items():
        model, _fe, meta = load_checkpoint(path, dev)
        model.eval()
        for c, idx in SAMPLES.items():
            assert int(y[idx]) == c, (idx, y[idx])
            x = torch.from_numpy(X[idx]).float().unsqueeze(0).to(dev)
            with torch.no_grad():
                r = model(x, return_all=True, training=False)
            tr = r['tfr_raw'][0].cpu().numpy()
            te = r['tfr_enhanced'][0].cpu().numpy()
            fr = r['freqs'].cpu().numpy()
            w = r['w_i'][0].mean(dim=-1).cpu().numpy()
            print(f'\n[{tag}] class{c}  w_i = '
                  + ' '.join(f'{v:.3f}' for v in w))
            for bname, (f0, f1) in BANDS.items():
                sr = band_stats(tr, fr, f0, f1)
                se = band_stats(te, fr, f0, f1)
                print(f'   {bname:<16s} raw peak {sr["peak_hz"]:6.2f} Hz '
                      f'share {sr["peak_share"]:.3f} sd {sr["sd_hz"]:5.2f} | '
                      f'enh peak {se["peak_hz"]:6.2f} Hz share '
                      f'{se["peak_share"]:.3f} sd {se["sd_hz"]:5.2f} | '
                      f'E比 {se["energy"]/max(sr["energy"],1e-12):.3f}')
                for k in ('peak_hz', 'peak_share', 'sd_hz', 'energy'):
                    rows.append(f'{tag},{c},{bname},{k},'
                                f'{sr[k]:.6f},{se[k]:.6f},'
                                f'{se[k]/max(abs(sr[k]),1e-12):.6f}')
            # 100/50 幅值比 (色阶是否反转)
            def amp(f0, f1):
                m = (fr >= f0) & (fr < f1)
                return float(te[m].max())
            ratio_raw = amp(99.0, 101.0) / max(amp(49.0, 51.0), 1e-12)
            print(f'   100/50 Hz 幅值比 (enh) = {ratio_raw:.3f}')
            rows.append(f'{tag},{c},ratio_100_50,enh,0,{ratio_raw:.6f},0')

    out = Path('results/band_census/gate_effect_3node.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print(f'\n已写入 {out}')


if __name__ == '__main__':
    main()