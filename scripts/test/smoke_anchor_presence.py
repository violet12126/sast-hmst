"""L_anchor / presence / presence-gated L_range 冒烟测试 (合成谱, 不读数据)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch                                                   # noqa: E402
from models.sast_losses import (                               # noqa: E402
    FreqEncoder, node_presence, total_sast_loss)
from models.sast_nodes import PUMP_TURBINE_REGIONS as R        # noqa: E402

print('regions:')
for r in R:
    print(f'  {r.name:<9s} [{r.f_min:.0f},{r.f_max:.0f}] '
          f'w∈[{r.w_range_low},{r.w_range_high}] '
          f'anchor={r.f_anchor}±{r.anchor_tol}')

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
B, F, T, N, M = 4, 1000, 60, 3, 10
freqs = torch.linspace(0, 500, F, device=dev)

mag = torch.full((B, F, T), 1e-3, device=dev)
mag[:, (freqs >= 99) & (freqs <= 101), :] += 1.0     # 100 Hz 线
mag[:, (freqs >= 46) & (freqs <= 48), :] += 0.4      # 47 Hz 鼓包

w_i = torch.full((B, N, T), 0.5, device=dev, requires_grad=True)
node_if = torch.stack([torch.full((B, T), 10.0, device=dev),
                       torch.full((B, T), 47.0, device=dev),
                       torch.full((B, T), 100.1, device=dev)])
enc = FreqEncoder(n_freq_bins=F).to(dev)
ef = torch.rand(B, M, T, 5, device=dev)
ge = torch.rand(B, M, T, device=dev)
es = torch.tensor([0, 0, 0, 1, 2, 2, 1, 3, 2, 3], device=dev)
ed = torch.tensor([1, 2, 3, 2, 1, 3, 3, 1, 3, 2], device=dev)
A = torch.rand(B, M, 4, T, device=dev, requires_grad=True)
y = torch.tensor([0, 1, 2, 3], device=dev)

loss, d = total_sast_loss(mag, mag, w_i, y, enc, ef, ge, es, ed,
                          node_if, freqs, A)
loss.backward()
print(f"\ntotal={d['total']:.4f}")
for k, v in d.items():
    if k != 'total':
        print(f'  {k:<18s} {v:.4f}')
print(f'grad w_i norm={w_i.grad.norm().item():.3e}  '
      f'grad A norm={A.grad.norm().item():.3e}')

flat = torch.full((B, F, T), 1e-3, device=dev)
flat[:, (freqs >= 2) & (freqs <= 3), :] += 1.0       # 只有 2 Hz 有能量
p_flat = node_presence(flat, freqs, R)
p_line = node_presence(mag, freqs, R)
print('\npresence 仅2Hz有能量 :',
      [round(p_flat[:, n, :].mean().item(), 3) for n in range(N)])
print('presence 含100/47Hz线:',
      [round(p_line[:, n, :].mean().item(), 3) for n in range(N)])

# 平坦谱: 2xBPF 缺席 -> w_low 应松弛, L_range 应比不给 presence 小
w_low = torch.tensor([r.w_range_low for r in R], device=dev)
w_high = torch.tensor([r.w_range_high for r in R], device=dev)
from models.sast_losses import w_range_loss, balance_loss            # noqa: E402
w_low_set = torch.full((B, N, T), 0.2, device=dev)
print('\n[平坦谱/分量缺席] w=0.2 时:')
print(f'  L_range 无门控      = {w_range_loss(w_low_set, w_low, w_high).item():.4f}')
print(f'  L_range 有门控      = {w_range_loss(w_low_set, w_low, w_high, presence=p_flat).item():.4f}')
print(f'  L_balance 无门控    = {balance_loss(w_low_set).item():.4f}')
print(f'  L_balance 有门控    = {balance_loss(w_low_set, presence=p_flat).item():.4f}')

# ── 锚越界时必须产生非零梯度 (否则锚形同虚设) ──
from models.sast_losses import freq_anchor_loss                          # noqa: E402
print('\n[锚越界检验] 把 BPF 带内能量搬到 43Hz (偏差 7Hz > 容差 4Hz):')
for f0 in (47.0, 43.0, 40.0):
    t = torch.full((2, F, T), 1e-3, device=dev, requires_grad=True)
    with torch.no_grad():
        t[:, (freqs >= f0 - 1) & (freqs <= f0 + 1), :] += 1.0
        t[:, (freqs >= 99) & (freqs <= 101), :] += 1.0
    la = freq_anchor_loss(t, freqs, R, presence=None)
    g = torch.autograd.grad(la, t, retain_graph=False)[0]
    print(f'  BPF 能量峰={f0:.0f}Hz  L_anchor={la.item():.4f}Hz  '
          f'|dL/dt|={g.abs().sum().item():.3e}')

print('\nSmoke test OK')