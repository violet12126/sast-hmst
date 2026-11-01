"""L_anchor / presence / presence-gated L_range 冒烟测试 (合成谱, 不读数据)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch                                                   # noqa: E402
from models.sast_losses import (                               # noqa: E402
    FreqEncoder, node_presence, total_sast_loss)
from models.sast_nodes import PUMP_TURBINE_REGIONS as R        # noqa: E402
from models.sast_graph import PHYSICS_EDGES, N_EDGES           # noqa: E402

print('regions:')
for r in R:
    print(f'  {r.name:<9s} [{r.f_min:.0f},{r.f_max:.0f}] '
          f'w∈[{r.w_range_low},{r.w_range_high}] '
          f'anchor={r.f_anchor}±{r.anchor_tol}')

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
B, F, T, N, M = 4, 1000, 60, len(R), N_EDGES
freqs = torch.linspace(0, 500, F, device=dev)

mag = torch.full((B, F, T), 1e-3, device=dev)
mag[:, (freqs >= 99) & (freqs <= 101), :] += 1.0     # 100 Hz 2xBPF 线
mag[:, (freqs >= 49.5) & (freqs <= 50.5), :] += 1.0  # 50 Hz 1xBPF 线
mag[:, (freqs >= 46) & (freqs <= 48), :] += 0.4      # 47 Hz 漂移鼓包

NODE_NAMES = [r.name for r in R]        # ['LOW_FREQ', 'BPF', '2xBPF']
I_LF, I_BPF, I_2X = (NODE_NAMES.index(n) for n in ('LOW_FREQ', 'BPF', '2xBPF'))
w_i = torch.full((B, N, T), 0.5, device=dev, requires_grad=True)
node_if = torch.stack([torch.full((B, T), 10.0, device=dev),      # LOW_FREQ
                       torch.full((B, T), 45.0, device=dev),      # BPF (41/50 之间)
                       torch.full((B, T), 100.1, device=dev)])    # 2xBPF
enc = FreqEncoder(n_freq_bins=F).to(dev)
ef = torch.rand(B, M, T, 5, device=dev)
ge = torch.rand(B, M, T, device=dev)
# 边索引直接由图定义导出 (3 节点拓扑共 10 条边), 不再硬编码
es = torch.tensor([e.src for e in PHYSICS_EDGES], device=dev)
ed = torch.tensor([e.dst for e in PHYSICS_EDGES], device=dev)
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
print('presence 含100/50/47Hz:',
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
# ⚠ 3 节点合并后**只有 2xBPF 带锚**: BPF 带同时装 41 Hz 非锁频峰与 50 Hz 锁定
#   线, 能量加权质心必然落在两者之间, 绝对锚会恒被违反 (见 sast_nodes)。
#   故本检验把 2xBPF 的峰移出锚带 [98.49, 101.49] 看梯度。
from models.sast_losses import freq_anchor_loss, harmonic_ratio_loss   # noqa: E402
from models.sast_nodes import band_centroid_if                         # noqa: E402
print('\n[锚越界检验] 把 2xBPF 带内能量峰移到 f0 '
      '(锚 99.990±1.5 → 容差带 [98.49, 101.49]); LOW_FREQ 另给无水力的参考:')
for f0 in (100.0, 101.0, 103.0, 106.0):
    t = torch.full((2, F, T), 1e-3, device=dev, requires_grad=True)
    with torch.no_grad():
        t[:, (freqs >= 2) & (freqs <= 3), :] += 1.0           # LOW_FREQ 有能量
        t[:, (freqs >= f0 - 0.5) & (freqs <= f0 + 0.5), :] += 1.0
    la = freq_anchor_loss(t, freqs, R, presence=None)
    g = torch.autograd.grad(la, t, retain_graph=False)[0]
    cents = band_centroid_if(t.detach(), freqs, R)[0, :, :].mean(dim=-1)
    print(f'  2xBPF 峰={f0:.1f}Hz  2xBPF质心={cents[I_2X]:.2f}Hz  '
          f'LOW_FREQ质心={cents[I_LF]:.2f}Hz  L_anchor={la.item():.4f}Hz  '
          f'|dL/dt|={g.abs().sum().item():.3e}')

# ── A2: L_ratio 必须对可微 if_hat 产生梯度 (对比 L_physics 的零梯度) ──
print('\n[A2 比值约束可微性] (f_BPF, f_2xBPF) 相对偏差超过 6% 时应有非零梯度:')
for f1, f2 in [(50.0, 100.0), (49.0, 104.0), (43.0, 104.0), (52.0, 95.0)]:
    t = torch.full((2, F, T), 1e-3, device=dev, requires_grad=True)
    with torch.no_grad():
        t[:, (freqs >= f1 - 0.4) & (freqs <= f1 + 0.4), :] += 1.0
        t[:, (freqs >= f2 - 0.4) & (freqs <= f2 + 0.4), :] += 1.0
    ih = band_centroid_if(t, freqs, R)
    fl, fh = ih[:, I_BPF, :].mean().item(), ih[:, I_2X, :].mean().item()
    lr = harmonic_ratio_loss(ih, R, presence=None, tol=0.06)
    g = torch.autograd.grad(lr, t, retain_graph=False)[0]
    print(f'  BPF={fl:6.2f}Hz  2xBPF={fh:6.2f}Hz  r_obs={fh / fl:.3f}  '
          f'L_ratio={lr.item():.4f}  |dL/dt|={g.abs().sum().item():.3e}')

print('\nSmoke test OK')