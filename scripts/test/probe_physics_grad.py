"""L_physics 的梯度流向检验 (合成常数边特征, 不读数据).

问题: physics_consistency_loss 的 ell 由 edge_feats (常数, 无 grad) 构造,
      唯一可微输入是 GAT 的 A_ij. 本脚本回答两个问题:
  1) ell 是否对任何参数可微? (答: 否, 只有 A 有 grad)
  2) hdL/dA_ij 的符号 —— 物理损失是把注意力推向还是推离 HARMONIC 边?

用法:
  python scripts/test/probe_physics_grad.py [--r-obs 2.13]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch                                                       # noqa: E402
from models.sast_graph import PHYSICS_EDGES, EdgeType               # noqa: E402
from models.sast_losses import physics_consistency_loss            # noqa: E402

_ap = argparse.ArgumentParser()
# 实测 (B4 拆节点**前**, 混装 BPF 带): node_if = [LOW_FREQ 13.0, BPF 47.0, 2xBPF 100.1]
#   → r_obs = 2.13。拆节点后 HARMONIC 边改接 BPF_LINE (48-53 Hz), 该值须在重训后
#   用 probe_node_if.py 重新测量; 这里默认值只用于说明梯度符号与拓扑无关的性质。
_ap.add_argument('--r-obs', type=float, default=100.1 / 47.0)
_args = _ap.parse_args()

M = len(PHYSICS_EDGES)
B, T = 2, 20
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

R_OBS = _args.r_obs
print(f'r_obs = {R_OBS:.4f}   (真值 2.0, 偏差 {100*(R_OBS-2)/2:+.1f}%)')

feats = torch.zeros(B, M, T, 5, device=dev)      # requires_grad=False — 关键
for m, e in enumerate(PHYSICS_EDGES):
    feats[:, m, :, 3] = e.w_type                  # dim3 = w_type
    if e.edge_type == EdgeType.INTEGER_HARMONIC:
        feats[:, m, :, 0] = R_OBS                 # dim0 = r_obs
    elif e.edge_type == EdgeType.CONDITION:
        feats[:, m, :, 0] = 0.5                   # dim0 = cond_sim
    elif e.edge_type == EdgeType.DRIFT:
        feats[:, m, :, 0] = 0.2                   # dim0 = Corr_E
    elif e.edge_type == EdgeType.ENERGY_COMPETITION:
        feats[:, m, :, 0] = 0.2                   # dim0 = -Corr_E

print(f'edge_feats.requires_grad = {feats.requires_grad}')

A = torch.full((B, M, 4, T), 1.0 / M, device=dev, requires_grad=True)
loss = physics_consistency_loss(A, feats, None, None)
g = torch.autograd.grad(loss, A)[0].mean(dim=(0, 2, 3))   # [M]

# 复算 ell 与 w*ell, 便于对照
ell = torch.zeros(M)
for m, e in enumerate(PHYSICS_EDGES):
    f0 = feats[0, m, 0, 0].item()
    if e.edge_type == EdgeType.INTEGER_HARMONIC:
        ell[m] = abs(f0 - 2.0) / 2.0
    else:
        ell[m] = 1.0 - max(-1.0, min(1.0, f0))
w = torch.tensor([e.w_type for e in PHYSICS_EDGES])

print(f'\nL_physics = {loss.item():.4f}')
print(f'{"edge":<34s} {"type":<18s} {"w":>5s} {"ell":>6s} {"w*ell":>7s} {"dL/dA":>10s}')
for m, e in enumerate(PHYSICS_EDGES):
    print(f'{e.description[:32]:<34s} {e.edge_type.name:<18s} '
          f'{w[m]:5.2f} {ell[m]:6.3f} {w[m]*ell[m]:7.3f} {g[m].item():+10.5f}')

harm = [m for m, e in enumerate(PHYSICS_EDGES)
        if e.edge_type == EdgeType.INTEGER_HARMONIC][0]
print(f'\n→ HARMONIC 边 dL/dA = {g[harm].item():+.5f}  '
      f'({"梯度下降会【增大】其注意力" if g[harm] < 0 else "梯度下降会【减小】其注意力"})')