"""SAST v4 物理损失 (完全解耦训练口径)
=====================================

v4 与 CNN 完全解耦 —— 这里**没有** SupCon / 分类项, 只有时频分布的物理约束:

    L = ½·[H2(BPF)  + λ_tear,BPF · L_tear(BPF)]
      + ½·[H2(2xBPF) + λ_tear,2x  · L_tear(2xBPF)]
      + λ_z · mean(z_w²)

  两项物理博弈: **聚能推力** H2(带内二维 Rényi 熵, 越低越集中) 对
  **时间撕裂阻力** L_tear(相对 S_base 的 50 ms 尺度 TV 增量平方 hinge)。
  两带必须**成对平均**再求和 —— renyi_2d_loss 旧口径返回的是两带均值,
  若外层再直接乘 λ, 实际权重会翻倍 (与 1D β 扫描不同口径)。

  两带 λ 相差约 10-20 倍是物理差异而非调参: BPF 是 46-47.5 与 50 Hz 双分量
  共存的拍频带 (时间调制真实存在, 宜弱正则), 2xBPF 是单音带 (时间碎裂等价于
  分量缺席, 宜强正则)。实测 1D 集中度增益 BPF≈0.33 与空转 2xBPF≈0.32 相同,
  无法用单一 λ 区分 (见 docs/SAST_v4_框架说明.md §8.9)。

  已移除: L_soft_ent (β≡0 于 LOW_FREQ, 恒为 0)、L_anchor / L_ratio
  (实测 mean|Δω| ≤ 0.10 Hz、熵降 <0.7%, 梯度可忽略)。

各项实现全部复用 v3 的 ``models.sast_losses``, 但**作用对象换成 v4 的量**:

  * ``tfr_enhanced`` = v4 的 T (双窗底图 + 分频带 β 重排)
  * ``tfr_raw``      = v4 的 S_base (未挤压的双窗底图, 作为熵降基准)
  * ``w_i``          = GAT 输出的 ``w_node`` [B, 3, T]

为什么用 S_base 而不是 S_long 做熵降基准: v4 的底图本身就是 α 混合的短/长窗谱,
「挤压前 vs 挤压后」的对比必须把 α 换窗这一步算进基准, 否则低频段 (α=1, 短窗)
的熵降会把换窗效应误记到挤压头上。LOW_FREQ 的 β 还有 q 调制, 用 S_base 才同口径。

**2026-11-13: 删除 L_absence (原 w_range_loss)**。GAT 的 w 头已改为真区间映射
(``MinimalPhysicsGAT._bounds``), ``w_node`` 逐帧从构造上落在 ``[w_low_eff,
w_high_eff]`` 内 (5 类真实样本实测越界 < 1e-4), 该罚项恒为 0, 属死项。
presence / absence 仍然保留 —— 它们现在只作**门控输入** (决定区间边界与
RE_2D 的缺席屏蔽), 不再以损失项形式出现。

presence / absence 由 **S_base** 计算 (必须传未挤压谱, 否则模型可制造尖峰自证存在)。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from models.sast_losses import (
    _h2_band,
    absence_gate,
    freq_anchor_loss,
    harmonic_ratio_loss,
    node_presence,
    region_weights,
    renyi_2d_loss,
    soft_entropy_hinge,
)
from models.sast_nodes import (
    PUMP_TURBINE_REGIONS,
    FreqRegion,
    band_centroid_if,
)


def node_presence_absence(tfr_raw: torch.Tensor, freqs: torch.Tensor,
                          regions: Optional[List[FreqRegion]] = None,
                          frac: float = 0.1
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """未增强谱 -> (presence [B,N,T], absence [B,N,T])。"""
    if regions is None:
        regions = PUMP_TURBINE_REGIONS
    p = node_presence(tfr_raw, freqs, regions, frac=frac)
    return p, absence_gate(p, regions)


def band_tear(tfr: torch.Tensor, tfr_base: torch.Tensor,
              freqs: torch.Tensor, region: 'FreqRegion',
              stride: int = 50, eps: float = 1e-8) -> torch.Tensor:
    """带内时间撕裂增量 (平方 hinge, 逐样本后对 batch 求均值)。

    TV_t = 带内逐帧谱按频率归一化后, 相隔 stride 帧的 L1 距离均值。
    减去未挤压底图的同一量 (detach), 只惩罚**重排额外制造**的碎裂:

        L = mean_b relu( TV_t(T) − TV_t(S_base) )²

    平方 hinge 保证对 β 有内部极小值; 线性 hinge 会退化为 bang-bang
    (实测 λ=20 就把所有类别压到 β=0)。
    """
    w_f = region_weights(freqs, [region])[0]
    m = w_f > 0.0
    t = (tfr[:, m, :] * w_f[m].view(1, -1, 1)).clamp_min(0.0)
    s = (tfr_base[:, m, :] * w_f[m].view(1, -1, 1)).clamp_min(0.0)
    p_t = t / t.sum(dim=1, keepdim=True).clamp_min(eps)
    p_s = s / s.sum(dim=1, keepdim=True).clamp_min(eps)
    tv_t = (p_t[:, :, stride:] - p_t[:, :, :-stride]).abs().sum(dim=1).mean(-1)
    tv_s = (p_s[:, :, stride:] - p_s[:, :, :-stride]).abs().sum(dim=1).mean(-1)
    return torch.relu(tv_t - tv_s.detach()).pow(2).mean()


def sast_v4_loss(tfr: torch.Tensor, tfr_base: torch.Tensor,
                 w_node: torch.Tensor, freqs: torch.Tensor,
                 regions: Optional[List[FreqRegion]] = None,
                 lambda_tear_bpf: float = 5.0,
                 lambda_tear_2x: float = 60.0,
                 tear_stride: int = 50,
                 z_w: Optional[torch.Tensor] = None,
                 lambda_z: float = 0.002,
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """v4 物理总损失 (两带成对平均)。

        L = ½·[H2(BPF)  + λ_t,BPF·tear(BPF)]
          + ½·[H2(2xBPF)+ λ_t,2x ·tear(2xBPF)]
          + λ_z·mean(z_w²)

    Args:
        tfr:       [B,F,T] 增强谱 T (可微)
        tfr_base:  [B,F,T] 未挤压底图 S_base (撕裂基准; 也用于诊断 presence)
        w_node:    [B,N,T] 挤压强度 (仅诊断, 不进损失)
        freqs:     [F] Hz
        lambda_tear_bpf / lambda_tear_2x: 两带撕裂权重。**必须分开设** ——
                   BPF 是双分量拍频带, 实测 λ=5 时 β*≈0.90-0.95; 2xBPF 是
                   单音带, λ=60 时 β*≈0.35/0.80/0.90/1.00/0.85 (空转..抽水)
        tear_stride: 时间步长 (帧), 默认 50 = 50 ms。1 ms 步长下相邻帧窗
                   重叠 99.9%, 撕裂量测不出来 (实测五类全为 0.03-0.04 Hz)
        z_w:       [B,N,Tc] mlp_w 未激活 logit, 供 L2 护栏
        lambda_z:  z_w 的 L2 权重, 默认 0.002 (只防爆, 不把 w 拽回中点)

    旧口径的 anchor / ratio / soft_ent 已移除: soft_ent 在 β≡0 的 LOW_FREQ 上
    恒为 0; anchor/ratio 实测梯度可忽略 (mean|Δω|≤0.10 Hz, 熵降 <0.7%)。
    """
    if regions is None:
        regions = PUMP_TURBINE_REGIONS
    rmap = {r.name: r for r in regions}
    r_bpf, r_2x = rmap['BPF'], rmap['2xBPF']

    h_bpf = _h2_band(tfr, freqs, r_bpf)
    h_2x = _h2_band(tfr, freqs, r_2x)
    t_bpf = band_tear(tfr, tfr_base, freqs, r_bpf, stride=tear_stride)
    t_2x = band_tear(tfr, tfr_base, freqs, r_2x, stride=tear_stride)

    # 两带成对平均: 与 1D β 扫描的 H2_band + λ·tear_band 严格 1:1 对齐
    l_phys = (0.5 * (h_bpf + lambda_tear_bpf * t_bpf)
              + 0.5 * (h_2x + lambda_tear_2x * t_2x))
    l_z = (z_w ** 2).mean() if (z_w is not None and lambda_z > 0) else         torch.zeros((), device=tfr.device)
    total = l_phys + lambda_z * l_z

    with torch.no_grad():
        parts: Dict[str, float] = {
            'total': float(total.detach()),
            're_2d': float((0.5 * (h_bpf + h_2x)).detach()),
            'h2_bpf': float(h_bpf.detach()), 'h2_2x': float(h_2x.detach()),
            'tear_bpf': float(t_bpf.detach()),
            'tear_2x': float(t_2x.detach()),
            'z_l2': float(l_z.detach()),
        }
        names = [r.name for r in regions]
        for i, nm in enumerate(names):
            parts[f'w_{nm}'] = float(w_node[:, i].mean().detach())
            parts[f'z_{nm}'] = (float(z_w[:, i].abs().mean().detach())
                                if z_w is not None else 0.0)
        presence, absence = node_presence_absence(tfr_base, freqs, regions)
        for i, nm in enumerate(names):
            parts[f'presence_{nm}'] = float(presence[:, i].mean().detach())
            parts[f'absence_{nm}'] = float(absence[:, i].mean().detach())
    return total, parts


@torch.no_grad()
def tf_quality(tfr: torch.Tensor, freqs: torch.Tensor,
               regions: Optional[List[FreqRegion]] = None,
               frac: float = 0.1) -> Dict[str, float]:
    """时频质量指标 (验证集报告用, 替代 KNN 精度)。

    逐节点给出:
      * ``conc``  —— 带内集中度 = 能量最高的 10% bin 占带内总能量的比例,
                     经 0.2 s 时间平滑 (与 node_presence 同口径)。越高越锐。
      * ``h2``    —— 带内 Rényi-2D 熵 (nats)。越低越集中。
      * ``if``    —— 带内能量质心 (Hz), 用于看锚/比值的收敛。

    注意 LOW_FREQ 的 conc/h2 **不应**被优化 (它是软带), 报告出来是为了
    确认「低频没有被压瘪」—— 硬带 conc 升高的同时低频 conc 应基本不变。
    """
    if regions is None:
        regions = PUMP_TURBINE_REGIONS
    out: Dict[str, float] = {}
    p = node_presence(tfr, freqs, regions, frac=frac)
    if_hat = band_centroid_if(tfr, freqs, regions)
    for i, r in enumerate(regions):
        out[f'conc_{r.name}'] = float(p[:, i].mean())
        out[f'h2_{r.name}'] = float(_h2_band(tfr, freqs, r))
        out[f'if_{r.name}'] = float(if_hat[:, i].mean())
    return out


if __name__ == '__main__':
    torch.manual_seed(0)
    from models.sast_v4 import SASTv4

    t = torch.arange(2000, dtype=torch.float64) / 1000.0
    x = (torch.cos(2 * torch.pi * 50.0 * t)
         + 0.6 * torch.cos(2 * torch.pi * 100.0 * t)
         + 0.3 * torch.cos(2 * torch.pi * 1.2 * t)
         + 0.2 * torch.randn(2000, dtype=torch.float64)).unsqueeze(0).float()

    m = SASTv4()
    T, d = m(x, return_detail=True)
    loss, parts = sast_v4_loss(T, d['S_base'], d['w_node'], d['freqs'])
    print(f'loss = {float(loss):.4f}')
    for k, v in parts.items():
        print(f'  {k:14s} {v:+.4f}')
    print('tf quality:', {k: round(v, 3)
                          for k, v in tf_quality(T, d['freqs']).items()})
    # 梯度必须能回到 GAT 参数
    loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in m.gat.parameters()
            if p.grad is not None)
    print(f'GAT 梯度范数和 = {g:.3e} (应 > 0)')
