"""
SAST 损失函数 (v3 — 自适应阶数 SST + 异构物理图)
===================================================

L_total = L_task + lambda_e * RE_2D + lambda_p * L_physics + lambda_s * L_smooth + lambda_b * L_balance

五项各司其职:
  L_task:    下游分类任务 (唯一有 GT 监督的项) — "TFR 必须对诊断有用"
  RE_2D:     Rényi 2D 熵 (自监督, Colominas & Meignen 2025 Eq.19) — "TFR 越集中越好"
             替代 v2 的逐节点加权 Rényi. 更简洁: GAT 通过最小化全局 RE_2D 学到
             "该挤的 bin 挤, 不该挤的不挤", 无需 per-region splitting.
  L_physics: 比值偏差 * 边类型权重 * 两端信任度 — "物理关系被满足时才可信任"
  L_smooth:  时序平滑 — 防止 w_i 在毫秒间剧烈跳变
  L_balance: 防退化 — 防止所有 w_i -> 0 或所有 w_i -> 1

变更 (v2 -> v3):
  - C_i -> w_i: 语义从"可压缩性"变为"IF信任度"
  - L_entropy: 加权 Rényi (per-region, C_i controlled) -> RE_2D (全局, 自监督)
  - 边数: M=1 -> M=6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, List

from models.sast_nodes import FreqRegion, PUMP_TURBINE_REGIONS
from models.sast_graph import PHYSICS_EDGES


# ═══════════════════════════════════════════════════════════════
# 0. Simple TFR Classifier (for L_task)
# ═══════════════════════════════════════════════════════════════

class TFRClassifier(nn.Module):
    """
    轻量 TFR 分类器: GlobalAvgPool -> MLP -> class logits.
    """

    def __init__(self, n_freq_bins: int, n_classes: int = 5,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.n_freq_bins = n_freq_bins
        self.n_classes = n_classes

        self.classifier = nn.Sequential(
            nn.Linear(n_freq_bins, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, tfr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tfr: [B, F, T] 时频表示 (非负幅值)

        Returns:
            logits: [B, n_classes]
        """
        feat = tfr.mean(dim=-1)
        feat = torch.log1p(feat)
        return self.classifier(feat)


# ═══════════════════════════════════════════════════════════════
# 1. L_task — 下游任务损失
# ═══════════════════════════════════════════════════════════════

def task_loss(tfr: torch.Tensor, y_true: torch.Tensor,
              classifier: nn.Module) -> torch.Tensor:
    """下游分类交叉熵损失."""
    logits = classifier(tfr)
    return F.cross_entropy(logits, y_true)


# ═══════════════════════════════════════════════════════════════
# 2. RE_2D — Rényi 2D 熵 (Colominas & Meignen 2025, Eq.19)
# ═══════════════════════════════════════════════════════════════

def renyi_2d_loss(tfr: torch.Tensor, alpha: int = 2,
                   eps: float = 1e-8) -> torch.Tensor:
    """
    Rényi 2D 熵 — 全局 TFR 集中度度量.

    RE_2D(M) = 1/(1-alpha) * log sum_m sum_n (|M[m,n]|/sum|M|)^alpha

    替代 v2 的 weighted_renyi_entropy_loss.
    优势:
      - 无需 per-region splitting
      - 无需 C_i/w_i 加权 (GAT 通过最小化 RE_2D 自然学到正确的 w_i)
      - 与 L_physics + L_task 交叉约束: 三者来自不同监督域

    Args:
        tfr:   [B, F, T] TFR 幅值 (非负)
        alpha: Rényi 阶数 (default 2, 论文推荐)
        eps:   数值保护

    Returns:
        scalar loss (lower = more concentrated)
    """
    B, F, T = tfr.shape
    tfr_pos = tfr.clamp(min=eps)
    total = tfr_pos.sum(dim=(1, 2), keepdim=True).clamp(min=eps)
    p = tfr_pos / total
    p_alpha = p ** alpha
    h = torch.log(p_alpha.sum(dim=(1, 2)).clamp(min=eps)) / (1.0 - alpha)
    return h.mean()


# ═══════════════════════════════════════════════════════════════
# 3. L_physics — 比值偏差物理一致性
# ═══════════════════════════════════════════════════════════════

def physics_consistency_loss(w_i: torch.Tensor,
                              edge_feats: torch.Tensor,
                              gate_edge: torch.Tensor,
                              edge_src: torch.Tensor,
                              edge_dst: torch.Tensor,
                              r_nom: Optional[torch.Tensor] = None,
                              eps: float = 1e-8) -> torch.Tensor:
    """
    比值偏差 * 边类型权重 * 两端信任度.

    L_physics = mean_edges [ w(type) * |r_obs - r_nom|/r_nom * w_src * w_dst ]

    自适应效应:
      - 整数倍频 (r_obs ~ r_nom): 偏差小 -> L 小 -> 允许高 w
      - 滑差 (r_obs 漂移): 偏差大 -> L 大 -> 迫使 w 降低
      - w 已降低: w_src*w_dst -> 0 -> L 自动变小 -> 梯度自消失

    注意: 仅 HARMONIC 边参与比值偏差计算.
          CONDITION 和 DRIFT 边由门控 (gate_edge) 处理.

    Args:
        w_i:        [B, N_phys, T] IF 可信度
        edge_feats: [B, M, T, 5] 边特征
        gate_edge:  [B, M, T] 边门控
        edge_src:   [M]
        edge_dst:   [M]
        r_nom:      [M] 标称比值
        eps:        数值保护

    Returns:
        scalar loss
    """
    B, N_phys, T = w_i.shape
    M = edge_feats.shape[1]
    device = w_i.device

    # ── 提取边特征 ──
    r_obs = edge_feats[:, :, :, 0]     # [B, M, T]
    w_type = edge_feats[:, :, :, 3]    # [B, M, T]

    if r_nom is None:
        r_nom = torch.tensor([e.r_nom for e in PHYSICS_EDGES],
                            device=device, dtype=torch.float32)

    r_nom_exp = r_nom.view(1, M, 1)    # [1, M, 1]

    # ── 比值偏差 ──
    ratio_dev = (r_obs - r_nom_exp).abs() / r_nom_exp.clamp(min=eps)

    # ── 两端 w_i (edge_src/dst 是图的 0-indexed, 物理节点从 1 开始) ──
    # 映射: graph idx -> phys idx (减去 1, OP=0 不参与 w_i)
    src_phys = edge_src - 1  # [M], -1 for OP edges
    dst_phys = edge_dst - 1  # [M]

    # 仅对物理节点索引有效的边 (>=0) 取 w_i
    src_valid = src_phys.clamp(0, N_phys - 1)
    dst_valid = dst_phys.clamp(0, N_phys - 1)
    w_src = w_i[:, src_valid, :]  # [B, M, T]
    w_dst = w_i[:, dst_valid, :]  # [B, M, T]
    w_edge = w_src * w_dst         # [B, M, T]

    # ── 逐边损失 ──
    per_edge = w_type * ratio_dev * w_edge * gate_edge

    total_weight = gate_edge.sum() + eps
    loss = per_edge.sum() / total_weight

    return loss


# ═══════════════════════════════════════════════════════════════
# 4. L_smooth — 时序平滑
# ═══════════════════════════════════════════════════════════════

def temporal_smoothness_loss(w_i: torch.Tensor,
                              A_ij: Optional[torch.Tensor] = None,
                              lambda_A: float = 0.1) -> torch.Tensor:
    """
    w_i(t) 和 A_ij(t) 的时序一致性.
    """
    device = w_i.device
    B, N, T = w_i.shape

    if T >= 2:
        w_diff = (w_i[:, :, 1:] - w_i[:, :, :-1])
        loss_w = (w_diff ** 2).mean()
    else:
        loss_w = torch.tensor(0.0, device=device)

    loss_attn = torch.tensor(0.0, device=device)
    if A_ij is not None and lambda_A > 0:
        _, _, _, T_a = A_ij.shape
        if T_a >= 2:
            attn_diff = (A_ij[:, :, :, 1:] - A_ij[:, :, :, :-1])
            loss_attn = (attn_diff ** 2).mean()

    return loss_w + lambda_A * loss_attn


# ═══════════════════════════════════════════════════════════════
# 5. L_balance — 防退化正则化
# ═══════════════════════════════════════════════════════════════

def balance_loss(w_i: torch.Tensor,
                  w_min: float = 0.3, w_max: float = 0.8) -> torch.Tensor:
    """
    防止所有 w_i -> 0 或所有 w_i -> 1.

    L_balance = max(0, w_min - w_mean) + max(0, w_mean - w_max)
    """
    w_mean = w_i.mean()
    return torch.clamp(w_min - w_mean, min=0.0) + torch.clamp(w_mean - w_max, min=0.0)


# ═══════════════════════════════════════════════════════════════
# 6. Total Loss
# ═══════════════════════════════════════════════════════════════

def total_sast_loss(tfr_raw: torch.Tensor,
                     tfr_enhanced: torch.Tensor,
                     w_i: torch.Tensor,
                     y_true: torch.Tensor,
                     classifier: nn.Module,
                     edge_feats: torch.Tensor,
                     gate_edge: torch.Tensor,
                     edge_src: torch.Tensor,
                     edge_dst: torch.Tensor,
                     node_if: torch.Tensor,
                     freqs: torch.Tensor,
                     A_ij: Optional[torch.Tensor] = None,
                     lambda_task: float = 1.0,
                     lambda_entropy: float = 0.1,
                     lambda_physics: float = 0.5,
                     lambda_smooth: float = 0.05,
                     lambda_balance: float = 0.01,
                     lambda_A: float = 0.1,
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    SAST v3 总损失.

    L_total = lambda_task*L_task + lambda_e*RE_2D + lambda_p*L_physics
            + lambda_s*L_smooth + lambda_b*L_balance

    Args:
        tfr_raw:      [B, F, T] 原始 STFT 幅值
        tfr_enhanced: [B, F, T] SAST 增强 TFR
        w_i:          [B, N_phys, T] IF 可信度
        y_true:       [B] 类别标签
        classifier:   TFRClassifier 模块
        edge_feats:   [B, M, T, 5] 边特征
        gate_edge:    [B, M, T] 边门控
        edge_src:     [M]
        edge_dst:     [M]
        node_if:      [B, N_phys, T] 节点 IF
        freqs:        [F] 频率轴
        A_ij:         [B, M, H, T] (可选)
        lambda_*:     各项权重系数

    Returns:
        total_loss:  scalar
        losses_dict: dict of individual loss values + diagnostics
    """
    device = w_i.device

    # ── 各分量 ──
    l_task = task_loss(tfr_enhanced, y_true, classifier)
    l_entropy = renyi_2d_loss(tfr_enhanced, alpha=2)
    l_physics = physics_consistency_loss(
        w_i, edge_feats, gate_edge, edge_src, edge_dst
    )
    l_smooth = temporal_smoothness_loss(w_i, A_ij, lambda_A=lambda_A)
    l_balance = balance_loss(w_i)

    # ── 加权求和 ──
    total = (lambda_task * l_task +
             lambda_entropy * l_entropy +
             lambda_physics * l_physics +
             lambda_smooth * l_smooth +
             lambda_balance * l_balance)

    # ── 诊断 ──
    losses_dict = {
        'total': total.item(),
        'task': l_task.item(),
        'entropy_2d': l_entropy.item(),
        'physics': l_physics.item(),
        'smooth': l_smooth.item(),
        'balance': l_balance.item(),
        'w_mean': w_i.mean().item(),
        'w_min': w_i.min().item(),
        'w_max': w_i.max().item(),
        'w_spread': (w_i.mean(dim=-1).max(dim=-1).values.mean() -
                      w_i.mean(dim=-1).min(dim=-1).values.mean()).item(),
    }

    return total, losses_dict


# ═══════════════════════════════════════════════════════════════
# 7. Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("SAST Losses v3 — Smoke Test (10-edge heterogeneous graph)")
    print("=" * 60)

    nB, nF, nT, nN, nM = 2, 256, 100, 3, 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tfr_raw = torch.rand(nB, nF, nT, device=device)
    tfr_enhanced = torch.rand(nB, nF, nT, device=device)
    w_i = torch.sigmoid(torch.randn(nB, nN, nT, device=device))
    y_true = torch.randint(0, 5, (nB,), device=device)
    node_if = torch.rand(nB, nN, nT, device=device) * 500
    freqs = torch.linspace(0, 500, nF, device=device)
    edge_feats = torch.rand(nB, nM, nT, 5, device=device)
    gate_edge = torch.rand(nB, nM, nT, device=device)
    edge_src = torch.tensor([0, 0, 0, 1, 2, 2, 1, 3, 2, 3], device=device)
    edge_dst = torch.tensor([1, 2, 3, 2, 1, 3, 3, 1, 3, 2], device=device)
    A_ij = torch.rand(nB, nM, 4, nT, device=device)

    classifier = TFRClassifier(n_freq_bins=nF).to(device)

    total, d = total_sast_loss(
        tfr_raw, tfr_enhanced, w_i, y_true, classifier,
        edge_feats, gate_edge, edge_src, edge_dst,
        node_if, freqs, A_ij,
    )

    print(f"Total loss: {d['total']:.4f}")
    for k, v in d.items():
        print(f"  {k:<20s} {v:.4f}")
    print("Done!")
