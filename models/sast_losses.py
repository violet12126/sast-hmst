"""
SAST 损失函数: 时频聚集性 + 软物理一致性 + 时间平滑

L = L_entropy + λ2·L_physics + λ3·L_smooth

设计原则:
  - L_entropy: 推动 TFR 能量集中 → GAT 不能全输出最宽的核
  - L_physics: 约束 A_ij 与观测边特征的一致性，以原型匹配门控加权
    → 匹配原型的边获得强物理监督，匿名边自由探索
  - L_smooth: 防止 C_i 和 A_ij 在时间轴上剧烈跳变

注意: SAST 不使用下游任务损失——SAST 是物理保真预处理步骤，不是分类器。
"""
import torch
import torch.nn.functional as F


# ============================================================
# 1. Rényi 熵损失: TFR 能量集中度
# ============================================================

def renyi_entropy_loss(tfr, alpha=3, eps=1e-8):
    """
    Rényi α-熵: 衡量 TFR 能量集中程度。值越低 → TFR 越锐利。

    H_α = 1/(1-α) · log( ∫∫ (TFR / ∫∫TFR)^α dω dt )

    α=3 对弱分量更敏感，比 Shannon 熵 (α→1) 更能推动能量集中。
    与 L_physics 形成自然平衡:
      - σ→0 (过窄): TFR 锐利但可能丢失滑差信息
      - σ→∞ (过宽): TFR 模糊 → 损害 L_entropy

    Args:
        tfr:  [B, F, T] 时频表示 (非负)
        alpha: Rényi 阶数 (默认 3)
        eps:   数值保护

    Returns:
        scalar: 平均 Rényi 熵
    """
    B, F, T = tfr.shape
    tfr_pos = tfr.abs().clamp(min=eps)
    total = tfr_pos.sum(dim=(1, 2), keepdim=True).clamp(min=eps)
    p = tfr_pos / total  # [B, F, T] 概率分布

    if alpha == 1:
        h = -(p * torch.log(p + eps)).sum(dim=(1, 2))
    else:
        p_alpha = p ** alpha
        h = torch.log(p_alpha.sum(dim=(1, 2)).clamp(min=eps)) / (1 - alpha)

    return h.mean()


# ============================================================
# 2. 软物理一致性损失: 原型匹配门控的边自洽约束
# ============================================================

def physics_consistency_loss(A_ij, edge_feats, gate, edge_src, edge_dst,
                             eps=1e-8):
    """
    软物理一致性损失: 约束 GAT 注意力权重与观测边特征的一致性。

    每条匿名边 (i,j) 的"物理一致性得分":
      consistency_ij = energy_corr_ij · exp(-r_std_ij) · confidence_ij

    这衡量: 能量相关性高 + 比值稳定 + 端点可信 → 边自洽。

    损失推动 A_ij 与该得分成比例，以门控 gate 加权:
      - gate_i, gate_j 都高 (两边都匹配原型): 强物理监督
      - gate_i 或 gate_j 低 (匿名边): 监督自动退火

    注意: 约束的是 A_ij (注意力权重)，不是 C_i。
          A_ij 代表"这条边的消息传递多重要"，
          它应该与边特征的物理一致性对齐。

    Args:
        A_ij:       [B, M, H, T] 注意力权重 (多头)
        edge_feats: [B, M, T, 4] 边特征 [r_obs, r_std, energy_corr, confidence]
        gate:       [B, K, T] 原型匹配门控 (0-1)
        edge_src:   [M] 边起点索引
        edge_dst:   [M] 边终点索引
        eps:        数值保护

    Returns:
        scalar: 加权物理一致性损失
    """
    B, M, H, T = A_ij.shape
    device = A_ij.device

    # ── 多头平均注意力 ──
    A_mean = A_ij.mean(dim=2)  # [B, M, T]

    # ── 边特征分量 ──
    r_std = edge_feats[:, :, :, 1]       # [B, M, T]
    energy_corr = edge_feats[:, :, :, 2]   # [B, M, T]
    confidence = edge_feats[:, :, :, 3]    # [B, M, T]

    # ── 观测一致性得分 ──
    # 比值越稳定 (r_std↓) + 能量越相关 (energy_corr↑) + 端点越可信 (confidence↑)
    # → 一致性越高 (consistency → 1)
    consistency = energy_corr.clamp(0, 1) \
        * torch.exp(-r_std) \
        * confidence.clamp(0, 1)
    consistency = consistency.clamp(0.0, 1.0)  # [B, M, T]

    # ── 门控加权 ──
    # 边的物理监督强度 = 两端节点原型匹配度的乘积
    gate_src = gate[:, edge_src, :]  # [B, M, T]
    gate_dst = gate[:, edge_dst, :]  # [B, M, T]
    w_gate = gate_src * gate_dst     # [B, M, T]
    # → gate 都高: w_gate ≈ 1 → 强物理监督
    # → gate 低:   w_gate ≈ 0 → 纯数据驱动

    # ── MSE: A_mean 应接近 consistency ──
    per_edge_error = (A_mean - consistency) ** 2  # [B, M, T]
    weighted_error = w_gate * per_edge_error

    # 归一化: 均值 over 有效边
    total_weight = w_gate.sum() + eps
    loss = weighted_error.sum() / total_weight

    return loss


# ============================================================
# 3. 时间平滑损失: C_i 和 A_ij 的时序一致性
# ============================================================

def temporal_smoothness_loss(C_i, A_ij=None, lambda_A=0.1):
    """
    时间平滑正则: 惩罚 C_i(t) 和 A_ij(t) 在时间轴上的剧烈跳变。

    物理可信度不应在毫秒级发生突变——C_i 和 A_ij 都应时序平滑。

    Args:
        C_i:     [B, K, T] Compressibility Token
        A_ij:    [B, M, H, T] 注意力权重 (可选)
        lambda_A: A_ij 平滑的相对权重 (默认 0.1)

    Returns:
        scalar: 时间平滑损失
    """
    device = C_i.device
    B, K, T = C_i.shape

    # ── C_i 平滑 ──
    if T >= 2:
        c_diff = (C_i[:, :, 1:] - C_i[:, :, :-1])  # [B, K, T-1]
        loss_c = (c_diff ** 2).mean()
    else:
        loss_c = torch.tensor(0.0, device=device)

    # ── A_ij 轻量平滑 ──
    loss_attn = torch.tensor(0.0, device=device)
    if A_ij is not None and lambda_A > 0:
        _, _, _, T_a = A_ij.shape
        if T_a >= 2:
            attn_diff = (A_ij[:, :, :, 1:] - A_ij[:, :, :, :-1])
            loss_attn = (attn_diff ** 2).mean()

    return loss_c + lambda_A * loss_attn


# ============================================================
# 4. 总损失
# ============================================================

def total_sast_loss(tfr_enhanced, A_ij, C_i, gate,
                    edge_feats, edge_src, edge_dst,
                    lambda_entropy=0.1, lambda_physics=0.5, lambda_smooth=0.01,
                    lambda_A=0.1):
    """
    SAST 总损失 (无任务损失, 纯物理自监督):

      L = L_entropy + λ2·L_physics + λ3·L_smooth

    Args:
        tfr_enhanced: [B, F, T] SAST 增强 TFR
        A_ij:         [B, M, H, T] 注意力权重
        C_i:          [B, K, T] Compressibility Token
        gate:         [B, K, T] 原型匹配门控
        edge_feats:   [B, M, T, 4] 观测边特征
        edge_src:     [M] 边起点
        edge_dst:     [M] 边终点
        lambda_*:     各项权重系数

    Returns:
        total_loss:  scalar
        losses_dict: dict of individual loss values
    """
    l_entropy = renyi_entropy_loss(tfr_enhanced)
    l_physics = physics_consistency_loss(
        A_ij, edge_feats, gate, edge_src, edge_dst
    )
    l_smooth = temporal_smoothness_loss(C_i, A_ij, lambda_A=lambda_A)

    total = l_entropy + lambda_physics * l_physics + lambda_smooth * l_smooth

    losses_dict = {
        'entropy': l_entropy.item(),
        'physics': l_physics.item(),
        'smooth': l_smooth.item(),
        'total': total.item(),
    }

    return total, losses_dict
