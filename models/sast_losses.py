"""
SAST 损失函数 (v4 - SupCon 自监督)
===================================================

L_total = λ_sc * L_supcon + λ_e * RE_2D + λ_p * L_physics
        + λ_s * L_smooth + λ_b * L_balance

五项各司其职:
  L_supcon:  监督对比 (Khosla 2020) - 同工况 TFR 表示拉近, 不同工况推远.
             用工况标签定义正负对但不分类. 经 z_freq->TFR->σ->w_i 监督 GAT.
  RE_2D:     Rényi 2D 熵 (自监督) - "TFR 越集中越好"
  L_physics: 比值偏差 * 边类型权重 * 两端信任度 - "物理关系被满足时才可信任"
  L_smooth:  时序平滑 - 防止 w_i 在毫秒间剧烈跳变
  L_balance: 防退化 - 防止所有 w_i -> 0 或所有 w_i -> 1

变更 (v3 -> v4):
  - L_task (分类 CE) -> L_supcon (监督对比, 工况标签定义正负对但不分类)
  - TFRClassifier -> FreqEncoder (TFR -> z_freq, LayerNorm 替代 BN, batch 小时更稳)
  - 去掉合成预训练 L_w (方案B 弃用)
  - 可微 reassigner 修复后, SupCon 经 TFR->σ->w_i 监督 GAT (此前梯度断裂)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, List

from models.sast_nodes import FreqRegion, PUMP_TURBINE_REGIONS
from models.sast_graph import PHYSICS_EDGES, EdgeType


# ═══════════════════════════════════════════════════════════════
# 1. FreqEncoder (for L_supcon) - TFR -> 归一化 z_freq
# ═══════════════════════════════════════════════════════════════

class FreqEncoder(nn.Module):
    """
    TFR -> 单位球面 embedding z_freq (供 SupCon 对比).

    GlobalAvgPool(T) -> log1p -> MLP -> L2 normalize.
    用 LayerNorm 替代 BatchNorm (batch 小时统计更稳).
    """

    def __init__(self, n_freq_bins: int, embed_dim: int = 128,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_freq_bins, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

    def forward(self, tfr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tfr: [B, F, T] 时频表示 (非负幅值)
        Returns:
            z_freq: [B, embed_dim] L2 归一化 embedding
        """
        feat = tfr.mean(dim=-1)        # [B, F] GlobalAvgPool over time
        feat = torch.log1p(feat)
        z = self.encoder(feat)         # [B, embed_dim]
        return F.normalize(z, dim=-1)  # 单位球面


# ═══════════════════════════════════════════════════════════════
# 2. L_supcon - 监督对比损失 (Khosla et al. 2020)
# ═══════════════════════════════════════════════════════════════

def supcon_loss(z: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.1) -> torch.Tensor:
    """
    Supervised Contrastive Loss.

    同工况样本的 z 拉近, 不同工况推远. 用工况标签定义正负对, 但不做分类
    (无分类头, 学的是工况判别表示).

    Args:
        z:           [B, D] L2 归一化 embedding
        labels:      [B] 工况标签
        temperature: 温度 (越小越聚焦硬正样本)

    Returns:
        scalar loss
    """
    device = z.device
    B = z.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 相似度矩阵 (z 已归一化, cos sim = z@z^T)
    sim = torch.matmul(z, z.T) / temperature          # [B, B]
    # 数值稳定
    sim_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - sim_max.detach()

    # 排除自身
    mask_self = torch.eye(B, dtype=torch.bool, device=device)
    logits = logits.masked_fill(mask_self, -1e9)

    # log p(j|i) = logits(i,j) - log(sum_a exp logits(i,a))
    exp = torch.exp(logits)
    log_prob = logits - torch.log(exp.sum(dim=1, keepdim=True) + 1e-8)

    # 正样本: 同工况 (且非自身)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & ~mask_self      # [B, B]
    has_pos = pos_mask.sum(dim=1) > 0

    if not has_pos.any():
        # batch 内无同工况正样本对 -> 不贡献
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 对每个 anchor, 在正样本上平均 log p
    pos_log_prob = (pos_mask.float() * log_prob).sum(dim=1) / \
                   pos_mask.sum(dim=1).clamp(min=1)
    loss = -pos_log_prob[has_pos].mean()
    return loss


# ═══════════════════════════════════════════════════════════════
# 3. RE_2D - Rényi 2D 熵 (Colominas & Meignen 2025, Eq.19)
# ═══════════════════════════════════════════════════════════════

def renyi_2d_loss(tfr: torch.Tensor,
                   freqs: torch.Tensor,
                   regions: Optional[List[FreqRegion]] = None,
                   alpha: int = 2,
                   eps: float = 1e-8) -> torch.Tensor:
    """
    Per-region 选择性 Renyi 2D 熵.

    HARMONIC/BLADE_PASS 节点频段 (BPF, 2xBPF): 最小化熵 (鼓励集中, 该硬挤)
    HYDRAULIC 节点频段 (LOW_FREQ): 不惩罚 (保留展宽, 该软挤)

    修复 v3 全局熵"鼓励所有能量集中"与选择性挤压的冲突:
    全局 RE_2D 会惩罚 LOW_FREQ 涡带展宽 (本应保留), 破坏选择性.

    Args:
        tfr:     [B, F, T] TFR 幅值 (非负)
        freqs:   [F] 频率轴 (Hz)
        regions: 物理节点频段 (默认 PUMP_TURBINE_REGIONS)
        alpha:   Renyi 阶数
        eps:     数值保护

    Returns:
        scalar (lower = HARMONIC 频段更集中; LOW_FREQ 不计入)
    """
    if regions is None:
        regions = PUMP_TURBINE_REGIONS
    device = tfr.device
    loss = torch.zeros((), device=device)
    n_squeeze = 0
    for region in regions:
        if region.f_type == 'HYDRAULIC':
            continue  # LOW_FREQ: 保留展宽, 不惩罚
        f_mask = (freqs >= region.f_min) & (freqs <= region.f_max)
        if not f_mask.any():
            continue
        tfr_r = tfr[:, f_mask, :]                      # [B, F_r, T]
        tfr_pos = tfr_r.clamp(min=eps)
        total = tfr_pos.sum(dim=(1, 2), keepdim=True).clamp(min=eps)
        p = tfr_pos / total
        h = torch.log((p ** alpha).sum(dim=(1, 2)).clamp(min=eps)) / (1.0 - alpha)
        loss = loss + h.mean()
        n_squeeze += 1
    return loss / max(1, n_squeeze)


# ═══════════════════════════════════════════════════════════════
# 4. L_physics - 比值偏差物理一致性
# ═══════════════════════════════════════════════════════════════

def physics_consistency_loss(A_ij: torch.Tensor,
                              edge_feats: torch.Tensor,
                              edge_src: torch.Tensor,
                              edge_dst: torch.Tensor,
                              r_nom: Optional[torch.Tensor] = None,
                              eps: float = 1e-8) -> torch.Tensor:
    """
    约束 GAT 注意力 A_ij 与物理一致性对齐 (设计文档 SAST_v2_design S4.3).

    L_physics = sum_edges [ w(type) * A_ij * l_ij ] / sum A_ij

    高注意力 + 物理不一致 -> 大惩罚 -> 推动 GAT 学会利用边特征分配注意力.
    按边类型差异化 l_ij (edge_feats dim0 语义按边类型):
      INTEGER_HARMONIC: |r_obs - r_nom|/r_nom   (比值须等于整数)
      CONDITION:        1 - cond_sim            (上下文匹配应高)
      DRIFT:            1 - Corr_E              (能量共变应正相关)
      ENERGY_COMPETITION: 1 - (-Corr_E)         (此消彼长应负相关; dim0=-Corr_E)

    修复 v3: 约束 A_ij (非 w_i), 避免 gate_edge x ratio_dev 互消导致 ~0.

    Args:
        A_ij:       [B, M, H, T] 或 [B, M, T] GAT 注意力权重 (可微)
        edge_feats: [B, M, T, 5] 边特征
        edge_src:   [M]
        edge_dst:   [M]
        r_nom:      [M] 标称比值
        eps:        数值保护

    Returns:
        scalar loss
    """
    device = A_ij.device
    A = A_ij.mean(dim=2) if A_ij.dim() == 4 else A_ij   # [B, M, T]
    B, M, T = A.shape

    feat0 = edge_feats[:, :, :, 0]      # [B, M, T] dim0 (语义按边类型)
    w_type = edge_feats[:, :, :, 3]     # [B, M, T]

    if r_nom is None:
        r_nom = torch.tensor([e.r_nom for e in PHYSICS_EDGES],
                            device=device, dtype=torch.float32)
    r_nom_list = r_nom.tolist()

    # 按边类型算 l_ij
    ell = torch.zeros(B, M, T, device=device)
    for m, e in enumerate(PHYSICS_EDGES):
        if e.edge_type == EdgeType.INTEGER_HARMONIC:
            ell[:, m, :] = (feat0[:, m, :] - r_nom_list[m]).abs() / max(r_nom_list[m], eps)
        elif e.edge_type == EdgeType.CONDITION:
            ell[:, m, :] = 1.0 - feat0[:, m, :].clamp(-1.0, 1.0)        # cond_sim
        elif e.edge_type == EdgeType.DRIFT:
            ell[:, m, :] = 1.0 - feat0[:, m, :].clamp(-1.0, 1.0)        # Corr_E (正应高)
        elif e.edge_type == EdgeType.ENERGY_COMPETITION:
            ell[:, m, :] = 1.0 - feat0[:, m, :].clamp(-1.0, 1.0)        # -Corr_E (大=负相关=自洽)

    per_edge = w_type * A * ell
    loss = per_edge.sum() / (A.sum() + eps)
    return loss


# ═══════════════════════════════════════════════════════════════
# 5. L_smooth - 时序平滑
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
# 6. L_balance - 防退化正则化
# ═══════════════════════════════════════════════════════════════

def balance_loss(w_i: torch.Tensor,
                  w_min: float = 0.3, w_max: float = 0.9) -> torch.Tensor:
    """
    per-node 防退化: 每节点时间均值 w_node in [w_min, w_max].

    L_balance = mean_nodes [ max(0, w_min - w_node) + max(0, w_node - w_max) ]
    per-node (非全局均值), 防单节点饱和到 1.0.
    """
    w_node = w_i.mean(dim=-1)  # [B, N_phys] per-node 时间均值
    l = torch.clamp(w_min - w_node, min=0.0) + torch.clamp(w_node - w_max, min=0.0)
    return l.mean()


def w_variance_loss(w_i: torch.Tensor, target_var: float = 0.05) -> torch.Tensor:
    """
    鼓励节点间 w_i 分化 (惩罚低方差).

    w_i 全同 (无分化) -> var=0 -> loss=target_var (惩罚)
    w_i 分化 -> var>target -> loss=0 (不惩罚)

    与 balance_loss 配合: balance 防单节点饱和, variance 防节点间全同.
    """
    w_node = w_i.mean(dim=-1)  # [B, N_phys]
    var = w_node.var(dim=-1)   # [B] 节点间方差
    return torch.relu(target_var - var).mean()


# ═══════════════════════════════════════════════════════════════
# 6b. Low-frequency sharpness loss (方法5)
# ═══════════════════════════════════════════════════════════════

def lowfreq_sharpness_loss(
    tfr_enhanced: torch.Tensor,
    freqs: torch.Tensor,
    node_if: torch.Tensor,
    regions: List[FreqRegion],
    alpha_renyi: float = 3.0,
    f_low: float = 2.0,
    f_high: float = 25.0,
) -> torch.Tensor:
    """
    低频锐化 loss (设计 §18 方法5).

    只在低频段 (LOW_FREQ: 2-25 Hz) 是简谐信号时鼓励 TFR 集中,
    宽带噪声则保留展宽 (不搞坏原始"LOW_FREQ 保留展宽"设计).

    机制:
      1. 提取 LOW_FREQ 频段 TFR 子矩阵
      2. 逐帧 tonality = 1/(1+CV(frame_energy))
         简谐: 能量集中少数 bin → CV 大 → tonality -> 1
         宽带噪声: 能量分散 → CV 小 -> tonality -> 0
      3. Rényi 熵 × tonality 加权 → 只在简谐时鼓励集中

    Args:
        tfr_enhanced: [B, F, T] SAST 增强 TFR
        freqs:        [F] 频率轴 (Hz)
        node_if:      [B, N_phys, T] 节点瞬时频率
        regions:      FreqRegion 列表 (找 LOW_FREQ)
        alpha_renyi:  Rényi α 参数
        f_low, f_high: LOW_FREQ 频段边界 (Hz), 默认 2-25

    Returns:
        scalar loss (越低越好)
    """
    B, F, T = tfr_enhanced.shape
    device = tfr_enhanced.device
    dtype = tfr_enhanced.dtype

    # 1. 频段 mask
    freq_mask = (freqs >= f_low) & (freqs <= f_high)
    n_low = freq_mask.sum().item()
    if n_low == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # 2. 提取低频子矩阵 [B, n_low, T]
    tfr_low = tfr_enhanced[:, freq_mask, :]
    eps = 1e-12

    # 3. 逐帧 tonality (CV-based)
    # E_t = sum_f |TFR(f,t)|² -> [B, T]
    E_t = (tfr_low ** 2).sum(dim=1) + eps
    # std_t = std across freq bins -> [B, T]
    std_t = tfr_low.std(dim=1)
    # CV = std/mean (能量越集中 CV 越大)
    mean_t = tfr_low.mean(dim=1) + eps
    cv_t = std_t / mean_t
    # tonality: CV 大 → tonality -> 1; CV 小 → tonality -> 0
    tonality = 1.0 / (1.0 + 1.0 / (cv_t + eps))  # 即 cv/(1+cv), bounded [0,1)

    # 4. 逐帧 Rényi 熵 (1D: freq 维)
    # P(f|t) = |TFR|^2 / E_t -> [B, n_low, T]
    P_ft = (tfr_low ** 2) / (E_t.unsqueeze(1) + eps)
    if alpha_renyi == 1.0:
        # Shannon limit
        re_t = -(P_ft * (P_ft + eps).log()).sum(dim=1)  # [B, T]
    else:
        # Rényi: 1/(1-α) * log(∑ P^α)
        P_alpha = P_ft ** alpha_renyi
        sum_P_alpha = P_alpha.sum(dim=1) + eps
        re_t = 1.0 / (1.0 - alpha_renyi) * torch.log(sum_P_alpha)  # [B, T]

    # 5. 加权: Rényi × tonality, 均值
    # tonality 高 (简谐) -> 全权重鼓励集中; tonality 低 (噪声) -> 几乎不惩罚
    loss = (re_t * tonality).mean()

    # 诊断: 平均 tonality (0=噪声, 1=纯简谐)
    return loss


# ═══════════════════════════════════════════════════════════════
# 7. Total Loss
# ═══════════════════════════════════════════════════════════════

def total_sast_loss(tfr_raw: torch.Tensor,
                     tfr_enhanced: torch.Tensor,
                     w_i: torch.Tensor,
                     y_true: torch.Tensor,
                     freq_encoder: nn.Module,
                     edge_feats: torch.Tensor,
                     gate_edge: torch.Tensor,
                     edge_src: torch.Tensor,
                     edge_dst: torch.Tensor,
                     node_if: torch.Tensor,
                     freqs: torch.Tensor,
                     A_ij: Optional[torch.Tensor] = None,
                     lambda_supcon: float = 1.0,
                     lambda_entropy: float = 0.1,
                     lambda_physics: float = 0.5,
                     lambda_smooth: float = 0.05,
                     lambda_balance: float = 0.01,
                     lambda_A: float = 0.1,
                     lambda_var: float = 0.5,
                     lambda_lowfreq: float = 0.05,
                     supcon_temperature: float = 0.1,
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    SAST v4 总损失 (SupCon 自监督).

    L_total = λ_sc*L_supcon + λ_e*RE_2D + λ_p*L_physics
            + λ_s*L_smooth + λ_b*L_balance + λ_var*L_var
            + λ_lf*L_lowfreq

    Args:
        tfr_raw:      [B, F, T] 原始 STFT 幅值
        tfr_enhanced: [B, F, T] SAST 增强 TFR
        w_i:          [B, N_phys, T] IF 可信度
        y_true:       [B] 工况标签 (SupCon 正负对定义)
        freq_encoder: FreqEncoder 模块
        edge_feats:   [B, M, T, 5] 边特征
        gate_edge:    [B, M, T] 边门控
        edge_src:     [M]
        edge_dst:     [M]
        node_if:      [B, N_phys, T] 节点 IF
        freqs:        [F] 频率轴
        A_ij:         [B, M, H, T] (可选)
        lambda_*:     各项权重系数
        lambda_lowfreq: 低频锐化 loss 权重 (默认 0.05)
        supcon_temperature: SupCon 温度

    Returns:
        total_loss:  scalar
        losses_dict: dict of individual loss values + diagnostics
    """
    # ── 各分量 ──
    z_freq = freq_encoder(tfr_enhanced)
    l_supcon = supcon_loss(z_freq, y_true, temperature=supcon_temperature)
    l_entropy = renyi_2d_loss(tfr_enhanced, freqs)
    if A_ij is not None:
        l_physics = physics_consistency_loss(
            A_ij, edge_feats, edge_src, edge_dst
        )
    else:
        l_physics = torch.tensor(0.0, device=w_i.device)
    l_smooth = temporal_smoothness_loss(w_i, A_ij, lambda_A=lambda_A)
    l_balance = balance_loss(w_i)
    l_var = w_variance_loss(w_i)
    l_lowfreq = lowfreq_sharpness_loss(
        tfr_enhanced, freqs, node_if, PUMP_TURBINE_REGIONS)

    # ── 加权求和 ──
    total = (lambda_supcon * l_supcon +
             lambda_entropy * l_entropy +
             lambda_physics * l_physics +
             lambda_smooth * l_smooth +
             lambda_balance * l_balance +
             lambda_var * l_var +
             lambda_lowfreq * l_lowfreq)

    # ── 诊断 ──
    losses_dict = {
        'total': total.item(),
        'supcon': l_supcon.item(),
        'entropy_2d': l_entropy.item(),
        'physics': l_physics.item(),
        'smooth': l_smooth.item(),
        'balance': l_balance.item(),
        'w_var': l_var.item(),
        'lowfreq_sharp': l_lowfreq.item(),
        'w_mean': w_i.mean().item(),
        'w_min': w_i.min().item(),
        'w_max': w_i.max().item(),
        'w_spread': (w_i.mean(dim=-1).max(dim=-1).values.mean() -
                     w_i.mean(dim=-1).min(dim=-1).values.mean()).item(),
    }

    return total, losses_dict


# ═══════════════════════════════════════════════════════════════
# 8. Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("SAST Losses v4 - Smoke Test (SupCon)")
    print("=" * 60)

    nB, nF, nT, nN, nM = 8, 256, 100, 3, 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tfr_raw = torch.rand(nB, nF, nT, device=device)
    tfr_enhanced = torch.rand(nB, nF, nT, device=device)
    w_i = torch.sigmoid(torch.randn(nB, nN, nT, device=device))
    y_true = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4], device=device)  # 含同工况对
    node_if = torch.rand(nB, nN, nT, device=device) * 500
    freqs = torch.linspace(0, 500, nF, device=device)
    edge_feats = torch.rand(nB, nM, nT, 5, device=device)
    gate_edge = torch.rand(nB, nM, nT, device=device)
    edge_src = torch.tensor([0, 0, 0, 1, 2, 2, 1, 3, 2, 3], device=device)
    edge_dst = torch.tensor([1, 2, 3, 2, 1, 3, 3, 1, 3, 2], device=device)
    A_ij = torch.rand(nB, nM, 4, nT, device=device)

    encoder = FreqEncoder(n_freq_bins=nF).to(device)

    total, d = total_sast_loss(
        tfr_raw, tfr_enhanced, w_i, y_true, encoder,
        edge_feats, gate_edge, edge_src, edge_dst,
        node_if, freqs, A_ij,
    )

    print(f"Total loss: {d['total']:.4f}")
    for k, v in d.items():
        print(f"  {k:<20s} {v:.4f}")
    print("Done!")
