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

from models.sast_nodes import (FreqRegion, PUMP_TURBINE_REGIONS, N_PHYSICS_NODES,
                               SOFT_F_TYPES, band_centroid_if, region_weights)
from models.sast_graph import PHYSICS_EDGES, N_EDGES, EdgeType


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

    BLADE_PASS/BLADE_HARMONIC 频段 (BPF_LINE, 2xBPF): 最小化熵 (鼓励集中, 该硬挤)
    SOFT_F_TYPES 频段 (LOW_FREQ, BPF_DRIFT): 不惩罚 (保留展宽, 该软挤)

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
        if region.f_type in SOFT_F_TYPES:
            continue  # LOW_FREQ / BPF_DRIFT: 保留展宽, 不惩罚
        # 区域支撑用统一真源的布尔掩码; 共享内部边的过渡带内 bin 按 w<1
        # 加权参与 (旧实现为两端皆闭的硬矩形)
        w_f = region_weights(freqs, [region])[0]        # [F]
        f_mask = w_f > 0.0
        if not f_mask.any():
            continue
        tfr_r = tfr[:, f_mask, :] * w_f[f_mask].view(1, -1, 1)   # [B, F_r, T]
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
# 5. L_consistency - 频段内时序一致性
# ═══════════════════════════════════════════════════════════════

def w_consistency_loss(w_i: torch.Tensor) -> torch.Tensor:
    """
    惩罚各物理节点内 w_i 沿时间的方差.

    核心思路: 一旦模型确定了频率脊线 (如 BPF ≈ 50Hz), 该频段内的挤压宽度
    w_i 在所有时间帧上应该差不多——不同频段可以有不同的 w_i, 但同一频段内部
    不能跳变.

    L_consistency = mean_{nodes} Var_{t} [w_i(n, t)]

    与 w_variance_loss 互补:
      - w_consistency_loss: 缩小节点内时间方差 (管内聚, 防跳变)
      - w_variance_loss:    扩大节点间均值差异 (管分化, 防全同退化)

    Args:
        w_i: [B, N_phys, T] IF 可信度

    Returns:
        scalar loss (越低 → 各频段内 w_i 越一致)
    """
    B, N, T = w_i.shape
    if T < 2:
        return torch.tensor(0.0, device=w_i.device)
    per_node_var = w_i.var(dim=2)    # [B, N_phys] 每节点沿时间的方差
    return per_node_var.mean()


# ═══════════════════════════════════════════════════════════════
# 6. L_balance - 防退化正则化
# ═══════════════════════════════════════════════════════════════

def balance_loss(w_i: torch.Tensor,
                  w_min=0.3, w_max=0.9,
                  presence: Optional[torch.Tensor] = None,
                  w_min_floor: float = 0.05) -> torch.Tensor:
    """
    per-node 防退化: 每节点时间均值 w_node in [w_min, w_max].

    L_balance = mean_nodes [ max(0, w_min - w_node) + max(0, w_node - w_max) ]
    per-node (非全局均值), 防单节点饱和到 1.0.

    分量存在度门控 (presence):
      原式假设"每个节点在每帧都应有分量", 对 2xBPF 在空转 (分量消失) 会强行
      要求 w_node >= w_min, 与"分量不存在时不该挤压"矛盾。
      给定 presence [B, N, T] 时下界按存在度插值:
        w_min_eff = w_min_floor + (w_min - w_min_floor) * mean_t(e_n)
      分量缺席 (e→0) → 下界松弛到 w_min_floor; 分量明显 (e→1) → 恢复 w_min。

    Args:
        w_i:         [B, N_phys, T]
        w_min/w_max: 标量或 [N_phys] 张量上下界
                     (B4 拆节点后各节点角色差异大: BPF_LINE∈[0.65,0.95],
                      BPF_DRIFT∈[0.10,0.50], 统一的 [0.3,0.9] 会把混装问题
                      从频带搬到损失里 —— 故按频段各自给界)
        presence:    [B, N_phys, T] 可选, 分量存在度
        w_min_floor: 存在度为 0 时的下界
    """
    w_node = w_i.mean(dim=-1)  # [B, N_phys] per-node 时间均值
    w_min_t = torch.as_tensor(w_min, device=w_i.device, dtype=w_i.dtype)
    w_max_t = torch.as_tensor(w_max, device=w_i.device, dtype=w_i.dtype)
    if presence is not None:
        e_mean = presence.mean(dim=-1)  # [B, N_phys]
        w_min_eff = w_min_floor + (w_min_t - w_min_floor) * e_mean
    else:
        w_min_eff = w_min_t
    l = torch.clamp(w_min_eff - w_node, min=0.0) + torch.clamp(w_node - w_max_t, min=0.0)
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
# 6b2. L_range - per-node 逐帧范围约束
# ═══════════════════════════════════════════════════════════════

def w_range_loss(w_i: torch.Tensor,
                 w_low: torch.Tensor,
                 w_high: torch.Tensor,
                 presence: Optional[torch.Tensor] = None,
                 w_low_min: float = 0.05) -> torch.Tensor:
    """
    惩罚 w_i 超出 per-node 目标范围的每一个元素 (逐帧, 非仅时间均值).

    与 w_consistency_loss 互补:
      - consistency: 缩小节点内时间方差 (平滑, 但快速跳变也能低方差)
      - range:       硬约束每个时间帧在物理合理范围内 (防跳变)

    L_range = mean[ clamp(w_low_eff - w_i, 0) + clamp(w_i - w_high, 0) ]

    分量存在度门控 (presence):
      w_range_low 隐含假设"该分量恒存在"。2xBPF 在空转工况分量消失
      (存在度 e≈0.10 vs 带负荷 0.32-0.34), 但现行下界 0.65 仍强制
      w≥0.65 → λ=round(w·4)≥3 轮硬挤压 → 凭空造出尖锐谱线。
      给定 presence [B, N, T] 时下界逐帧插值:
        w_low_eff = w_low_min + e_n(t) · (w_low - w_low_min)
      分量缺席 → 允许 w 降到 w_low_min (几乎不挤压)。

    Args:
        w_i:      [B, N_phys, T] IF 可信度
        w_low:    [N_phys] per-node 下界 (如 LOW_FREQ=0.10, BPF=0.55)
        w_high:   [N_phys] per-node 上界 (如 LOW_FREQ=0.45, BPF=1.00)
        presence: [B, N_phys, T] 可选, 分量存在度 (0=缺席, 1=显著)
        w_low_min: 存在度为 0 时的下界

    Returns:
        scalar loss (→0 表示所有帧都在有效范围内)
    """
    w_high_exp = w_high.view(1, -1, 1).to(w_i.device)   # [1, N_phys, 1]
    if presence is not None:
        w_low_base = w_low.view(1, -1, 1).to(w_i.device)     # [1, N_phys, 1]
        w_low_exp = w_low_min + presence.to(w_i.dtype) * (w_low_base - w_low_min)
    else:
        w_low_exp = w_low.view(1, -1, 1).to(w_i.device)
    l = torch.clamp(w_low_exp - w_i, min=0.0) + torch.clamp(w_i - w_high_exp, min=0.0)
    return l.mean()


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
# 6c. 节点分量存在度 (presence) — 供 w_range / balance / anchor 门控
# ═══════════════════════════════════════════════════════════════

def node_presence(tfr_mag: torch.Tensor,
                  freqs: torch.Tensor,
                  regions: Optional[List[FreqRegion]] = None,
                  frac: float = 0.1,
                  eps: float = 1e-12) -> torch.Tensor:
    """
    逐节点逐帧的"分量存在度" e ∈ [0, 1].

    动机: w_range_low / balance 的下界都隐含"该分量恒存在"的假设。
    实测 2xBPF 节点在空转工况分量消失 (带内能量峰平坦, 存在度 0.10),
    但下界仍强制 w≥0.65 → λ≥3 轮硬挤压 → 凭空造出尖锐谱线。
    用一个纯粹由输入谱决定的标量把"分量是否存在"显式表达出来。

    定义 (能量集中度, 与频率轴分辨率无关):
      c_n(t) = 带内能量最高的 k 个 bin 的能量 / 带内总能量,  k = ceil(frac·F_n)
      p      = 权重最大的 k 个 bin 的权重占比  (平坦谱时的期望占比;
               硬边界下 w≡1 退化为 k / F_n)
      e_n(t) = clamp((c - p)/(1 - p), 0, 1)
      平坦宽带谱 → c≈p → e≈0 ; 独立谱线 → c→1 → e→1

    频带权重 w_n(f) 见 sast_nodes.region_weights (半开区间 + 共享内部边
    的单位分解过渡带)。过渡带内 bin 按 w<1 加权 —— 旧实现是两端皆闭的
    硬矩形, 使空转的漂移峰右翼 (48.0-49.0 Hz) 整体计入 BPF_LINE, 造出
    presence_e = 0.317 这个反常的"全类最高"值。

    注意: 必须传**原始** (未增强) 谱, 否则模型可通过制造尖锐谱线提高 e。

    Args:
        tfr_mag: [B, F, T] 幅值谱 (或复数 STFT, 内部取模)
        freqs:   [F] 频率轴 (Hz)
        regions: FreqRegion 列表
        frac:    计入"峰"的 bin 比例

    Returns:
        [B, N_phys, T] 存在度
    """
    if regions is None:
        regions = PUMP_TURBINE_REGIONS
    if torch.is_complex(tfr_mag):
        tfr_mag = tfr_mag.abs()
    B, F, T = tfr_mag.shape
    dtype = tfr_mag.dtype
    e = torch.zeros(B, len(regions), T, device=tfr_mag.device, dtype=dtype)

    for n, r in enumerate(regions):
        # 区域支撑用统一真源 (半开区间 + 共享内部边过渡带)
        w_in = region_weights(freqs, [r])[0]             # [F]
        m = w_in > 0.0
        Fn = int(m.sum().item())
        if Fn == 0:
            continue
        w_in = w_in[m]                                   # [Fn]
        # 过渡带内的 bin 按 w<1 加权 (旧实现为两端皆闭的硬矩形)
        band = (tfr_mag[:, m, :] * w_in.view(1, -1, 1)).clamp(min=0.0)  # [B, Fn, T]
        kk = min(max(1, int(math.ceil(frac * Fn))), Fn)
        tot = band.sum(dim=1)                            # [B, T]
        top = band.topk(kk, dim=1).values.sum(dim=1)     # [B, T]
        c = top / tot.clamp(min=eps)
        # 平坦谱基线: 权重最大的 kk 个 bin 的权重占比 (硬边界下 = kk / Fn)
        p = float(w_in.topk(kk).values.sum().item() /
                  w_in.sum().clamp(min=eps).item())
        e[:, n, :] = ((c - p) / max(eps, 1.0 - p)).clamp(0.0, 1.0)
    return e


# ═══════════════════════════════════════════════════════════════
# 6d. 绝对频率锚 (L_anchor)
# ═══════════════════════════════════════════════════════════════

def freq_anchor_loss(tfr_enhanced: torch.Tensor,
                     freqs: torch.Tensor,
                     regions: Optional[List[FreqRegion]] = None,
                     presence: Optional[torch.Tensor] = None,
                     eps: float = 1e-12) -> torch.Tensor:
    """
    绝对频率锚 — 打破比值约束的尺度不变性 (设计讨论 §"谐波边").

    为什么需要:
      边约束用的是比值 r_obs = f_dst / f_src (INTEGER_HARMONIC 要求 = 2.0)。
      比值是尺度不变量: 若 BPF 与 2xBPF 的频率同时被高估 (或低估) 同样的比例,
      r_obs 不变 → 该约束对"共模频率偏差"完全无感, 无法把节点 IF 拉回真值。
      实测 r_obs = 100.1/47 ≈ 2.13 而非 2.00, 谐波边门控因此被压到 0.25-0.44。
      绝对锚提供唯一的外部参考系。

    作用对象 (关键):
      锚**不能**作用在 node_if 上 —— node_if 由 MSST 的 omega_final 解析得到,
      对网络参数不可微, 锚在它上面只会得到一个常数 (零梯度、无任何训练作用)。
      故本函数锚在**增强 TFR 的可微带内能量质心**上:
        E(f,t) = |TFR_enhanced(f,t)|²,  w = E/Σ_f E
        f_cent(t) = Σ_f w·f          (带内能量质心, 对参数可微)
      惩罚 f_cent 超出 [f_anchor ± anchor_tol] 的漂移 (flat-bottom hinge):

        L_anchor = Σ_n gate_n(t) · clamp(|f_cent_n(t) - f_anchor_n| - tol_n, 0)

      flat-bottom 的意义: 真实的工况漂移 (水头/负荷引起的峰位移动) 不应被惩罚,
      只有越界漂移才罚 → 锚是"护栏"而非"硬目标"。

    Args:
        tfr_enhanced: [B, F, T] 增强 TFR (可微)
        freqs:        [F] 频率轴 (Hz)
        regions:      FreqRegion 列表 (读 f_anchor / anchor_tol / 频段)
        presence:     [B, N_phys, T] 可选, 分量存在度门控 (缺席时不锚)

    Returns:
        scalar loss (Hz 量纲)
    """
    if regions is None:
        regions = PUMP_TURBINE_REGIONS
    dtype = tfr_enhanced.dtype
    total = torch.zeros((), device=tfr_enhanced.device, dtype=dtype)
    if_hat = band_centroid_if(tfr_enhanced, freqs, regions, eps=eps)   # [B, N, T]

    for n, r in enumerate(regions):
        if r.f_anchor is None:
            continue
        dev = (if_hat[:, n, :] - float(r.f_anchor)).abs() - float(r.anchor_tol)
        l_bt = dev.clamp(min=0.0)                               # [B, T] Hz
        if presence is not None:
            total = total + (l_bt * presence[:, n, :].to(dtype)).mean()
        else:
            total = total + l_bt.mean()
    return total


# ═══════════════════════════════════════════════════════════════
# 6e. 谐波比值一致性 (L_ratio) — 作用在**可微**节点频率上
# ═══════════════════════════════════════════════════════════════

def harmonic_ratio_loss(if_hat: torch.Tensor,
                        regions: Optional[List[FreqRegion]] = None,
                        presence: Optional[torch.Tensor] = None,
                        tol: float = 0.06,
                        eps: float = 1e-8) -> torch.Tensor:
    """
    INTEGER_HARMONIC 边的比值一致性 — A2 的核心: 让约束真正连到频率上.

    为什么需要独立的这一项 (而不是复用 L_physics):
      L_physics 的 ℓ 由 edge_feats 构造, 而 edge_feats 来自 node_if (观测 IF),
      对网络参数不可微 —— 于是 L_physics 在频率量上的梯度恒为零,
      模型可以在完全不满足 "f(2xBPF) = 2·f(BPF_LINE)" 的条件下把它降到最低,
      唯一可下降方向是重塑 GAT 注意力。约束等于没接上。

      本项改作用在 if_hat (增强 TFR 的带内能量质心) 上:
        if_hat 经 w_i → σ_i → 重排, 对参数可微 → 约束有控制力。

    公式 (对每条 r_nom > 0 的边):
      ell_m = clamp( |f_dst/f_src − r_nom| / r_nom − tol, 0 )
      L_ratio = Σ_m mean_t[ ell_m · min(e_src, e_dst) ]

    flat-bottom (tol): 真实的工况漂移 (峰位在带内小幅移动) 不该被罚,
    只有相对偏差超过 tol 才算"谐波关系被破坏"。与 L_anchor 的 flat-bottom 同源。

    注意: 比值是尺度不变量 —— 共模频率偏差由 L_anchor 负责, 二者互补。

    Args:
        if_hat:   [B, N_phys, T] 可微节点频率 (Hz), 由 band_centroid_if 得到
        regions:  FreqRegion 列表 (仅用于维度对齐校验)
        presence: [B, N_phys, T] 可选, 分量存在度门控
        tol:      允许的相对偏差 (flat-bottom 半宽), 默认 6%

    Returns:
        scalar loss (无量纲)
    """
    if regions is None:
        regions = PUMP_TURBINE_REGIONS
    dtype = if_hat.dtype
    total = torch.zeros((), device=if_hat.device, dtype=dtype)
    n_edges = 0

    for e in PHYSICS_EDGES:
        if e.edge_type != EdgeType.INTEGER_HARMONIC or e.r_nom <= 0:
            continue
        src = e.src - 1      # OP=0 → 物理节点 1..N 映射到 0..N-1
        dst = e.dst - 1
        f_src = if_hat[:, src, :]
        f_dst = if_hat[:, dst, :]
        r_obs = f_dst / f_src.clamp(min=eps)
        dev = (r_obs - float(e.r_nom)).abs() / float(e.r_nom)
        ell = (dev - tol).clamp(min=0.0)                        # [B, T]
        if presence is not None:
            gate = torch.minimum(presence[:, src, :], presence[:, dst, :]).to(dtype)
            ell = ell * gate
        total = total + ell.mean()
        n_edges += 1

    return total / max(1, n_edges)


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
                     lambda_balance: float = 0.01,
                     lambda_var: float = 0.5,
                     lambda_consistency: float = 0.3,
                     lambda_lowfreq: float = 0.05,
                     lambda_range: float = 0.3,
                     lambda_anchor: float = 0.2,
                     lambda_ratio: float = 0.2,
                     ratio_tol: float = 0.06,
                     supcon_temperature: float = 0.1,
                     regions: Optional[List[FreqRegion]] = None,
                     presence_frac: float = 0.1,
                     w_low_min: float = 0.05,
                     freq_constraints_active: bool = False,
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    SAST v4 总损失 (SupCon 自监督).

    L_total = λ_sc*L_supcon + λ_e*RE_2D + λ_p*L_physics
            + λ_b*L_balance + λ_var*L_var + λ_c*L_consistency
            + λ_lf*L_lowfreq + λ_r*L_range + λ_a*L_anchor + λ_q*L_ratio

    ⚠ 方案 A (freq_constraints_active=False, 默认): L_physics / L_anchor /
      L_ratio 这三项**不进入** L_total, 仅作为诊断量照旧计算并写入 losses_dict。
      理由 (详见 docs §14.2 及本轮结论):
        1. 重排落点 omega_hat_int 完全不可学 —— ReassignerFunction.backward 的
           三条路径皆返回 None, 只有 σ 有梯度。因此基于 IF_hat 的锚/比值约束
           无法校准瞬时频率, 只能经 w_i → σ_i 反向操纵**测量窗口内的能量质心**,
           属于伪物理约束。
        2. L_anchor 结构性失效: 实测五类 BPF_LINE 带内质心仅跨 0.74 Hz
           (49.294/50.033/49.801/49.517/49.945), 窄于 ±1.5 Hz 容差, 锚值
           49.995 天然落在该范围内 —— 对**任何**工况都不会触发。
        3. 保留 losses_dict 中的 physics/anchor/ratio 与 if_hat_* 诊断,
           使"锚在不被约束时也天然满足"成为可复现的实验证据而非断言。
      需要复现消融时传 freq_constraints_active=True 即可 (一个开关)。

    Args:
        tfr_raw:      [B, F, T] 原始 STFT 幅值 (用于估计分量存在度, 须未增强)
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
        lambda_consistency: w_i 频段内时序一致性权重 (默认 0.3)
        lambda_lowfreq: 低频锐化 loss 权重 (默认 0.05)
        lambda_range:   w_i per-node 逐帧范围约束权重 (默认 0.3)
        lambda_anchor:  绝对频率锚权重 (默认 0.2)
        lambda_ratio:   谐波比值一致性权重 (默认 0.2); 作用在可微 if_hat 上
        ratio_tol:      比值允许的相对偏差 (flat-bottom 半宽, 默认 6%)
        supcon_temperature: SupCon 温度
        regions:      物理节点频段定义 (默认 PUMP_TURBINE_REGIONS)
        presence_frac: 存在度估计计入"峰"的 bin 比例 (默认 0.1)
        w_low_min:     分量缺席时 w_range 下界松弛到的值 (默认 0.05)
        freq_constraints_active: 是否让 L_physics/L_anchor/L_ratio 参与反向图。
                         默认 False (方案 A: 降级为诊断, 见上方说明)

    Returns:
        total_loss:  scalar
        losses_dict: dict of individual loss values + diagnostics
    """
    if regions is None:
        regions = PUMP_TURBINE_REGIONS

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

    # ── 分量存在度 (由原始谱决定, 门控下面三项) ──
    presence = node_presence(tfr_raw, freqs, regions, frac=presence_frac)

    # ── per-node 逐帧范围约束 (下界按存在度松弛) ──
    w_low = torch.tensor([r.w_range_low for r in regions],
                          device=w_i.device, dtype=torch.float32)
    w_high = torch.tensor([r.w_range_high for r in regions],
                           device=w_i.device, dtype=torch.float32)
    # B4 拆节点后各节点角色差异大 (BPF_LINE 该硬挤 / BPF_DRIFT 该留宽),
    # balance 与 range 共用同一套 per-node 边界, 不再用全局 [0.3, 0.9]。
    l_balance = balance_loss(w_i, w_min=w_low, w_max=w_high,
                             presence=presence, w_min_floor=w_low_min)
    l_var = w_variance_loss(w_i)
    l_consistency = w_consistency_loss(w_i)
    r_low = regions[0]                      # LOW_FREQ 频段随定义走
    l_lowfreq = lowfreq_sharpness_loss(
        tfr_enhanced, freqs, node_if, regions,
        f_low=r_low.f_min, f_high=r_low.f_max)
    l_range = w_range_loss(w_i, w_low, w_high,
                           presence=presence, w_low_min=w_low_min)
    # ── 可微节点频率 (A2): 约束的落点, 经 w_i→σ_i→重排对参数可微 ──
    if_hat = band_centroid_if(tfr_enhanced, freqs, regions)
    # ── 绝对频率锚: 打破比值约束的尺度不变性 ──
    l_anchor = freq_anchor_loss(tfr_enhanced, freqs, regions, presence=presence)
    # ── 谐波比值一致性: 作用在 if_hat 上, 让约束真正连到频率 ──
    l_ratio = harmonic_ratio_loss(if_hat, regions, presence=presence, tol=ratio_tol)

    # ── 加权求和 ──
    total = (lambda_supcon * l_supcon +
             lambda_entropy * l_entropy +
             lambda_balance * l_balance +
             lambda_var * l_var +
             lambda_consistency * l_consistency +
             lambda_lowfreq * l_lowfreq +
             lambda_range * l_range)

    # ── 方案 A: 频率约束 (伪物理约束) 是否参与反向图 ──
    # 三项照旧在下方诊断中计算。降级时只把它们的加权贡献从 total 里摘掉,
    # 不动 losses_dict —— 这样日志仍能看到 if_hat_* / anchor / ratio 的实际值,
    # 供论文论证"锚在不被约束时也天然满足"。
    w_physics = lambda_physics if freq_constraints_active else 0.0
    w_anchor = lambda_anchor if freq_constraints_active else 0.0
    w_ratio = lambda_ratio if freq_constraints_active else 0.0
    total = (total +
             w_physics * l_physics +
             w_anchor * l_anchor +
             w_ratio * l_ratio)

    # ── 诊断 ──
    losses_dict = {
        'total': total.item(),
        'supcon': l_supcon.item(),
        'entropy_2d': l_entropy.item(),
        'physics': l_physics.item(),
        'balance': l_balance.item(),
        'w_var': l_var.item(),
        'w_consistency': l_consistency.item(),
        'lowfreq_sharp': l_lowfreq.item(),
        'w_range': l_range.item(),
        'anchor': l_anchor.item(),
        'ratio': l_ratio.item(),
        # 生效乘数 (方案 A 下为 0.0) —— 让日志自证哪几项真正参与训练
        'w_physics': w_physics,
        'w_anchor': w_anchor,
        'w_ratio': w_ratio,
        'w_mean': w_i.mean().item(),
        'w_min': w_i.min().item(),
        'w_max': w_i.max().item(),
        'w_spread': (w_i.mean(dim=-1).max(dim=-1).values.mean() -
                     w_i.mean(dim=-1).min(dim=-1).values.mean()).item(),
    }
    # 逐节点存在度 (batch 均值) — 用于确认"分量缺席"是否被识别
    p_mean = presence.mean(dim=-1).mean(dim=0)   # [N_phys]
    for n, r in enumerate(regions):
        losses_dict[f'presence_{r.name}'] = p_mean[n].item()
    # 逐节点可微 IF (batch/时间均值) — 确认 A2 把约束接到了帧率上
    ih_mean = if_hat.mean(dim=(0, 2))   # [N_phys]
    for n, r in enumerate(regions):
        losses_dict[f'if_hat_{r.name}'] = ih_mean[n].item()

    return total, losses_dict


# ═══════════════════════════════════════════════════════════════
# 8. Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("SAST Losses v4 - Smoke Test (SupCon)")
    print("=" * 60)

    nB, nF, nT = 8, 256, 100
    nN, nM = N_PHYSICS_NODES, N_EDGES
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tfr_raw = torch.rand(nB, nF, nT, device=device)
    tfr_enhanced = torch.rand(nB, nF, nT, device=device)
    w_i = torch.sigmoid(torch.randn(nB, nN, nT, device=device))
    y_true = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4], device=device)  # 含同工况对
    node_if = torch.rand(nB, nN, nT, device=device) * 500
    freqs = torch.linspace(0, 500, nF, device=device)
    edge_feats = torch.rand(nB, nM, nT, 5, device=device)
    gate_edge = torch.rand(nB, nM, nT, device=device)
    # 边索引直接由图定义导出 (B4 拆节点后 11 条边), 不再硬编码
    edge_src = torch.tensor([e.src for e in PHYSICS_EDGES], device=device)
    edge_dst = torch.tensor([e.dst for e in PHYSICS_EDGES], device=device)
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
