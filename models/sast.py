"""
SAST: Structure-Aware Synchrosqueezing Transform (v3)
=====================================================
异构物理图 + 静态原型软匹配 + 自适应阶数 SST + 稀疏高斯重排.

核心架构:
  Signal -> MSST (N_max, save_trajectory) -> Node Features + omegas
  -> StaticPrototypeMatcher (soft matching) -> OperatingToken
  -> PPM (ratio-gated + type-embed) -> GAT (6 edges) -> w_i
  -> PerBinOrderSelector (w_i + convergence + structure) -> N*(eta,b)
  -> SparseGaussianReassigner (banded sparse A_n) -> TFR_sast

物理图 (异构, 4 节点 / 6 边):
  OP (virtual) --CONDITION--> LOW_FREQ, BPF, 2xBPF
  LOW_FREQ <--DRIFT--> BPF
  BPF --HARMONIC(r=2.0)--> 2xBPF

GAT 输出 w_i -> 四参数策略:
  w_i   : IF 可信度 (GAT 直接输出)
  sigma_i: 核宽 = sigma_min + (1-w_i)*Delta_sigma (连续, 可微)
  lambda_i: 迭代深度 = round(w_i * N_max) (离散, 推理用)
  IF 阶数: omegas 索引 = round(w_i * (N_max-1)) (离散, 推理用)

模块分工:
  models/tfr.py        - MSST (omega estimation + trajectory)
  models/sast_nodes.py - MSSTNodeExtractor (per-region aggregation)
  models/sast_graph.py - Heterogeneous physics graph topology + edge features
  models/sast.py       - StaticPrototypeMatcher + PPM + GAT + PerBinOrderSelector
                         + SparseGaussianReassigner + SAST (this file)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Dict, List, Tuple

from models.tfr import msst
from models.sast_nodes import (
    MSSTNodeExtractor, NodeFeatures, PUMP_TURBINE_REGIONS,
    N_PHYSICS_NODES, FreqRegion,
)
from models.sast_graph import (
    PHYSICS_EDGES, N_EDGES, N_NODES,
    PHYSICS_NODE_INDICES,
    compute_edge_features, get_graph_summary,
    HARMONIC_EDGE_INDICES, CONDITION_EDGE_INDICES, DRIFT_EDGE_INDICES,
    COMPETITION_EDGE_INDICES,
)


# ═══════════════════════════════════════════════════════════════
# 0. Static Prototypes (from EDA on 5_dataset.npz)
# ═══════════════════════════════════════════════════════════════

STATIC_PROTOTYPES = torch.tensor([
    [0.037636, 0.208980, 0.026644, 0.986240],  # Class 0: No-load
    [0.068068, 0.832580, 0.010149, 1.987957],  # Class 1: Low load
    [0.046170, 0.269881, 0.303809, 0.001379],  # Class 2: Mid load
    [0.013093, 0.153476, 0.652464, -0.651019], # Class 3: High load
    [0.021357, 0.685910, 0.099116, 0.850605],  # Class 4: Pumping
], dtype=torch.float64)

N_PROTOTYPES = STATIC_PROTOTYPES.shape[0]  # 5
PROTOTYPE_DIM = STATIC_PROTOTYPES.shape[1]  # 4


# ═══════════════════════════════════════════════════════════════
# 1. StaticPrototypeMatcher
# ═══════════════════════════════════════════════════════════════

class StaticPrototypeMatcher(nn.Module):
    """
    离线统计的 5 个工况原型 -> 在线软匹配.

    不预测工况类别. 仅计算当前帧的能量分布 V_obs 与 5 个 frozen prototype
    的余弦相似度 -> softmax -> 注意力权重 alpha.
    alpha 不直接使用, 而是用于加权混合可学习的 prototype 解释嵌入 P_embed.

    Args:
        d_cond:    cond_ctx 输出维度
        temperature: softmax 温度 (越小越偏向 hard assignment)
    """

    def __init__(self, d_cond: int = 32, temperature: float = 0.1):
        super().__init__()
        self.d_cond = d_cond
        self.temperature = temperature

        # Frozen prototypes from EDA: [5, 4]
        self.register_buffer('prototypes', STATIC_PROTOTYPES.clone())

        # Learnable prototype interpretation embeddings: [5, d_cond]
        # 模型学习"像原型 k"如何转化为挤压策略
        self.P_embed = nn.Parameter(torch.randn(N_PROTOTYPES, d_cond) * 0.02)

        # V_obs 投影: 4 -> d_cond (对齐到 prototype 空间)
        self.obs_proj = nn.Sequential(
            nn.Linear(PROTOTYPE_DIM, d_cond),
            nn.LayerNorm(d_cond),
            nn.GELU(),
        )

    def forward(self, V_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            V_obs: [B, T, 4]  每帧实时能量比例向量
                   [R_LF, R_BPF, R_2xBPF, log10(E_BPF/E_2xBPF)]

        Returns:
            cond_ctx: [B, T, d_cond]  工况上下文嵌入
            alpha:    [B, T, 5]       原型注意力权重 (诊断用)
        """
        B, T, _ = V_obs.shape
        device = V_obs.device

        # 余弦相似度
        proto = self.prototypes.to(device).to(V_obs.dtype)        # [5, 4]
        proto_norm = F.normalize(proto, dim=-1)                    # [5, 4]
        obs_norm = F.normalize(V_obs, dim=-1)                      # [B, T, 4]

        sim = torch.matmul(obs_norm, proto_norm.T)                 # [B, T, 5]
        alpha = F.softmax(sim / self.temperature, dim=-1)          # [B, T, 5]

        # 加权混合 prototype 解释嵌入
        P_emb = self.P_embed.to(device).to(V_obs.dtype)            # [5, d_cond]
        cond_ctx = torch.matmul(alpha, P_emb)                       # [B, T, d_cond]

        return cond_ctx, alpha

    def compute_V_obs(self, node_energy: torch.Tensor,
                       freqs: torch.Tensor) -> torch.Tensor:
        """
        从节点能量 + 频率轴实时计算 V_obs (4 维能量比例向量).

        Args:
            node_energy: [B, N_phys, T]  节点对数能量
            freqs:       [F] 频率轴 (Hz)

        Returns:
            V_obs: [B, T, 4]  [R_LF, R_BPF, R_2xBPF, log10(E_BPF/E_2xBPF)]
        """
        B, N, T = node_energy.shape
        device = node_energy.device

        # node_energy 已经是对数能量, 转回线性用于比例计算
        E_lin = torch.exp(node_energy.clamp(max=20.0))  # [B, N, T]
        E_total = E_lin.sum(dim=1, keepdim=True).clamp(min=1e-12)

        R_LF = E_lin[:, 0, :] / E_total[:, 0, :]      # [B, T]
        R_BPF = E_lin[:, 1, :] / E_total[:, 0, :]     # [B, T]
        R_2xBPF = E_lin[:, 2, :] / E_total[:, 0, :]   # [B, T]

        eps = 1e-12
        log_ratio = torch.log10(
            E_lin[:, 1, :].clamp(min=eps) / E_lin[:, 2, :].clamp(min=eps)
        )  # [B, T]

        V_obs = torch.stack([R_LF, R_BPF, R_2xBPF, log_ratio], dim=-1)  # [B, T, 4]
        return V_obs


# ═══════════════════════════════════════════════════════════════
# 2. PhysicsPrototypeMemory (updated for heterogeneous graph)
# ═══════════════════════════════════════════════════════════════

class PhysicsPrototypeMemory(nn.Module):
    """
    物理原型记忆库 (异构图版).

    变更 (vs v2):
      - 新增 OP 虚拟节点: 从 cond_ctx 投影得到 h_OP
      - CONDITION 边门控: cond_sim = cosine(cond_ctx, node_proj)
      - DRIFT 边门控: energy correlation based
      - HARMONIC 边门控: ratio-gated (unchanged)

    Args:
        d_h:              隐层维度
        f_type_embed_dim: 频率类型嵌入维度
        temperature:       比值门控温度 tau
        regions:           频率区域定义
    """

    def __init__(self, d_h: int = 128, f_type_embed_dim: int = 16,
                 temperature: float = 0.08,
                 regions: Optional[List[FreqRegion]] = None):
        super().__init__()
        self.regions = regions if regions is not None else PUMP_TURBINE_REGIONS
        self.N_phys = len(self.regions)
        self.N_total = N_NODES  # 4 (OP + 3 physical)
        self.temperature = temperature

        # ── C_prior: 每物理节点的先验信任度 (可学习 logit) ──
        c_prior_init = torch.tensor([r.C_prior for r in self.regions])
        self.C_prior_logit = nn.Parameter(
            torch.logit(c_prior_init.clamp(0.01, 0.99))
        )  # [N_phys]

        # ── 频率类型嵌入 (仅物理节点) ──
        f_types = [r.f_type for r in self.regions]
        unique_types = sorted(set(f_types))
        self.type_to_idx = {t: i for i, t in enumerate(unique_types)}
        self.f_type_embed = nn.Embedding(len(unique_types), f_type_embed_dim)
        type_idx = torch.tensor([self.type_to_idx[t] for t in f_types])
        self.register_buffer('type_idx', type_idx)  # [N_phys]

        # ── 可学习原型嵌入 (物理节点) ──
        self.prototype_embed = nn.Parameter(
            torch.randn(self.N_phys, d_h) * 0.02
        )  # [N_phys, d_h]

        # ── OP 虚拟节点投影 ──
        self.op_proj = nn.Sequential(
            nn.Linear(32, d_h),  # cond_ctx dim
            nn.LayerNorm(d_h),
            nn.GELU(),
        )

        # ── 标称比值 (仅 HARMONIC 边) ──
        r_nom_list = [e.r_nom for e in PHYSICS_EDGES]
        self.register_buffer('r_nom', torch.tensor(r_nom_list))  # [M]
        self.register_buffer('w_type', torch.tensor(
            [e.w_type for e in PHYSICS_EDGES]
        ))  # [M]

        # ── 节点特征投影 (物理节点: 4 -> d_h) ──
        self.node_proj = nn.Sequential(
            nn.Linear(4, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, d_h),
        )

        # 类型增强投影
        self.type_proj = nn.Linear(f_type_embed_dim, d_h, bias=False)

        # 交叉注意力: node -> prototype
        self.W_q = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.W_v = nn.Linear(d_h, d_h, bias=False)

        # CONDITION 边相似度投影
        self.cond_sim_proj = nn.Linear(d_h, d_h, bias=False)

        # GAT 输入投影 (h + C_prior)
        self.gat_input_proj = nn.Linear(d_h + 1, d_h)

    def get_C_prior(self) -> torch.Tensor:
        """返回当前学习的 C_prior 值 [N_phys]."""
        return torch.sigmoid(self.C_prior_logit).detach()

    def compute_edge_gates(self, r_obs: torch.Tensor,
                           h_phys: torch.Tensor,
                           cond_ctx: torch.Tensor,
                           drft_feats: torch.Tensor,
                           comp_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算逐边门控 (异构: 不同类型使用不同门控).

        Args:
            r_obs:      [B, M, T] 实测比值 (仅 HARMONIC 边有意义)
            h_phys:     [B, N_phys, d_h] 物理节点特征
            cond_ctx:   [B, T, d_cond] 工况上下文
            drft_feats: [B, M, T] DRIFT 边特征 (dim 0 = Corr_E)
            comp_feats: [B, M, T] COMPETITION 边特征 (dim 0 = -Corr_E)

        Returns:
            gate_edge: [B, M, T] 逐边门控 (0-1)
        """
        B = r_obs.shape[0]
        T = r_obs.shape[2]
        M = r_obs.shape[1]
        device = r_obs.device

        gate_edge = torch.zeros(B, M, T, device=device)

        # ── HARMONIC: 比值门控 ──
        for m in HARMONIC_EDGE_INDICES:
            r_nom_m = self.r_nom[m]
            ratio_dist = (r_obs[:, m, :] - r_nom_m).abs() / max(r_nom_m, 1e-8)
            gate_edge[:, m, :] = torch.exp(-ratio_dist / self.temperature)

        # ── CONDITION: cond_ctx 相似度门控 ──
        # h_phys projected for similarity with cond_ctx
        h_for_cond = self.cond_sim_proj(h_phys)  # [B, N_phys, d_h]
        cond_proj = self.op_proj(cond_ctx)        # [B, T, d_h]
        cond_sim = torch.zeros(B, len(CONDITION_EDGE_INDICES), T, device=device)

        for i, m in enumerate(CONDITION_EDGE_INDICES):
            e = PHYSICS_EDGES[m]
            dst_phys = e.dst - 1  # OP=0, physical nodes 1,2,3 -> 0,1,2

            # cosine similarity: h_phys[dst] vs cond_ctx
            h_dst = h_for_cond[:, dst_phys, :].unsqueeze(1)  # [B, 1, d_h]
            cond_t = cond_proj                                   # [B, T, d_h]
            sim = F.cosine_similarity(
                h_dst.expand(-1, T, -1), cond_t, dim=-1
            )  # [B, T]
            gate_edge[:, m, :] = torch.sigmoid(sim / 0.1)
            cond_sim[:, i, :] = sim  # raw cosine similarity (pre-sigmoid)

        # ── DRIFT: 能量相关门控 ──
        for m in DRIFT_EDGE_INDICES:
            corr_e = drft_feats[:, m, :]  # [B, T]  dim 0 = Corr_E
            # |Corr_E| -> gate: 高相关 -> 高门控
            gate_edge[:, m, :] = torch.sigmoid((corr_e.abs() - 0.3) / 0.1)

        # ── ENERGY_COMPETITION: 负能量相关门控 ──
        # comp_feats[:, m, 0] = -Corr_E (越大越"此消彼长")
        if comp_feats is not None:
            for m in COMPETITION_EDGE_INDICES:
                neg_corr = comp_feats[:, m, :]  # [B, T]
                # -Corr_E > 0.3 -> 显著负相关 -> 竞争关系成立 -> gate -> 1
                gate_edge[:, m, :] = torch.sigmoid((neg_corr - 0.3) / 0.1)

        return gate_edge.clamp(0.0, 1.0), cond_sim

    def forward(self, node_feats_raw: torch.Tensor,
                node_if: torch.Tensor,
                r_obs: torch.Tensor,
                cond_ctx: torch.Tensor,
                drft_feats: torch.Tensor,
                comp_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            node_feats_raw: [B, N_phys, 4] 原始节点特征
            node_if:        [B, N_phys] 节点 IF (Hz) — 当前帧
            r_obs:          [B, M] 实测边比值 — 当前帧
            cond_ctx:       [B, d_cond] 工况上下文 — 当前帧
            drft_feats:     [B, M] DRIFT 边特征 — 当前帧
            comp_feats:     [B, M] COMPETITION 边特征 — 当前帧 (dim=-Corr_E)

        Returns:
            h_enhanced:  [B, N_total, d_h]  含 OP + 物理节点
            C_prior:     [B, N_phys]
            gate_edge:   [B, M]
            gate_node:   [B, N_total]
        """
        B = node_feats_raw.shape[0]
        device = node_feats_raw.device

        # ── C_prior (物理节点) ──
        C_prior_raw = torch.sigmoid(self.C_prior_logit)  # [N_phys]
        # 工况调制: cond_ctx -> per-node bias
        cond_mod = self.op_proj(cond_ctx)[:, :self.N_phys]  # [B, N_phys]
        cond_mod = cond_mod.mean(dim=-1, keepdim=True).expand(-1, self.N_phys)
        # Simple additive modulation
        C_prior = (C_prior_raw.unsqueeze(0) + 0.1 * torch.tanh(cond_mod)).clamp(0.01, 0.99)
        # [B, N_phys]

        # ── 物理节点特征投影 ──
        h_raw = self.node_proj(node_feats_raw)  # [B, N_phys, d_h]

        # ── OP 虚拟节点特征 ──
        h_op = self.op_proj(cond_ctx)  # [B, d_h]

        # ── 类型增强 ──
        type_emb = self.f_type_embed(self.type_idx.to(device))    # [N_phys, f_dim]
        type_feat = self.type_proj(type_emb)                       # [N_phys, d_h]
        proto_feat = self.prototype_embed + type_feat              # [N_phys, d_h]

        # 交叉注意力: physical nodes -> prototypes
        Q = self.W_q(h_raw)             # [B, N_phys, d_h]
        K_p = self.W_k(proto_feat)      # [N_phys, d_h]
        V_p = self.W_v(proto_feat)      # [N_phys, d_h]

        attn_scores = torch.matmul(Q, K_p.T) / math.sqrt(h_raw.shape[-1])
        self_bias = torch.eye(self.N_phys, device=device) * 3.0
        attn_scores = attn_scores + self_bias.unsqueeze(0)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N_phys, N_phys]
        proto_context = torch.matmul(attn_weights, V_p)  # [B, N_phys, d_h]

        # ── 门控融合 ──
        edge_src_t = torch.tensor([e.src for e in PHYSICS_EDGES], device=device)
        edge_dst_t = torch.tensor([e.dst for e in PHYSICS_EDGES], device=device)

        # 边门控
        comp_t = comp_feats.unsqueeze(-1) if comp_feats is not None else None
        gate_edge, cond_sim = self.compute_edge_gates(
            r_obs.unsqueeze(-1), h_raw, cond_ctx.unsqueeze(1),
            drft_feats.unsqueeze(-1), comp_t
        )
        gate_edge = gate_edge.squeeze(-1)  # [B, M]
        cond_sim = cond_sim.squeeze(-1)     # [B, len(CONDITION_EDGES)]

        # 逐节点门控 (含 OP) - scatter_reduce (非 inplace, autograd 安全)
        src_idx = edge_src_t.unsqueeze(0).expand(B, -1)  # [B, M]
        dst_idx = edge_dst_t.unsqueeze(0).expand(B, -1)
        gate_node = torch.zeros(B, self.N_total, device=device)
        gate_node = gate_node.scatter_reduce(
            1, src_idx, gate_edge, reduce='amax', include_self=True)
        gate_node = gate_node.scatter_reduce(
            1, dst_idx, gate_edge, reduce='amax', include_self=True)
        gate_node = gate_node.clamp(0.0, 1.0)

        # 门控融合物理节点
        gate_exp = gate_node[:, 1:].unsqueeze(-1)  # [B, N_phys, 1]
        h_phys_enhanced = (1 - gate_exp) * h_raw + gate_exp * proto_context
        # [B, N_phys, d_h]

        # 组装: [OP | PHYS_0 | PHYS_1 | PHYS_2]
        h_enhanced = torch.cat([
            h_op.unsqueeze(1),           # [B, 1, d_h]
            h_phys_enhanced,             # [B, N_phys, d_h]
        ], dim=1)  # [B, N_total, d_h]

        return h_enhanced, C_prior, gate_edge, gate_node, cond_sim


# ═══════════════════════════════════════════════════════════════
# 3. EdgeConditionedGAT (mostly unchanged)
# ═══════════════════════════════════════════════════════════════

def scatter_softmax(scores: torch.Tensor, indices: torch.Tensor,
                    N: int) -> torch.Tensor:
    """沿 scatter 索引做分组 softmax (多头并行)."""
    B, M, H = scores.shape
    device = scores.device
    idx_exp = indices.view(1, M, 1).expand(B, -1, H)

    max_per_group = torch.zeros(B, N, H, device=device)
    max_per_group = max_per_group.scatter_reduce(
        1, idx_exp, scores, reduce='amax', include_self=False
    )
    scores_max = scores - max_per_group[:, indices]
    exp_scores = torch.exp(scores_max)

    sum_exp = torch.zeros(B, N, H, device=device)
    sum_exp = sum_exp.scatter_add(1, idx_exp, exp_scores)
    probs = exp_scores / (sum_exp[:, indices].clamp(min=1e-8))
    return probs


class EdgeConditionedGATLayer(nn.Module):
    """单层边条件图注意力."""

    def __init__(self, d_h: int, d_e: int = 5, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert d_h % n_heads == 0
        self.d_h = d_h
        self.d_e = d_e
        self.n_heads = n_heads
        self.d_head = d_h // n_heads
        self.dropout_rate = dropout

        self.W_q = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.W_v = nn.Linear(d_h, d_h, bias=False)
        self.W_e = nn.Linear(d_e, d_h, bias=False)

        self.attn_a = nn.Parameter(
            torch.randn(n_heads, 3 * self.d_head) * 0.02
        )
        self.out_proj = nn.Linear(d_h, d_h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_feats: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor):
        B, N, _ = h.shape
        M = edge_feats.shape[1]
        device = h.device

        Q = self.W_q(h).view(B, N, self.n_heads, self.d_head)
        K = self.W_k(h).view(B, N, self.n_heads, self.d_head)
        V = self.W_v(h).view(B, N, self.n_heads, self.d_head)
        E = self.W_e(edge_feats).view(B, M, self.n_heads, self.d_head)

        Q_dst = Q[:, edge_dst]
        K_src = K[:, edge_src]
        cat_feat = torch.cat([Q_dst, K_src, E], dim=-1)

        attn_logits = (cat_feat * self.attn_a.view(1, 1, self.n_heads, -1)
                      ).sum(dim=-1)
        attn_scores = F.leaky_relu(attn_logits, negative_slope=0.2)
        attn_weights = scatter_softmax(attn_scores, edge_dst, N)

        V_src = V[:, edge_src]
        msg = attn_weights.unsqueeze(-1) * V_src

        h_new = torch.zeros(B, N, self.n_heads, self.d_head, device=device)
        dst_exp = edge_dst.view(1, M, 1, 1).expand(B, -1, self.n_heads, self.d_head)
        h_new = h_new.scatter_add(1, dst_exp, msg)
        h_new = h_new.reshape(B, N, self.d_h)

        h_out = F.relu(self.out_proj(h_new))
        h_out = self.dropout(h_out)
        return h_out, attn_weights


class EdgeConditionedGAT(nn.Module):
    """L 层 Edge-Conditioned GAT -> w_i (IF trust)."""

    def __init__(self, d_h: int = 128, d_e: int = 5, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_h = d_h
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            EdgeConditionedGATLayer(d_h, d_e, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_h) for _ in range(n_layers)
        ])

        # w_i 输出头: h_i^(L) -> w_i in (0, 1)  per physical node
        self.mlp_w = nn.Sequential(
            nn.Linear(d_h, d_h // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h // 2, 1),
        )

    def forward(self, h: torch.Tensor, edge_feats: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            w_i:  [B, N_phys] IF 可信度 (仅物理节点, 不含 OP)
            A_ij: [B, M, H] 注意力权重
        """
        A_ij = None
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            residual = h
            h_new, attn = layer(h, edge_feats, edge_src, edge_dst)
            h = norm(residual + h_new)
            if i == self.n_layers - 1:
                A_ij = attn

        # 仅从物理节点输出 w_i
        h_phys = h[:, 1:, :]  # [B, N_phys, d_h] — skip OP
        w_logits = self.mlp_w(h_phys).squeeze(-1)  # [B, N_phys]
        w_i = torch.sigmoid(w_logits)
        return w_i, A_ij


# ═══════════════════════════════════════════════════════════════
# 4. SqueezeIterationController (formerly PerBinOrderSelector)
# ═══════════════════════════════════════════════════════════════

class SqueezeIterationController(nn.Module):
    """
    挤压迭代次数控制.

    关键洞察: MSST 的 lookup 迭代对所有 bin 无害 (总是朝能量集中方向走),
    因此所有 bin 统一用 omega_5 (最优 IF)。GAT 控制的是挤压轮数:
      - 2xBPF: lambda=1 (一轮到位, 已收敛)
      - BPF:    lambda=3~5 (多轮, 每轮重估 IF 提升 SNR)
      - LOW_FREQ: lambda=0 (不挤)

    节点级 lambda_i = round(w_i * N_max)
    bin 级 ridge_factor = energy_ratio * ridge_decay (脊线 bin→全参与, 远离 bin→跳过)
    """

    def __init__(self, N_max: int = 5):
        super().__init__()
        self.N_max = N_max

    def forward(self, w_i: torch.Tensor,
                tfr_mag: torch.Tensor, freqs: torch.Tensor,
                node_if: torch.Tensor, bw_expected: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            w_i:         [B, N_phys, T]  IF 可信度
            tfr_mag:     [B, F, T]  STFT 幅值
            freqs:       [F] 频率轴 (Hz)
            node_if:     [B, N_phys, T]  节点 IF (Hz)
            bw_expected: [N_phys] 预期带宽 (Hz)

        Returns:
            lambda_i:     [B, N_phys, T]  节点级挤压轮数 (0..N_max)
            ridge_factor: [B, F, T]  逐 bin 挤压参与因子 (0..1)
        """
        B, N_phys, T = w_i.shape
        F = freqs.shape[0]
        device = w_i.device
        eps = 1e-8

        # ── lambda_i: 节点级挤压轮数 ──
        lambda_i = torch.round(w_i * self.N_max).long()  # [B, N_phys, T]
        lambda_i = lambda_i.clamp(0, self.N_max)

        # ── ridge_factor: bin 级参与因子 ──
        # 逐 bin 分配到最近物理节点
        freqs_exp = freqs.view(1, 1, F, 1)           # [1, 1, F, 1]
        node_if_exp = node_if.unsqueeze(2)             # [B, N_phys, 1, T]
        dist = (freqs_exp - node_if_exp).abs()         # [B, N_phys, F, T]
        i_star = dist.argmin(dim=1)                     # [B, F, T]

        # 每帧, 每个节点频段内的脊线位置 (能量最大 bin)
        # 需要在节点频段内找 argmax — 简化为全局频段内能量最大
        ridge_factor = torch.zeros(B, F, T, device=device)
        bw_exp = bw_expected.view(1, N_phys, 1).to(device)  # [1, N_phys, 1]

        for n in range(N_phys):
            node_mask = (i_star == n)  # [B, F, T]

            # 该节点的脊线 (逐帧, 在 assigned bin 内找 max energy)
            tfr_masked = tfr_mag.clone()
            tfr_masked[~node_mask] = 0.0

            # 脊线位置 (energy-weighted mean 而非 argmax, 更鲁棒)
            ridge_energy = tfr_masked.max(dim=1).values  # [B, T] — 脊线能量
            ridge_pos_f = (tfr_masked * freqs_exp.squeeze(1).squeeze(1).view(1, F, 1)).sum(dim=1) / \
                          tfr_masked.sum(dim=1).clamp(min=eps)  # [B, T] — energy-weighted freq

            # 对所有 bin 计算到脊线的距离
            f_all = freqs.view(1, F, 1)  # [1, F, 1]
            ridge_f_exp = ridge_pos_f.unsqueeze(1)  # [B, 1, T]
            ridge_dist = (f_all - ridge_f_exp).abs() / bw_exp[:, n:n+1, :].clamp(min=eps)  # [B, F, T]
            ridge_decay = torch.exp(-ridge_dist ** 2 / 2.0)

            # 能量比
            max_e = ridge_energy.unsqueeze(1).clamp(min=eps)  # [B, 1, T]
            energy_ratio = tfr_mag / max_e

            ridge_factor = ridge_factor + node_mask.float() * energy_ratio * ridge_decay

        ridge_factor = ridge_factor.clamp(0.0, 1.0)

        return lambda_i, ridge_factor


# ═══════════════════════════════════════════════════════════════
# 5. SparseGaussianReassigner (replaces AdaptiveSqueeze)
# ═══════════════════════════════════════════════════════════════

class SparseGaussianReassigner(nn.Module):
    """
    论文式稀疏矩阵软重排: s_n = A_n(sigma, N*) · f_n

    每列 m 的重排矩阵 A_n 仅有 2*ceil(3*sigma)+1 个非零元素,
    沿频率轴呈高斯分布, 中心位于 omega_hat[N*(m)].

    实现: scatter_add with pre-computed Gaussian weights.

    Args:
        sigma_min:     最小核宽 (bin)
        sigma_max:     最大核宽 (bin)
        n_sigma_levels: sigma 量化级别
        kernel_radius: 核半径倍数 (default 3.0 -> 3*sigma)
    """

    def __init__(self, sigma_min: float = 0.5, sigma_max: float = 15.0,
                 n_sigma_levels: int = 20, kernel_radius: float = 3.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n_sigma_levels = n_sigma_levels
        self.kernel_radius = kernel_radius

    def forward(self, tfr_mag: torch.Tensor, omegas: torch.Tensor,
                sigma_sq: torch.Tensor, freqs: torch.Tensor,
                ridge_factor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        连续可微稀疏高斯重排.

        s_n(target) = sum_eta w_k(eta; sigma) * f_n(eta)
        其中 w_k = exp(-0.5*(k/sigma)^2) / Z, sigma = sigma_sq[eta] (连续, 可微).

        可微性 (修复 v3 旧实现的梯度断裂):
          - 权重 w_k 连续依赖 sigma_sq -> 梯度可经高斯核流向 sigma_sq -> w_i
          - 目标位置 omega_hat 来自 MSST (离散, 常量), 不携带梯度 (符合设计:
            "scatter index detach", IF 本身不参与梯度, 见 SAST_v2_design §可微性)
          - 归一化核 (sum_k w_k = 1): 频率求和守恒, 不改变每帧总能量

        替代旧的 argmin level 量化 (level_idx=argmin -> 离散, tfr_enhanced 对
        sigma_sq 的梯度恒为 0, tfr_enhanced.requires_grad==False).

        Args:
            tfr_mag:      [B, F, T] STFT 幅值
            omegas:       [B, N_max, F, T] IF 轨迹 (int, 1-indexed, 0=invalid)
            sigma_sq:     [B, F, T] 逐 bin 核宽 (bin) - 可微
            freqs:        [F] 频率轴 (保留接口, 未使用)
            ridge_factor: [B, F, T] 逐 bin 挤压参与因子 (0..1, 可选)

        Returns:
            tfr_enhanced: [B, F, T] 重排后的 TFR (对 sigma_sq 可微)
        """
        B, F, T = tfr_mag.shape
        device = tfr_mag.device
        eps = 1e-8

        # ── IF 目标位置 (来自 MSST, 常量, 不参与梯度) ──
        # 所有 bin 统一使用 omega_final (最高阶 MSST lookup 的 IF)
        omega_final = omegas[:, -1, :, :].float()                # [B, F, T]
        omega_hat = (omega_final - 1.0).clamp(0, F - 1)          # 0-indexed
        omega_hat_int = omega_hat.round().long()                 # 离散目标 bin

        # ── 加权输入 ──
        if ridge_factor is not None:
            tfr_weighted = tfr_mag * ridge_factor                # 远离脊线的 bin 被衰减
        else:
            tfr_weighted = tfr_mag

        # ── 连续高斯核: sigma_sq 直接进权重 (可微) ──
        sigma = sigma_sq.clamp(self.sigma_min, self.sigma_max)   # [B, F, T]
        K = int(np.ceil(self.kernel_radius * self.sigma_max))    # 固定最大半径
        offsets_int = torch.arange(-K, K + 1, device=device).to(torch.long).tolist()

        # 两遍循环 (避免 [2K+1, B, F, T] 大张量, 省 ~91x 内存):
        #   Pass 1: Z = sum_k exp(-0.5*(k/sigma)^2)   [B, F, T]
        #   Pass 2: scatter w_k = exp(...)/Z * tfr_weighted
        # w_k 连续依赖 sigma -> 梯度回流到 sigma_sq -> w_i
        Z = torch.zeros(B, F, T, device=device, dtype=sigma.dtype)
        for k in offsets_int:
            ratio = (k / sigma).clamp(-30.0, 30.0)
            Z = Z + torch.exp(-0.5 * ratio ** 2)
        Z = Z + eps

        tfr_enhanced = torch.zeros(B, F, T, device=device, dtype=tfr_mag.dtype)
        for k in offsets_int:
            ratio = (k / sigma).clamp(-30.0, 30.0)
            w_k = torch.exp(-0.5 * ratio ** 2) / Z              # [B, F, T] 可微
            target = (omega_hat_int + k).clamp(0, F - 1)        # [B, F, T] long
            tfr_enhanced.scatter_add_(1, target, w_k * tfr_weighted)

        return tfr_enhanced


# ═══════════════════════════════════════════════════════════════
# 6. SAST — Top-level Module
# ═══════════════════════════════════════════════════════════════

class SAST(nn.Module):
    """
    Structure-Aware Synchrosqueezing Transform (v3).

    数据流:
      Signal -> MSST (N_max=5, save_trajectory) -> NodeFeatures + omegas
      -> V_obs from node energy -> StaticPrototypeMatcher -> cond_ctx, alpha
      -> Per-frame: PPM (OP + phys nodes) -> GAT (6 edges) -> w_i
      -> sigma_i = sigma_min + (1-w_i)*Delta_sigma
      -> PerBinOrderSelector -> N_star (推理用)
      -> SparseGaussianReassigner(omegas[N_star], sigma_i) -> TFR_sast

    训练时: 统一使用 omegas[-1] (最高阶), 仅 sigma_i 可微
    推理时: 四参数全启用 (sigma_i + lambda_i + IF order + w_i)

    Args:
        fs:               采样率 (Hz)
        freq_regions:     物理节点频率区域定义
        d_h:              GAT 隐层维度
        n_heads:          注意力头数
        n_layers:         GAT 层数
        sigma_min:        最小核宽 (bin)
        sigma_max:        最大核宽 (bin)
        N_max:            MSST 最大迭代次数
        msst_num:         MSST 迭代次数 (训练时使用, 应 >= N_max)
        d_cond:           工况上下文维度
        prototype_temperature: 原型匹配 softmax 温度
    """

    def __init__(self, fs: int = 1000,
                 freq_regions: Optional[List[FreqRegion]] = None,
                 d_h: int = 128, n_heads: int = 4, n_layers: int = 2,
                 sigma_min: float = 0.5, sigma_max: float = 15.0,
                 N_max: int = 4, msst_num: int = 4,
                 msst_hlength: Optional[int] = None,
                 d_cond: int = 32,
                 f_type_embed_dim: int = 16,
                 ppn_temperature: float = 0.08,
                 prototype_temperature: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()
        self.fs = fs
        self.d_h = d_h
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N_max = N_max
        self.msst_num = max(msst_num, N_max)

        # ── 频率区域 ──
        self.regions = freq_regions if freq_regions is not None else PUMP_TURBINE_REGIONS
        self.N_phys = len(self.regions)

        # ── MSST Node Extractor (CPU, numpy) ──
        self.node_extractor = MSSTNodeExtractor(
            fs=fs, freq_regions=self.regions,
            msst_hlength=msst_hlength, msst_num=self.msst_num,
        )

        # ── Physics Graph ──
        self.edges = PHYSICS_EDGES
        self.M_edges = N_EDGES
        edge_src_list = [e.src for e in self.edges]
        edge_dst_list = [e.dst for e in self.edges]
        self.register_buffer('edge_src',
                            torch.tensor(edge_src_list, dtype=torch.long))
        self.register_buffer('edge_dst',
                            torch.tensor(edge_dst_list, dtype=torch.long))

        # ── Static Prototype Matcher ──
        self.prototype_matcher = StaticPrototypeMatcher(
            d_cond=d_cond, temperature=prototype_temperature,
        )

        # ── Physics Prototype Memory ──
        self.ppm = PhysicsPrototypeMemory(
            d_h=d_h, f_type_embed_dim=f_type_embed_dim,
            temperature=ppn_temperature, regions=self.regions,
        )

        # ── Edge-Conditioned GAT ──
        self.gat = EdgeConditionedGAT(
            d_h=d_h, d_e=5, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout,
        )

        # ── Squeeze Iteration Controller (推理用) ──
        self.sqz_controller = SqueezeIterationController(N_max=N_max)

        # ── Sparse Gaussian Reassigner ──
        self.reassigner = SparseGaussianReassigner(
            sigma_min=sigma_min, sigma_max=sigma_max,
        )

    def _extract_nodes_single(self, x_np: np.ndarray) -> NodeFeatures:
        """对单条信号运行 MSST + 节点特征提取 (CPU, numpy)."""
        return self.node_extractor(x_np)

    def forward(self, x: torch.Tensor,
                training: bool = True,
                return_all: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x:          [B, T] 或 [T] 原始信号
            training:   是否训练模式 (True: 固定 N_max 阶; False: 逐 bin 自适应)
            return_all: 是否返回完整诊断量

        Returns:
            dict with keys:
              tfr_enhanced  [B, F, T]  SAST 增强 TFR
              tfr_raw       [B, F, T]  原始 STFT 幅度
              w_i           [B, N_phys, T]  IF 可信度
              sigma_sq      [B, F, T]     逐 bin 核宽
              alpha         [B, T, 5]      原型注意力权重
              A_ij          [B, M, H, T]   注意力权重
              gate_edge     [B, M, T]      边门控
              ...
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, T_in = x.shape
        device = next(self.parameters()).device  # model's device

        # ═══════════════════════════════════════════════════════
        # Step 1+2: MSST + Node Extraction
        # ═══════════════════════════════════════════════════════
        if x.is_cuda:
            # ── GPU fast path: batch MSST + node extraction on GPU ──
            gpu_nodes = self.node_extractor.extract_gpu(x)
            node_if = gpu_nodes['node_if']               # [B, N_phys, T]
            node_energy = gpu_nodes['node_energy']       # [B, N_phys, T]
            node_bw = gpu_nodes['node_bw']               # [B, N_phys, T]
            node_persist = gpu_nodes['node_persist']     # [B, N_phys]
            tfr_mag = gpu_nodes['tfr_stft'].abs()        # [B, F, T]
            freqs = gpu_nodes['freqs']                   # [F]
            omegas = gpu_nodes['omegas'].long()          # [B, N_max, F, T]

            F_bins = len(freqs)
            T_msst = tfr_mag.shape[-1]

            # Edge features (CPU compute_graph still needs numpy — transfer once)
            node_if_np = node_if.cpu().numpy()
            node_energy_np = node_energy.cpu().numpy()
            node_persist_np = node_persist.cpu().numpy()
            node_bw_np = node_bw.cpu().numpy()
        else:
            # ── CPU fallback: per-sample numpy MSST ──
            all_nodes: List[NodeFeatures] = []
            all_tfr_mag = []
            all_freqs = None

            for b in range(B):
                x_np = x[b].cpu().numpy()
                nodes = self._extract_nodes_single(x_np)
                all_nodes.append(nodes)
                all_tfr_mag.append(np.abs(nodes.tfr_stft))
                if all_freqs is None:
                    all_freqs = nodes.freqs.copy()

            T_msst = all_nodes[0].T
            F_bins = len(all_freqs)

            node_if_np = np.stack([n.if_hz for n in all_nodes])
            node_energy_np = np.stack([n.energy for n in all_nodes])
            node_bw_np = np.stack([n.bandwidth for n in all_nodes])
            node_persist_np = np.stack([n.persistence for n in all_nodes])
            tfr_mag_np = np.stack(all_tfr_mag)
            N_max_actual = len(all_nodes[0].omegas) if all_nodes[0].omegas else 0
            omegas_np = np.stack([np.stack(n.omegas) for n in all_nodes])

            node_if = torch.from_numpy(node_if_np).float().to(device)
            node_energy = torch.from_numpy(node_energy_np).float().to(device)
            node_bw = torch.from_numpy(node_bw_np).float().to(device)
            node_persist = torch.from_numpy(node_persist_np).float().to(device)
            tfr_mag = torch.from_numpy(tfr_mag_np).float().to(device)
            omegas = torch.from_numpy(omegas_np).long().to(device)
            freqs = torch.from_numpy(all_freqs).float().to(device)

        fs_half = self.fs / 2.0

        # ═══════════════════════════════════════════════════════
        # Step 3: V_obs -> StaticPrototypeMatcher -> cond_ctx
        # ═══════════════════════════════════════════════════════
        V_obs = self.prototype_matcher.compute_V_obs(node_energy, freqs)
        cond_ctx, alpha = self.prototype_matcher(V_obs)  # [B, T, d_cond], [B, T, 5]

        # ═══════════════════════════════════════════════════════
        # Step 4: Edge features (numpy -> torch, with cond_ctx)
        # ═══════════════════════════════════════════════════════
        edge_feats_list = []
        r_obs_list = []
        for b in range(B):
            ef = compute_edge_features(
                node_if_np[b], node_energy_np[b], node_persist_np[b],
                node_bw=node_bw_np[b], edges=self.edges,
                window_size=5, fs=self.fs,
            )
            edge_feats_list.append(ef['edge_feats'])  # [M, T, 5]
            r_obs_list.append(ef['edge_feats'][:, :, 0])  # [M, T] — HARMONIC uses dim 0

        edge_feats_np = np.stack(edge_feats_list)  # [B, M, T, 5]
        r_obs_np = np.stack(r_obs_list)            # [B, M, T]
        edge_feats_t = torch.from_numpy(edge_feats_np).float().to(device)
        r_obs_t = torch.from_numpy(r_obs_np).float().to(device)

        # ═══════════════════════════════════════════════════════
        # Step 5: Per-frame PPM -> GAT -> w_i
        # ═══════════════════════════════════════════════════════
        w_i_frames = []
        A_ij_frames = []
        gate_edge_frames = []
        gate_node_frames = []

        for t in range(T_msst):
            # 原始节点特征: [f_norm, log_E, bw_norm, persistence]
            f_norm = node_if[:, :, t] / fs_half          # [B, N_phys]
            log_E = node_energy[:, :, t]                  # [B, N_phys]
            bw_norm = node_bw[:, :, t] / fs_half          # [B, N_phys]
            persist = node_persist                        # [B, N_phys]

            raw_feats = torch.stack([f_norm, log_E, bw_norm, persist], dim=-1)
            # [B, N_phys, 4]

            # PPM: 原型增强 + 边门控
            # drft_feats: edge_feats dim 0 (Corr_E for DRIFT edges)
            # comp_feats: edge_feats dim 0 (-Corr_E for COMPETITION edges)
            h_enhanced, C_prior_t, gate_edge_t, gate_node_t, cond_sim_t = self.ppm(
                raw_feats,
                node_if[:, :, t],           # [B, N_phys]
                r_obs_t[:, :, t],           # [B, M]
                cond_ctx[:, t, :],          # [B, d_cond]
                edge_feats_t[:, :, t, 0],   # [B, M] — DRIFT dim 0
                edge_feats_t[:, :, t, 0],   # [B, M] — COMPETITION dim 0 (-Corr_E)
            )
            # h_enhanced: [B, N_total, d_h]

            # 注入 C_prior -> h_enhanced (OP padded, 物理节点 concatenated)
            h_op_padded = F.pad(h_enhanced[:, :1, :], (0, 1))     # [B, 1, d_h+1]
            h_phys_cat = torch.cat([
                h_enhanced[:, 1:, :], C_prior_t.unsqueeze(-1)
            ], dim=-1)                                             # [B, N_phys, d_h+1]
            h_cat = torch.cat([h_op_padded, h_phys_cat], dim=1)   # [B, N_total, d_h+1]

            h_gat_in = self.ppm.gat_input_proj(h_cat)  # [B, N_total, d_h]

            # ── Inject cond_sim into CONDITION edge features ──
            edge_feats_frame = edge_feats_t[:, :, t, :].clone()  # [B, M, 5]
            for i, m in enumerate(CONDITION_EDGE_INDICES):
                edge_feats_frame[:, m, 0] = cond_sim_t[:, i]   # raw cos sim

            # GAT
            w_i_t, A_ij_t = self.gat(
                h_gat_in, edge_feats_frame,
                self.edge_src, self.edge_dst,
            )

            w_i_frames.append(w_i_t)
            A_ij_frames.append(A_ij_t)
            gate_edge_frames.append(gate_edge_t)
            gate_node_frames.append(gate_node_t)

        w_i = torch.stack(w_i_frames, dim=-1)              # [B, N_phys, T]
        A_ij = torch.stack(A_ij_frames, dim=-1)            # [B, M, H, T]
        gate_edge = torch.stack(gate_edge_frames, dim=-1)  # [B, M, T]
        gate_node = torch.stack(gate_node_frames, dim=-1)  # [B, N_total, T]

        # ═══════════════════════════════════════════════════════
        # Step 6: w_i -> sigma_i, N_star, order_idx
        # ═══════════════════════════════════════════════════════
        # sigma_i: per-node -> broadcast to per-bin
        delta = self.sigma_max - self.sigma_min
        sigma_i = self.sigma_min + (1.0 - w_i) * delta  # [B, N_phys, T]

        # 逐 bin sigma: 分配到最近物理节点
        freqs_exp = freqs.view(1, 1, F_bins, 1).to(device)
        node_if_exp = node_if.unsqueeze(2)
        dist = (freqs_exp - node_if_exp).abs()
        i_star = dist.argmin(dim=1)  # [B, F_bins, T]

        B_idx_f = torch.arange(B, device=device).view(B, 1, 1).expand(-1, F_bins, T_msst)
        T_idx_f = torch.arange(T_msst, device=device).view(1, 1, T_msst).expand(B, F_bins, -1)
        sigma_sq = sigma_i[B_idx_f, i_star, T_idx_f]  # [B, F_bins, T]

        # ── Squeeze iteration control (推理时) ──
        lambda_sqz = None
        if not training:
            bw_expected = torch.tensor([r.bw_expected for r in self.regions],
                                      device=device, dtype=torch.float32)
            lambda_sqz, ridge_factor = self.sqz_controller(
                w_i, tfr_mag, freqs, node_if, bw_expected)
        else:
            ridge_factor = None  # 训练时所有 bin 全参与

        # ═══════════════════════════════════════════════════════
        # Step 7: Sparse Gaussian Reassignment
        # 所有 bin 统一使用 omega_5 (最优 IF), sigma + ridge_factor 控制挤压
        # ═══════════════════════════════════════════════════════
        tfr_enhanced = self.reassigner(
            tfr_mag, omegas, sigma_sq, freqs, ridge_factor,
        )

        # ── 输出 ──
        result = {
            'tfr_enhanced': tfr_enhanced,
            'tfr_raw': tfr_mag,
            'w_i': w_i,
            'sigma_sq': sigma_sq,
            'alpha': alpha,            # [B, T, 5] 原型注意力
            'cond_ctx': cond_ctx,      # [B, T, d_cond]
            'A_ij': A_ij,
            'gate_edge': gate_edge,
            'gate_node': gate_node,
            'node_if': node_if,
            'node_energy': node_energy,
            'node_bw': node_bw,
            'freqs': freqs,
            't_axis': torch.arange(T_msst, device=device, dtype=torch.float32) / self.fs,
            'edge_src': self.edge_src,
            'edge_dst': self.edge_dst,
            'edge_feats': edge_feats_t,
        }
        if not training:
            result['lambda_sqz'] = lambda_sqz       # [B, N_phys, T] — 挤压轮数
            result['ridge_factor'] = ridge_factor   # [B, F, T] — bin 参与因子
        return result

    def get_freq_features(self, x: torch.Tensor) -> torch.Tensor:
        """DCMR 桥接: 时间池化 SAST TFR -> 增强频域特征."""
        out = self.forward(x)
        tfr = out['tfr_enhanced']
        freq_feat = tfr.mean(dim=-1) + tfr.max(dim=-1).values
        return freq_feat

    def get_C_prior(self) -> torch.Tensor:
        """返回当前学习的 C_prior 值."""
        return self.ppm.get_C_prior()


# ═══════════════════════════════════════════════════════════════
# 7. Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("SAST v3 — Smoke Test (heterogeneous graph)")
    print("=" * 60)
    print(get_graph_summary())

    # 合成测试信号
    fs = 1000
    T_end = 1.0
    t = np.arange(0, T_end, 1 / fs)
    N_sig = len(t)

    sig = (np.sin(2 * np.pi * 48 * t + 0.15 * np.sin(2 * np.pi * 3 * t)) +
           0.6 * np.sin(2 * np.pi * 96 * t) +
           0.25 * np.sin(2 * np.pi * 12 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t)))
    sig = sig.astype(np.float64)

    print(f"\nSignal: T={N_sig}, fs={fs} Hz")
    print("Components: BPF(~48 Hz, FM) + 2xBPF(96 Hz, clean) + LOW_FREQ(~12 Hz)")
    print("Running SAST v3 forward pass...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = SAST(fs=fs, d_h=64, n_heads=4, n_layers=2, N_max=3, msst_num=3).to(device)
    model.eval()

    x_t = torch.from_numpy(sig).float().unsqueeze(0).to(device)

    with torch.no_grad():
        result = model(x_t, training=True, return_all=True)

    print(f"\nResults:")
    print(f"  tfr_enhanced: {result['tfr_enhanced'].shape}")
    print(f"  w_i:          {result['w_i'].shape}")
    print(f"  alpha:        {result['alpha'].shape}  (prototype attention)")
    print(f"  sigma_sq:     {result['sigma_sq'].shape}")
    print(f"  A_ij:         {result['A_ij'].shape}")
    print(f"  gate_edge:    {result['gate_edge'].shape}")
    print(f"  gate_node:    {result['gate_node'].shape}")

    w_i = result['w_i'][0].cpu().numpy()  # [N_phys, T]
    alpha = result['alpha'][0].cpu().numpy()  # [T, 5]

    print(f"\nPer-node w_i (time-mean):")
    node_names_phys = ['LOW_FREQ', 'BPF', '2xBPF']
    for i, name in enumerate(node_names_phys):
        print(f"  {name:<12s} w_i={w_i[i].mean():.3f} +/- {w_i[i].std():.3f}")

    print(f"\nPrototype attention (time-mean):")
    proto_names = ['No-load', 'Low load', 'Mid load', 'High load', 'Pumping']
    for k, name in enumerate(proto_names):
        print(f"  Proto {k} ({name:<12s}): alpha={alpha[:, k].mean():.3f}")

    print("\nDone!")
