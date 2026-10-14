"""
SAST: Structure-Aware Synchrosqueezing Transform (v3)
=====================================================
异构物理图 + 静态原型软匹配 + 自适应阶数 SST + 稀疏高斯重排.

核心架构:
  Signal -> MSST (num, save_trajectory) -> Node Features + omegas [B,num,F,T]
  -> StaticPrototypeMatcher (soft matching) -> OperatingToken
  -> PPM (ratio-gated + type-embed) -> GAT (11 edges) -> w_i
  -> SparseGaussianReassigner(sigma_i, 软门控 g_m) -> TFR_sast
     ↑ 无 per-bin IF 层选择器 (历史上写作 PerBinOrderSelector, 从未实现;
       实测最优 IF 层恒为 omegas[-1], 详见 papers/SAST_讲解文档.md §5.1)

物理图 (异构, 5 图节点 = 4 物理 + 1 OP / 11 边):
  OP (virtual) --CONDITION--> LOW_FREQ, BPF_DRIFT, BPF_LINE, 2xBPF
  LOW_FREQ <--DRIFT--> BPF_DRIFT
  BPF_LINE --HARMONIC(r=2.0)--> 2xBPF

GAT 输出 w_i -> 两参数策略 (自适应次数改造后的现状):
  w_i     : IF 可信度 (GAT 直接输出)
  sigma_i : 核宽 = sigma_min + (1-w_i)*Delta_sigma   (连续, 可微) —— 控「挤多宽」
  g_m     : 轮数软门控 = clamp(w_i*M-(m-1), 0, 1)   (连续, 可微) —— 控「挤几轮」

  ⚠ 旧稿在此列的「lambda_i (离散, 推理用)」与「IF 阶数 (离散, 推理用)」均已删除:
    - lambda_i 的整数路径训练时不启用、round() 不可微、且被 clamp(1,..) 卡在 ≥1
      (推理永远到不了 0 轮), 已由上面的连续门控取代;
    - 「IF 阶数」从未实现。omegas[k] 是固定点 lookup 迭代层而非相位导数阶,
      且实测最优层恒为 omegas[-1] (即已在用的那层), 详见图结构文档与
      papers/SAST_讲解文档.md §5.1。

模块分工:
  models/tfr.py        - MSST (omega estimation + trajectory)
  models/sast_nodes.py - MSSTNodeExtractor (per-region aggregation)
  models/sast_graph.py - Heterogeneous physics graph topology + edge features
  models/sast.py       - StaticPrototypeMatcher + PPM + GAT + SparseGaussianReassigner
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
    NODE_LOW_FREQ, NODE_BPF_DRIFT, NODE_BPF_LINE, NODE_2XBPF,
    compute_edge_features, get_graph_summary,
    HARMONIC_EDGE_INDICES, CONDITION_EDGE_INDICES, DRIFT_EDGE_INDICES,
    COMPETITION_EDGE_INDICES,
)


# ═══════════════════════════════════════════════════════════════
# 0. Static Prototypes (5 × 6, 由 scripts/analysis/build_static_prototypes.py 重算)
# ═══════════════════════════════════════════════════════════════
# B4 拆节点 (4 物理节点) 后按 compute_v_obs 的同一条计算路径重算, 由构造保证
# 与在线特征同一归一化。旧 5×4 表 (3 节点时期 EDA) 有三处问题, 均已修正:
#   1. 前三列之和 0.273/0.911/0.620/0.819/0.806, 而在线的 E_total = Σ 带能量
#      使前三列恒和为 1.0 —— 余弦相似度对"R 块 vs log 比值列"的相对权重敏感,
#      该不一致会系统性扭曲 alpha。
#   2. 旧代码 E_lin = exp(log1p(x)) = 1 + x (应为 expm1), 每个节点被加 +1 地板。
#   3. E_bpf = E_DRIFT + E_LINE 的重新合并抹掉了 B4 拆分最有判别力的特征。
# 列序 = [R_LF, R_DRIFT, R_LINE, R_2x, log10(E_DRIFT/E_LINE), log10((E_DRIFT+E_LINE)/E_2x)]
# 计算口径: train 划分 (val_split=0.15, seed=42 -> N=6575, 已排除验证集),
#   class 0 全量 196 样本, 其余各类抽 400; 逐列取中位数后把 R 块重新归一化到 1。
STATIC_PROTOTYPES = torch.tensor([
    [0.138863, 0.671624, 0.112759, 0.076755,  0.738565,  1.014024],  # Class 0: 空转
    [0.078291, 0.414625, 0.494852, 0.012232, -0.080094,  1.868020],  # Class 1: 低负荷
    [0.098077, 0.209527, 0.118380, 0.574016,  0.181772, -0.205221],  # Class 2: 中负荷
    [0.016501, 0.093141, 0.037490, 0.852869,  0.389718, -0.778678],  # Class 3: 高负荷
    [0.049762, 0.455254, 0.345705, 0.149280,  0.118111,  0.746714],  # Class 4: 抽水
], dtype=torch.float64)

N_PROTOTYPES = STATIC_PROTOTYPES.shape[0]  # 5
PROTOTYPE_DIM = STATIC_PROTOTYPES.shape[1]  # 6


# ═══════════════════════════════════════════════════════════════
# 0b. 在线工况特征向量 V_obs (模块级纯函数)
# ═══════════════════════════════════════════════════════════════
# 做成模块级函数, 是为了让 scripts/analysis/build_static_prototypes.py 能在
# **不实例化模型**的前提下复用同一条计算路径 —— 否则该脚本需要
# StaticPrototypeMatcher, 而后者又要 STATIC_PROTOTYPES 的形状, 形成循环依赖。
# StaticPrototypeMatcher.compute_V_obs 只是本函数的薄委托。
def compute_v_obs(node_energy: torch.Tensor) -> torch.Tensor:
    """
    从节点对数能量算 6 维工况特征向量 V_obs。

    V_obs = [ R_LF, R_DRIFT, R_LINE, R_2x,
              log10(E_DRIFT / E_LINE),            # 漂移/线 判别子
              log10((E_DRIFT + E_LINE) / E_2x) ]  # 谐波比值 (保留旧语义)

    注意 (与 node_energy 的换算):
      node_energy = log1p(band_energy), 故精确逆运算是 **expm1**。
      旧实现用 torch.exp(...), 实际得到 1 + band_energy —— 每个节点都被加了
      +1 地板。对强带可忽略, 对弱带会把能量占比系统性地抬向 1/3 (旧原型
      类 1 的 BPF 族占比 0.0101 即受此影响)。此处改用 expm1。

    注意 (归一化一致性):
      旧冻结原型前三列之和为 0.273/0.911/0.620/0.819/0.806, 而本函数的
      E_total = Σ 带能量 使前三列恒和为 1.0 —— 离线 EDA 与在线代码用了不同
      归一化。余弦相似度**不是**对"前三列整体 vs log 比值列"相对权重不变的,
      故旧 alpha 被系统性扭曲。现已按 4 节点重算原型, 由构造与本节一致。

    Args:
        node_energy: [B, N_phys, T] 节点对数能量 (= log1p(带能量))

    Returns:
        [B, T, 6]
    """
    eps = 1e-12
    # node_energy = log1p(band_energy) -> band_energy = expm1(node_energy)
    E_lin = torch.expm1(node_energy.clamp(max=20.0))  # [B, N, T] ≥ 0

    E_lf = E_lin[:, NODE_LOW_FREQ - 1, :]
    E_drift = E_lin[:, NODE_BPF_DRIFT - 1, :]
    E_line = E_lin[:, NODE_BPF_LINE - 1, :]
    E_2x = E_lin[:, NODE_2XBPF - 1, :]
    E_total = (E_lf + E_drift + E_line + E_2x).clamp(min=eps)

    R_LF = E_lf / E_total                        # [B, T]
    R_DRIFT = E_drift / E_total                  # [B, T]
    R_LINE = E_line / E_total                    # [B, T]
    R_2xBPF = E_2x / E_total                     # [B, T]

    # 漂移/线 判别子 —— B4 拆节点带来的最有判别力的特征。旧实现为对齐
    # 3 节点时期的原型把二者合并回 "BPF 族", 恰好把它抹掉了。
    log_drift_line = torch.log10(
        E_drift.clamp(min=eps) / E_line.clamp(min=eps))
    # 谐波比值: 编码 BPF_LINE → 2xBPF 的整数倍频关系 (r_nom = 2.0)
    log_harm = torch.log10(
        (E_drift + E_line).clamp(min=eps) / E_2x.clamp(min=eps))

    return torch.stack(
        [R_LF, R_DRIFT, R_LINE, R_2xBPF, log_drift_line, log_harm], dim=-1)


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

        # Frozen prototypes (4 节点重算, 见 compute_v_obs): [5, 6]
        self.register_buffer('prototypes', STATIC_PROTOTYPES.clone())

        # Learnable prototype interpretation embeddings: [5, d_cond]
        # 模型学习"像原型 k"如何转化为挤压策略
        self.P_embed = nn.Parameter(torch.randn(N_PROTOTYPES, d_cond) * 0.02)

        # V_obs 投影: PROTOTYPE_DIM (6) -> d_cond (对齐到 prototype 空间)
        # 注: 本模块在 forward 中**未被调用** —— alpha 直接来自 cosine,
        # 故 obs_proj 是 dead code (其参数量随 PROTOTYPE_DIM 变化, 但不影响前向)
        self.obs_proj = nn.Sequential(
            nn.Linear(PROTOTYPE_DIM, d_cond),
            nn.LayerNorm(d_cond),
            nn.GELU(),
        )

    def forward(self, V_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            V_obs: [B, T, 6]  每帧实时工况特征向量
                   [R_LF, R_DRIFT, R_LINE, R_2x,
                    log10(E_DRIFT/E_LINE), log10((E_DRIFT+E_LINE)/E_2x)]

        Returns:
            cond_ctx: [B, T, d_cond]  工况上下文嵌入
            alpha:    [B, T, 5]       原型注意力权重 (诊断用)
        """
        B, T, _ = V_obs.shape
        device = V_obs.device

        # 余弦相似度
        proto = self.prototypes.to(device).to(V_obs.dtype)        # [5, 6]
        proto_norm = F.normalize(proto, dim=-1)                    # [5, 6]
        obs_norm = F.normalize(V_obs, dim=-1)                      # [B, T, 6]

        sim = torch.matmul(obs_norm, proto_norm.T)                 # [B, T, 5]
        alpha = F.softmax(sim / self.temperature, dim=-1)          # [B, T, 5]

        # 加权混合 prototype 解释嵌入
        P_emb = self.P_embed.to(device).to(V_obs.dtype)            # [5, d_cond]
        cond_ctx = torch.matmul(alpha, P_emb)                       # [B, T, d_cond]

        return cond_ctx, alpha

    def compute_V_obs(self, node_energy: torch.Tensor,
                       freqs: torch.Tensor) -> torch.Tensor:
        """薄委托到模块级 compute_v_obs (见其 docstring)。

        freqs 形参保留仅为兼容既有调用方 (sast.py Step 3 / ONNX 导出)。
        """
        del freqs
        return compute_v_obs(node_energy)


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
        self.N_total = N_NODES  # 5 (OP + 4 physical, B4 拆节点后)
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
# 4. RidgeParticipationController (formerly SqueezeIterationController,
#    originally PerBinOrderSelector)
# ═══════════════════════════════════════════════════════════════

class SqueezeIterationController(nn.Module):
    """
    bin 级挤压参与因子 (ridge_factor) 计算.

    ⚠ 语义收窄 (自适应次数改造):
      本类原来还输出节点级**整数**轮数 lambda_i = round(w_i·N_max), 供推理时
      构造 n_sqz_per_bin 硬掩码。那条路径的问题是:
        1. 训练时根本不调用本类 (训练只跑 1 轮), 训推不一致;
        2. round() 不可微, w_i 拿不到来自「轮数」的梯度;
        3. n_sqz_per_bin 结果被 clamp(1, ...) 卡在 ≥1, 推理**永远不可能 0 轮**,
           与设计文档 §18.0 写的「λ=0 (LOW_FREQ) 不挤」直接矛盾。
      现在轮数改由 SparseGaussianReassigner 内的**可微软门控** g_m 承担
      (见那里的注释), 本类只保留空间参与因子 ridge_factor。
    """

    def __init__(self, N_max: int = 5, min_sqz: int = 1):
        super().__init__()
        # N_max / min_sqz 保留仅为兼容旧 checkpoint 的构造调用, 不再参与计算
        self.N_max = N_max
        self.min_sqz = min_sqz

    def forward(self, w_i: torch.Tensor,
                tfr_mag: torch.Tensor, freqs: torch.Tensor,
                node_if: torch.Tensor, bw_expected: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            w_i:         [B, N_phys, T]  IF 可信度 (本函数不再使用, 保留签名)
            tfr_mag:     [B, F, T]  STFT 幅值 (同上)
            freqs:       [F] 频率轴 (Hz)
            node_if:     [B, N_phys, T]  节点 IF (Hz)
            bw_expected: [N_phys] 预期带宽 (Hz)

        Returns:
            ridge_factor: [B, F, T]  逐 bin 空间参与因子 (ridge_floor..1)
        """
        B, N_phys, T = w_i.shape
        F = freqs.shape[0]
        device = w_i.device
        eps = 1e-8

        # ── ridge_factor: bin 级参与因子 ──
        # 基于 bin 频率到节点 IF 的归一化距离 (替代 energy-weighted ridge).
        # 加宽衰减 (sigma=2x bw) + ridge_floor → 弱谐波也能轻度参与挤压.
        freqs_exp = freqs.view(1, 1, F, 1)           # [1, 1, F, 1]
        node_if_exp = node_if.unsqueeze(2)             # [B, N_phys, 1, T]
        dist = (freqs_exp - node_if_exp).abs()         # [B, N_phys, F, T]
        i_star = dist.argmin(dim=1)                     # [B, F, T]

        ridge_factor = torch.zeros(B, F, T, device=device)
        bw_exp = bw_expected.view(1, N_phys, 1).to(device)  # [1, N_phys, 1]

        for n in range(N_phys):
            node_mask = (i_star == n)  # [B, F, T]

            # 脊线 = 节点 IF (GAT 已跟踪, 无需 energy-weighted)
            ridge_pos_f = node_if[:, n, :]  # [B, T]

            # 归一化距离: |freq - IF| / (2 * bw) — 加宽到 2x 预期带宽
            f_all = freqs.view(1, F, 1)  # [1, F, 1]
            ridge_f_exp = ridge_pos_f.unsqueeze(1)  # [B, 1, T]
            ridge_dist = (f_all - ridge_f_exp).abs() / \
                         (2.0 * bw_exp[:, n:n+1, :]).clamp(min=eps)  # [B, F, T]
            # 放宽衰减: sigma=2 → fainter harmonics still get partial participation
            ridge_decay = torch.exp(-ridge_dist ** 2 / 8.0)

            ridge_factor = ridge_factor + node_mask.float() * ridge_decay

        # ridge_floor=0.15: 远端谐波也能参与, 至少 ~15% 每轮
        ridge_factor = ridge_factor.clamp(0.15, 1.0)

        return ridge_factor


# ═══════════════════════════════════════════════════════════════
# 5. SparseGaussianReassigner (replaces AdaptiveSqueeze)
# ═══════════════════════════════════════════════════════════════

# C++ CUDA kernel (可选, 加速 循环; 编译: python deploy/setup_msst_kernels.py build_ext --inplace)
try:
    import deploy.reassigner as _reassigner_cpp
    _HAS_REASSIGNER_CPP = True
except Exception:
    _HAS_REASSIGNER_CPP = False

# 转置布局 [B,T,F] 优化版 (动态 K_eff=ceil(6σ), 比 base 快 2~7x)
try:
    import deploy.reassigner_opt as _reassigner_opt_cpp
    _HAS_REASSIGNER_OPT = True
except Exception:
    _HAS_REASSIGNER_OPT = False


class ReassignerFunction(torch.autograd.Function):
    """可微软高斯重排 (C++ CUDA kernel 加速, Python fallback).

    C++ kernel (deploy/reassigner.cu): 一次 launch, 每 thread 一个 (b,f,t),
    省 Python 91 循环. float32 + CUDA 时用 C++, 否则 Python 循环 fallback.
    """

    @staticmethod
    def forward(ctx, tfr_weighted, sigma, omega_hat_int, K, F_dim):
        B, F, T = tfr_weighted.shape
        tfr_enhanced = torch.zeros(B, F, T, device=tfr_weighted.device, dtype=tfr_weighted.dtype)
        use_cuda = (tfr_weighted.is_cuda and tfr_weighted.dtype == torch.float32
                    and sigma.dtype == torch.float32)
        use_opt = use_cuda and _HAS_REASSIGNER_OPT
        use_base = use_cuda and _HAS_REASSIGNER_CPP and not use_opt
        if use_opt:
            # 转置布局优化核期望 [B,T,F]; 转置 -> 调用 -> 转回 (多 2 次 contiguous 拷贝)
            tw = tfr_weighted.transpose(1, 2).contiguous()
            sg = sigma.transpose(1, 2).contiguous()
            oh = omega_hat_int.transpose(1, 2).contiguous()
            out_bTf = torch.zeros(B, T, F, device=tfr_weighted.device, dtype=torch.float32)
            _reassigner_opt_cpp.forward(tw, sg, oh, out_bTf, K)
            tfr_enhanced = out_bTf.transpose(1, 2).contiguous()
        elif use_base:
            _reassigner_cpp.forward(tfr_weighted, sigma, omega_hat_int, tfr_enhanced, K)
        else:
            dtype = sigma.dtype
            eps = 1e-8
            offsets = list(range(-K, K + 1))
            Z = torch.zeros(B, F, T, device=tfr_weighted.device, dtype=dtype)
            for k in offsets:
                r = k / sigma
                Z = Z + torch.exp(-0.5 * r * r)
            Z = Z + eps
            for k in offsets:
                r = k / sigma
                w_k = torch.exp(-0.5 * r * r) / Z
                target = (omega_hat_int + k).clamp(0, F - 1)
                tfr_enhanced.scatter_add_(1, target, w_k * tfr_weighted)
        ctx.save_for_backward(sigma, tfr_weighted, omega_hat_int)
        ctx.K = K
        ctx.F = F
        ctx.use_opt = use_opt
        ctx.use_base = use_base
        return tfr_enhanced

    @staticmethod
    def backward(ctx, grad_out):
        sigma, tfr_weighted, omega_hat_int = ctx.saved_tensors
        K = ctx.K
        F = ctx.F
        B, _, T = sigma.shape
        if ctx.use_opt:
            go = grad_out.transpose(1, 2).contiguous()
            sg = sigma.transpose(1, 2).contiguous()
            tw = tfr_weighted.transpose(1, 2).contiguous()
            oh = omega_hat_int.transpose(1, 2).contiguous()
            gs_bTf = torch.zeros(B, T, F, device=sigma.device, dtype=torch.float32)
            _reassigner_opt_cpp.backward(go, sg, tw, oh, gs_bTf, K)
            grad_sigma = gs_bTf.transpose(1, 2).contiguous()
            return None, grad_sigma, None, None, None
        if ctx.use_base:
            grad_out = grad_out.contiguous()
            sigma = sigma.contiguous()
            tfr_weighted = tfr_weighted.contiguous()
            omega_hat_int = omega_hat_int.contiguous()
            grad_sigma = torch.zeros_like(sigma)
            _reassigner_cpp.backward(grad_out, sigma, tfr_weighted, omega_hat_int, grad_sigma, K)
            return None, grad_sigma, None, None, None
        # Python fallback
        dtype = sigma.dtype
        eps = 1e-8
        offsets = list(range(-K, K + 1))
        sigma3 = sigma * sigma * sigma
        Z = torch.zeros_like(sigma)
        dZ = torch.zeros_like(sigma)
        for k in offsets:
            r = k / sigma
            e = torch.exp(-0.5 * r * r)
            Z = Z + e
            dZ = dZ + e * float(k * k)
        Z = Z + eps
        dZ = dZ / sigma3
        inv_Z = 1.0 / Z
        grad_sigma = torch.zeros_like(sigma)
        for k in offsets:
            r = k / sigma
            e = torch.exp(-0.5 * r * r)
            w_k = e * inv_Z
            dw_k = w_k * (float(k * k) / sigma3 - dZ * inv_Z)
            target = (omega_hat_int + k).clamp(0, F - 1)
            grad_at = grad_out.gather(1, target)
            grad_sigma = grad_sigma + grad_at * tfr_weighted * dw_k
        return None, grad_sigma, None, None, None


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
                 n_sigma_levels: int = 20, kernel_radius: float = 3.0,
                 N_max: int = 4):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n_sigma_levels = n_sigma_levels
        self.kernel_radius = kernel_radius
        self.N_max = N_max  # 推理多轮挤压最大轮数

    def forward(self, tfr_mag: torch.Tensor, omegas: torch.Tensor,
                sigma_sq: torch.Tensor, freqs: torch.Tensor,
                ridge_factor: Optional[torch.Tensor] = None,
                w_bin: Optional[torch.Tensor] = None,
                M: int = 3, gate_mode: str = 'clamp') -> torch.Tensor:
        """
        连续可微软高斯重排 (通过自定义 autograd Function 省显存).

        可微性: ReassignerFunction 前向/反向都循环重算, 不保存 91 个 [B,F,T]
        中间. sigma_sq 直接进高斯核权重 -> 梯度回流到 w_i.

        ── 自适应挤压轮数 (软门控递推, 训练与推理同一代码路径) ──
        轮数不再由 round(w_i·M) 的**整数掩码**决定 (那条路径训练时不启用、
        且不可微, 见 SqueezeIterationController 的说明), 而由可微软门控:

            r   = clamp(w_i · M, 0, M)           期望轮数 (连续)
            g_m = clamp(r - (m-1), 0, 1)         满轮取 1, 末轮取小数, 之后恒 0
            g_m_bin = gather(g_m, i_star) · ridge_factor

          w_i=0.25, M=3 → r=0.75 → g=(0.75, 0, 0): 不到一整轮, 保留宽峰
          w_i=0.90, M=3 → r=2.70 → g=(1, 1, 0.70): 近乎完整三轮, 极致锐化

        递推为**输出混合**而非对源谱加权:
            S ← (1 - g_m_bin)·S + g_m_bin·Reassign(S)
        理由: ReassignerFunction.backward 对其第一个输入恒返回 None, 源码谱
        拿不到梯度。若写成 S' = Reassign(S⊙g) + S⊙(1-g), autograd 只能看到
        ∂S'/∂g = -S, 而真值是 A_θS - S (A_θ 为列随机重排矩阵) —— 误差恰好
        删掉「能量真聚到目标 bin」那一项, 在重排确实浓缩处会**翻号**。输出
        混合形式下 ∂S'/∂g = R - S, 两者都是前向张量, 直接偏导精确。
        ⚠ 这是**语义变更**而非纯梯度修正: R(S⊙g) ≠ g⊙R(S) (源门控 vs 目标
        门控), 二者在 σ 较大时不等价; 实测差值见 scripts/test 的对照。
        根治方案是给 ReassignerFunction 补 tfr_weighted 的 VJP (scatter 的
        伴随 gather), 但需改三个 CUDA kernel, 已登记为后续项。

        Args:
            tfr_mag:     [B, F, T] STFT 幅值
            omegas:      [B, N_max, F, T] IF 轨迹 (只用 [:, -1])
            sigma_sq:    [B, F, T] 逐 bin 核宽 (bin)
            freqs:       [F] 频率轴 (Hz)
            ridge_factor:[B, F, T] 逐 bin 空间参与因子 (None = 全 1)
            w_bin:       [B, F, T] 逐 bin IF 可信度 (w_i 经 i_star 广播);
                         None 时退化为单轮完整重排 (等价旧训练路径)
            M:           最大轮数
            gate_mode:   'clamp' (推荐, 上述) | 'sigmoid'
        """
        B, F, T = tfr_mag.shape
        # IF 目标位置 (常量, 不参与梯度)
        omega_final = omegas[:, -1, :, :].float()
        omega_hat_int = (omega_final - 1.0).clamp(0, F - 1).round().long()
        # sigma (clamp, 反向自动)
        sigma = sigma_sq.clamp(self.sigma_min, self.sigma_max)
        K = int(np.ceil(self.kernel_radius * self.sigma_max))

        if w_bin is None:
            # 无门控: 单轮完整重排 (保留给探针/消融使用)
            tfr_weighted = tfr_mag * ridge_factor if ridge_factor is not None else tfr_mag
            return ReassignerFunction.apply(tfr_weighted, sigma, omega_hat_int, K, F)

        # ── 软门控递推 (训练与推理同一路径) ──
        ridge = ridge_factor if ridge_factor is not None else 1.0
        tfr_cur = tfr_mag
        for m in range(1, M + 1):
            if gate_mode == 'clamp':
                g_m = (w_bin * M - (m - 1)).clamp(0.0, 1.0)      # [B, F, T]
            elif gate_mode == 'sigmoid':
                tau = 4.0
                g_m = torch.sigmoid(tau * (w_bin * M - m))
            else:
                raise ValueError(f'未知 gate_mode: {gate_mode}')
            part = g_m * ridge
            # 早停: 该轮没有任何 bin 需要参与就不做这次重排 (与旧实现一致)。
            # 未训练模型 w_i≈0.5 时最后一轮恒为 0, 直接省掉 1/3 重排开销。
            if float(part.max()) <= 0.0:
                break
            moved = ReassignerFunction.apply(tfr_cur, sigma, omega_hat_int, K, F)
            tfr_cur = (1.0 - part) * tfr_cur + part * moved
        return tfr_cur


# ═══════════════════════════════════════════════════════════════
# 6. SAST — Top-level Module
# ═══════════════════════════════════════════════════════════════

class SAST(nn.Module):
    """
    Structure-Aware Synchrosqueezing Transform (v3).

    数据流:
      Signal -> MSST (num, save_trajectory) -> NodeFeatures + omegas [B,num,F,T]
      -> V_obs from node energy -> StaticPrototypeMatcher -> cond_ctx, alpha
      -> Per-frame: PPM (OP + phys nodes) -> GAT (11 edges) -> w_i
      -> sigma_i = sigma_min + (1-w_i)*Delta_sigma      控「挤多宽」
      -> 软门控 g_m = clamp(w_i*M - (m-1), 0, 1)         控「挤几轮」
      -> SparseGaussianReassigner(omegas[-1], sigma_i, g_m) -> TFR_sast

    ⚠ 命名与历史 (B4 之后的自适应次数改造):
      - `omegas[k]` **不是** Taylor 相位导数阶, 而是同一个一阶 IF 估计的固定点
        迭代层: omega_{k+1}[eta,b] = omega_k[omega_k[eta,b], b]。历史上曾称为
        「阶/order」, 且 docstring 里写过的 `PerBinOrderSelector -> N_star ->
        omegas[N_star]` **在代码中从未存在**(只活在注释与论文里); 生产路径一直
        只用 omegas[:, -1]。旧的 SqueezeIterationController 输出整数轮数
        lambda_i, 只在推理启用, 训练固定 1 轮 —— 现已由可微软门控取代。
      - 训练与推理现在走**完全同一条**挤压路径 (无 `if not training` 分支)。

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
                 dropout: float = 0.1,
                 n_sqz_max: int = 3,
                 gate_mode: str = 'clamp',
                 train_ridge: bool = True,
                 gate_detach: bool = False,
                 sigma_detach: bool = False):
        super().__init__()
        self.fs = fs
        self.d_h = d_h
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N_max = N_max
        self.msst_num = max(msst_num, N_max)
        # M = 软门控最大轮数 (取代旧的整数 n_sqz_per_bin)
        self.n_sqz_max = n_sqz_max
        # 'clamp' (推荐) | 'sigmoid'; 见 SparseGaussianReassigner.forward
        self.gate_mode = gate_mode
        # 消融开关: False = 训练时不做 ridge 加权 (复现旧的「全 bin 满参与」)
        self.train_ridge = train_ridge
        # 是否把全尺寸的挤压诊断张量 ([B,F,T] 级) 放进返回字典。默认关:
        # 它们在 B=10/T=2000 下各占 76 MB, 训练时白占显存。
        self.emit_squeeze_diag = False
        # ── 消融开关 (回答「门控是新的监督信号, 还是 w_range 的重参数化」) ──
        # gate_detach: 断开门控路径 -> 只剩 sigma 路径 (σ-only)
        # sigma_detach: 断开 sigma 路径 -> 只剩门控路径 (gate-only)
        # 两者都开 = 两条路径都断 (预期 w_i 只剩 w_range/balance 先验)
        self.gate_detach = gate_detach
        self.sigma_detach = sigma_detach

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
        self.sqz_controller = SqueezeIterationController(N_max=N_max, min_sqz=1)

        # ── Sparse Gaussian Reassigner ──
        self.reassigner = SparseGaussianReassigner(
            sigma_min=sigma_min, sigma_max=sigma_max,
            N_max=n_sqz_max,
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
            training:   **已不影响计算路径** (仅为兼容调用方保留)。挤压现在
                        训练与推理走同一条可微软门控递推; 残差只来自
                        reassigner 的 atomicAdd 非确定性。详见
                        SparseGaussianReassigner.forward 的说明。
            return_all: **已废弃 (no-op)** —— 本方法始终返回完整诊断字典

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

            # (edge features 现在用 torch 向量化, 不转 numpy)
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
        if node_if.is_cuda:
            from models.sast_graph import compute_edge_features_torch
            edge_feats_t = compute_edge_features_torch(
                node_if, node_energy, node_persist, node_bw,
                edges=self.edges, window_size=5, fs=self.fs)  # [B, M, T, 5]
            r_obs_t = edge_feats_t[..., 0]  # [B, M, T]
        else:
            edge_feats_list = []
            r_obs_list = []
            for b in range(B):
                ef = compute_edge_features(
                    node_if_np[b], node_energy_np[b], node_persist_np[b],
                    node_bw=node_bw_np[b], edges=self.edges,
                    window_size=5, fs=self.fs,
                )
                edge_feats_list.append(ef['edge_feats'])
                r_obs_list.append(ef['edge_feats'][:, :, 0])
            edge_feats_np = np.stack(edge_feats_list)
            r_obs_np = np.stack(r_obs_list)
            edge_feats_t = torch.from_numpy(edge_feats_np).float().to(device)
            r_obs_t = torch.from_numpy(r_obs_np).float().to(device)

        # ═══════════════════════════════════════════════════════
        # Step 5: Per-frame PPM -> GAT -> w_i
        # ═══════════════════════════════════════════════════════
        # Step 5: 向量化 PPM -> GAT -> w_i (T 维合并到 batch, 消除 per-frame Python 循环)
        N_phys = node_if.shape[1]
        M_edges = edge_feats_t.shape[1]
        d_cond = cond_ctx.shape[-1]
        BT = B * T_msst
        f_norm = node_if / fs_half
        persist_exp = node_persist.unsqueeze(-1).expand(-1, -1, T_msst)
        raw_feats = torch.stack([f_norm, node_energy, node_bw / fs_half, persist_exp], dim=-1)
        raw_feats_bt = raw_feats.permute(0, 2, 1, 3).reshape(BT, N_phys, 4)
        node_if_bt = node_if.permute(0, 2, 1).reshape(BT, N_phys)
        r_obs_bt = r_obs_t.permute(0, 2, 1).reshape(BT, M_edges)
        cond_ctx_bt = cond_ctx.reshape(BT, d_cond)
        drft_bt = edge_feats_t[:, :, :, 0].permute(0, 2, 1).reshape(BT, M_edges)
        h_enhanced, C_prior_t, gate_edge_bt, gate_node_bt, cond_sim_bt = self.ppm(
            raw_feats_bt, node_if_bt, r_obs_bt, cond_ctx_bt, drft_bt, drft_bt)
        h_op_padded = F.pad(h_enhanced[:, :1, :], (0, 1))
        h_phys_cat = torch.cat([h_enhanced[:, 1:, :], C_prior_t.unsqueeze(-1)], dim=-1)
        h_cat = torch.cat([h_op_padded, h_phys_cat], dim=1)
        h_gat_in = self.ppm.gat_input_proj(h_cat)
        edge_feats_bt = edge_feats_t.permute(0, 2, 1, 3).reshape(BT, M_edges, 5)
        for i, m in enumerate(CONDITION_EDGE_INDICES):
            edge_feats_bt[:, m, 0] = cond_sim_bt[:, i]
        w_i_bt, A_ij_bt = self.gat(h_gat_in, edge_feats_bt, self.edge_src, self.edge_dst)
        w_i = w_i_bt.reshape(B, T_msst, N_phys).permute(0, 2, 1)
        A_ij = A_ij_bt.reshape(B, T_msst, M_edges, -1).permute(0, 2, 3, 1)
        gate_edge = gate_edge_bt.reshape(B, T_msst, M_edges).permute(0, 2, 1)
        gate_node = gate_node_bt.reshape(B, T_msst, -1).permute(0, 2, 1)  # [B, N_total, T]

        # ═══════════════════════════════════════════════════════
        # Step 6: w_i -> sigma_i (挤多宽) + 软门控 g_m (挤几轮)
        #   (旧标题写作 "N_star, order_idx" —— 那两项目前都不存在, 见类 docstring)
        # ═══════════════════════════════════════════════════════
        # sigma_i: per-node -> broadcast to per-bin
        # 原线性映射 sigma = sigma_min + (1-w_i)*(sigma_max-sigma_min) 会让保守节点
        # (如 LOW_FREQ, w_i≈0.3) 得到 sigma≈10.7 bin (≈5.3 Hz), 高斯重排把低频
        # 脊线摊到 ±3σ≈±16 Hz, 峰值稀释 ~27x, 低频基线被"摊没"。
        # 治本收紧: 把"软挤压"的有效核宽上限压到 sigma_soft_cap, 使低频脊线
        # 保持软挤压(不锐化)但不再过度展宽; BPF/2xBPF 的 w_i→1 仍逼近 sigma_min。
        sigma_soft_cap = 3.0   # 保守端最大核宽 (bin) ≈ 1.5 Hz
        delta = min(self.sigma_max, sigma_soft_cap) - self.sigma_min
        sigma_i = self.sigma_min + (1.0 - w_i) * delta  # [B, N_phys, T]

        # 逐 bin sigma: 分配到最近物理节点
        freqs_exp = freqs.view(1, 1, F_bins, 1).to(device)
        node_if_exp = node_if.unsqueeze(2)
        dist = (freqs_exp - node_if_exp).abs()
        i_star = dist.argmin(dim=1)  # [B, F_bins, T]

        B_idx_f = torch.arange(B, device=device).view(B, 1, 1).expand(-1, F_bins, T_msst)
        T_idx_f = torch.arange(T_msst, device=device).view(1, 1, T_msst).expand(B, F_bins, -1)
        sigma_sq = sigma_i[B_idx_f, i_star, T_idx_f]  # [B, F_bins, T]

        # ── 挤压参与因子与门控 (训练与推理**同一路径**) ──
        # 旧实现: 推理才算 ridge_factor / n_sqz_per_bin, 训练走单轮全参与分支
        # —— 训推不一致, 且「挤压轮数」在训练中从未获得梯度。现统一。
        bw_expected = torch.tensor([r.bw_expected for r in self.regions],
                                   device=device, dtype=torch.float32)
        ridge_factor = self.sqz_controller(
            w_i, tfr_mag, freqs, node_if, bw_expected)      # [B, F, T]
        if not self.train_ridge:
            ridge_factor = None      # 消融: 回到旧训练行为 (全 bin 满参与)
        # w_i [B, N_phys, T] 经最近节点分配到 [B, F, T], 供软门控使用
        w_bin = w_i[B_idx_f, i_star, T_idx_f]                # [B, F, T]
        if self.gate_detach:          # 消融: 只剩 sigma 路径
            w_bin = w_bin.detach()
        if self.sigma_detach:         # 消融: 只剩门控路径
            sigma_sq = sigma_sq.detach()

        # ═══════════════════════════════════════════════════════
        # Step 7: Sparse Gaussian Reassignment
        # 所有 bin 统一使用 omega_5 (最优 IF); sigma 控「挤多宽」,
        # 可微软门控 g_m 控「挤几轮」(取代旧的整数 n_sqz_per_bin)
        # ═══════════════════════════════════════════════════════
        tfr_enhanced = self.reassigner(
            tfr_mag, omegas, sigma_sq, freqs, ridge_factor,
            w_bin=w_bin, M=self.n_sqz_max, gate_mode=self.gate_mode,
        )

        # ── MSST 硬挤压 (可视化对比基准) ──
        # 与 reassigner 共用 omega_final (omegas[:,-1]), 唯一区别:
        # 硬挤压 (单点 scatter) vs 软高斯重排 (sigma 控宽). 不参与反传.
        with torch.no_grad():
            omega_final_msst = omegas[:, -1, :, :].float()
            valid_msst = omega_final_msst >= 1
            idx_msst = (omega_final_msst - 1.0).clamp(0, F_bins - 1).round().long()
            tfr_msst = torch.zeros_like(tfr_mag)
            tfr_msst.scatter_add_(1, idx_msst, tfr_mag * valid_msst.to(tfr_mag.dtype))

        # ── 输出 ──
        result = {
            'tfr_enhanced': tfr_enhanced,
            'tfr_raw': tfr_mag,
            'tfr_msst': tfr_msst,       # [B, F, T] MSST 硬挤压 (对比基准)
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
        # 挤压控制的诊断量。⚠ 只放**小张量**: 全尺寸的 gate [M,B,F,T] 在
        # B=3/T=2000 下是 228 MB, 每次前向都建会把 8 GB 卡顶到边缘 (实测
        # 7477/8188 MiB, 训练近乎停滞)。这里只报节点级"有效轮数" Σ_m g_m,
        # 由 w_i 在节点空间直接算, 与 bin 空间的门控等价且只有一个向量。
        with torch.no_grad():
            r_node = (w_i * self.n_sqz_max).clamp(0.0, self.n_sqz_max)
            # 逐轮 Σ_m clamp(r-(m-1),0,1) 在该 w_i 下的值 = r 本身 (恒等)
            result['gate_rounds_node'] = r_node.mean(dim=(0, 2))   # [N_phys]
        if self.emit_squeeze_diag:
            result['ridge_factor'] = ridge_factor      # [B,F,T]
            result['w_bin'] = w_bin                    # [B,F,T]
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

    print(f"\nPer-node w_i (time-mean) + 有效挤压轮数:")
    node_names_phys = [r.name for r in PUMP_TURBINE_REGIONS]
    _gr = result.get('gate_rounds_node')
    for i, name in enumerate(node_names_phys):
        _tail = (f"  Σ_m g_m={float(_gr[i]):.2f} (M={model.n_sqz_max})"
                 if _gr is not None else '')
        print(f"  {name:<12s} w_i={w_i[i].mean():.3f} +/- {w_i[i].std():.3f}"
              f"{_tail}")

    print(f"\nPrototype attention (time-mean):")
    proto_names = ['No-load', 'Low load', 'Mid load', 'High load', 'Pumping']
    for k, name in enumerate(proto_names):
        print(f"  Proto {k} ({name:<12s}): alpha={alpha[:, k].mean():.3f}")

    print("\nDone!")
