"""
SAST Physics Graph — 异构物理图 (工况上下文 + 倍频 + 能量耦合)
============================================================

设计依据:
  - 实测数据频谱分析 (5 类, 9669 样本, fs=1000 Hz)
  - Colominas & Meignen (2025) 自适应阶数 SST 框架
  - 用户指定图结构: OP Token → 3 频率节点 + DRIFT + HARMONIC

图结构:
                    ┌─────────────────────┐
                    │   Operating Token    │  ← 虚拟节点 (NODE_OP = 0)
                    │   工况上下文嵌入       │
                    └──────┬──┬──┬─────────┘
              CONDITION   │  │  │  CONDITION
              (w=0.6)     │  │  │  (w=0.6)
                           ▼  ▼  ▼
          ┌──────────┐  ┌──────────┐  ┌──────────┐
          │ LOW_FREQ │  │   BPF    │  │  2×BPF   │
          │  (idx=1) │  │  (idx=2) │  │  (idx=3) │
          └────┬─────┘  └──┬───┬───┘  └──────────┘
               │           │   │            ▲
               │    DRIFT  │   └────────────┘
               │   (w=0.15)│   HARMONIC (r=2.0, w=0.8)
               │           ▼
               └──────────►┘
             [能量耦合: Corr_E, E_ratio, bw_coupling]

节点 (4 个):
  [0] OP:        虚拟节点, 来自 StaticPrototypeMatcher 的 soft matching 嵌入
  [1] LOW_FREQ:  水力分量 (涡带/压力脉动), 2-25 Hz
  [2] BPF:       叶片通过频率, 42-55 Hz
  [3] 2×BPF:     二倍叶片通过频率, 90-105 Hz

边 (6 条有向):
  OP → LOW_FREQ  : CONDITION, w=0.6
  OP → BPF       : CONDITION, w=0.6
  OP → 2×BPF     : CONDITION, w=0.6
  LOW_FREQ → BPF : DRIFT, w=0.15  (涡带能量 <-> BPF 调制深度)
  BPF → LOW_FREQ : DRIFT, w=0.15  (反向: 调制反馈)
  BPF → 2×BPF    : HARMONIC, r=2.0, w=0.8
"""

import numpy as np
from typing import Dict, List, NamedTuple, Optional
from enum import Enum
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════
# Edge Types
# ═══════════════════════════════════════════════════════════════

class EdgeType(Enum):
    """物理边类型 — 决定约束强度和特征计算方式。"""
    INTEGER_HARMONIC = 'INTEGER_HARMONIC'   # 整数倍频 (BPF → 2×BPF, r=2.0)
    CONDITION = 'CONDITION'                 # 工况上下文广播 (OP → 频率节点)
    DRIFT = 'DRIFT'                         # 能量耦合 / 调制关系 (非倍频, 正向共变)
    ENERGY_COMPETITION = 'ENERGY_COMPETITION'  # 能量竞争 (此消彼长, 负相关)


@dataclass
class PhysicsEdge:
    """物理图中的一条有向边。"""
    src: int           # 源节点索引
    dst: int           # 目标节点索引
    edge_type: EdgeType
    r_nom: float       # 标称比值 f_dst / f_src (CONDITION/DRIFT 边为 0.0)
    w_type: float      # 约束权重 (0-1)
    description: str   # 人类可读描述


# ═══════════════════════════════════════════════════════════════
# Node Indices (fixed)
# ═══════════════════════════════════════════════════════════════

NODE_OP = 0          # 虚拟节点: 工况上下文
NODE_LOW_FREQ = 1
NODE_BPF = 2
NODE_2XBPF = 3

NODE_NAMES = ['OP', 'LOW_FREQ', 'BPF', '2xBPF']
N_NODES = len(NODE_NAMES)

# 物理节点 (不含 OP)
PHYSICS_NODE_INDICES = [NODE_LOW_FREQ, NODE_BPF, NODE_2XBPF]
N_PHYSICS_NODES = len(PHYSICS_NODE_INDICES)


# ═══════════════════════════════════════════════════════════════
# Physics Graph Definition
# ═══════════════════════════════════════════════════════════════

PHYSICS_EDGES: List[PhysicsEdge] = [
    # ── CONDITION: 工况上下文广播 (OP → 频率节点) ──
    PhysicsEdge(NODE_OP, NODE_LOW_FREQ, EdgeType.CONDITION,
                0.0, 0.6, 'OP→LOW_FREQ: 工况决定低频强弱'),
    PhysicsEdge(NODE_OP, NODE_BPF, EdgeType.CONDITION,
                0.0, 0.6, 'OP→BPF: 工况决定 BPF 主导程度'),
    PhysicsEdge(NODE_OP, NODE_2XBPF, EdgeType.CONDITION,
                0.0, 0.6, 'OP→2xBPF: 工况决定 RSI 强弱'),

    # ── DRIFT: 能量耦合 / 调制关系 (LOW_FREQ <-> BPF) ──
    # 涡带增强 -> 流道紊乱 -> BPF 被调制展宽; 两者能量共变
    PhysicsEdge(NODE_LOW_FREQ, NODE_BPF, EdgeType.DRIFT,
                0.0, 0.15, 'LOW_FREQ→BPF: 涡带能量→BPF调制'),
    PhysicsEdge(NODE_BPF, NODE_LOW_FREQ, EdgeType.DRIFT,
                0.0, 0.15, 'BPF→LOW_FREQ: 调制反馈'),

    # ── HARMONIC: 唯一确定性倍频 (BPF → 2×BPF, r=2.0) ──
    PhysicsEdge(NODE_BPF, NODE_2XBPF, EdgeType.INTEGER_HARMONIC,
                2.0, 0.8, '2xBPF = 2×BPF'),

    # ── ENERGY_COMPETITION: 此消彼长 — 流道能量守恒 ──
    # 低负荷: 涡带主导 → LOW_FREQ↑, BPF↑, 2×BPF↓
    # 高负荷: 动静干涉主导 → LOW_FREQ↓, BPF↓, 2×BPF↑
    # 边特征: [-Corr_E, E_ratio_campAB, E_ratio_stability, w_type, p_min]
    PhysicsEdge(NODE_LOW_FREQ, NODE_2XBPF, EdgeType.ENERGY_COMPETITION,
                0.0, 0.25, 'LOW_FREQ vs 2xBPF: 涡带↔RSI 此消彼长'),
    PhysicsEdge(NODE_2XBPF, NODE_LOW_FREQ, EdgeType.ENERGY_COMPETITION,
                0.0, 0.25, '2xBPF vs LOW_FREQ: 反向'),
    PhysicsEdge(NODE_BPF, NODE_2XBPF, EdgeType.ENERGY_COMPETITION,
                0.0, 0.20, 'BPF vs 2xBPF: BPF↔RSI 此消彼长'),
    PhysicsEdge(NODE_2XBPF, NODE_BPF, EdgeType.ENERGY_COMPETITION,
                0.0, 0.20, '2xBPF vs BPF: 反向'),
]

N_EDGES = len(PHYSICS_EDGES)

# 边类型分组 (用于特征计算)
HARMONIC_EDGE_INDICES = [i for i, e in enumerate(PHYSICS_EDGES)
                         if e.edge_type == EdgeType.INTEGER_HARMONIC]
CONDITION_EDGE_INDICES = [i for i, e in enumerate(PHYSICS_EDGES)
                          if e.edge_type == EdgeType.CONDITION]
DRIFT_EDGE_INDICES = [i for i, e in enumerate(PHYSICS_EDGES)
                      if e.edge_type == EdgeType.DRIFT]
COMPETITION_EDGE_INDICES = [i for i, e in enumerate(PHYSICS_EDGES)
                            if e.edge_type == EdgeType.ENERGY_COMPETITION]


# ═══════════════════════════════════════════════════════════════
# Edge Feature Computation
# ═══════════════════════════════════════════════════════════════

def compute_edge_features(node_if: np.ndarray,
                          node_energy: np.ndarray,
                          node_persistence: np.ndarray,
                          node_bw: Optional[np.ndarray] = None,
                          cond_ctx: Optional[np.ndarray] = None,
                          edges: Optional[List[PhysicsEdge]] = None,
                          window_size: int = 5,
                          fs: float = 1000.0) -> Dict:
    """
    从节点特征计算物理图的边特征。

    边特征维度: 统一为 d_e=5, 不同类型边用不同的前几维语义:

    HARMONIC edges:
      e_ij(t) = [r_obs, σ_r, Corr_E, w_type, persistence_min]
      语义:    比值   稳定性 能量相关  边权重  最小持续性

    DRIFT edges:
      e_ij(t) = [Corr_E, E_ratio, bw_coupling, w_type, persistence_min]
      语义:    能量相关  能量比  带宽耦合    边权重  最小持续性

    CONDITION edges:
      e_ij(t) = [cond_sim, 0, 0, w_type, 0]
      语义:     cond_ctx 与节点特征的匹配度

    Args:
        node_if:          [N_phys, T] 节点 IF (Hz) — 不含 OP
        node_energy:      [N_phys, T] 节点对数能量
        node_persistence: [N_phys] 节点持续性 (0-1)
        node_bw:          [N_phys, T] 节点带宽 (Hz)
        cond_ctx:         [d_cond] 或 [T, d_cond] 工况上下文嵌入
        edges:            边定义列表
        window_size:      局部统计窗口半径 (帧数)
        fs:               采样率 (保留参数, 未使用)

    Returns:
        dict:
          edge_src:    [M] 源节点索引
          edge_dst:    [M] 目标节点索引
          r_nom:       [M] 标称比值
          w_type:      [M] 边类型权重
          edge_feats:  [M, T, 5] 边特征时间序列
    """
    if edges is None:
        edges = PHYSICS_EDGES

    N_phys, T = node_if.shape
    M = len(edges)
    eps = 1e-8

    edge_src = np.array([e.src for e in edges], dtype=np.int64)
    edge_dst = np.array([e.dst for e in edges], dtype=np.int64)
    r_nom = np.array([e.r_nom for e in edges], dtype=np.float64)
    w_type = np.array([e.w_type for e in edges], dtype=np.float64)

    # 预分配 edge_feats
    edge_feats = np.zeros((M, T, 5), dtype=np.float64)

    # ── HARMONIC edges: 比值特征 ──
    for m in HARMONIC_EDGE_INDICES:
        e = edges[m]
        src_phys = e.src - 1  # OP=0, 物理节点从 1 开始 → 减 1 得 0-indexed
        dst_phys = e.dst - 1

        f_src = node_if[src_phys, :]
        f_dst = node_if[dst_phys, :]
        r_obs = f_dst / np.maximum(f_src, eps)
        r_std = _running_std(r_obs.reshape(1, -1), window_size).ravel()
        e_src = node_energy[src_phys, :]
        e_dst = node_energy[dst_phys, :]
        corr_e = _running_pearson(e_src.reshape(1, -1), e_dst.reshape(1, -1),
                                  window_size).ravel()
        p_min = min(node_persistence[src_phys], node_persistence[dst_phys])

        edge_feats[m, :, 0] = r_obs
        edge_feats[m, :, 1] = r_std
        edge_feats[m, :, 2] = corr_e
        edge_feats[m, :, 3] = w_type[m]
        edge_feats[m, :, 4] = p_min

    # ── DRIFT edges: 能量耦合特征 ──
    for m in DRIFT_EDGE_INDICES:
        e = edges[m]
        src_phys = e.src - 1
        dst_phys = e.dst - 1

        e_src = node_energy[src_phys, :]
        e_dst = node_energy[dst_phys, :]
        corr_e = _running_pearson(e_src.reshape(1, -1), e_dst.reshape(1, -1),
                                  window_size).ravel()
        # 能量比 (对数域): log(E_src / E_dst)
        e_ratio = np.log(np.maximum(e_src, eps) / np.maximum(e_dst, eps))
        # 带宽耦合: Corr(bw_src, bw_dst)
        bw_coupling = 0.0
        if node_bw is not None:
            bw_src = node_bw[src_phys, :]
            bw_dst = node_bw[dst_phys, :]
            bw_coupling = _running_pearson(bw_src.reshape(1, -1),
                                           bw_dst.reshape(1, -1),
                                           window_size).ravel()

        p_min = min(node_persistence[src_phys], node_persistence[dst_phys])

        edge_feats[m, :, 0] = corr_e
        edge_feats[m, :, 1] = e_ratio
        edge_feats[m, :, 2] = bw_coupling if isinstance(bw_coupling, np.ndarray) else np.full(T, bw_coupling)
        edge_feats[m, :, 3] = w_type[m]
        edge_feats[m, :, 4] = p_min

    # ── ENERGY_COMPETITION edges: 此消彼长 (负能量相关) ──
    for m in COMPETITION_EDGE_INDICES:
        e = edges[m]
        src_phys = e.src - 1
        dst_phys = e.dst - 1

        e_src = node_energy[src_phys, :]
        e_dst = node_energy[dst_phys, :]
        # 负能量相关: Corr_E 越负 → 竞争关系越强 → 边越"可信"
        corr_e = _running_pearson(e_src.reshape(1, -1), e_dst.reshape(1, -1),
                                  window_size).ravel()
        # 阵营能量比: log(E_campA / E_campB) — 量化"谁占主导"
        e_ratio = np.log(np.maximum(e_src, eps) / np.maximum(e_dst, eps))
        # 阵营能量比的窗口稳定性 (样本内工况固定, 稳态→稳定, 瞬态→波动)
        e_ratio_std = _running_std(e_ratio.reshape(1, -1), window_size).ravel()
        e_ratio_stability = np.exp(-e_ratio_std / 0.5)  # [0,1], 1=非常稳定
        p_min = min(node_persistence[src_phys], node_persistence[dst_phys])

        # 注意: 竞争边用 -corr_e 作为主特征 (正相关=坏, 负相关=好)
        edge_feats[m, :, 0] = -corr_e            # 负相关的程度 (越大越"此消彼长")
        edge_feats[m, :, 1] = e_ratio             # 当前阵营能量比
        edge_feats[m, :, 2] = e_ratio_stability   # 阵营能量比稳定性
        edge_feats[m, :, 3] = w_type[m]
        edge_feats[m, :, 4] = p_min

    # ── CONDITION edges: 上下文匹配特征 ──
    # OP 节点特征由 cond_ctx 提供, 频率节点由观测特征提供
    # cond_sim = cosine_sim(cond_ctx, node_feat_projection)
    # 此处留为占位, 实际由 PPM 在 forward 时填充
    for m in CONDITION_EDGE_INDICES:
        edge_feats[m, :, 3] = w_type[m]
        # dim 0: cond_sim (由调用方在训练循环中填充)
        # dim 1-2: 0 (reserved)
        # dim 4: 0 (reserved)

    return {
        'edge_src': edge_src,
        'edge_dst': edge_dst,
        'r_nom': r_nom,
        'w_type': w_type,
        'edge_feats': edge_feats,
        'N_nodes': N_NODES,
        'N_edges': M,
    }


# ═══════════════════════════════════════════════════════════════
# Running statistics helpers
# ═══════════════════════════════════════════════════════════════

def _running_std(x: np.ndarray, window_size: int) -> np.ndarray:
    """沿 axis=1 计算运行窗口标准差。"""
    M, T = x.shape
    if T < 2 * window_size + 1:
        result = np.std(x, axis=1, keepdims=True)
        return np.tile(result, (1, T))

    result = np.zeros((M, T), dtype=np.float64)
    w = 2 * window_size + 1
    x_pad = np.pad(x, ((0, 0), (window_size, window_size)), mode='edge')
    for t in range(T):
        window = x_pad[:, t:t + w]
        result[:, t] = np.std(window, axis=1)
    return result


def _running_pearson(x: np.ndarray, y: np.ndarray, window_size: int) -> np.ndarray:
    """沿 axis=1 计算运行窗口 Pearson 相关系数。"""
    M, T = x.shape
    eps = 1e-8
    if T < 2 * window_size + 1:
        return np.zeros((M, T), dtype=np.float64)

    result = np.zeros((M, T), dtype=np.float64)
    w = 2 * window_size + 1
    x_pad = np.pad(x, ((0, 0), (window_size, window_size)), mode='edge')
    y_pad = np.pad(y, ((0, 0), (window_size, window_size)), mode='edge')
    for t in range(T):
        xw = x_pad[:, t:t + w]
        yw = y_pad[:, t:t + w]
        xc = xw - xw.mean(axis=1, keepdims=True)
        yc = yw - yw.mean(axis=1, keepdims=True)
        cov = (xc * yc).sum(axis=1)
        denom = np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))
        result[:, t] = np.clip(cov / np.maximum(denom, eps), -1.0, 1.0)
    return result


# ═══════════════════════════════════════════════════════════════
# Graph summary (for diagnostics)
# ═══════════════════════════════════════════════════════════════

def get_graph_summary() -> str:
    """返回物理图的人类可读摘要。"""
    lines = [
        "Physics Graph: Heterogeneous Pump-Turbine Structure",
        "=" * 60,
        f"Nodes ({N_NODES}):",
    ]
    for i, name in enumerate(NODE_NAMES):
        tag = "(virtual)" if i == NODE_OP else "(physical)"
        lines.append(f"  [{i}] {name} {tag}")
    lines.append(f"\nEdges ({N_EDGES}):")
    for e in PHYSICS_EDGES:
        src_name = NODE_NAMES[e.src]
        dst_name = NODE_NAMES[e.dst]
        r_str = f"r={e.r_nom:.1f}" if e.r_nom > 0 else ""
        lines.append(
            f"  {src_name:>12s} → {dst_name:<12s}  "
            f"{r_str:>8s}  w={e.w_type:.2f}  "
            f"[{e.edge_type.value}]"
        )
    lines.append("\nDesign notes:")
    lines.append("  - OP: virtual node, soft-matched from 5 static prototypes (EDA)")
    lines.append("  - CONDITION edges: broadcast operating context to frequency nodes")
    lines.append("  - DRIFT edges: energy coupling (vortex <-> BPF modulation), non-harmonic")
    lines.append("  - HARMONIC edge: BPF → 2×BPF (r=2.0), sole deterministic relationship")
    lines.append("  - No self-loops; no classifier-based condition prediction")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(get_graph_summary())

    # Smoke test: random node features → edge features
    np.random.seed(42)
    T = 100
    N_phys = N_PHYSICS_NODES
    node_if = np.random.randn(N_phys, T) * 5 + np.array([
        [12.0], [48.0], [96.0]
    ])
    node_energy = np.abs(np.random.randn(N_phys, T))
    node_persistence = np.random.rand(N_phys)
    node_bw = np.abs(np.random.randn(N_phys, T)) * 5 + np.array([
        [10.0], [12.0], [2.0]
    ])
    cond_ctx = np.random.randn(32)

    result = compute_edge_features(node_if, node_energy, node_persistence,
                                   node_bw=node_bw, cond_ctx=cond_ctx)
    print(f"\nEdge features shape: {result['edge_feats'].shape}")
    print(f"  Expected: ({N_EDGES}, {T}, 5)")
    assert result['edge_feats'].shape == (N_EDGES, T, 5)
    print("  OK!")

    # Check edge type counts
    from collections import Counter
    type_counts = Counter(e.edge_type for e in PHYSICS_EDGES)
    print(f"\nEdge type counts: {dict(type_counts)}")
    print(f"  CONDITION: {type_counts[EdgeType.CONDITION]} (expected 3)")
    print(f"  DRIFT:     {type_counts[EdgeType.DRIFT]} (expected 2)")
    print(f"  HARMONIC:  {type_counts[EdgeType.INTEGER_HARMONIC]} (expected 1)")
    print(f"  COMPETITION: {type_counts[EdgeType.ENERGY_COMPETITION]} (expected 4)")


# ═══════════════════════════════════════════════════════════════
# torch 向量化版 (GPU, 替代 numpy compute_edge_features 的 per-batch 循环)
# ═══════════════════════════════════════════════════════════════
import torch
import torch.nn.functional as F


def _running_std_torch(x, window):
    """x [..., T] -> [..., T] 滑动窗口 std (匹配 numpy _running_std, ddof=0)."""
    *shape, T = x.shape
    W = 2 * window + 1
    if T < W:
        s = x.std(dim=-1, keepdim=True, unbiased=False)
        return s.expand(*shape, T)
    x_pad = F.pad(x, (window, window), mode='replicate')
    x_unf = x_pad.unfold(-1, W, 1)              # [..., T, W]
    return x_unf.std(dim=-1, unbiased=False)    # [..., T]


def _running_pearson_torch(x, y, window):
    """x, y [..., T] -> [..., T] 滑动窗口 Pearson (匹配 numpy _running_pearson)."""
    *shape, T = x.shape
    W = 2 * window + 1
    if T < W:
        return torch.zeros(*shape, T, device=x.device, dtype=x.dtype)
    x_pad = F.pad(x, (window, window), mode='replicate')
    y_pad = F.pad(y, (window, window), mode='replicate')
    x_unf = x_pad.unfold(-1, W, 1)              # [..., T, W]
    y_unf = y_pad.unfold(-1, W, 1)
    xm = x_unf - x_unf.mean(dim=-1, keepdim=True)
    ym = y_unf - y_unf.mean(dim=-1, keepdim=True)
    cov = (xm * ym).sum(dim=-1)
    denom = torch.sqrt((xm ** 2).sum(dim=-1) * (ym ** 2).sum(dim=-1))
    corr = cov / denom.clamp(min=1e-8)
    return corr.clamp(-1.0, 1.0)                # [..., T] (匹配 numpy clip)


def compute_edge_features_torch(node_if, node_energy, node_persist, node_bw,
                                 edges=None, window_size=5, fs=1000.0):
    """
    torch 向量化 compute_edge_features (GPU, 一次算所有 B).

    Args:
        node_if:        [B, N, T] GPU
        node_energy:    [B, N, T]
        node_persist:   [B, N]
        node_bw:        [B, N, T]
        edges:          物理边列表
        window_size:    滑动窗口半径
        fs:             采样率

    Returns:
        edge_feats: [B, M, T, 5] GPU (dim0 语义按边类型, 同 numpy 版)
    """
    if edges is None:
        edges = PHYSICS_EDGES
    B, N, T = node_if.shape
    M = len(edges)
    device = node_if.device
    dtype = node_if.dtype
    eps = 1e-8

    edge_feats = torch.zeros(B, M, T, 5, device=device, dtype=dtype)

    for m, e in enumerate(edges):
        src = e.src - 1   # 0-indexed (OP=0, phys 1..3 -> 0..2)
        dst = e.dst - 1
        f_src = node_if[:, src, :]
        f_dst = node_if[:, dst, :]
        e_src = node_energy[:, src, :]
        e_dst = node_energy[:, dst, :]
        p_min = torch.min(node_persist[:, src], node_persist[:, dst])
        w_t = torch.tensor(float(e.w_type), device=device, dtype=dtype)

        if e.edge_type == EdgeType.INTEGER_HARMONIC:
            r_obs = f_dst / f_src.clamp(min=eps)
            edge_feats[:, m, :, 0] = r_obs
            edge_feats[:, m, :, 1] = _running_std_torch(r_obs, window_size)
            edge_feats[:, m, :, 2] = _running_pearson_torch(e_src, e_dst, window_size)
            edge_feats[:, m, :, 3] = w_t
            edge_feats[:, m, :, 4] = p_min.unsqueeze(-1)
        elif e.edge_type == EdgeType.DRIFT:
            edge_feats[:, m, :, 0] = _running_pearson_torch(e_src, e_dst, window_size)
            edge_feats[:, m, :, 1] = torch.log(e_src.clamp(min=eps) / e_dst.clamp(min=eps))
            edge_feats[:, m, :, 2] = _running_pearson_torch(
                node_bw[:, src, :], node_bw[:, dst, :], window_size)
            edge_feats[:, m, :, 3] = w_t
            edge_feats[:, m, :, 4] = p_min.unsqueeze(-1)
        elif e.edge_type == EdgeType.ENERGY_COMPETITION:
            corr_e = _running_pearson_torch(e_src, e_dst, window_size)
            e_ratio = torch.log(e_src.clamp(min=eps) / e_dst.clamp(min=eps))
            e_ratio_std = _running_std_torch(e_ratio, window_size)
            edge_feats[:, m, :, 0] = -corr_e
            edge_feats[:, m, :, 1] = e_ratio
            edge_feats[:, m, :, 2] = torch.exp(-e_ratio_std / 0.5)
            edge_feats[:, m, :, 3] = w_t
            edge_feats[:, m, :, 4] = p_min.unsqueeze(-1)
        elif e.edge_type == EdgeType.CONDITION:
            edge_feats[:, m, :, 3] = w_t
            # dim 0 (cond_sim) 由 SAST.forward Step5 填充

    return edge_feats

