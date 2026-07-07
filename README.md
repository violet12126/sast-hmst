# SAST-HMST: Structure-Aware Synchrosqueezing with High-Order Multi-Synchrosqueezing

基于高阶多同步压缩变换（HMST）的结构感知时频分析框架，面向旋转机械故障诊断的可解释自监督表征学习。

## 概述

本项目实现了两条互补的技术路线：

| 模块 | 论文 | 功能 |
|------|------|------|
| **HMST** | Bao et al. (2023) "High-Order Multisynchrosqueezing Transform" | N 阶瞬时频率估计 + M 次幅值挤压 → 高精度 TFR |
| **SAST** | 本项目原创 | 物理原型记忆 + 匿名全连接图 + GAT 信任分配 → 自适应挤压 |

### HMST (High-Order Multi-Synchrosqueezing Transform)

- **N=1**: 一阶 IF（标准 SST 级精度）
- **N=2**: 二阶 IF（通过 2×2 上三角矩阵求解，对线性调频无偏）
- **M 次挤压**: 迭代能量集中，sparsity 从 0% → 73%
- **幅值累加**: 避免 PyTorch 非调制 STFT 的相位不一致问题（与 ssqueezepy 验证 corr=0.9999）

### SAST (Structure-Aware Synchrosqueezing Transform)

```
Signal → HMST IF → BlindRidgeExtractor → Anonymous Graph
  → PhysicsPrototypeMemory → EdgeConditionedGAT → C_i
  → AdaptiveSqueeze → Physical TFR
```

GAT 不做 IF 修正（避免错位），只输出 Compressibility Token C_i 控制挤压带宽。



## 脚本说明

### 1. 绘制 HMST 时频图

```bash
# 需要 5_dataset.npz 在根目录
python plot_hmst_tfr.py
# 输出: hmst_figures/ (7 张图: 5×逐类 + IF 叠加 + 汇总)
```

### 2. 训练 SAST

```bash
python train_sast.py --epochs 20 --lr 0.001 --batch_size 4 --hmst_order 2
# 输出: sast_checkpoints/sast_v2_model.pt
```

### 3. 推理：用训练好的 SAST 绘制 TFR

```bash
# 三面板对比图 (STFT vs SAST TFR + C_i 决策曲线)
python infer_sast.py -c sast_checkpoints/sast_v2_model.pt \
    -d 5_dataset.npz --class 1 -o sast_tfr.png

# 六面板全诊断 (含 σ_sq 带宽热力图 + A_ij 因果推理探针)
python infer_sast.py -c sast_checkpoints/sast_v2_model.pt \
    -d 5_dataset.npz --class 1 --mode full -o sast_full.png

# 批量导出增强特征 → 下游分类/聚类
python infer_sast.py -c sast_checkpoints/sast_v2_model.pt \
    -d 5_dataset.npz --mode batch -o sast_features.npy
```

### 4. Python API 调用

```python
import torch
from models.sast import compute_hmst, compute_hmst_if, SAST
from infer_sast import load_sast, infer_sast

# --- HMST: 时频分析 ---
x = torch.randn(1, 2000)  # [B, T]
tfr, IF, mag = compute_hmst(x, fs=1000, order=2, M=2)

# --- SAST: 加载训练好的模型 + 推理 ---
model = load_sast('sast_v2_model.pt')
tfr = infer_sast(model, x)         # 仅增强 TFR [F, T]
results = model(x, return_all=True) # 完整诊断量
# results['C_i']    — Compressibility Token (每条脊线的可信度)
# results['A_ij']   — Edge Attention (因果推理诊断探针)
# results['gate']   — 原型匹配门控
# results['sigma_i']— 逐脊线挤压带宽
```

## 文件结构

```
sast-hmst/
├── models/
│   ├── sast.py           # HMST + SAST 完整实现 (~1500 行)
│   └── sast_losses.py    # SAST 损失函数
├── plot_hmst_tfr.py      # WSST vs HMST N=1 vs N=2 可视化
├── train_sast.py         # SAST 训练脚本
├── infer_sast.py         # SAST 推理 + 可视化 (训练后使用)
├── docs/
│   ├── cuda_deploy_plan.md   # CUDA 部署方案
│   ├── architecture.md       # 架构总览
│   ├── parameters.md         # 参数参考
│   └── visualization.md      # 可视化说明
└── papers/
    ├── HMST_paper.md         # Bao et al. 2023 论文笔记
    ├── SAST_v2_design.md     # SAST v2 设计文档
    └── 发展脉络_HMST_to_SAST.md  # 发展脉络
```

## 引用

- Bao, W. et al. "Application of High-Order Multisynchrosqueezing Transform in Fault Diagnosis of Pump-Turbines." *Measurement*, 2023.
- Daubechies, I., Lu, J., & Wu, H.-T. "Synchrosqueezed wavelet transforms: An empirical mode decomposition-like tool." *Applied and Computational Harmonic Analysis*, 2011.
- Veličković, P. et al. "Graph Attention Networks." *ICLR*, 2018.


