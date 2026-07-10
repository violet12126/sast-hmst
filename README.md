# SAST-HMST: Structure-Aware Synchrosqueezing with High-Order Multi-Synchrosqueezing

基于高阶多同步压缩变换（HMST）的结构感知时频分析框架，面向**水泵水轮机**故障诊断的可解释自监督表征学习。

## 概述

本项目实现两条互补技术路线：

| 模块 | 论文 | 功能 |
|------|------|------|
| **HMST** | Bao et al. (2023) | N 阶瞬时频率估计 + M 次幅值挤压 → 高精度 TFR |
| **SAST** | 本项目原创 | 物理原型记忆 + 匿名全连接图 + GAT 信任分配 → 自适应挤压 |

### HMST — 高阶瞬时频率估计

- **N=1**: 一阶 IF（标准 SST 级精度）
- **N=2**: 二阶 IF（对线性调频无偏，**实际使用推荐**）
- **N=3**: 三阶 IF（对二次调频/强时变 IF 无偏，**需较大 n_fft**）
- **M 次挤压**: 迭代能量集中，M 越大脊线越锐（CUDA kernel ~52× 加速单次 squeeze）

### SAST — 物理原型引导的自适应挤压

```
Signal → HMST IF → BlindRidgeExtractor → Anonymous Graph
  → PhysicsPrototypeMemory → EdgeConditionedGAT → C_i
  → AdaptiveSqueeze → Physical TFR
```

GAT 只输出 Compressibility Token C_i 控制挤压带宽，不做 IF 修正（避免错位）。

### 物理原型库 — 水泵水轮机频率

基于 **Z_r=9 转轮叶片, Z_s=20 活动导叶, f_r=5.56 Hz**：

| 原型 | 公式 | 频率 (Hz) | 故障类型 | 置信度 |
|------|------|-----------|----------|--------|
| fr | 333.3÷60 | 5.56 | ROTATION | 0.90 |
| RSI_low | 0.43×fr | 2.4 | VORTEX_ROPE | 0.30 |
| RSI_turb | 1.5×fr | 8.35 | TURBULENCE | 0.30 |
| BPF | 9×fr | 50.0 | BLADE_PASS | 1.00 |
| RSI (2×BPF) | ⌊20÷9⌋×BPF | 100.0 | RSI | 1.00 |
| GPF | 20×fr | 111.1 | GUIDE_VANE | 0.95 |
| 3×BPF | 3×BPF | 150.0 | BLADE_HARMONIC | 1.00 |

> RSI = ν × BPF, ν = ⌊Z_s/Z_r⌋ = 2 → 与 2×BPF 同频。五类流态 FFT 验证一致。

---

## 快速开始

### 依赖

```bash
conda create -n sast python=3.10 -y && conda activate sast
pip install torch>=2.1.0 numpy scipy matplotlib scikit-learn ssqueezepy ninja openpyxl
```

Windows 上 CUDA C++ 编译需要 VS BuildTools + CUDA Toolkit，详见 [DEV_SETUP.md](DEV_SETUP.md)。

### 数据集准备

```bash
# 从原始 Excel 提取样本 (data/LTai1~5.xlsx → 5_dataset.npz)
python scripts/extract_dataset.py

# 如需 train/test 划分:
python scripts/extract_dataset.py --split
```

### 绘制时频图

```bash
python plot_hmst_tfr.py
# 输出: hmst_figures/ (7 张图)
#   5×逐类 1×4 (WSST | STFT | HMST N=1 | HMST N=2)
#   + 1×IF 叠加 (N=1 vs N=2)
#   + 1×5×4 全类汇总
```

### 训练 SAST

```bash
python train_sast.py --epochs 20 --lr 0.001 --batch_size 4 --hmst_order 2
# 输出: sast_checkpoints/sast_v2_model.pt
```

### 推理

```bash
# 三面板对比 (STFT vs SAST TFR + C_i)
python infer_sast.py -c sast_checkpoints/sast_v2_model.pt \
    -d 5_dataset.npz --class 1 -o sast_tfr.png

# 六面板全诊断 (含 σ_sq 带宽热力图 + A_ij 因果探针)
python infer_sast.py -c sast_checkpoints/sast_v2_model.pt \
    -d 5_dataset.npz --class 1 --mode full -o sast_full.png

# 批量导出特征 → 下游分类
python infer_sast.py -c sast_checkpoints/sast_v2_model.pt \
    -d 5_dataset.npz --mode batch -o sast_features.npy
```

### Python API

```python
import torch
from models.sast import compute_hmst, compute_hmst_if

# HMST: N=1/2/3, M 可调
x = torch.randn(1, 2000)       # [B, T]
tfr, IF, mag = compute_hmst(x, fs=1000, order=2, M=8)

# SAST
from infer_sast import load_sast, infer_sast
model = load_sast('sast_v2_model.pt')
tfr = infer_sast(model, x)           # 增强 TFR [F, T]
results = model(x, return_all=True)  # 完整诊断量 (C_i, A_ij, gate, sigma_i)
```

### CUDA 加速

编译一次，之后任意终端免环境直接用：

```bash
# 首次: 在 MSVC/vcvars 环境中编译 (Windows) 或 gcc+nvcc (Linux/Jetson)
# Windows: 见 DEV_SETUP.md 的 MSVC 环境配置
python deploy/compile_hmst_cuda.py          # → JIT 缓存
cp $TORCH_EXTENSIONS_CACHE/.../hmst_cuda_ext.pyd deploy/   # → deploy/

# 之后: 任意终端直接运行, 自动加载 deploy/ 下的预编译扩展
python plot_hmst_tfr.py   # → "CUDA squeeze kernel: 已加载 ✓"
```

| 方案 | squeeze 单次 | 完整 HMST (N=2, M=8) | 5 类绘图 |
|------|-------------|----------------------|----------|
| GPU + CUDA kernel | 0.17 ms | 23 ms | ~230 ms |
| CPU 全程 | 619 ms | 849 ms | ~8.5 s |
| 加速比 | **~3,700×** | **~37×** | **~37×** |

> 性能数据基于 GTX 1650。Jetson Orin 部署: `cd deploy && TORCH_CUDA_ARCH_LIST="8.7" python setup_hmst.py`

---

## 文件结构

```
sast-hmst/
├── models/
│   ├── sast.py              # HMST (N=1/2/3) + SAST 完整实现
│   └── sast_losses.py       # SAST 损失函数
├── deploy/
│   ├── hmst_squeeze.cu       # CUDA squeeze kernel 源码
│   ├── compile_hmst_cuda.py  # JIT 编译脚本
│   └── setup_hmst.py         # setuptools 预编译 (Jetson)
├── scripts/
│   └── extract_dataset.py    # Excel → NPZ 数据集提取
├── plot_hmst_tfr.py          # WSST vs STFT vs HMST 可视化
├── train_sast.py             # SAST 训练脚本
├── infer_sast.py             # SAST 推理 + 可视化
├── DEV_SETUP.md              # 开发环境详细配置 (CUDA/MSVC/编码问题)
├── papers/
│   ├── HMST_paper.md         # Bao et al. 2023 论文
│   ├── SAST_v2_design.md     # SAST v2 设计文档
│   └── 发展脉络_HMST_to_SAST.md
└── docs/
    ├── cuda_deploy_plan.md
    ├── architecture.md
    ├── parameters.md
    └── visualization.md
```

## 引用

- Bao, W. et al. "Application of High-Order Multisynchrosqueezing Transform in Fault Diagnosis." *IEEE Trans. Instrum. Meas.*, 2023.
- Daubechies, I., Lu, J., & Wu, H.-T. "Synchrosqueezed wavelet transforms." *Appl. Comput. Harmon. Anal.*, 2011.
- Veličković, P. et al. "Graph Attention Networks." *ICLR*, 2018.
