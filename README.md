# SAST: Structure-Aware Synchrosqueezing Transform

基于结构感知同步压缩变换（SAST）的自适应时频分析框架，面向**水泵水轮机**故障诊断。

## 项目结构

```
sast-hmst/
├── models/                    # 核心模型
│   ├── tfr.py                # MSST, WSST2, SST2 等 TFR 基座 + CUDA STFT wrapper
│   ├── sast_nodes.py         # MSSTNodeExtractor — 从 omega_final 提取节点特征
│   ├── sast_graph.py         # 异构物理图 (10 边, 4 类型) + 边特征计算
│   ├── sast.py               # SAST 主模块 (PPM + GAT + SqueezeIterationController)
│   └── sast_losses.py        # 损失函数 (RE_2D + L_physics + L_smooth + L_balance)
├── deploy/                    # CUDA kernel
│   ├── msst_stft.cu          # cuFFT modulated-form STFT 
│   ├── msst_squeeze_hard.cu  # 硬最近邻挤压
│   ├── msst_squeeze_linear.cu# 线性插值挤压
│   └── setup_msst_kernels.py # 编译脚本
├── train_sast.py              # 训练入口
├── infer_sast.py              # 推理入口
├── papers/                    # 设计文档
│   ├── SAST_完整设计文档.md   # 主设计文档 
│   └── # Adaptive order synchrosqueezing transform.md  # Colominas & Meignen 2025 论文笔记
├── scripts/
│   ├── plot/                 # 可视化
│   │   └── plot_msst_tfr.py              # WSST2 / ssqueezepy / MSST / FFT 对比
│   └── test/                 # 测试
│       ├── test_squeeze_compare.py        # 挤压精度对比
│       └── test_torch_stft_v2.py          # Torch STFT + Squeeze 对比
└── docs/                      # 补充文档
```

## 快速开始

### 编译 CUDA kernel

```bash
python deploy/setup_msst_kernels.py build_ext --inplace
```

### 运行 MSST 时频图对比

```bash
python scripts/plot/plot_msst_tfr.py                # WSST2 / ssq / MSST / FFT 对比
```


