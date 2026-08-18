# SAST Jetson 部署指南 (TRT + CUDA 重排)

> 目标: 把完整 SAST 前向 (信号 → 增强 TFR) 部署到 Jetson.
> 最后更新: 2026-08-07

---

## 1. 部署架构

```
┌──────────────────────────────────────────────────────────────┐
│  Jetson 推理 (信号 → 增强 TFR)                                │
│                                                              │
│  Signal [1, T=2000]                                          │
│    │                                                         │
│    ├─ TRT engine (sast_net.engine, FP16)                     │
│    │     MSST→节点提取→PPM→GAT→sigma/ridge                   │
│    │     → tfr_mag, sigma, omega_hat_int,                    │
│    │       ridge_factor, n_sqz_per_bin        [网络部分]     │
│    │                                                         │
│    └─ CUDA kernel (reassigner.cu, 单次 launch)               │
│          多轮高斯重排 (91 循环在 kernel 内)                    │
│          ← 5 个中间量 → tfr_enhanced [F, T]                  │
│                                                              │
│  为什么重排不进 ONNX?                                        │
│    91 次高斯循环若导出 ONNX 会被 unroll 成 7000+ 节点巨图      │
│    → OOM. CUDA kernel 单次 launch 完成, 无中间张量累积.       │
└──────────────────────────────────────────────────────────────┘
```

## 2. 为什么不用 modelopt INT8

- modelopt 的 INT8 fake-quant 依赖其 CUDA 扩展 (`modelopt_cuda_ext`)
- 在 Windows + VS2026 下, 该扩展编译失败 (nvcc 只支持 VS2022, 链接器报 LNK2019)
- 解决方案: **导出 FP32 ONNX, 在 Jetson 上用 TRT 构建 FP16 engine**
  (TRT 的 FP16 是自动转换, 无需 modelopt; FP16 在 Jetson Orin 上吞吐 ~2x)

---

## 3. PC 端: 导出 ONNX

在 Windows (modelopt_env 有 torch 2.11 + onnx; 或 my_work) 运行:

```bash
cd deploy
# 用 modelopt_env (torch 2.11, onnx 1.22):
PYTHONIOENCODING=utf-8 D:/Anaconda/envs/modelopt_env/python.exe \
    export_sast_onnx.py \
    --checkpoint ../sast_checkpoints/sast_v3_e050.pt \
    --data ../5_dataset.npz \
    --output sast_net.onnx \
    --samples 8
```

输出:
- `sast_net.onnx` (单文件 ~13MB, 权重已内联)
- 5 个输出: `tfr_mag, sigma, omega_hat_int, ridge_factor, n_sqz_per_bin`

### 数值验证 (可选, 需 onnxruntime 支持 float64 Atan)

```bash
conda run -n my_work python export_sast_onnx.py --verify-only \
    --checkpoint ../sast_checkpoints/sast_v3_e050.pt \
    --data ../5_dataset.npz --output sast_net.onnx --samples 4
```

(本机 onnxruntime 1.23 的 CPU EP 不支持 float64 Atan, 脚本会回退到
torch.export 图验证, 数值等价, 已验证 maxdiff ≤ 1e-6 PASS)

---

## 4. Jetson 端: 构建 TRT engine

把 `sast_net.onnx` + `deploy/reassigner.cu` + `deploy/setup_msst_kernels.py` 拷到 Jetson.

### 4.1 环境 (JetPack)

```bash
# JetPack 6.0 自带: Python3, TensorRT 8.6, CUDA 12.x, cuDNN
sudo apt install python3-pip pycuda  # pycuda 用于 TRT 推理 IO
pip3 install numpy
```

### 4.2 构建 engine (FP16)

```bash
cd deploy
# 方式 A: Python API (推荐)
python3 build_trt_sast.py --onnx sast_net.onnx --output sast_net.engine \
    --precision fp16 --workspace 1073741824

# 方式 B: trtexec (最可靠, 若 Python API 版本问题)
/usr/src/tensorrt/bin/trtexec --onnx=sast_net.onnx --saveEngine=sast_net.engine \
    --fp16 --memPoolSize=workspace:1024

# INT8 (可选, 需校准):
python3 build_trt_sast.py --onnx sast_net.onnx --output sast_net_int8.engine \
    --precision int8 --calib ../5_dataset.npz
```

### 4.3 编译重排 CUDA kernel

```bash
# 在 Jetson (Linux) 上重新编译, 输出 .so:
python3 setup_msst_kernels.py build_ext --inplace
# 注意: 重排 kernel 需 sm_87 (Orin) 或实际架构,
# 设置 TORCH_CUDA_ARCH_LIST="8.7" (Orin) / "8.6" (AGX Xavier)
```

---

## 5. Jetson 端: 推理

```bash
python3 sast_jetson_infer.py \
    --engine sast_net.engine \
    --signal signal.npy \
    --output tfr_enhanced.npy
```

Python API:

```python
from sast_jetson_infer import SastJetson
sast = SastJetson('sast_net.engine', n_sqz_max=4)
tfr = sast(signal)   # [F, T] 增强 TFR
```

---

## 6. 已知限制与问题

| 问题 | 说明 | 解决方案 |
|------|------|---------|
| ONNX 含 `DFT` 算子 (复数 FFT) | TRT 对 DFT 支持取决于 JetPack 版本 | JetPack 6.x (TRT 8.6+) 支持; 若报 unsupported, 把 MSST 的 FFT 移到 Jetson 端用 cuFFT (见 CUDA-HMST deploy plan §4) |
| float64 Atan | onnxruntime CPU EP 不支持; TRT 不确定 | 导出为 FP16 engine 时 TRT 自动转; 若 TRT 也报错, 需把相位估计改 float32 (精度降至 bin 级, 需重测) |
| 动态形状 | engine 固定输入 [1, 2000] | 若需变长, 用 TRT 优化 profile + dynamic_axes 重导出 |
| 多轮重排每轮分配 | 4 轮循环, 每轮 kernel launch | 可用 CUDA Graph 录制 4 轮, 消除 launch 开销 (docs plan §5.2.3) |

---

## 7. 文件清单

| 文件 | 用途 |
|------|------|
| `deploy/sast_export_model.py` | 可导出的纯 torch SAST 网络部分 (复用训练权重) |
| `deploy/export_sast_onnx.py` | PC 端: checkpoint → FP32 ONNX + 数值验证 |
| `deploy/build_trt_sast.py` | Jetson: ONNX → TRT engine (fp32/fp16/int8) |
| `deploy/trt_calibrator.py` | INT8 校准器 (EntropyCalibrator2) |
| `deploy/sast_jetson_infer.py` | Jetson 推理: TRT 网络部分 + CUDA 重排 → TFR |
| `deploy/reassigner.cu` | 高斯重排 CUDA kernel (91 循环单次 launch) |
| `deploy/setup_msst_kernels.py` | 编译上述 kernel (Windows .pyd / Linux .so) |

## 8. 验证结果

- ✅ 网络部分 ONNX: 0.9MB (未内联) / 12.9MB (内联后), 与 PyTorch maxdiff ≤ 1e-6
- ✅ 端到端管线 (ONNX 网络部分 + CUDA 重排) vs 原始 SAST: **maxdiff = 0.0, 能量比 1.0**
- ✅ `tfr_mag/sigma/omega_hat_int/ridge_factor/n_sqz_per_bin` 全部一致
