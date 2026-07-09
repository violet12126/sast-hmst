# CUDA-HMST + CUDA-SAST 部署方案

> 目标：将 HMST 时频分析 + SAST 自适应挤压完整管线部署到 GPU (RTX 4060 / Jetson Orin)，实现训练加速 10-50× + 推理延迟 <10ms。
>
> 最后更新：2026-07-08

---

## 目录

1. [性能瓶颈剖析](#1-性能瓶颈剖析)
2. [CUDA-HMST：挤压算子加速](#2-cuda-hmst挤压算子加速)
3. [CUDA-SAST：GAT 批量化 + 图构建向量化](#3-cuda-sastgat-批量化--图构建向量化)
4. [ONNX/TensorRT 导出策略](#4-onnxtensorrt-导出策略)
5. [Jetson Orin 部署适配](#5-jetson-orin-部署适配)
6. [分阶段实施路线](#6-分阶段实施路线)

---

## 1. 性能瓶颈剖析

### 1.1 HMST 管线 (单次前向, B=1, T=2000, F=257)

```
Signal [1, 2000]
  │
  ├─ 1. compute_hmst_if: 3× torch.stft (cuFFT) .............. ~2ms  ✅ 已CUDA
  ├─ 2. N=2 矩阵求解 (复数伪逆 + 频域差分) .................. ~0.1ms ✅ 已CUDA
  └─ 3. _hmst_squeeze × M=2 ................................. ~15ms ❌ CPU Python循环
       └─ for b in B:                        1×
            for j in T_if:                   14×
              for i in F:                    257×
                → 3,598 Python iter/squeeze
                → 每 iter: threshold→round→if→accumulate
```

**瓶颈结论**：HMST 总延迟 ~17ms，其中 88% 花在 `_hmst_squeeze` 的 Python 三重循环。`torch.stft` 已走 cuFFT 无需改动。

### 1.2 SAST 管线 (单次前向, B=1, T=2000, K=6 ridges)

```
Signal [1, 2000]
  │
  ├─ 1. compute_hmst_if (高阶IF) ............................. ~2ms  ✅
  ├─ 2. BlindRidgeExtractor: for t in T_if .................. ~3ms  ❌ 逐帧串行
  │     └─ _find_local_maxima (max_pool1d) + topk per frame
  ├─ 3. build_anonymous_graph: _local_std/_local_pearson .... ~2ms  ⚠️ unfold已向量化但大显存
  ├─ 4. for t in T_if: PPM + GAT ............................ ~8ms  ❌ 逐帧串行 GAT
  │     └─ PhysicsPrototypeMemory: cross-attn [B,K,dh]→[B,K,dh]
  │     └─ EdgeConditionedGAT: 2-layer multi-head GAT
  └─ 5. AdaptiveSqueeze: _gaussian_blur_along_freq ........... ~2ms  ⚠️ 20×conv1d离散化
```

**瓶颈结论**：SAST 总延迟 ~17ms，60% 花在逐帧 PPM+GAT 串行循环，18% 在 BlindRidgeExtractor。

---

## 2. CUDA-HMST：挤压算子加速

### 2.1 方案 A：PyTorch scatter_add 向量化 (推荐首步)

将三重循环替换为单次 `scatter_add_` 调用：

```python
def _hmst_squeeze_cuda(mag, IF, freqs_hz, gamma=1e-6):
    """
    CUDA 向量化挤压: scatter_add 替代 Python 三重循环。

    mag: [B, F, T], IF: [B, F, T], freqs_hz: [F]
    Returns: Tx [B, F, T]
    """
    B, F, T = mag.shape
    device = mag.device
    f0 = freqs_hz[0]
    df = freqs_hz[1] - freqs_hz[0]

    # 1. 阈值掩码 + 目标 bin 索引 [B, F, T]
    mask = mag >= gamma
    k_idx = ((IF - f0) / df).round().long().clamp(0, F - 1)  # [B, F, T]

    # 2. 构建 flat 索引: Tx_flat[k * T + t] += mag[f * T + t]
    t_base = torch.arange(T, device=device).view(1, 1, T)   # [1, 1, T]
    dst_flat = (k_idx * T + t_base).long()                    # [B, F, T]
    src_flat = mag.reshape(B, -1)                             # [B, F*T]
    dst_flat = dst_flat.reshape(B, -1)                        # [B, F*T]
    mask_flat = mask.reshape(B, -1)                           # [B, F*T]

    # 3. scatter_add: 重复索引自动求和 (PyTorch 保证确定性)
    Tx_flat = torch.zeros(B, F * T, device=device, dtype=mag.dtype)
    for b in range(B):
        Tx_flat[b].scatter_add_(
            0,
            dst_flat[b].masked_fill(~mask_flat[b], 0),
            src_flat[b].masked_fill(~mask_flat[b], 0),
        )

    return Tx_flat.reshape(B, F, T)
```

**收益**：Python 3598 iter × 3μs → 1 次 GPU kernel launch ~50μs。**预期加速 ~200×**（HMST squeeze 部分）。

**代价**：显存临时增加 `[B, F*T]` (~3.6K floats per batch element)，可忽略。

### 2.2 方案 B：Triton 自定义 Kernel (追求极致)

当 scatter_add 的原子竞争成为瓶颈时（M≥4、大批量下），替换为 Triton kernel：

```python
import triton
import triton.language as tl

@triton.jit
def _hmst_squeeze_kernel(
    mag_ptr, IF_ptr, Tx_ptr,
    B, F, T, f0, df, gamma,
    BLOCK_SIZE: tl.constexpr
):
    """Triton squeeze: 每 thread 处理一个 (b, f, t) 元素，atomic_add 到目标 bin。"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < B * F * T

    # 解算 (b, f, t) 坐标
    t = offsets % T
    f = (offsets // T) % F
    b = offsets // (F * T)

    val = tl.load(mag_ptr + offsets, mask=mask)
    if_val = tl.load(IF_ptr + offsets, mask=mask)

    # 阈值 + bin 映射
    valid = (val >= gamma) & mask
    k = tl.math.round((if_val - f0) / df).to(tl.int32)
    in_range = (k >= 0) & (k < F)
    write_mask = valid & in_range

    # 目标位置: Tx[b, k, t]
    dst = b * F * T + k * T + t
    tl.atomic_add(Tx_ptr + dst, val.to(Tx_ptr.dtype.element_ty), mask=write_mask)
```

**收益**：避免 PyTorch scatter_add 的中间张量分配，单次 kernel launch。**预期比方案 A 再快 2-3×**。

**代价**：引入 Triton 依赖 (`pip install triton`)，需要 sm_80+ GPU（RTX 4060 的 Ada Lovelace 支持）。

### 2.3 方案 C：CUDA C++ Native Kernel (Jetson 必需)

Jetson Orin 不支持 Triton，需要原生 CUDA C++ kernel：

```cpp
// hmst_squeeze.cu
#include <torch/extension.h>

__global__ void hmst_squeeze_kernel(
    const float* __restrict__ mag,
    const float* __restrict__ IF,
    float* __restrict__ Tx,
    int B, int F, int T, float f0, float df, float gamma
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * F * T;
    if (idx >= total) return;

    float val = mag[idx];
    if (val < gamma) return;

    int t = idx % T;
    int f = (idx / T) % F;
    int b = idx / (F * T);

    float w = IF[idx];
    int k = __float2int_rn((w - f0) / df);  // round to nearest
    if (k < 0 || k >= F) return;

    // 原子累加: dst = b*F*T + k*T + t
    atomicAdd(&Tx[b * F * T + k * T + t], val);
}

// PyTorch binding
torch::Tensor hmst_squeeze_cuda(
    torch::Tensor mag, torch::Tensor IF,
    torch::Tensor freqs_hz, float gamma
) {
    int B = mag.size(0), F = mag.size(1), T = mag.size(2);
    float f0 = freqs_hz[0].item<float>();
    float df = (freqs_hz[1] - freqs_hz[0]).item<float>();

    auto Tx = torch::zeros({B, F, T}, mag.options());

    int total = B * F * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    hmst_squeeze_kernel<<<blocks, threads>>>(
        mag.data_ptr<float>(), IF.data_ptr<float>(),
        Tx.data_ptr<float>(), B, F, T, f0, df, gamma
    );

    return Tx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hmst_squeeze_cuda", &hmst_squeeze_cuda, "HMST Squeeze (CUDA)");
}
```

编译脚本：
```python
# setup_hmst_cuda.py
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
setup(
    name='hmst_cuda',
    ext_modules=[CUDAExtension('hmst_cuda', ['hmst_squeeze.cu'])],
    cmdclass={'build_ext': BuildExtension},
)
```

### 2.4 HMST 加速汇总

| 方案 | 实现难度 | HMST延迟 (T=2000, M=2) | 加速比 | 适用平台 |
|------|---------|----------------------|--------|---------|
| 当前 (CPU循环) | - | ~17ms | 1× | 所有 |
| A: scatter_add | ⭐ 低 (~20行) | ~3ms | ~6× | CUDA通用 |
| B: Triton | ⭐⭐ 中 (~80行) | ~1.5ms | ~11× | sm_80+ |
| **C: CUDA C++** | ⭐⭐⭐ 高 (~120行) | **~1ms** | **~17×** | **全部GPU incl. Jetson** |

**推荐**：先实施方案 A 快速验证，训练用。部署到 Jetson 时再写方案 C。

---

## 3. CUDA-SAST：GAT 批量化 + 图构建向量化

### 3.1 GAT 跨时间批量化 (最高优先级)

**现状**：`SAST.forward()` 逐帧调用 PPM + GAT：
```python
for t in range(T_if):                    # T_if ≈ 14
    C_i_t, A_ij_t = self.gat(            # 每次 2-layer GAT
        self.ppm(raw_feats[:,:,t], ...), # PPM cross-attention
        edge_feats[:,:,t,:], edge_src, edge_dst, C_prior
    )
```

**优化**：将所有时间帧堆叠为 batch 维，一次前向：
```python
# 前: [B, K, T_if] → for t → [B, K] × T_if 次
# 后: [B*T_if, K] → GAT → [B*T_if, K] 一次

# Step 1: PPM 跨时间批量化
# node_feats_raw: [B, K, T_if, 4] → [B*T_if, K, 4]
# node_freqs:     [B, K, T_if]    → [B*T_if, K]
h_raw = node_feats_raw.permute(0, 2, 1, 3).reshape(B * T_if, K, 4)
f_obs = ridge_freq.permute(0, 2, 1).reshape(B * T_if, K)
h_enh, gate, C_prior = self.ppm(h_raw, f_obs)
# h_enh: [B*T_if, K, d_h]

# Step 2: 边特征也跨时间展开
e_feat = edge_feats.permute(0, 2, 1, 3).reshape(B * T_if, M, 4)

# Step 3: GAT 一次前向
C_i, A_ij = self.gat(h_enh, e_feat, edge_src, edge_dst, C_prior)
# C_i: [B*T_if, K] → reshape → [B, K, T_if]
C_i = C_i.reshape(B, T_if, K).permute(0, 2, 1)
# A_ij: [B*T_if, M, H] → reshape → [B, M, H, T_if]
```

**PPM 适配注意事项**：
- `prototype_embed` 和 `f_nom` 不随帧变 → 无需展开，直接与 `[B*T_if, K, d_h]` 做交叉注意力
- `best_proto = argmax(match_score)` 结果形状变为 `[B*T_if, K]`
- 频率门控温度 τ 保持不变

**收益**：14 次 kernel launch → 1 次，消除 Python 循环开销。**预期加速 ~10×**（GAT 部分）。

### 3.2 BlindRidgeExtractor 向量化

**现状**：逐帧 max_pool1d + topk + 贪心跟踪。

**优化**：
- `_find_local_maxima` 已通过 `max_pool1d` 在频率轴向量化，但每帧独立调用
- 将所有帧堆叠为 `[B*T, 1, F]`，单次 `max_pool1d` + `topk`

```python
def forward_vectorized(self, mag, freqs):
    """向量化盲脊线提取: 全时间帧并行。"""
    B, F, T = mag.shape

    # 全帧并行峰值检测
    mag_bt = mag.permute(0, 2, 1).reshape(B * T, 1, F)  # [B*T, 1, F]
    mag_pooled = F.max_pool1d(mag_bt, 2*self.min_dist+1, stride=1,
                              padding=self.min_dist)
    is_peak = (mag_bt == mag_pooled).squeeze(1)  # [B*T, F]

    # 掩码非峰值
    masked = torch.where(is_peak, mag_bt.squeeze(1),
                         torch.full_like(mag_bt.squeeze(1), -float('inf')))
    # → [B*T, F]

    # top-K per frame
    topk_vals, topk_idx = torch.topk(masked, self.K, dim=1)  # [B*T, K]
    ridge_freq = freqs[topk_idx].reshape(B, T, self.K).permute(0, 2, 1)
    # [B, K, T]

    # 贪心跟踪: 用 cumulative max persistence (向量化)
    # ...
```

**收益**：T 次独立的 max_pool1d kernel launch → 1 次。**预期加速 ~5×**（BlindRidge 部分）。

### 3.3 Gaussian Blur 优化

**现状**：20 个离散 sigma 级别，每级一次 `conv1d` → 20 次 kernel launch。

**优化**：用 `torch.nn.functional.interpolate` 的 anti-alias 高斯滤波，或将 20 个 kernel 合并为一次 grouped conv1d：

```python
# 单次 grouped conv: groups = B*T, 每个 (1,1,F) 用对应 sigma 的 kernel
kernel_weights = torch.stack(kernels).to(device)  # [L, 1, K_size]
# 对每个 (b,t) 选对应 kernel → 构建 [B*T, 1, K_size] kernel bank
selected_kernels = kernel_weights[level_idx.reshape(-1)]  # [B*T, 1, K_size]
# grouped conv1d: groups = B*T
F.conv1d(mag_bt, selected_kernels, groups=B*T, padding=self.pad)
```

**收益**：20 次 → 1 次 conv1d。**预期加速 ~8×**（AdaptiveSqueeze 部分）。

### 3.4 SAST 加速汇总

| 优化 | 目标模块 | 当前延迟 | 优化后 | 加速比 | 难度 |
|------|---------|---------|--------|--------|------|
| GAT 跨时间批量化 | `SAST.forward()` 循环 | ~8ms | ~0.8ms | ~10× | ⭐⭐ |
| BlindRidge 向量化 | `BlindRidgeExtractor` | ~3ms | ~0.6ms | ~5× | ⭐⭐ |
| Gaussian Blur 合并 | `AdaptiveSqueeze` | ~2ms | ~0.3ms | ~7× | ⭐⭐ |
| **SAST 总计** | | **~17ms** | **~3.7ms** | **~4.6×** | |

加上 HMST squeeze 加速（~1ms），**全管线从 ~34ms → ~5ms**。

---

## 4. ONNX/TensorRT 导出策略

### 4.1 HMST 可导出性分析

| 算子 | ONNX支持 | 注意事项 |
|------|---------|---------|
| `torch.stft` | ✅ opset≥17 (DFT-17) | `return_complex=True` 需 ONNX≥1.14 |
| `torch.fft.fft/ifft` | ✅ opset≥17 | 窗函数微分用 |
| 复数运算 (`+`, `*`, `/`, `.conj()`) | ✅ | ONNX 原生支持 complex64 |
| `scatter_add_` | ✅ (ScatterElements) | PyTorch→ONNX 自动映射 |
| `torch.where`, `clamp` | ✅ | 无条件分支 |
| `round().long()` | ⚠️ | ONNX Round 语义与 PyTorch 一致 (banker's rounding) |

**HMST 导出策略**：`compute_hmst_if` + `_hmst_squeeze_cuda` (scatter_add 版本) 全部可 ONNX trace。导出为单一 `HMSTEncoder.onnx`。

### 4.2 SAST 可导出性分析

| 算子 | ONNX支持 | 阻断？ |
|------|---------|--------|
| `max_pool1d` | ✅ | |
| `topk` | ✅ opset≥11 | |
| `scatter_add` | ✅ | PPM 交叉注意力 |
| `F.softmax` | ✅ | |
| `GELU` | ✅ opset≥20 原生 |
| `LayerNorm` | ✅ |
| `F.conv1d` | ✅ | Gaussian blur |
| `F.pad` | ✅ |
| `unfold` | ⚠️ | `_local_std` / `_local_pearson_corr` 用了 unfold，ONNX 映射可能有问题 |
| `searchsorted` (CWT版) | ❌ | 仅 CWT 版使用 |

**关键阻断点**：`unfold` 在 ONNX 中无直接等价算子。解决方案：
- 用 `F.pad` + `as_strided` 替代（但 as_strided 也可能有问题）
- **推荐**：导出时用 `torch.jit.script` 替代 ONNX (对 SAST 这种含动态索引的图网络更友好)

### 4.3 推荐导出策略

```
┌─────────────────────────────────────────────────┐
│                  导出分两段                       │
├─────────────────────────────────────────────────┤
│ Stage 1: HMST前端                                │
│  Signal → compute_hmst_if + squeeze → TFR        │
│  → ONNX ✅ (算子全兼容)                           │
│  → TensorRT ✅ (cuFFT + ScatterElements)          │
├─────────────────────────────────────────────────┤
│ Stage 2: SAST后端 (可选)                          │
│  TFR → BlindRidge → GAT → AdaptiveSqueeze        │
│  → TorchScript ⚠️ (unfold阻断ONNX)                │
│  → 或保留 PyTorch + CUDA Graph                   │
└─────────────────────────────────────────────────┘
```

**如果只需要 HMST TFR 输出**（例如用于特征提取），仅导出 Stage 1 即可，完全 ONNX + TensorRT 兼容。

---

## 5. Jetson Orin 部署适配

### 5.1 硬件约束

| 约束 | Jetson Orin Nano 8GB | RTX 4060 (训练) |
|------|---------------------|-----------------|
| GPU 架构 | Ampere sm_87 | Ada Lovelace sm_89 |
| 显存 | 8GB LPDDR5 (共享) | 8GB GDDR6 |
| FP32 | ~1 TFLOPS | ~15 TFLOPS |
| FP16 | 20 TFLOPS | ~30 TFLOPS |
| CUDA Cores | 1024 | 3072 |
| cuFFT 支持 | ✅ | ✅ |
| Triton 支持 | ❌ (需 sm_80+，架构支持但包不官方支持) | ✅ |

### 5.2 Jetson 特化优化

1. **CUDA C++ squeeze kernel**（§2.3）：Triton 不可用，必须手写 CUDA
2. **FP16 混合精度**：`torch.stft` 内部 cuFFT 必须 FP32；squeeze 和 GAT 可用 FP16
3. **CUDA Graph**：对于固定 T 的推理，录制整个 HMST pipeline 为 CUDA Graph，消除 kernel launch overhead (~2ms)
4. **显存管理**：Jetson 是统一内存架构，避免 `cudaMalloc` 碎片化 → 用 `torch.cuda.empty_cache()` + 固定大小 buffer pool

### 5.3 预期延迟 (Jetson Orin)

| 管线阶段 | PyTorch FP32 | +CUDA squeeze | +CUDA Graph | +FP16 |
|----------|-------------|---------------|-------------|-------|
| HMST IF + squeeze | ~80ms | ~5ms | ~3ms | ~2ms |
| SAST (GAT+Squeeze) | ~120ms | ~25ms | ~18ms | ~10ms |
| **HMST + SAST 全管线** | **~200ms** | **~30ms** | **~21ms** | **~12ms** |

> HMST-only 推理可到 **2ms**，满足近实时要求。

---

## 6. 分阶段实施路线

```
Phase 1: PyTorch 向量化 (本周, 无需CUDA C++)
├── □ 1a. _hmst_squeeze → scatter_add 向量化 (~20行)
├── □ 1b. SAST.forward() GAT 跨时间批量化 (~40行)
├── □ 1c. BlindRidgeExtractor 全帧并行 (~30行)
├── □ 1d. Gaussian blur grouped conv1d (~15行)
├── □ 验证: B=4 训练吞吐 vs baseline
└── □ 预期: 训练加速 5-10×, 推理延迟 ~8ms (RTX 4060)

Phase 2: CUDA Kernel (下周, 用于部署)
├── □ 2a. hmst_squeeze.cu CUDA kernel + pybind11
├── □ 2b. setup.py 编译脚本
├── □ 2c. 与 PyTorch 版本的一致性测试 (tolerance=1e-5)
├── □ 2d. Profile: Nsight Systems 分析 kernel 效率
└── □ 预期: squeeze 再加速 3-5×

Phase 3: ONNX/TensorRT 导出
├── □ 3a. HMST ONNX 导出脚本
├── □ 3b. TensorRT engine 构建 (FP16 + INT8)
├── □ 3c. 精度验证 (vs PyTorch, tolerance=1e-3 for FP16)
├── □ 3d. CUDA Graph 录制
└── □ 预期: Jetson Orin 全管线 ~12ms

Phase 4: Jetson 板端部署
├── □ 4a. JetPack 6.0 + TensorRT 8.6 环境
├── □ 4b. 交叉编译 CUDA kernel (sm_87)
├── □ 4c. 端到端延迟 + 功耗测试
└── □ 4d. 集成到生产系统
```

---

## 附录 A：关键代码位置

| 文件 | 函数/类 | 行号 | 瓶颈类型 |
|------|--------|------|---------|
| `models/sast.py` | `_hmst_squeeze` | ~283 | Python 三重循环 |
| `models/sast.py` | `compute_hmst` | ~380 | 调 squeeze M 次 |
| `models/sast.py` | `SAST.forward` | ~1232 | 逐帧 GAT 循环 |
| `models/sast.py` | `BlindRidgeExtractor.forward` | ~458 | 逐帧峰值搜索 |
| `models/sast.py` | `AdaptiveSqueeze._gaussian_blur_along_freq` | ~1058 | 20×conv1d |

## 附录 B：依赖与工具

| 工具 | 用途 | 安装 |
|------|------|------|
| `torch.utils.cpp_extension` | CUDA kernel 编译 | PyTorch 内置 |
| `triton` | GPU kernel (Python DSL) | `pip install triton` |
| `onnx` + `onnxruntime-gpu` | ONNX 导出/验证 | `pip install onnx onnxruntime-gpu` |
| `tensorrt` | Jetson GPU 推理 | JetPack 内置 |
| `nsight-systems` | CUDA profiling | `apt install nsight-systems` |
| `pybind11` | C++/Python 绑定 | `pip install pybind11` |
