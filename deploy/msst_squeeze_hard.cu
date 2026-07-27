/*
 * msst_squeeze_hard.cu — MSST 硬最近邻挤压 (Hard Nearest-Neighbor Squeeze)
 * ========================================================================
 *
 * 精确复刻 numpy MSST 的硬挤压行为:
 *   - IF 已被 round 到整数 bin 索引 (1-indexed, 0=invalid)
 *   - 每个 TF 系数完整分配到唯一目标 bin (无插值)
 *   - 与 models/tfr.py 的 msst() Step 4 输出完全一致
 *
 * 输入:
 *   mag:      [B, F, T]   STFT 幅值 (或前次挤压结果)
 *   IF:       [B, F, T]   瞬时频率 bin 索引 (int32, 1-indexed, 0=invalid)
 *                          对应 MSST omega_final: (bin-1)*fs/N = Hz
 *   gamma:                幅度阈值
 *
 * 输出:
 *   Tx: [B, F, T]  挤压后 TFR (幅度)
 *
 * 每个 (b,i,j) 元素由一个 CUDA 线程处理:
 *   1. 读取 mag[b,i,j], IF[b,i,j] (int bin index)
 *   2. 若 IF[b,i,j] == 0 → 跳过 (invalid bin)
 *   3. 若 mag[b,i,j] < gamma → 跳过 (噪声)
 *   4. k = IF[b,i,j] - 1 → 0-indexed target bin
 *   5. atomicAdd(Tx[b,k,j], mag[b,i,j])
 *
 * 与 msst_squeeze_linear.cu 的区别:
 *   本文件: k ∈ Z (整数 bin), 100% 能量去 1 个 bin — 匹配 numpy MSST
 *   linear: k ∈ R (连续 Hz), 按距离分配去 2 个 bin — 抑制量化阶梯
 *
 * 编译:
 *   python deploy/setup_msst_kernels.py
 *
 * Author: SAST Project
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// ── CUDA Kernel: Hard Nearest-Neighbor Squeeze ─────────────────

__global__ void msst_squeeze_hard_kernel(
    const float* __restrict__ mag,      // [B, F, T]
    const int32_t* __restrict__ IF,     // [B, F, T] — 1-indexed bin, 0=invalid
    float* __restrict__ Tx,             // [B, F, T] (output, zero-initialized)
    int B, int F, int T,
    float gamma
) {
    // ── 1D grid: 每个线程处理一个 (b, f, t) ──
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * F * T;
    if (idx >= total) return;

    // ── 解算三维坐标 ──
    int t = idx % T;
    int b = idx / (F * T);

    // ── 读 IF (1-indexed bin index) ──
    int k = IF[idx];  // 1-indexed, 0 means invalid
    if (k <= 0 || k > F) return;  // invalid IF

    // ── 读幅值 ──
    float val = mag[idx];
    if (val < gamma) return;  // 噪声 bin, 跳过

    // ── 硬挤压: 直接加到目标 bin (k-1 → 0-indexed) ──
    int dst = b * F * T + (k - 1) * T + t;
    atomicAdd(&Tx[dst], val);
}


// ── PyTorch 包装函数 (CPU 端) ──────────────────────────────────

torch::Tensor msst_squeeze_hard_cuda(
    torch::Tensor mag,
    torch::Tensor IF,
    float gamma
) {
    // ── 输入校验 ──
    TORCH_CHECK(mag.is_cuda(),  "mag must be on CUDA");
    TORCH_CHECK(IF.is_cuda(),   "IF must be on CUDA");
    TORCH_CHECK(mag.dim() == 3, "mag must be [B, F, T]");
    TORCH_CHECK(IF.sizes() == mag.sizes(), "IF shape must match mag");
    TORCH_CHECK(IF.dtype() == torch::kInt32, "IF must be int32 (1-indexed bin)");

    // 确保连续内存布局
    auto mag_contig = mag.contiguous();
    auto IF_contig  = IF.contiguous();

    int B = mag_contig.size(0);
    int F = mag_contig.size(1);
    int T = mag_contig.size(2);

    // ── 分配输出 (零初始化) ──
    auto Tx = torch::zeros({B, F, T}, mag_contig.options());

    // ── Launch kernel ──
    int total = B * F * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    msst_squeeze_hard_kernel<<<blocks, threads>>>(
        mag_contig.data_ptr<float>(),
        IF_contig.data_ptr<int32_t>(),
        Tx.data_ptr<float>(),
        B, F, T,
        gamma
    );

    // ── 检查 kernel 错误 ──
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "msst_squeeze_hard_kernel failed: ",
                cudaGetErrorString(err));

    return Tx;
}


// ── pybind11 模块注册 ──────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("msst_squeeze_hard", &msst_squeeze_hard_cuda,
          "MSST hard nearest-neighbor squeeze (matches numpy MSST exactly)\n\n"
          "Args:\n"
          "  mag:   [B, F, T] magnitude TFR (float32)\n"
          "  IF:    [B, F, T] IF bin indices (int32, 1-indexed, 0=invalid)\n"
          "  gamma: amplitude threshold (default 0.0001)\n\n"
          "Returns:\n"
          "  Tx: [B, F, T] squeezed TFR (float32)\n\n"
          "Note: IF must be pre-rounded integer bin indices, matching\n"
          "      the numpy MSST omega_final format. Each coefficient goes\n"
          "      to exactly ONE target bin (no interpolation).",
          py::arg("mag"),
          py::arg("IF"),
          py::arg("gamma") = 1e-4f);
}
