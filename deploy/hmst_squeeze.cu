/*
 * hmst_squeeze.cu — HMST 同步压缩 CUDA kernel
 * ==============================================
 *
 * 替代 Python 三重循环，将 3,598 次独立 GPU kernel launch 合并为 1 次。
 *
 * 输入:
 *   mag:      [B, F, T]   STFT 幅值 (或前次挤压结果)
 *   IF:       [B, F, T]   瞬时频率估计 (Hz)，所有 M 次迭代共用
 *   freqs_hz: [F]         频率网格 (Hz)，均匀间隔
 *   gamma:                幅度阈值
 *
 * 输出:
 *   Tx: [B, F, T]  挤压后 TFR
 *
 * 每个 (b,i,j) 元素由一个 CUDA 线程处理:
 *   1. 读取 mag[b,i,j], IF[b,i,j]
 *   2. 若 mag < gamma → 跳过 (噪声 bin)
 *   3. k = round((IF - f0) / df)   → 目标频率 bin
 *   4. atomicAdd(&Tx[b,k,j], mag)   → 原子累加
 *
 * 编译 (x86):
 *   见 deploy/setup_hmst.py
 *
 * 编译 (Jetson Orin, sm_87):
 *   TORCH_CUDA_ARCH_LIST="8.7" python deploy/setup_hmst.py
 *
 * Author: TFDCL Project
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// ── CUDA Kernel ─────────────────────────────────────────────

__global__ void hmst_squeeze_kernel(
    const float* __restrict__ mag,      // [B, F, T]
    const float* __restrict__ IF,       // [B, F, T]
    float* __restrict__ Tx,             // [B, F, T] (output, zero-initialized)
    int B, int F, int T,
    float f0, float inv_df,             // f0 = freqs[0], inv_df = 1.0 / df
    float gamma
) {
    // ── 1D grid: 每个线程处理一个 (b, f, t) ──
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * F * T;
    if (idx >= total) return;

    // ── 解算三维坐标 ──
    int t = idx % T;
    int f = (idx / T) % F;
    int b = idx / (F * T);

    // ── 读输入 ──
    float val = mag[idx];
    if (val < gamma) return;  // 噪声 bin, 跳过

    float w = IF[idx];  // 瞬时频率 (Hz)

    // ── 频率 → bin 索引 ──
    // k = round((w - f0) / df)
    int k = __float2int_rn((w - f0) * inv_df);

    if (k < 0 || k >= F) return;  // 越界保护 (IF 估计极端异常时)

    // ── 原子累加: Tx[b, k, t] += val ──
    int dst = b * F * T + k * T + t;
    atomicAdd(&Tx[dst], val);
}


// ── PyTorch 包装函数 (CPU 端) ──────────────────────────────

torch::Tensor hmst_squeeze_cuda(
    torch::Tensor mag,
    torch::Tensor IF,
    torch::Tensor freqs_hz,
    float gamma
) {
    // ── 输入校验 ──
    TORCH_CHECK(mag.is_cuda(),  "mag must be on CUDA");
    TORCH_CHECK(IF.is_cuda(),   "IF must be on CUDA");
    TORCH_CHECK(mag.dim() == 3, "mag must be [B, F, T]");
    TORCH_CHECK(IF.sizes() == mag.sizes(), "IF shape must match mag");

    // 确保连续内存布局 (避免 stride 导致的非法访存)
    auto mag_contig = mag.contiguous();
    auto IF_contig  = IF.contiguous();

    int B = mag_contig.size(0);
    int F = mag_contig.size(1);
    int T = mag_contig.size(2);

    float f0 = freqs_hz[0].item<float>();
    float df = (freqs_hz[1] - freqs_hz[0]).item<float>();
    float inv_df = 1.0f / df;

    // ── 分配输出 (零初始化) ──
    auto Tx = torch::zeros({B, F, T}, mag_contig.options());

    // ── Launch kernel ──
    int total = B * F * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    hmst_squeeze_kernel<<<blocks, threads>>>(
        mag_contig.data_ptr<float>(),
        IF_contig.data_ptr<float>(),
        Tx.data_ptr<float>(),
        B, F, T,
        f0, inv_df,
        gamma
    );

    // ── 检查 kernel 错误 ──
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "hmst_squeeze_kernel failed: ",
                cudaGetErrorString(err));

    return Tx;
}


// ── pybind11 模块注册 ──────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hmst_squeeze", &hmst_squeeze_cuda,
          "HMST synchrosqueezing (CUDA-accelerated)\n\n"
          "Args:\n"
          "  mag:      [B, F, T] magnitude TFR\n"
          "  IF:       [B, F, T] IF estimates (Hz)\n"
          "  freqs_hz: [F] frequency grid (Hz)\n"
          "  gamma:    amplitude threshold\n\n"
          "Returns:\n"
          "  Tx: [B, F, T] squeezed TFR",
          py::arg("mag"),
          py::arg("IF"),
          py::arg("freqs_hz"),
          py::arg("gamma") = 1e-6f);
}
