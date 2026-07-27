/*
 * msst_squeeze_linear.cu — MSST 线性插值挤压 (Linear Interpolation Squeeze)
 * ========================================================================
 *
 * 升级版 MSST 挤压: 连续 IF + 线性插值分配到相邻双 bin.
 *
 * 与硬最近邻的区别:
 *   硬最近邻: IF 被 round 到整数 → 每个系数 → 1 个 bin → 量化阶梯/散斑
 *   线性插值: IF 保留连续 Hz → 每个系数 → 2 个 bin → 平滑 TFR, 抑制阶梯
 *
 * 输入:
 *   mag:      [B, F, T]   STFT 幅值 (float32)
 *   IF:       [B, F, T]   瞬时频率 (Hz, float32, 连续值)
 *   freqs_hz: [F]         频率网格 (Hz), 均匀间隔, f0..f0+(F-1)*df
 *   gamma:                幅度阈值
 *
 * 输出:
 *   Tx: [B, F, T]  挤压后 TFR (float32)
 *
 * 每个 (b,i,j) 元素由一个 CUDA 线程处理:
 *   1. 读取 mag[b,i,j], IF[b,i,j] (Hz)
 *   2. 若 mag < gamma → 跳过 (噪声)
 *   3. k_float = (IF - f0) / df → 连续 bin 索引
 *   4. k_floor = floor(k_float), alpha = k_float - k_floor
 *   5. Tx[k_floor]     += (1 - alpha) * val
 *      Tx[k_floor + 1] += alpha * val
 *
 * 编译:
 *   python deploy/setup_msst_kernels.py
 *
 * Author: SAST Project (refactored from hmst_squeeze.cu)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// ── CUDA Kernel: Linear Interpolation Squeeze ──────────────────

__global__ void msst_squeeze_linear_kernel(
    const float* __restrict__ mag,      // [B, F, T]
    const float* __restrict__ IF,       // [B, F, T] — IF in Hz (continuous)
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

    // ── 读幅值 ──
    float val = mag[idx];
    if (val < gamma) return;  // 噪声 bin, 跳过

    // ── 读 IF (Hz) → 连续 bin 索引 ──
    float w = IF[idx];  // 瞬时频率 (Hz)
    if (w <= 0.0f) return;  // invalid IF

    // k_float = (w - f0) / df
    float k_float = (w - f0) * inv_df;
    int k_floor = __float2int_rd(k_float);  // floor
    float alpha = k_float - (float)k_floor;  // 小数部分 ∈ [0, 1)

    // ── 线性插值分配到两个相邻 bin ──
    // Tx[k_floor] += (1 - alpha) * val
    if (k_floor >= 0 && k_floor < F) {
        int dst_lo = b * F * T + k_floor * T + t;
        atomicAdd(&Tx[dst_lo], (1.0f - alpha) * val);
    }
    // Tx[k_floor + 1] += alpha * val
    if (k_floor + 1 >= 0 && k_floor + 1 < F) {
        int dst_hi = b * F * T + (k_floor + 1) * T + t;
        atomicAdd(&Tx[dst_hi], alpha * val);
    }
}


// ── PyTorch 包装函数 (CPU 端) ──────────────────────────────────

torch::Tensor msst_squeeze_linear_cuda(
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

    // 确保连续内存布局
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

    msst_squeeze_linear_kernel<<<blocks, threads>>>(
        mag_contig.data_ptr<float>(),
        IF_contig.data_ptr<float>(),
        Tx.data_ptr<float>(),
        B, F, T,
        f0, inv_df,
        gamma
    );

    // ── 检查 kernel 错误 ──
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "msst_squeeze_linear_kernel failed: ",
                cudaGetErrorString(err));

    return Tx;
}


// ── pybind11 模块注册 ──────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("msst_squeeze_linear", &msst_squeeze_linear_cuda,
          "MSST linear interpolation squeeze (continuous IF, 2-bin assignment)\n\n"
          "Args:\n"
          "  mag:      [B, F, T] magnitude TFR (float32)\n"
          "  IF:       [B, F, T] IF estimates in Hz (float32, continuous)\n"
          "  freqs_hz: [F] frequency grid (Hz), uniformly spaced\n"
          "  gamma:    amplitude threshold (default 1e-6)\n\n"
          "Returns:\n"
          "  Tx: [B, F, T] squeezed TFR (float32)\n\n"
          "Note: IF values are continuous Hz (NOT rounded to integer bins).\n"
          "      Each coefficient is split between two adjacent bins via\n"
          "      linear interpolation, suppressing quantization artifacts\n"
          "      that occur with hard nearest-neighbor squeeze.",
          py::arg("mag"),
          py::arg("IF"),
          py::arg("freqs_hz"),
          py::arg("gamma") = 1e-6f);
}
