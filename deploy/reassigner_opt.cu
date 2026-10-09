#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ── 布局转置为 [B, T, F] 后的 forward kernel ──
__global__ void reassigner_forward_kernel_transposed(
    const float* __restrict__ tfr_weighted,
    const float* __restrict__ sigma,
    const int64_t*  __restrict__ omega_hat,
    float* __restrict__ tfr_enhanced,
    int B, int T, int F, int K) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * F;
    if (idx >= total) return;
    
    // 解析 [B, T, F] 的坐标
    int f = idx % F;
    int tmp = idx / F;
    int t = tmp % T;
    int b = tmp / T;

    float sig = sigma[idx];
    float tfr_w = tfr_weighted[idx];
    int omega = (int)omega_hat[idx];

    // WHY 动态 K_eff: 高斯权重 exp(-0.5*(k/σ)²) 衰减极快, |k| > 6σ 时权重 < 1.5e-8,
    //   float32 累加 < 1e-7 会被淹没(无数学意义)。按 σ 截断循环, σ 小则大幅缩短
    //   (σ=2 时 91 次 → 25 次), 省下的 atomicAdd 次数正是本 kernel 的主要开销。
    int K_eff = (int)ceilf(6.0f * sig);
    if (K_eff > K) K_eff = K;   // 不超过全局 K
    if (K_eff < 1) K_eff = 1;   // 至少中心点

    // 计算归一化系数 Z（同一 K_eff 范围, 保证 Σw_k=1 能量守恒）
    float Z = 0.0f;
    for (int k = -K_eff; k <= K_eff; k++) {
        float r = (float)k / sig;
        Z += expf(-0.5f * r * r);
    }
    Z += 1e-8f;
    float inv_Z = 1.0f / Z;

    // Scatter 写入到 [B, T, F] 的目标位置
    for (int k = -K_eff; k <= K_eff; k++) {
        float r = (float)k / sig;
        float w_k = expf(-0.5f * r * r) * inv_Z;
        
        int target = omega + k;
        // 钳位保证不越界
        target = target < 0 ? 0 : (target >= F ? F - 1 : target);
        
        // 注意地址计算公式变了：b * (T * F) + t * F + target
        atomicAdd(&tfr_enhanced[b * T * F + t * F + target], w_k * tfr_w);
    }
}

// ── 布局转置为 [B, T, F] 后的 backward kernel ──
__global__ void reassigner_backward_kernel_transposed(
    const float* __restrict__ grad_out,
    const float* __restrict__ sigma,
    const float* __restrict__ tfr_weighted,
    const int64_t*  __restrict__ omega_hat,
    float* __restrict__ grad_sigma,
    int B, int T, int F, int K) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * F;
    if (idx >= total) return;
    
    int f = idx % F;
    int tmp = idx / F;
    int t = tmp % T;
    int b = tmp / T;

    float sig = sigma[idx];
    float tfr_w = tfr_weighted[idx];
    int omega = (int)omega_hat[idx];

    // 动态 K_eff（与 forward 一致, 保证 grad 截断范围与 forward 严格对应）
    int K_eff = (int)ceilf(6.0f * sig);
    if (K_eff > K) K_eff = K;
    if (K_eff < 1) K_eff = 1;

    float sig3 = sig * sig * sig;
    float Z = 0.0f, dZ = 0.0f;
    
    for (int k = -K_eff; k <= K_eff; k++) {
        float r = (float)k / sig;
        float e = expf(-0.5f * r * r);
        Z += e;
        dZ += e * (float)(k * k);
    }
    Z += 1e-8f;
    dZ = dZ / sig3;
    float inv_Z = 1.0f / Z;
    float gs = 0.0f;

    for (int k = -K_eff; k <= K_eff; k++) {
        float r = (float)k / sig;
        float e = expf(-0.5f * r * r);
        float w_k = e * inv_Z;
        float dw_k = w_k * ((float)(k * k) / sig3 - dZ * inv_Z);
        
        int target = omega + k;
        target = target < 0 ? 0 : (target >= F ? F - 1 : target);
        
        // 同样注意 grad_out 读取的地址计算
        float grad_at = grad_out[b * T * F + t * F + target];
        gs += grad_at * tfr_w * dw_k;
    }
    grad_sigma[idx] = gs;
}

void reassigner_forward_cuda(torch::Tensor tfr_weighted, torch::Tensor sigma,
                             torch::Tensor omega_hat, torch::Tensor tfr_enhanced, int K) {
    // 此时传入的张量应该是 [B, T, F] 格式
    int B = tfr_weighted.size(0);
    int T = tfr_weighted.size(1);
    int F = tfr_weighted.size(2);
    int total = B * T * F;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    reassigner_forward_kernel_transposed<<<blocks, threads>>>(
        tfr_weighted.data_ptr<float>(), sigma.data_ptr<float>(),
        omega_hat.data_ptr<int64_t>(), tfr_enhanced.data_ptr<float>(),
        B, T, F, K);
}

void reassigner_backward_cuda(torch::Tensor grad_out, torch::Tensor sigma,
                              torch::Tensor tfr_weighted, torch::Tensor omega_hat,
                              torch::Tensor grad_sigma, int K) {
    int B = grad_out.size(0);
    int T = grad_out.size(1);
    int F = grad_out.size(2);
    int total = B * T * F;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    reassigner_backward_kernel_transposed<<<blocks, threads>>>(
        grad_out.data_ptr<float>(), sigma.data_ptr<float>(),
        tfr_weighted.data_ptr<float>(), omega_hat.data_ptr<int64_t>(),
        grad_sigma.data_ptr<float>(), B, T, F, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reassigner_forward_cuda, "reassigner forward transposed");
    m.def("backward", &reassigner_backward_cuda, "reassigner backward transposed");
}