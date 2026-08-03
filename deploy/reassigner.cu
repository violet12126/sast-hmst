// reassigner.cu - 可微软高斯重排 CUDA kernel (forward + backward)
// 替代 Python ReassignerFunction 的 91 循环, 一次 kernel launch.
//
// forward: tfr_enhanced[omega_hat+k] += (exp(-0.5*(k/sigma)^2)/Z) * tfr_weighted
// backward: grad_sigma = sum_k grad_out[omega_hat+k] * tfr_weighted * dw_k/dsigma
//
// 输入 dtype: tfr_weighted/sigma/grad float32, omega_hat int64

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ── forward kernel: 每 thread 一个 (b, f, t) 源 bin ──
__global__ void reassigner_forward_kernel(
    const float* __restrict__ tfr_weighted,
    const float* __restrict__ sigma,
    const int64_t*  __restrict__ omega_hat,
    float* __restrict__ tfr_enhanced,
    int B, int F, int T, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * F * T;
    if (idx >= total) return;
    int t = idx % T;
    int b = idx / (F * T);

    float sig = sigma[idx];
    float tfr_w = tfr_weighted[idx];
    int omega = (int)omega_hat[idx];

    // Z = sum_k exp(-0.5*(k/sig)^2)
    float Z = 0.0f;
    #pragma unroll 8
    for (int k = -K; k <= K; k++) {
        float r = (float)k / sig;
        Z += expf(-0.5f * r * r);
    }
    Z += 1e-8f;

    // scatter w_k * tfr_w to target (atomicAdd)
    for (int k = -K; k <= K; k++) {
        float r = (float)k / sig;
        float w_k = expf(-0.5f * r * r) / Z;
        int target = omega + k;
        if (target < 0) target = 0;
        else if (target >= F) target = F - 1;
        atomicAdd(&tfr_enhanced[b * F * T + target * T + t], w_k * tfr_w);
    }
}

// ── backward kernel: grad_sigma = sum_k grad_out[target] * tfr_weighted * dw_k ──
__global__ void reassigner_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ sigma,
    const float* __restrict__ tfr_weighted,
    const int64_t*  __restrict__ omega_hat,
    float* __restrict__ grad_sigma,
    int B, int F, int T, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * F * T;
    if (idx >= total) return;
    int t = idx % T;
    int b = idx / (F * T);

    float sig_f = sigma[idx];
    float tfr_w_f = tfr_weighted[idx];
    int omega = (int)omega_hat[idx];
    // double 内部累加: backward 的 dw_k = w_k*(k^2/sig^3 - dZ/Z) 是两个接近大数相减,
    // float32 灾难抵消丢精度 (grad 差 6 数量级). 用 double 累加, 写回 float.
    double sig = (double)sig_f;
    double sig3 = sig * sig * sig;

    // Z + dZ/dsigma = (1/sig^3) sum_k exp_k * k^2
    double Z = 0.0, dZ = 0.0;
    for (int k = -K; k <= K; k++) {
        double r = (double)k / sig;
        double e = exp(-0.5 * r * r);
        Z += e;
        dZ += e * (double)(k * k);
    }
    Z += 1e-8;
    dZ = dZ / sig3;
    double inv_Z = 1.0 / Z;

    // grad_sigma = sum_k grad_out[target] * tfr_w * dw_k
    // dw_k = w_k * (k^2/sig^3 - dZ/Z)
    double gs = 0.0;
    for (int k = -K; k <= K; k++) {
        double r = (double)k / sig;
        double e = exp(-0.5 * r * r);
        double w_k = e * inv_Z;
        double dw_k = w_k * ((double)(k * k) / sig3 - dZ * inv_Z);
        int target = omega + k;
        if (target < 0) target = 0;
        else if (target >= F) target = F - 1;
        double grad_at = (double)grad_out[b * F * T + target * T + t];
        gs += grad_at * (double)tfr_w_f * dw_k;
    }
    grad_sigma[idx] = (float)gs;
}

// ── C++ wrappers ──
void reassigner_forward_cuda(torch::Tensor tfr_weighted, torch::Tensor sigma,
                             torch::Tensor omega_hat, torch::Tensor tfr_enhanced, int K) {
    int B = tfr_weighted.size(0);
    int F = tfr_weighted.size(1);
    int T = tfr_weighted.size(2);
    int total = B * F * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reassigner_forward_kernel<<<blocks, threads>>>(
        tfr_weighted.data_ptr<float>(), sigma.data_ptr<float>(),
        omega_hat.data_ptr<int64_t>(), tfr_enhanced.data_ptr<float>(),
        B, F, T, K);
}

void reassigner_backward_cuda(torch::Tensor grad_out, torch::Tensor sigma,
                              torch::Tensor tfr_weighted, torch::Tensor omega_hat,
                              torch::Tensor grad_sigma, int K) {
    int B = grad_out.size(0);
    int F = grad_out.size(1);
    int T = grad_out.size(2);
    int total = B * F * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reassigner_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(), sigma.data_ptr<float>(),
        tfr_weighted.data_ptr<float>(), omega_hat.data_ptr<int64_t>(),
        grad_sigma.data_ptr<float>(), B, F, T, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reassigner_forward_cuda, "reassigner forward (CUDA)");
    m.def("backward", &reassigner_backward_cuda, "reassigner backward (CUDA)");
}
