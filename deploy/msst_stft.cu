/*
 * msst_stft.cu - CUDA Modulated-Form STFT (exact match numpy msst() Step 1)
 * ==========================================================================
 *
 * 复刻 models/tfr.py msst() 的 Step 1:
 *   1. 逐列填充 tfr_pre[N, tcol]: x[t+tau] * conj(h[Lh+tau]) at cyclic index (N+tau)%N
 *   2. cuFFT batch 1D FFT over columns
 *   3. 取前 neta = round(N/2) 行 (positive frequencies)
 *
 * Memory layout: ROW-MAJOR (C-order) - matches torch/numpy.
 *   tfr_pre[i, j] = tfr_pre_flat[i * tcol + j]
 *   Column j has stride tcol between elements.
 *
 * 输入:
 *   x:       [N] double  - signal
 *   h:       [hlength] double - Gaussian window (real)
 *   N, hlength, Lh, neta, tcol: STFT parameters
 *
 * 输出:
 *   tfr: [neta, tcol] complex128 - STFT (positive frequencies only)
 *
 * 编译:
 *   python deploy/setup_msst_kernels.py build_ext --inplace
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>

// Helper: min of three integers
__device__ inline int min3(int a, int b, int c) {
    int m = (a < b) ? a : b;
    return (m < c) ? m : c;
}

// Fill tfr_pre kernel: one block per time column
__global__ void fill_tfr_pre_kernel(
    const double* __restrict__ x,
    const double* __restrict__ h,
    cuDoubleComplex* __restrict__ tfr_pre,
    int N, int hlength, int Lh, int neta, int tcol
) {
    int col = blockIdx.x;
    if (col >= tcol) return;

    int tau_min = -min3(neta - 1, Lh, col);
    int tau_max = min3(neta - 1, Lh, N - 1 - col);
    int n_tau = tau_max - tau_min + 1;
    if (n_tau <= 0) return;

    int tid = threadIdx.x;
    if (tid >= n_tau) return;

    int tau = tau_min + tid;
    int row = (N + tau) % N;
    int sig_idx = col + tau;
    int win_idx = Lh + tau;

    double val = x[sig_idx] * h[win_idx];
    // Row-major: row * tcol + col
    tfr_pre[row * tcol + col] = make_cuDoubleComplex(val, 0.0);
}

// PyTorch wrapper
torch::Tensor msst_stft_cuda(
    torch::Tensor x,
    torch::Tensor h_window,
    int N, int hlength, int Lh, int neta, int tcol
) {
    TORCH_CHECK(x.is_cuda(),      "x must be on CUDA");
    TORCH_CHECK(h_window.is_cuda(), "h_window must be on CUDA");
    TORCH_CHECK(x.dim() == 1,     "x must be 1D");
    TORCH_CHECK(x.dtype() == torch::kFloat64, "x must be float64");

    auto x_contig = x.contiguous();
    auto h_contig = h_window.contiguous();

    // Allocate tfr_pre [N * tcol] flat, row-major
    auto options = torch::TensorOptions()
        .dtype(torch::kComplexDouble)
        .device(x.device());
    auto tfr_pre = torch::zeros({N * tcol}, options);

    // Launch fill kernel
    // Max n_tau ~ 2*Lh+1 ~ 2*256+1 = 513 for hlength=min(N,512)=513
    int threads = 1024;
    int blocks = tcol;

    fill_tfr_pre_kernel<<<blocks, threads>>>(
        x_contig.data_ptr<double>(),
        h_contig.data_ptr<double>(),
        reinterpret_cast<cuDoubleComplex*>(tfr_pre.data_ptr()),
        N, hlength, Lh, neta, tcol
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fill_tfr_pre_kernel failed: ", cudaGetErrorString(err));

    // cuFFT: batch 1D FFT over N-point columns
    // Row-major [N, tcol]: column j is at indices j, j+tcol, j+2*tcol, ...
    // Stride between consecutive elements of same column = tcol
    // Distance between consecutive columns = 1
    cufftHandle plan;
    int n_fft[1] = {N};
    int inembed[1] = {N};

    cufftResult cufft_err = cufftPlanMany(
        &plan, 1, n_fft,
        inembed, tcol, 1,   // istride=tcol, idist=1
        inembed, tcol, 1,   // ostride=tcol, odist=1
        CUFFT_Z2Z, tcol
    );
    TORCH_CHECK(cufft_err == CUFFT_SUCCESS, "cufftPlanMany failed");

    cufft_err = cufftExecZ2Z(
        plan,
        reinterpret_cast<cuDoubleComplex*>(tfr_pre.data_ptr()),
        reinterpret_cast<cuDoubleComplex*>(tfr_pre.data_ptr()),
        CUFFT_FORWARD
    );
    TORCH_CHECK(cufft_err == CUFFT_SUCCESS, "cufftExecZ2Z failed");

    cufftDestroy(plan);

    // Extract first neta rows: [neta, tcol] row-major
    auto tfr_pre_2d = tfr_pre.view({N, tcol});
    auto tfr = tfr_pre_2d.slice(0, 0, neta).clone();

    return tfr;
}

// pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("msst_stft_cuda", &msst_stft_cuda,
          "CUDA modulated-form STFT (exact match numpy msst() Step 1)\n\n"
          "Args:\n"
          "  x:        [N] float64 signal\n"
          "  h_window: [hlength] float64 Gaussian window\n"
          "  N, hlength, Lh, neta, tcol: STFT parameters\n\n"
          "Returns:\n"
          "  tfr: [neta, tcol] complex128 STFT (positive frequencies only)",
          py::arg("x"),
          py::arg("h_window"),
          py::arg("N"),
          py::arg("hlength"),
          py::arg("Lh"),
          py::arg("neta"),
          py::arg("tcol"));
}
