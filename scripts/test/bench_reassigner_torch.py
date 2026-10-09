"""
临时脚手架: 测"原版 torch 实现"(Python fallback) 的重排算子耗时
=================================================================
直接复刻 models/sast.py ReassignerFunction 的 Python fallback 逻辑
(91 次 Python 循环 + 每次独立的 torch.exp / scatter_add_ kernel),
用 torch.cuda.Event 计时, 与 reassigner_opt.cu 的 CUDA kernel 耗时对比。

用法:
  D:/Anaconda/envs/my_work/python.exe scripts/test/bench_reassigner_torch.py --sigma 3.0
  D:/Anaconda/envs/my_work/python.exe scripts/test/bench_reassigner_torch.py --sigma 0.5

注意: 本脚本只测 torch 算子路径, 不 import deploy.*, 可独立运行。
(可删; 仅用于给 nsys_interview_qa.md 补充"原版 torch 实现"的真实耗时数据)
"""
import argparse
import torch


def fwd_torch(tfr_weighted, sigma, omega_hat_int, K, F_dim):
    """复刻 ReassignerFunction.forward 的 Python fallback (models/sast.py L664-677)."""
    B, F, T = tfr_weighted.shape
    tfr_enhanced = torch.zeros(B, F, T, device=tfr_weighted.device, dtype=tfr_weighted.dtype)
    dtype = sigma.dtype
    eps = 1e-8
    offsets = list(range(-K, K + 1))
    Z = torch.zeros(B, F, T, device=tfr_weighted.device, dtype=dtype)
    for k in offsets:
        r = k / sigma
        Z = Z + torch.exp(-0.5 * r * r)
    Z = Z + eps
    for k in offsets:
        r = k / sigma
        w_k = torch.exp(-0.5 * r * r) / Z
        target = (omega_hat_int + k).clamp(0, F - 1)
        tfr_enhanced.scatter_add_(1, target, w_k * tfr_weighted)
    return tfr_enhanced


def bwd_torch(grad_out, sigma, tfr_weighted, omega_hat_int, K, F_dim):
    """复刻 ReassignerFunction.backward 的 Python fallback (models/sast.py L708-732)."""
    B, F, T = sigma.shape
    dtype = sigma.dtype
    eps = 1e-8
    offsets = list(range(-K, K + 1))
    sigma3 = sigma * sigma * sigma
    Z = torch.zeros_like(sigma)
    dZ = torch.zeros_like(sigma)
    for k in offsets:
        r = k / sigma
        e = torch.exp(-0.5 * r * r)
        Z = Z + e
        dZ = dZ + e * float(k * k)
    Z = Z + eps
    dZ = dZ / sigma3
    inv_Z = 1.0 / Z
    grad_sigma = torch.zeros_like(sigma)
    for k in offsets:
        r = k / sigma
        e = torch.exp(-0.5 * r * r)
        w_k = e * inv_Z
        dw_k = w_k * (float(k * k) / sigma3 - dZ * inv_Z)
        target = (omega_hat_int + k).clamp(0, F - 1)
        grad_at = grad_out.gather(1, target)
        grad_sigma = grad_sigma + grad_at * tfr_weighted * dw_k
    return grad_sigma


def timeit(fn, iters):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--B', type=int, default=1)
    p.add_argument('--T', type=int, default=256)
    p.add_argument('--F', type=int, default=501)
    p.add_argument('--K', type=int, default=45)
    p.add_argument('--sigma', type=float, default=3.0)
    p.add_argument('--iters', type=int, default=20)
    args = p.parse_args()

    device = 'cuda'
    if not torch.cuda.is_available():
        print("CUDA not available, abort.")
        return

    torch.manual_seed(0)
    B, T, F, K = args.B, args.T, args.F, args.K
    tfr = torch.rand(B, F, T, device=device, dtype=torch.float32)
    sigma = torch.full((B, F, T), args.sigma, device=device, dtype=torch.float32)
    omega = torch.randint(0, F, (B, F, T), device=device, dtype=torch.int64)

    out = fwd_torch(tfr, sigma, omega, K, F)
    grad_out = torch.rand_like(out)
    gs = bwd_torch(grad_out, sigma, tfr, omega, K, F)

    # warmup
    for _ in range(2):
        out = fwd_torch(tfr, sigma, omega, K, F)
        gs = bwd_torch(grad_out, sigma, tfr, omega, K, F)
    torch.cuda.synchronize()

    fwd_ms = timeit(lambda: fwd_torch(tfr, sigma, omega, K, F), args.iters)
    bwd_ms = timeit(lambda: bwd_torch(grad_out, sigma, tfr, omega, K, F), args.iters)
    print(f"torch-fallback  B={B} T={T} F={F} K={K} sigma={args.sigma}  "
          f"fwd={fwd_ms*1000:8.1f} us   bwd={bwd_ms*1000:8.1f} us   "
          f"(91 循环, 与 sigma 无关)")


if __name__ == '__main__':
    main()
