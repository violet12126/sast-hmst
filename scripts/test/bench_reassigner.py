"""
Standalone microbenchmark for the reassigner CUDA kernels
==========================================================
直接对重排算子做隔离计时 (对比 reassigner.cu [B,F,T] vs reassigner_opt.cu [B,T,F]),
并作为 nsys / ncu 的驱动脚本 (配合 --iters 1 减少采样量)。

用法:
  # 纯计时 (torch.cuda.Event, 比 perf_counter 更精确)
  python scripts/test/bench_reassigner.py
  python scripts/test/bench_reassigner.py --opt-only --iters 200
  python scripts/test/bench_reassigner.py --sigma 3.0 --sigma-max 15.0

  # nsys 采样 (少迭代, 便于查看 kernel 时间线)
  nsys profile -o reassigner_prof --trace cuda,nvtx \
      python scripts/test/bench_reassigner.py --opt-only --iters 20

注意:
  - reassigner.cu   期望 [B, F, T] (模型当前实际使用的版本)
  - reassigner_opt.cu 期望 [B, T, F] (转置布局 + 动态 K_eff=ceil(6σ))
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 必须先 import torch (pyd 依赖 torch 的 DLL)
import deploy.reassigner as reassigner          # [B, F, T]
import deploy.reassigner_opt as reassigner_opt  # [B, T, F]


def _make_tensors(B, T, F, sigma_val, device):
    """构造两种布局下语义等价的输入张量."""
    torch.manual_seed(0)
    # [B, F, T] (非 opt)
    tfr_bfT = torch.rand(B, F, T, device=device, dtype=torch.float32)
    sigma_bfT = torch.full((B, F, T), sigma_val, device=device, dtype=torch.float32)
    omega_bfT = torch.randint(0, F, (B, F, T), device=device, dtype=torch.int64)

    # [B, T, F] (opt, = 上面的转置)
    tfr_bTf = tfr_bfT.transpose(1, 2).contiguous()
    sigma_bTf = sigma_bfT.transpose(1, 2).contiguous()
    omega_bTf = omega_bfT.transpose(1, 2).contiguous()
    return (tfr_bfT, sigma_bfT, omega_bfT), (tfr_bTf, sigma_bTf, omega_bTf)


def bench(module, tensors, out_shape, K, iters, name, device):
    """用 torch.cuda.Event 计时 forward + backward."""
    tfr, sigma, omega = tensors
    out = torch.zeros(out_shape, device=device, dtype=torch.float32)
    grad_out = torch.rand_like(out)
    grad_sigma = torch.zeros_like(sigma)

    # warmup
    for _ in range(5):
        out.zero_()
        module.forward(tfr, sigma, omega, out, K)
        module.backward(grad_out, sigma, tfr, omega, grad_sigma, K)
    torch.cuda.synchronize()

    # forward timing
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        out.zero_()
        module.forward(tfr, sigma, omega, out, K)
    e.record()
    torch.cuda.synchronize()
    fwd_ms = s.elapsed_time(e) / iters

    # backward timing
    s.record()
    for _ in range(iters):
        module.backward(grad_out, sigma, tfr, omega, grad_sigma, K)
    e.record()
    torch.cuda.synchronize()
    bwd_ms = s.elapsed_time(e) / iters

    print(f"  {name:<18s} fwd={fwd_ms*1000:8.1f} us   bwd={bwd_ms*1000:8.1f} us   "
          f"total={total_elements(tfr)/1e6:.2f}M elems")
    return fwd_ms, bwd_ms


def total_elements(t):
    return t.numel()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--B', type=int, default=1)
    p.add_argument('--T', type=int, default=256)     # STFT 帧数
    p.add_argument('--F', type=int, default=501)     # 频点 (fs=1000, n_fft=1000)
    p.add_argument('--K', type=int, default=None,
                   help='半窗半径; 默认=ceil(3*sigma_max)=45')
    p.add_argument('--sigma', type=float, default=3.0)
    p.add_argument('--sigma-max', type=float, default=15.0)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--opt-only', action='store_true',
                   help='仅测 reassigner_opt (nsys 采样用)')
    args = p.parse_args()

    device = 'cuda'
    if not torch.cuda.is_available():
        print("CUDA not available, abort.")
        return

    K = args.K if args.K is not None else int(3.0 * args.sigma_max)
    (tfr_bfT, sigma_bfT, omega_bfT), (tfr_bTf, sigma_bTf, omega_bTf) = \
        _make_tensors(args.B, args.T, args.F, args.sigma, device)

    print(f"Reassigner microbenchmark  B={args.B} T={args.T} F={args.F} "
          f"K={K} sigma={args.sigma}  iters={args.iters}")
    print(f"  total elements = {args.B*args.T*args.F:,}")

    if args.opt_only:
        bench(reassigner_opt, (tfr_bTf, sigma_bTf, omega_bTf),
              (args.B, args.T, args.F), K, args.iters, "opt [B,T,F]", device)
    else:
        bench(reassigner, (tfr_bfT, sigma_bfT, omega_bfT),
              (args.B, args.F, args.T), K, args.iters, "base [B,F,T]", device)
        bench(reassigner_opt, (tfr_bTf, sigma_bTf, omega_bTf),
              (args.B, args.T, args.F), K, args.iters, "opt [B,T,F]", device)


if __name__ == '__main__':
    main()
