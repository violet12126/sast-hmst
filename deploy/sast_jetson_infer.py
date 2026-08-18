"""
SAST Jetson 推理封装: TRT engine (网络部分) + CUDA kernel (重排)
===================================================================

部署架构:
  Signal [1, T]
    │
    ├─ TRT engine (sast_net.engine):  MSST→节点提取→PPM→GAT→sigma/ridge
    │    → tfr_mag, sigma, omega_hat_int, ridge_factor, n_sqz_per_bin
    │
    └─ CUDA kernel (reassigner.cu):   多轮高斯重排
         ← 5 个中间量 → tfr_enhanced [F, T]

为什么重排用 CUDA kernel 而非 ONNX:
  SAST 的 91 次高斯循环若导出 ONNX 会 unroll 成 7000+ 节点巨图 → OOM.
  CUDA kernel 单次 launch 完成 91 循环, 无中间张量累积 (正是 reassigner.cu 的意义).

用法 (Jetson):
  python3 sast_jetson_infer.py --engine sast_net.engine \
      --signal signal.npy --output tfr.npy

  # Python API
  from sast_jetson_infer import SastJetson
  sast = SastJetson('sast_net.engine')
  tfr = sast(signal)   # [F, T]
"""
import argparse
import os
import sys
from typing import Optional

import numpy as np

# 复用重排 CUDA kernel (Jetson 上需先编译 reassigner.cu)
try:
    import torch
    import deploy.reassigner as _reassigner_cpp
    _HAS_REASSIGNER = True
except Exception:
    _HAS_REASSIGNER = False


# ═══════════════════════════════════════════════════════════════
# 重排: CUDA kernel 版 (无 Python 91 循环)
# ═══════════════════════════════════════════════════════════════

def _reassign_cuda(tfr_mag, sigma, omega_hat_int, K, F_dim, n_sqz_per_bin,
                   ridge_factor, n_sqz_max, device):
    """多轮重排, 用 reassigner.cu kernel (每轮单次 launch).

    Args:
        全部 [1, F, T] CUDA float32 tensors (omega_hat_int/n_sqz_per_bin int64)
    Returns:
        [1, F, T] float32 增强 TFR
    """
    tfr_cur = tfr_mag.contiguous()
    sigma_f = sigma.contiguous().float()
    omega_f = omega_hat_int.contiguous().long()
    for r in range(1, n_sqz_max + 1):
        src_mask = (n_sqz_per_bin >= r).float()
        part = (src_mask * ridge_factor).contiguous()
        moved = torch.zeros_like(tfr_cur)
        _reassigner_cpp.forward(tfr_cur * part, sigma_f, omega_f, moved, K)
        tfr_cur = moved + tfr_cur * (1.0 - part)
    return tfr_cur


# ═══════════════════════════════════════════════════════════════
# TRT engine runner
# ═══════════════════════════════════════════════════════════════

class _TrtRunner:
    """加载 .engine, 分配 IO 内存, run 一次网络部分前向."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._bindings = {}
        self._alloc_buffers()

    def _alloc_buffers(self):
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        self._cuda = cuda
        self._n_inputs = 0
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            np_dtype = np.float32 if dtype.name == 'FLOAT' else np.int64
            nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
            buf = cuda.mem_alloc(nbytes)
            self._bindings[name] = buf
            if self.engine.get_tensor_mode(name) == 'INPUT':
                self._n_inputs += 1
                self.input_name = name
                self.input_shape = shape
                self.input_nbytes = nbytes
        self._output_names = [self.engine.get_tensor_name(i)
                              for i in range(self.engine.num_io_tensors)
                              if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == 'OUTPUT']

    def run(self, signal_np: np.ndarray) -> dict:
        """signal_np: [1, T] float32 → dict of [1,F,T] outputs."""
        cuda = self._cuda
        # 上传输入
        cuda.memcpy_htod(self._bindings[self.input_name],
                         np.ascontiguousarray(signal_np))
        # 绑定所有 IO
        for name, buf in self._bindings.items():
            self.context.set_tensor_address(name, int(buf))
        self.context.execute_async_v3(0)
        cuda.Context.synchronize()
        # 下载输出
        out = {}
        for name in self._output_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = np.float32 if self.engine.get_tensor_dtype(name).name == 'FLOAT' else np.int64
            buf = np.empty(int(np.prod(shape)), dtype=dtype)
            cuda.memcpy_dtoh(buf, self._bindings[name])
            out[name] = buf.reshape(shape)
        return out


# ═══════════════════════════════════════════════════════════════
# 顶层封装
# ═══════════════════════════════════════════════════════════════

class SastJetson:
    """Jetson SAST 推理: TRT 网络部分 + CUDA 重排, 输出增强 TFR."""

    def __init__(self, engine_path: str, n_sqz_max: int = 4,
                 sigma_min: float = 0.5, sigma_max: float = 15.0,
                 device_id: int = 0):
        if not _HAS_REASSIGNER:
            raise RuntimeError('reassigner CUDA kernel 未加载. 请先在 Jetson 上编译: '
                               'cd deploy && python setup_msst_kernels.py build_ext --inplace')
        import torch
        self.device = torch.device(f'cuda:{device_id}')
        self.n_sqz_max = n_sqz_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._trt = _TrtRunner(engine_path)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """signal: [T] or [1, T] float32 → tfr_enhanced [F, T]."""
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal[None, :]

        # ── 1. TRT 网络部分 ──
        mid = self._trt.run(signal)

        # ── 2. 组装 CUDA 重排输入 ──
        torch.cuda.set_device(self.device)
        tfr_mag = torch.from_numpy(mid['tfr_mag']).to(self.device)
        sigma = torch.from_numpy(mid['sigma']).to(self.device).clamp(self.sigma_min, self.sigma_max)
        omega_hat = torch.from_numpy(mid['omega_hat_int']).to(self.device).long()
        ridge = torch.from_numpy(mid['ridge_factor']).to(self.device)
        n_sqz = torch.from_numpy(mid['n_sqz_per_bin']).to(self.device).long()

        F_dim = tfr_mag.shape[1]
        K = int(np.ceil(3.0 * self.sigma_max))

        # ── 3. 多轮重排 (CUDA kernel) ──
        tfr_enh = _reassign_cuda(tfr_mag, sigma, omega_hat, K, F_dim,
                                 n_sqz, ridge, self.n_sqz_max, self.device)
        return tfr_enh[0].cpu().numpy()


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='SAST Jetson 推理 (TRT + CUDA 重排)')
    p.add_argument('--engine', required=True, help='TRT engine (build_trt_sast.py 输出)')
    p.add_argument('--signal', required=True, help='输入信号 .npy [T]')
    p.add_argument('--output', default='tfr_enhanced.npy', help='输出 TFR .npy')
    p.add_argument('--n-sqz-max', type=int, default=4)
    p.add_argument('--device', type=int, default=0)
    args = p.parse_args()

    signal = np.load(args.signal)
    print(f'Signal: {signal.shape}, dtype={signal.dtype}')

    sast = SastJetson(args.engine, n_sqz_max=args.n_sqz_max, device_id=args.device)
    tfr = sast(signal)
    np.save(args.output, tfr)
    print(f'TFR: {tfr.shape}, saved to {args.output}')
    print(f'  energy: {tfr.sum():.2f}')


if __name__ == '__main__':
    main()
