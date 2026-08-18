"""
SAST 网络部分 INT8 校准器 (TRT EntropyCalibrator2)
====================================================

从校准数据估计每个激活张量的分布, 计算 INT8 量化 scale.
校准数据: npy/npz, 含 'train_X' (或 'calib_X'), [N, T] 信号.

用法 (由 build_trt_sast.py --calib 调用):
  calibrator = SastEntropyCalibrator('calib.npz', batch_size=8)
  config.int8_calibrator = calibrator
"""
import os
from typing import Optional

import numpy as np
import tensorrt as trt

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    _HAS_PYCUDA = True
except ImportError:
    _HAS_PYCUDA = False


class SastEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_path: str, batch_size: int = 8,
                 max_len: int = 2000, cache_file: str = 'sast_int8.cache'):
        super().__init__()
        if not _HAS_PYCUDA:
            raise RuntimeError('INT8 校准需要 pycuda: pip install pycuda')

        # 加载数据
        if data_path.endswith('.npz'):
            d = np.load(data_path, allow_pickle=True)
            X = d.get('train_X', d.get('calib_X', d['X']))
        else:  # .npy
            X = np.load(data_path)
        if X.ndim == 3:
            X = X[:, :, 0]
        X = X[:, :max_len].astype(np.float32)
        print(f'[calib] {X.shape} samples x {max_len}')

        self.batch_size = batch_size
        self.n_batches = len(X) // batch_size
        self.cur_batch = 0
        self.cache_file = cache_file

        # 预加载全部到 GPU (校准集通常小)
        self.device_buffers = []
        for i in range(self.n_batches):
            batch = X[i * batch_size:(i + 1) * batch_size]
            if not batch.any():
                continue
            buf = cuda.mem_alloc(batch.nbytes)
            cuda.memcpy_htod(buf, np.ascontiguousarray(batch))
            self.device_buffers.append(buf)
        self._n = len(self.device_buffers)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names: list, input_names: list = None) -> Optional[list]:
        """返回当前 batch 的 GPU 指针列表; 耗尽返回 None."""
        if self.cur_batch >= self._n:
            return None
        buf = self.device_buffers[self.cur_batch]
        self.cur_batch += 1
        return [int(buf)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f'[calib] cache saved: {self.cache_file}')
