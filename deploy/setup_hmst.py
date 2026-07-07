"""
编译 HMST CUDA 扩展
===================

用法:
  # 本地 GPU (自动检测架构)
  python deploy/setup_hmst.py

  # Jetson Orin (sm_87)
  TORCH_CUDA_ARCH_LIST="8.7" python deploy/setup_hmst.py

  # 开发模式 (JIT, 无需预编译)
  # sast.py 中已内置 JIT fallback, 首次 import 时自动编译

输出: deploy/hmst_cuda_ext.so (或 .pyd on Windows)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

if __name__ == '__main__':
    # 从环境变量获取目标架构, 或自动检测
    arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
    if arch_list:
        print(f"[setup_hmst] Targeting CUDA arch: {arch_list}")
    else:
        print("[setup_hmst] Auto-detecting CUDA arch (PyTorch default)")

    setup(
        name='hmst_cuda_ext',
        ext_modules=[
            CUDAExtension(
                name='hmst_cuda_ext',
                sources=['hmst_squeeze.cu'],
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': ['-O3', '--use_fast_math'],
                },
            ),
        ],
        cmdclass={
            'build_ext': BuildExtension,
        },
        # 输出到 deploy/ 目录
        script_args=['build_ext', '--inplace'],
    )
