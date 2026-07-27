"""
编译 MSST CUDA 扩展 (STFT + 硬挤压 + 线性挤压)
================================================

输出三个独立 CUDA 扩展:
  - msst_stft:           modulated-form STFT (精确匹配 numpy msst() Step 1, cuFFT)
  - msst_squeeze_hard:   硬最近邻挤压 (匹配 numpy MSST)
  - msst_squeeze_linear: 线性插值挤压 (连续 IF, 双 bin)

用法:
  # 本地 GPU (自动检测架构)
  python deploy/setup_msst_kernels.py build_ext --inplace

  # 指定 CUDA 架构
  TORCH_CUDA_ARCH_LIST="8.6" python deploy/setup_msst_kernels.py build_ext --inplace

输出: deploy/*.pyd (Windows) 或 *.so (Linux)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys

if __name__ == '__main__':
    arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
    if arch_list:
        print(f"[setup_msst] Targeting CUDA arch: {arch_list}")
    else:
        print("[setup_msst] Auto-detecting CUDA arch (PyTorch default)")

    build_dir = os.path.dirname(os.path.abspath(__file__))

    setup(
        name='msst_cuda_kernels',
        ext_modules=[
            CUDAExtension(
                name='msst_squeeze_hard',
                sources=[os.path.join(build_dir, 'msst_squeeze_hard.cu')],
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': ['-O3', '--use_fast_math', '-allow-unsupported-compiler',
          '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'],
                },
            ),
            CUDAExtension(
                name='msst_squeeze_linear',
                sources=[os.path.join(build_dir, 'msst_squeeze_linear.cu')],
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': ['-O3', '--use_fast_math', '-allow-unsupported-compiler',
          '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'],
                },
            ),
            CUDAExtension(
                name='msst_stft',
                sources=[os.path.join(build_dir, 'msst_stft.cu')],
                libraries=['cufft'],
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': ['-O3', '-allow-unsupported-compiler',
          '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'],
                },
            ),
        ],
        cmdclass={
            'build_ext': BuildExtension,
        },
    )
    print("[setup_msst] Usage: python deploy/setup_msst_kernels.py build_ext --inplace")
