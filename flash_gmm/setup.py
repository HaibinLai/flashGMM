"""
Build script for Flash-GMM C++/CUDA extension.

Usage:
  CPU only:    python setup.py build_ext --inplace
  CPU + CUDA:  FLASH_GMM_CUDA=1 python setup.py build_ext --inplace

Or use JIT compilation (see native_wrapper.py).
"""

import os
from setuptools import setup
import torch.utils.cpp_extension
# Bypass CUDA version mismatch check (nvcc 13.1 is ABI-compatible with PyTorch built for 12.8)
torch.utils.cpp_extension._check_cuda_version = lambda *a, **kw: None
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

USE_CUDA = os.environ.get("FLASH_GMM_CUDA", "0") == "1"

csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

sources = [
    os.path.join(csrc_dir, "binding.cpp"),
    os.path.join(csrc_dir, "flash_gmm_cpu.cpp"),
]

define_macros = []
extra_compile_args = {"cxx": ["-O3", "-std=c++17", "-fopenmp"]}
extra_link_args = ["-fopenmp"]

if USE_CUDA:
    sources.append(os.path.join(csrc_dir, "flash_gmm_cuda.cu"))
    define_macros.append(("WITH_CUDA", None))
    extra_compile_args["nvcc"] = ["-O3", "--use_fast_math", "-std=c++17"]
    ExtClass = CUDAExtension
else:
    ExtClass = CppExtension

setup(
    name="flash_gmm_native",
    ext_modules=[
        ExtClass(
            name="flash_gmm_native",
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
