from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda-flash-attention-lite",
    version="0.1.0",
    description="Educational CUDA C++ FlashAttention forward kernel",
    py_modules=["flash_attention"],
    ext_modules=[
        CUDAExtension(
            name="flash_attention_cuda",
            sources=["csrc/flash_attention.cpp", "csrc/flash_attention.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
