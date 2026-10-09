import pytest
import torch

try:
    from flash_attention import flash_attention  # noqa: F401
    import flash_attention_cuda  # noqa: F401
    _EXTENSION_AVAILABLE = True
except Exception:
    _EXTENSION_AVAILABLE = False

requires_cuda_extension = pytest.mark.skipif(
    not torch.cuda.is_available() or not _EXTENSION_AVAILABLE,
    reason="requires an NVIDIA GPU and successfully built CUDA extension",
)
