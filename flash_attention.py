"""Python entry point for the simplified CUDA FlashAttention extension."""
import torch

try:
    import flash_attention_cuda
except ImportError as exc:
    flash_attention_cuda = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """Compute attention for contiguous [B, H, S, D] CUDA tensors."""
    if flash_attention_cuda is None:
        raise RuntimeError(
            "flash_attention_cuda extension is not built. Run python -m pip install -e . "
            "--no-build-isolation with CUDA-enabled PyTorch and nvcc available."
        ) from _IMPORT_ERROR
    return flash_attention_cuda.forward(q, k, v, causal)
