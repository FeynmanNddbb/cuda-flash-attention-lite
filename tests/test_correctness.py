import pytest
import torch
import torch.nn.functional as F

from conftest import requires_cuda_extension
from flash_attention import flash_attention


@pytest.mark.parametrize("dtype,atol,rtol", [
    (torch.float32, 2e-4, 2e-4),
    (torch.float16, 3e-3, 3e-3),
])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 2, 17, 32), (1, 2, 64, 64)])
@requires_cuda_extension
def test_matches_torch_sdpa(dtype, atol, rtol, causal, shape):
    torch.manual_seed(2026)
    q, k, v = [torch.randn(shape, device="cuda", dtype=dtype).contiguous() for _ in range(3)]
    actual = flash_attention(q, k, v, causal=causal)
    expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@requires_cuda_extension
def test_output_shape_dtype_and_device(dtype):
    x = torch.randn((2, 3, 9, 16), device="cuda", dtype=dtype)
    out = flash_attention(x, x, x, causal=False)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
