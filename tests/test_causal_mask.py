import torch

from conftest import requires_cuda_extension
from flash_attention import flash_attention
from traditional_attention import traditional_attention


@requires_cuda_extension
def test_causal_mask_matches_traditional_reference_for_non_multiple_of_tile():
    torch.manual_seed(7)
    shape = (1, 1, 35, 48)
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.float32)
        for _ in range(3)
    ]
    actual = flash_attention(q.contiguous(), k.contiguous(), v.contiguous(), causal=True)
    expected = traditional_attention(q, k, v, causal=True)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@requires_cuda_extension
def test_causal_first_row_only_uses_first_key():
    q = torch.randn((1, 1, 5, 16), device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = flash_attention(q.contiguous(), k.contiguous(), v.contiguous(), causal=True)
    torch.testing.assert_close(out[0, 0, 0], v[0, 0, 0], atol=1e-5, rtol=1e-5)
