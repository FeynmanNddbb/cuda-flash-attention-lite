"""Naive PyTorch implementation of traditional scaled dot-product attention.

This intentionally materializes the full [B, H, S, S] score matrix and is used
as the correctness/performance baseline for the custom FlashAttention kernel.
"""
import math
import torch


def traditional_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Compute standard attention as QK^T -> scale -> mask -> softmax -> V."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, H, S, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have equal shapes")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must share a device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must share a dtype")

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        seq_len = q.size(-2)
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)
