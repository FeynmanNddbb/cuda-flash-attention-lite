#!/usr/bin/env python3
"""Compare explicit traditional attention with the custom FlashAttention CUDA kernel."""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from flash_attention import flash_attention
from traditional_attention import traditional_attention


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", default="128,256,512", help="Comma-separated sequence lengths")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()
    try:
        args.seq_lens = [int(item.strip()) for item in args.seq_lens.split(",") if item.strip()]
    except ValueError:
        parser.error("--seq-lens must be comma-separated positive integers")
    if not args.seq_lens or any(item <= 0 for item in args.seq_lens):
        parser.error("--seq-lens must contain positive integers")
    if min(args.batch, args.heads, args.head_dim, args.warmup, args.repeats) <= 0:
        parser.error("batch, heads, head-dim, warmup, and repeats must be positive")
    if args.head_dim > 128:
        parser.error("head-dim must be <= 128 in this implementation")
    return args


def measure(fn, q, k, v, warmup, repeats):
    for _ in range(warmup):
        result = fn(q, k, v)
        del result
    torch.cuda.synchronize()

    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        result = fn(q, k, v)
        del result
    end.record()
    torch.cuda.synchronize()

    latency_ms = start.elapsed_time(end) / repeats
    peak_delta_bytes = max(0, torch.cuda.max_memory_allocated() - baseline_bytes)
    return latency_ms, peak_delta_bytes


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run on a machine with an NVIDIA GPU.")

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    results = []

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}; dtype: {args.dtype}")
    print(f"batch={args.batch}, heads={args.heads}, head_dim={args.head_dim}, causal={args.causal}")
    print("Baseline: traditional attention explicitly materializes the full [B, H, S, S] score matrix.")
    print("Memory: peak PyTorch-allocated bytes above the pre-call baseline, not total device memory.")

    for seq_len in args.seq_lens:
        shape = (args.batch, args.heads, seq_len, args.head_dim)
        q, k, v = [torch.randn(shape, device=device, dtype=dtype).contiguous() for _ in range(3)]
        custom_fn = lambda a, b, c: flash_attention(a, b, c, causal=args.causal)
        traditional_fn = lambda a, b, c: traditional_attention(a, b, c, causal=args.causal)

        with torch.inference_mode():
            custom_out = custom_fn(q, k, v)
            traditional_out = traditional_fn(q, k, v)
            torch.cuda.synchronize()
            max_abs_error = (custom_out.float() - traditional_out.float()).abs().max().item()
            del custom_out, traditional_out

            custom_ms, custom_peak = measure(custom_fn, q, k, v, args.warmup, args.repeats)
            traditional_ms, traditional_peak = measure(
                traditional_fn, q, k, v, args.warmup, args.repeats
            )

        speedup = traditional_ms / custom_ms if custom_ms > 0 else float("nan")
        row = {
            "seq_len": seq_len,
            "shape": list(shape),
            "dtype": args.dtype,
            "causal": args.causal,
            "traditional_attention_latency_ms": traditional_ms,
            "flash_attention_latency_ms": custom_ms,
            "speedup_traditional_over_flash": speedup,
            "max_abs_error_vs_traditional": max_abs_error,
            "traditional_peak_allocated_delta_bytes": traditional_peak,
            "flash_peak_allocated_delta_bytes": custom_peak,
        }
        results.append(row)
        print(
            f"S={seq_len:5d} | traditional={traditional_ms:9.4f} ms "
            f"| flash={custom_ms:9.4f} ms | traditional/flash={speedup:7.3f}x "
            f"| max_abs_err={max_abs_error:.6g} "
            f"| peak_delta={traditional_peak}/{custom_peak} B (traditional/flash)"
        )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "args": {
            "batch": args.batch,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "causal": args.causal,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved JSON report to: {args.output}")


if __name__ == "__main__":
    main()
