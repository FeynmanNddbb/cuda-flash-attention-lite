# CUDA FlashAttention Lite

Educational CUDA C++ implementation of the forward pass of scaled dot-product attention with causal masking. The project focuses on tiled K/V loads, shared-memory reuse, and numerically stable online softmax without materializing the full S x S score matrix.

**Important:** this is an instructional baseline, not a production FlashAttention implementation. Performance and correctness data must be measured on an NVIDIA GPU; no placeholder speedup is presented as a real result.

## Features

- PyTorch CUDA extension for contiguous [B, H, S, D] tensors.
- FP32 and FP16 input; FP32 score/softmax/accumulation.
- Optional causal mask (keys after the query index are excluded).
- Tiled K/V staging in shared memory (tile size 32); online softmax update.
- PyTorch SDPA correctness tests and CUDA-event benchmark.
- Nsight Compute / Nsight Systems profiling notes.

## Limitations

The simple kernel assigns one CUDA block to each query row. It uses a straightforward reduction and serial output-row update to keep the algorithm readable; this is not tuned for tensor cores or high occupancy and may be slower than PyTorch SDPA. Equal Q/K/V shapes are required, head dimension is at most 128, dropout/backward/GQA are not implemented.

## Build requirements

Linux, NVIDIA GPU + compatible driver, CUDA Toolkit/nvcc compatible with installed PyTorch, Python 3.10+, CUDA-enabled PyTorch.

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install -e .
```

Install a CUDA-enabled PyTorch build that matches your environment before running the last command.

## Correctness tests

```bash
pytest -q tests
```

Tests compare against `torch.nn.functional.scaled_dot_product_attention` for FP32/FP16 and causal/non-causal modes. Tests are skipped if CUDA or the extension is unavailable. Record the exact GPU, driver, toolkit, PyTorch version, shape, dtype, tolerance, and test output before claiming results.

## Benchmark

```bash
python benchmarks/benchmark.py --seq-lens 128,512,1024 --batch 1 --heads 8 --head-dim 64 --dtype fp16 --causal
```

The benchmark warms up, synchronizes, measures CUDA-event latency, and reports PyTorch peak allocated memory delta. This metric does not account for every possible device allocation; use Nsight for deeper analysis.

```bash
ncu --set full python benchmarks/benchmark.py --seq-lens 512 --dtype fp16 --causal
nsys profile python benchmarks/benchmark.py --seq-lens 512 --dtype fp16 --causal
```

## Online softmax

For a new tile with scores s, running maximum m, normalization sum l, and unnormalized output o:

```text
m_new = max(m, max(s))
alpha = exp(m - m_new)
p     = exp(s - m_new)
l_new = alpha * l + sum(p)
o_new = alpha * o + sum(p * V)
output = o / l
```

Masked scores are set to negative infinity. The implementation stores only a K/V tile and row statistics rather than the full sequence-by-sequence score matrix.

## Result reporting template

| GPU | dtype | B/H/S/D | causal | SDPA ms | custom ms | speedup | max abs error |
|---|---|---|---|---:|---:|---:|---:|
| Fill after running benchmark | fp16 | 1/8/1024/64 | yes | — | — | — | — |

Do not use illustrative values as measurements. If discussing memory, state whether it is allocated tensor memory, peak framework allocation, or profiler-observed device memory. The theoretical FP32 score matrix size for one head is S x S x 4 bytes; SDPA may use a fused kernel that never allocates that matrix.

## Layout

- `include/flash_attention.h`: C++ interface
- `csrc/flash_attention.cpp`: extension binding/validation
- `csrc/flash_attention.cu`: CUDA kernel
- `flash_attention.py`: Python entry point
- `tests/`: numerical and causal correctness tests
- `benchmarks/`: reproducible benchmark
- `docs/`: algorithm/profiling/result notes

MIT License. See `LICENSE`.
