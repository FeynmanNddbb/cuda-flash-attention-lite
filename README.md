# CUDA FlashAttention Lite

Educational CUDA C++ implementation of the forward pass of scaled dot-product attention with causal masking. The project demonstrates tiled K/V loads, shared-memory reuse, and numerically stable online softmax without materializing the full S x S attention score matrix.

**Status:** learning/research implementation. Correctness and performance numbers must be measured on the target NVIDIA GPU. No placeholder speedup is presented as a real result.

## Features

- PyTorch CUDA extension for contiguous [B, H, S, D] tensors.
- FP32 and FP16 input; FP32 score, softmax, and accumulation in the custom kernel.
- Optional causal mask (keys after the query index are excluded).
- K/V staging in shared memory (tile size 32) and online softmax updates.
- Correctness comparison against a conventional attention implementation.
- Performance comparison against conventional attention that explicitly materializes the full [B, H, S, S] score matrix.
- CUDA-event latency, peak PyTorch allocated memory delta, max absolute error, speedup, JSON report, and Nsight profiling notes.

## Limitations

This is a **simplified teaching kernel**, not a production FlashAttention replacement. One CUDA block handles one query row; score dot products loop over the head dimension. The implementation is not tuned for tensor cores or high occupancy and may be slower than optimized libraries. Equal Q/K/V shapes are required, head dimension is at most 128, and dropout/backward/GQA are not implemented.

## Build requirements

Linux, an NVIDIA GPU with a compatible driver, CUDA Toolkit/nvcc compatible with the installed PyTorch build, Python 3.10+, and a supported C++ compiler.

    nvidia-smi
    nvcc --version
    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

Install a CUDA-enabled PyTorch build that matches your system first, then build the extension in the same environment:

    python -m pip install -r requirements-test.txt
    python -m pip install -e . --no-build-isolation

If CUDA is installed in a non-standard location, set CUDA_HOME before building.

## Correctness tests

    pytest -v tests

Tests compare the custom kernel with explicit conventional attention for FP32/FP16 and causal/non-causal modes, including sequence lengths that are not multiples of the K/V tile. They skip when CUDA or the compiled extension is unavailable. Record the GPU, driver, CUDA Toolkit, PyTorch version, dtype, shape, tolerance, and full test output before claiming results.

## Performance comparison: conventional attention vs FlashAttention

    python benchmarks/benchmark.py --seq-lens 128,256,512,1024 --batch 1 --heads 8 --head-dim 64 --dtype fp16 --causal --warmup 10 --repeats 50 --output benchmark-results/fp16-causal.json

The baseline performs the conventional sequence explicitly:

    scores = Q @ K.transpose(-2, -1)
    scores = scores / sqrt(head_dim)
    scores = scores.masked_fill(causal_mask, -inf)  # causal mode only
    probabilities = softmax(scores, dim=-1)
    output = probabilities @ V

This baseline materializes the full [B, H, S, S] score tensor and the softmax result. The custom FlashAttention kernel instead processes K/V tiles and uses online softmax without storing the full score matrix.

The benchmark warms up both implementations, measures latency using CUDA events, and reports max absolute output error, conventional/custom speedup, and peak change in PyTorch allocated memory. A speedup above 1 means the custom implementation was faster for that shape and environment. The memory metric is peak PyTorch-allocated memory above the pre-call baseline, not total device memory. Larger sequence lengths can require substantial memory for the conventional baseline and may run out of memory; increase them gradually.

For kernel-level profiling, use Nsight Compute, for example:

    ncu --set basic python benchmarks/benchmark.py --seq-lens 256 --dtype fp16 --causal

For system timelines, use Nsight Systems:

    nsys profile --stats=true python benchmarks/benchmark.py --seq-lens 256 --dtype fp16 --causal

Profiler options and permissions vary by installed version. Compare normal benchmark runs separately from instrumented profiler runs.

## Algorithm sketch

For the running maximum m, normalization sum l, output accumulator o, and scores s from a new tile:

    m_new = max(m, max(s))
    alpha = exp(m - m_new)
    p = exp(s - m_new)
    l_new = alpha*l + sum(p)
    o_new = alpha*o + sum(p*V)
    output = o/l

Masked positions receive negative-infinity scores and zero probability. Only a K/V tile and per-row statistics are stored rather than the complete sequence-by-sequence score matrix.

## Result reporting template

| GPU / software stack | dtype | B/H/S/D | causal | traditional ms | FlashAttention ms | speedup | max absolute error | peak allocation delta |
|---|---|---|---|---:|---:|---:|---:|---:|
| Fill from actual run | fp16 | 1/8/512/64 | yes | — | — | — | — | — |

Do not copy illustrative values from a draft resume. Memory statements must distinguish theoretical score-matrix size, framework peak allocated memory, and profiler-observed device memory.

## Layout

- include/flash_attention.h: C++ interface
- csrc/flash_attention.cpp: extension binding and input validation
- csrc/flash_attention.cu: CUDA forward kernel
- flash_attention.py: Python entry point
- traditional_attention.py: conventional baseline and reference implementation
- tests/: correctness and causal-mask tests
- benchmarks/: reproducible conventional-attention comparison
- docs/: implementation notes and results workflow

MIT License. See LICENSE.
