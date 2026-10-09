# Profiling and result collection

## 1. Record the environment

    nvidia-smi
    nvcc --version
    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

Also record the GPU model, driver version, CUDA Toolkit version, PyTorch version, and GPU compute capability.

## 2. Run correctness first

    python -m pip install -r requirements-test.txt
    python -m pip install -e . --no-build-isolation
    pytest -v tests

Tests compare the custom implementation with explicit conventional attention. Do not report latency or speedup for a build that has not passed correctness tests. Record any skips: tests skip when CUDA or the compiled extension is unavailable.

## 3. Run the performance comparison

    python benchmarks/benchmark.py --seq-lens 128,256,512,1024 --batch 1 --heads 8 --head-dim 64 --dtype fp16 --causal --warmup 10 --repeats 50 --output benchmark-results/fp16-causal.json
    python benchmarks/benchmark.py --seq-lens 128,256,512,1024 --batch 1 --heads 8 --head-dim 64 --dtype fp32 --warmup 10 --repeats 50 --output benchmark-results/fp32-noncausal.json

The baseline is a conventional implementation that explicitly calculates QK^T, scales the scores, applies an optional causal mask, runs softmax, and multiplies the probabilities by V. It materializes the full [B, H, S, S] score matrix.

The reported speedup is traditional attention latency divided by custom FlashAttention latency. Values above 1 mean the custom implementation was faster; values below 1 mean traditional attention was faster. Results vary with GPU, driver, CUDA Toolkit, PyTorch, dtype, and tensor shape.

The memory field is peak PyTorch-allocated memory above the baseline during repeated calls. It is not the complete memory used by the process or GPU. The conventional baseline intentionally materializes the attention score matrix, so memory usage grows quadratically with sequence length and longer runs can exhaust GPU memory. Increase sequence lengths gradually.

## 4. Profile the kernel

    ncu --set basic python benchmarks/benchmark.py --seq-lens 256 --batch 1 --heads 2 --head-dim 64 --dtype fp16
    nsys profile --stats=true python benchmarks/benchmark.py --seq-lens 256 --batch 1 --heads 2 --head-dim 64 --dtype fp16

Profiler permissions and option names can vary by installed version. Keep correctness/latency runs separate from heavily instrumented profiler runs.

## Result table template

| GPU / software stack | dtype | B/H/S/D | causal | traditional ms | FlashAttention ms | speedup | max absolute error | peak allocation delta |
|---|---|---|---|---:|---:|---:|---:|---:|
| Fill from actual run | fp16 | 1/8/512/64 | yes | — | — | — | — | — |

Do not copy illustrative values from a draft resume. Keep the JSON output and terminal log so every reported number can be reproduced. When discussing memory, specify whether a figure is theoretical score-matrix size, framework peak allocated memory, or profiler-observed device memory.
