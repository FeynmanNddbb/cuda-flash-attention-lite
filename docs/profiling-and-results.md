# Profiling and result collection

## 1. Record the environment

Before benchmarking, save this output with the results:

    nvidia-smi
    nvcc --version
    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

Also record the GPU model, driver version, CUDA Toolkit version, PyTorch version and GPU compute capability.

## 2. Run correctness first

    python -m pip install -r requirements-test.txt
    pytest -v tests

Do not report latency or speedup for a build that has not passed correctness tests. Record any skips: tests skip when CUDA or the compiled extension is unavailable.

## 3. Run the comparison benchmark

    python benchmarks/benchmark.py --seq-lens 128,256,512,1024 --batch 1 --heads 8 --head-dim 64 --dtype fp16 --causal --warmup 10 --repeats 50 --output benchmark-results/fp16-causal.json
    python benchmarks/benchmark.py --seq-lens 128,256,512,1024 --batch 1 --heads 8 --head-dim 64 --dtype fp32 --warmup 10 --repeats 50 --output benchmark-results/fp32-noncausal.json

The printed speedup is SDPA latency divided by custom-kernel latency. Values above 1 mean the custom kernel was faster; values below 1 mean SDPA was faster. Results vary with GPU, driver, CUDA Toolkit, PyTorch, tensor shape, and SDPA backend.

The memory field is peak PyTorch-allocated memory above the baseline during repeated calls. It is not the complete memory used by the process or GPU. PyTorch SDPA may select a fused backend, so do not assume it materializes the full attention-score matrix.

## 4. Profile the kernel

    ncu --set basic python benchmarks/benchmark.py --seq-lens 256 --batch 1 --heads 2 --head-dim 64 --dtype fp16
    nsys profile --stats=true python benchmarks/benchmark.py --seq-lens 256 --batch 1 --heads 2 --head-dim 64 --dtype fp16

Profiler permissions and option names can vary by installed version. Keep correctness/latency runs separate from heavily instrumented profiler runs.

## Resume/reporting template

| GPU / software stack | dtype | B/H/S/D | causal | SDPA ms | custom ms | speedup | max absolute error | peak allocation delta |
|---|---|---|---|---:|---:|---:|---:|---:|
| Fill from actual run | fp16 | 1/8/512/64 | yes | — | — | — | — | — |

Do not copy illustrative values from a draft resume. Keep the JSON output and terminal log so every reported number can be reproduced. When discussing memory savings, specify whether the claim is theoretical score-matrix size, framework peak allocated memory, or profiler-observed device memory.
