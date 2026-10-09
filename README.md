# CUDA FlashAttention Lite

> 使用 CUDA C++ 实现简化版 FlashAttention 前向算子，并通过传统 Attention 进行正确性与性能对比。

[![CUDA](https://img.shields.io/badge/CUDA-C%2B%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Extension-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目介绍

传统 Attention 通常按以下步骤计算：

1. 计算注意力分数：QK 的转置乘积，再除以 sqrt(D)。
2. 可选地应用因果掩码（Causal Mask）。
3. 对分数执行 Softmax。
4. 将概率矩阵与 V 相乘，得到最终输出。

其中，完整注意力分数矩阵的形状为 [B, H, S, S]。当序列长度 S 增长时，这个中间矩阵的空间开销会按 S 的平方增长。

本项目使用 CUDA C++ 实现一个便于学习的 FlashAttention 前向算子。它按 tile 分块读取 K 和 V，将数据暂存在共享内存中，并使用 Online Softmax 逐块更新归一化统计量与输出累加值，避免显式保存完整的注意力分数矩阵。

项目同时提供传统 Attention 参考实现，以及正确性测试、性能对比和 JSON 结果导出脚本。

## 项目亮点

| 方向 | 实现内容 | 能验证什么 |
|---|---|---|
| CUDA Kernel | 分块加载 K/V、共享内存复用 | 理解数据搬运与线程协作 |
| Online Softmax | 逐 tile 更新最大值、分母和输出累加值 | 避免保存完整分数矩阵 |
| Causal Mask | 屏蔽当前位置之后的 Key | 检查因果注意力语义 |
| 正确性测试 | 与传统 Attention 的显式计算结果对比 | 检查数值误差与边界情况 |
| 性能测试 | CUDA Event 计时并比较两种实现 | 得到延迟、加速比和显存分配数据 |
| 性能分析 | 提供 Nsight Compute / Nsight Systems 命令 | 进一步定位 Kernel 瓶颈 |

## 实现原理

### 传统 Attention

传统参考实现显式计算整个分数矩阵：

~~~python
scores = Q @ K.transpose(-2, -1)
scores = scores / sqrt(head_dim)
scores = apply_causal_mask_if_needed(scores)
probabilities = softmax(scores, dim=-1)
output = probabilities @ V
~~~

这份实现位于 **traditional_attention.py**，用于作为本项目的数值正确性参考和性能 baseline。

### 简化版 FlashAttention

自定义 CUDA Kernel 的主要步骤如下：

1. 一个 CUDA Block 负责一个 Query 行。
2. 多个线程协作，将当前 K/V tile 加载到共享内存。
3. 计算当前 tile 的注意力分数，并按需应用 Causal Mask。
4. 使用 Online Softmax 更新运行最大值、归一化分母和输出累加值。
5. 处理后续 tile，最终将累加结果除以归一化分母。

Online Softmax 的更新逻辑可以概括为：

~~~text
m_new = max(m, max(scores))
alpha = exp(m - m_new)
p     = exp(scores - m_new)
l_new = alpha * l + sum(p)
o_new = alpha * o + sum(p * V)
output = o / l
~~~

这里的 m 是运行最大值，l 是归一化分母，o 是未归一化的输出累加值。实际代码对分数、Softmax 统计量和输出累加使用 FP32，再将结果转换回输入 dtype。

> 当前实现的目标是展示核心算法，而不是追求生产级性能。每个 Block 只处理一个 Query 行，点积计算仍有逐维循环，暂未针对 Tensor Core、occupancy 和多 Query 行复用进行充分优化。

## 项目结构

~~~text
cuda-flash-attention-lite/
├── csrc/
│   ├── flash_attention.cpp       # PyTorch 扩展绑定与输入检查
│   └── flash_attention.cu        # CUDA 前向 Kernel
├── include/
│   └── flash_attention.h         # C++ 接口
├── tests/
│   ├── conftest.py               # 测试环境检查
│   ├── test_correctness.py       # 数值正确性对比
│   └── test_causal_mask.py       # 因果掩码边界测试
├── benchmarks/
│   └── benchmark.py              # 传统 Attention 与自定义 Kernel 对比
├── docs/
│   ├── implementation.md         # 算法与 Kernel 说明
│   └── profiling-and-results.md  # 测试和性能分析指南
├── scripts/
│   ├── run_tests.sh              # 构建并执行测试
│   └── run_benchmark.sh          # 性能测试快捷脚本
├── flash_attention.py            # 自定义算子的 Python 入口
├── traditional_attention.py      # 传统 Attention baseline
├── setup.py                      # CUDA Extension 构建配置
└── README.md
~~~

## 快速开始

### 1. 检查环境

建议使用 Linux 和 NVIDIA GPU。需要安装兼容的 NVIDIA Driver、CUDA Toolkit、Python 3.10+，以及 CUDA 版本的 PyTorch。

先执行以下命令检查环境：

~~~bash
nvidia-smi
nvcc --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
~~~

如果 CUDA 不可用，先检查 PyTorch 是否为 CUDA 版本，以及驱动、CUDA Toolkit 是否与当前环境匹配。

### 2. 克隆仓库

~~~bash
git clone https://github.com/FeynmanNddbb/cuda-flash-attention-lite.git
cd cuda-flash-attention-lite
~~~

### 3. 安装依赖并编译

先安装与你的驱动和 CUDA 环境匹配的 CUDA 版 PyTorch，然后执行：

~~~bash
python -m pip install -r requirements-test.txt
python -m pip install -e . --no-build-isolation
~~~

如果 CUDA Toolkit 安装在非标准路径，可以设置 CUDA_HOME。编译失败时，优先检查 nvcc 是否存在、PyTorch 的 CUDA 构建版本，以及 C++ 编译器兼容性。

### 4. 运行一个简单示例

编译成功后，在项目根目录运行：

~~~python
import torch
from flash_attention import flash_attention

batch, heads, seq_len, head_dim = 1, 2, 128, 64
q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

output = flash_attention(q.contiguous(), k.contiguous(), v.contiguous(), causal=True)
print("Output shape:", tuple(output.shape))
print("Output dtype:", output.dtype)
~~~

## 正确性测试

执行：

~~~bash
pytest -v tests
~~~

也可以使用项目脚本，它会安装测试依赖、构建 CUDA 扩展并运行测试：

~~~bash
bash scripts/run_tests.sh
~~~

测试覆盖以下场景：

- FP32 和 FP16 输入。
- Causal 与 Non-causal 两种模式。
- 不同序列长度，包括不是 32 倍数的序列长度。
- 输出形状、dtype、device 检查。
- 因果掩码首行检查：第 0 个 Query 只能关注第 0 个 Key。

测试会将自定义 Kernel 的输出与 **traditional_attention.py** 的显式计算结果进行比较，并使用设定的 atol / rtol 检查数值误差。

如果 CUDA 或扩展不可用，测试可能显示 skipped。请阅读完整的 pytest 结果；skipped 不代表测试通过。

## 性能对比

### Baseline 是什么？

本项目使用传统 Attention 作为性能 baseline，不使用 PyTorch SDPA 作为性能对照。

| 传统 Attention | 自定义 FlashAttention |
|---|---|
| 显式创建完整的注意力分数矩阵 | 分块处理 K/V |
| 对分数矩阵执行 Softmax | 使用 Online Softmax 更新统计量 |
| 中间矩阵空间规模随序列长度平方增长 | 不显式保存完整的分数矩阵 |
| 作为数值参考和性能基线 | 作为待测的 CUDA 实现 |

### 运行基准测试

建议先从较短的序列开始：

~~~bash
python benchmarks/benchmark.py \
  --seq-lens 128,256,512,1024 \
  --batch 1 \
  --heads 8 \
  --head-dim 64 \
  --dtype fp16 \
  --causal \
  --warmup 10 \
  --repeats 50 \
  --output benchmark-results/fp16-causal.json
~~~

移除 **--causal** 参数可测试非因果模式；将 **--dtype fp16** 改为 **--dtype fp32** 可测试 FP32。

也可以执行快捷脚本：

~~~bash
bash scripts/run_benchmark.sh
~~~

基准程序会先比较两种实现的输出误差，再分别预热、使用 CUDA Event 测量平均单次延迟。输出指标如下：

| 指标 | 含义 |
|---|---|
| traditional_attention_latency_ms | 传统 Attention 平均单次延迟 |
| flash_attention_latency_ms | 自定义 CUDA Kernel 平均单次延迟 |
| speedup_traditional_over_flash | 传统延迟除以自定义 Kernel 延迟；大于 1 表示本次测量中自定义 Kernel 更快 |
| max_abs_error_vs_traditional | 两种实现输出的最大绝对误差 |
| traditional_peak_allocated_delta_bytes | 传统实现的峰值 PyTorch 显存分配增量 |
| flash_peak_allocated_delta_bytes | 自定义实现的峰值 PyTorch 显存分配增量 |

结果会输出到终端；如果指定了 output 参数，也会保存成 JSON 文件。

> **显存提醒：** 传统 baseline 会显式分配完整的 [B, H, S, S] 分数矩阵。序列长度越大，显存占用增长越快。建议逐步增加序列长度；出现 CUDA Out of Memory 时，先减小序列长度、batch 或 heads。

本项目报告的是 PyTorch 记录到的峰值分配增量，不是整个进程的物理显存占用。不要用理论矩阵大小代替实际测试数据。

## 使用 Nsight 分析性能

使用 Nsight Compute 查看 Kernel 指标：

~~~bash
ncu --set basic python benchmarks/benchmark.py --seq-lens 256 --batch 1 --heads 2 --head-dim 64 --dtype fp16
~~~

使用 Nsight Systems 查看调用时间线：

~~~bash
nsys profile --stats=true python benchmarks/benchmark.py --seq-lens 256 --batch 1 --heads 2 --head-dim 64 --dtype fp16
~~~

建议先跑完正确性测试，再记录不带 profiler 的普通延迟，最后单独进行性能剖析。不同版本的 Nsight 可能需要调整参数或权限设置。

## 如何记录结果

每次实验建议记录：

- GPU 型号、驱动版本、CUDA Toolkit 版本、PyTorch 版本。
- 输入 dtype、B/H/S/D、是否启用 Causal Mask。
- pytest 完整输出和测试是否通过。
- 传统 Attention 延迟、自定义 Kernel 延迟、加速比、最大绝对误差。
- 两种实现的显存分配数据和 benchmark JSON 文件。

结果表格模板：

| GPU / 软件环境 | dtype | B/H/S/D | Causal | 传统延迟 (ms) | 自定义延迟 (ms) | 加速比 | 最大绝对误差 |
|---|---|---|---|---:|---:|---:|---:|
| 待实测填写 | FP16 | 1/8/512/64 | 是 | — | — | — | — |

仓库不预填示例加速比。请以实际运行日志和 JSON 报告为准，再将真实结果用于项目总结或简历。

## 已知限制与后续方向

当前版本更偏向教学和验证，可能慢于高度优化的注意力实现。主要限制包括：

- 只实现前向计算，不包含反向传播。
- Q/K/V 必须形状一致且连续。
- 只支持 FP32 和 FP16，head dimension 最大为 128。
- 每个 CUDA Block 处理一行 Query，尚未完成充分的并行度与访存优化。
- 未实现 Dropout、GQA/MQA 等功能。

后续可以尝试多 Query 行分块、Warp 协作归约、向量化访存、tile 尺寸调优，并使用 Nsight 验证每项优化是否真正带来收益。

## License

本项目采用 [MIT License](LICENSE)。
