<div align="center">

# CUDA FlashAttention Lite

**从传统 Attention 到分块 CUDA Attention：一个可编译、可验证、可测量的 AI Infra 学习项目**

<p>
  <img src="https://img.shields.io/badge/CUDA-C%2B%2B-76B900?logo=nvidia&logoColor=white" alt="CUDA C++" />
  <img src="https://img.shields.io/badge/PyTorch-CUDA%20Extension-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch CUDA Extension" />
  <img src="https://img.shields.io/badge/Focus-FlashAttention-blue" alt="FlashAttention" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License" />
</p>

<p>
  <a href="#项目亮点">项目亮点</a> ·
  <a href="#原理概览">原理概览</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#正确性测试">正确性测试</a> ·
  <a href="#性能对比">性能对比</a> ·
  <a href="#已知限制">已知限制</a>
</p>

</div>

---

## 项目简介

在标准 Transformer Attention 中，输入张量 \(Q,K,V\) 的形状通常为 \([B,H,S,D]\)。传统实现先计算完整的注意力分数矩阵：

\[
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{D}}\right)V
\]

其中 \(S\) 为序列长度。显式构建的注意力分数矩阵具有 \(S\times S\) 的空间规模，序列越长，中间张量的内存开销越大。

本项目使用 **CUDA C++ + PyTorch CUDA Extension** 实现一个简化版 FlashAttention 前向算子，通过 K/V 分块加载、共享内存复用和 Online Softmax，在不显式保存完整注意力分数矩阵的情况下计算输出。项目同时提供传统 Attention 参考实现，用于对比数值正确性、执行延迟和 PyTorch 显存分配增量。

> **项目定位：** 面向 CUDA、AI Infra 和大模型推理优化入门者的可读型实现。它帮助理解 FlashAttention 的核心思路，不是生产级 FlashAttention 的替代品。仓库当前不预设加速比；性能结论必须由目标 GPU 上的实测数据支撑。

## 项目亮点

<table>
  <tr>
    <td width="50%">
      <h3>01 · 内存优化思路</h3>
      不物化完整的 \(S\times S\) 分数矩阵，按 tile 处理 K/V，并使用 Online Softmax 持续更新结果。
    </td>
    <td width="50%">
      <h3>02 · CUDA C++ 实践</h3>
      从 PyTorch Tensor、C++ 扩展绑定到 CUDA Kernel，串起一个可构建的算子工程。
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>03 · 正确性有参照</h3>
      用独立的传统 Attention 实现做对照，覆盖 FP32、FP16、因果掩码和非 tile 对齐的序列长度。
    </td>
    <td width="50%">
      <h3>04 · 性能可复现</h3>
      CUDA Event 计时，输出最大绝对误差、平均延迟、加速比、显存分配增量，并支持保存 JSON。
    </td>
  </tr>
</table>

### 当前实现范围

- CUDA 前向算子，支持输入形状 \([B,H,S,D]\)，且 Q/K/V 形状一致。
- 支持 FP32 和 FP16 输入；自定义 Kernel 的分数、Softmax 统计量和累加使用 FP32。
- 支持可选 Causal Mask：位置 \(i\) 只能关注 \(j\leq i\) 的 token。
- K/V tile 大小为 32，使用共享内存暂存 tile。
- 使用 CUDA Event 做重复测量，提供传统 Attention 与自定义 Kernel 的性能对比程序。
- 暂不支持反向传播、Dropout、GQA/MQA 和不等长 Q/K/V。

## 原理概览

### 传统 Attention：显式计算中间矩阵

\`\`\`text
Q, K, V
   │
   ▼
scores = Q × Kᵀ / √D        → 形状 [B, H, S, S]
   │
   ▼
可选 Causal Mask
   │
   ▼
softmax(scores)
   │
   ▼
output = probabilities × V
\`\`\`

参考实现位于 \`traditional_attention.py\`。它会显式创建完整分数矩阵和 Softmax 概率矩阵，作为本项目的正确性与性能 baseline。

### 简化版 FlashAttention：分块 + Online Softmax

\`\`\`text
             Q 的一行
                │
                ▼
        按 tile 遍历 K / V
                │
                ▼
       共享内存加载当前 tile
                │
                ▼
       计算当前 tile 的分数
                │
                ▼
       应用 Causal Mask（可选）
                │
                ▼
       Online Softmax 更新状态
       ┌────────────────────┐
       │ 当前最大值 m        │
       │ 归一化分母 l        │
       │ 加权输出累加器 o    │
       └────────────────────┘
                │
          继续下一个 tile
                │
                ▼
             输出 o / l
\`\`\`

Online Softmax 的核心是：当新 tile 到来时，更新运行最大值 \(m\)、归一化分母 \(l\) 和加权输出累加器 \(o\)，不需要保存全部历史分数。

\`\`\`text
m_new = max(m, max(scores))
alpha = exp(m - m_new)
p     = exp(scores - m_new)
l_new = alpha * l + sum(p)
o_new = alpha * o + sum(p * V)
output = o / l
\`\`\`

该实现主要为了清晰展示算法流程。当前 Kernel 每个 CUDA Block 负责一个 query 行，尚未针对 Tensor Core、occupancy、向量化访存和多 query 行复用进行充分优化。

## 项目结构

\`\`\`text
cuda-flash-attention-lite/
├── csrc/
│   ├── flash_attention.cpp       # PyTorch 扩展绑定、输入校验
│   └── flash_attention.cu        # CUDA 前向 Kernel
├── include/
│   └── flash_attention.h         # C++ 接口
├── tests/
│   ├── conftest.py               # CUDA 扩展测试条件
│   ├── test_correctness.py       # 传统 Attention 数值对比
│   └── test_causal_mask.py       # Causal Mask 边界测试
├── benchmarks/
│   └── benchmark.py              # 正确性检查与性能对比
├── docs/
│   ├── implementation.md         # 算法与 Kernel 设计说明
│   └── profiling-and-results.md  # 测试、剖析与结果记录指南
├── scripts/
│   ├── run_tests.sh              # 编译并运行正确性测试
│   └── run_benchmark.sh          # 基准测试快捷脚本
├── flash_attention.py            # 自定义算子的 Python 入口
├── traditional_attention.py      # 传统 Attention baseline
├── setup.py                      # CUDA Extension 构建配置
└── README.md
\`\`\`

## 快速开始

### 1. 环境要求

建议在 Linux + NVIDIA GPU 环境中运行：

| 组件 | 要求 |
|---|---|
| GPU | 支持 CUDA 的 NVIDIA GPU |
| NVIDIA Driver | 与本机 CUDA/PyTorch 环境兼容 |
| CUDA Toolkit | 提供 \`nvcc\`，并与已安装的 PyTorch CUDA 构建兼容 |
| Python | 3.10+ |
| PyTorch | CUDA 版本，不是仅 CPU 版本 |
| C++ 编译器 | 与当前 CUDA Toolkit 兼容 |

先检查环境：

\`\`\`bash
nvidia-smi
nvcc --version

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
\`\`\`

如果 \`torch.cuda.is_available()\` 为 \`False\`，请先解决 PyTorch、驱动或 CUDA 环境问题，再继续编译。

### 2. 获取项目

\`\`\`bash
git clone https://github.com/FeynmanNddbb/cuda-flash-attention-lite.git
cd cuda-flash-attention-lite
\`\`\`

### 3. 安装测试依赖并编译

先安装与你的驱动、CUDA Toolkit 和 Python 环境匹配的 **CUDA 版 PyTorch**。PyTorch 的安装命令依平台而异，请按本机环境选择合适版本；确认 CUDA 可用后执行：

\`\`\`bash
python -m pip install -r requirements-test.txt
python -m pip install -e . --no-build-isolation
\`\`\`

如果 CUDA Toolkit 位于非标准路径，必要时设置 \`CUDA_HOME\`。编译失败时，优先检查 \`nvcc\` 是否存在、PyTorch CUDA 版本是否匹配，以及 C++ 编译器是否受当前 Toolkit 支持。

### 4. 快速调用

扩展编译完成后，可以在 Python 中调用自定义算子：

\`\`\`python
import torch
from flash_attention import flash_attention

B, H, S, D = 1, 2, 128, 64
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

out = flash_attention(q.contiguous(), k.contiguous(), v.contiguous(), causal=True)
print("output shape:", tuple(out.shape))
print("output dtype:", out.dtype)
\`\`\`

## 正确性测试

先运行测试，再进行正式性能对比：

\`\`\`bash
pytest -v tests
\`\`\`

也可以使用仓库脚本，该脚本会安装测试依赖、构建扩展并运行测试：

\`\`\`bash
bash scripts/run_tests.sh
\`\`\`

### 测试覆盖

| 测试项 | 验证内容 |
|---|---|
| FP32 / FP16 | 不同 dtype 的输出数值误差 |
| Causal / Non-causal | 两种注意力模式 |
| 多种序列长度 | 包括非 32 倍数的序列长度 |
| Shape / dtype / device | 输出结构与输入约定 |
| 因果首行 | 第 0 行只能关注第 0 个 Key |

测试通过意味着**当前硬件与软件环境下的这些测试用例通过**，不代表所有 GPU、形状或 dtype 组合均已验证。若 CUDA 或扩展不可用，部分测试会显示为 skipped；请查看完整 pytest 输出，不要把 skipped 当作 passed。

## 性能对比

### 1. Baseline 定义

本项目将性能 baseline 明确设为 **传统 Attention 显式实现**，而不是调用经过融合优化的 PyTorch SDPA。

\`\`\`text
传统 Attention：
QKᵀ → scale → Mask（可选）→ Softmax → × V

自定义 FlashAttention：
分块加载 K/V → 分块分数 → Online Softmax → 累积输出
\`\`\`

两种实现使用相同的 Q/K/V 输入、dtype 和因果掩码配置。基准脚本会先比较输出误差，然后分别进行预热与 CUDA Event 计时。

### 2. 运行基准测试

推荐先从较短序列开始：

\`\`\`bash
python benchmarks/benchmark.py \\
  --seq-lens 128,256,512,1024 \\
  --batch 1 \\
  --heads 8 \\
  --head-dim 64 \\
  --dtype fp16 \\
  --causal \\
  --warmup 10 \\
  --repeats 50 \\
  --output benchmark-results/fp16-causal.json
\`\`\`

如果希望测试非因果模式，移除 \`--causal\` 即可；如果希望使用 FP32，将 \`--dtype fp16\` 改为 \`--dtype fp32\`。也可以直接运行：

\`\`\`bash
bash scripts/run_benchmark.sh
\`\`\`

输出指标包括：

| 指标 | 含义 |
|---|---|
| \`traditional_attention_latency_ms\` | 传统 Attention 平均单次延迟 |
| \`flash_attention_latency_ms\` | 自定义 CUDA Kernel 平均单次延迟 |
| \`speedup_traditional_over_flash\` | 传统延迟 ÷ 自定义 Kernel 延迟；大于 1 表示本次测量中自定义 Kernel 更快 |
| \`max_abs_error_vs_traditional\` | 两种实现输出的最大绝对误差 |
| \`traditional_peak_allocated_delta_bytes\` | 传统实现的峰值 PyTorch 显存分配增量 |
| \`flash_peak_allocated_delta_bytes\` | 自定义实现的峰值 PyTorch 显存分配增量 |

> **长序列注意：** 传统 baseline 会真实物化 \([B,H,S,S]\) 分数矩阵，显存随 \(S^2\) 增长。请逐步增大序列长度；如果发生 CUDA Out of Memory，缩小序列长度或 batch/head 数后再测。

显存统计是 PyTorch 统计到的峰值分配增量，不等于进程总显存或设备物理显存。不要把理论矩阵大小直接当作测量结果，也不要在没有运行记录的情况下填写加速比。

### 3. 使用 Nsight 分析

查看 Kernel 级性能计数器：

\`\`\`bash
ncu --set basic python benchmarks/benchmark.py --seq-lens 256 --batch 1 --heads 2 --head-dim 64 --dtype fp16
\`\`\`

查看 CUDA 调用与系统时间线：

\`\`\`bash
nsys profile --stats=true python benchmarks/benchmark.py --seq-lens 256 --batch 1 --heads 2 --head-dim 64 --dtype fp16
\`\`\`

不同版本的 Nsight 可能需要调整参数或权限。建议先完成正确性测试，再进行普通计时，最后单独进行 profiler 分析。

## 如何解读显存开销

单个 Attention Score 矩阵的理论大小为：

\[
\text{Memory}=B\times H\times S^2\times \text{每元素字节数}
\]

例如，当 \(B=1,H=1,S=4096\) 时，**单个** FP32 分数矩阵理论上约为 64 MiB；FP16 则约为 32 MiB。传统实现还会产生概率矩阵等中间结果，因此实际峰值与这些理论值不同。

FlashAttention 的核心优势是避免保存完整的 \(S\times S\) 注意力分数矩阵，而不是承诺任何环境下都达到固定的显存比例或固定加速比。实际收益应以目标设备上的 profiler 和 benchmark 结果为准。

## 结果记录模板

每次正式测试建议记录 GPU 型号、驱动、CUDA Toolkit、PyTorch 版本、dtype、张量形状、是否启用因果掩码、测试命令、pytest 输出和 benchmark JSON。可以按下面的格式整理结果：

| GPU / 软件环境 | dtype | B/H/S/D | causal | 传统延迟 (ms) | 自定义延迟 (ms) | 加速比 | 最大绝对误差 |
|---|---|---|---|---:|---:|---:|---:|
| 待实测填写 | FP16 | 1/8/512/64 | 是 | — | — | — | — |

**仓库当前不提供虚构的性能成绩。** 请先在目标 NVIDIA GPU 上运行并保存报告，再把真实结果补充到表格。

## 已知限制与后续优化方向

当前版本优先保证代码可读，主要用于理解 Attention 的内存访问、Online Softmax 和 CUDA 扩展开发。它尚不是高性能实现，可能慢于高度优化的库。

后续可以围绕以下方向迭代：

- 让多个 Warp 协作完成 QK 点积和归约。
- 一个 CTA 处理多个 Query 行，增加 K/V tile 的复用。
- 优化全局内存访问、tile 尺寸、寄存器占用和 occupancy。
- 借助 Nsight 定位访存、计算和 Kernel launch 开销。
- 在前向正确性稳定后，再扩展反向传播和更多输入形状。

## License

本项目采用 [MIT License](LICENSE)。

---

<div align="center">

**Learn the algorithm · Inspect the kernel · Measure the result**

[GitHub Repository](https://github.com/FeynmanNddbb/cuda-flash-attention-lite)

</div>
