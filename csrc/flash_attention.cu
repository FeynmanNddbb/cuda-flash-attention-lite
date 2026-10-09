#include "flash_attention.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

constexpr int kTile = 32;
constexpr int kMaxHeadDim = 128;
constexpr int kThreads = 128;

// Educational baseline: one CTA owns one query row. Threads cooperatively load
// a K/V tile to shared memory and compute one score per key. Thread 0 performs
// the online-softmax update and writes the output row. Clarity is prioritized
// over production-level occupancy and parallelism.
template <typename scalar_t>
__global__ void flash_attention_forward_kernel(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, scalar_t* __restrict__ out,
    int batch_heads, int seq_len, int head_dim, bool causal) {
  const int row = blockIdx.x;
  if (row >= batch_heads * seq_len) return;

  const int query_index = row % seq_len;
  const int bh_index = row / seq_len;
  const scalar_t* q_row = q + static_cast<long long>(row) * head_dim;
  const scalar_t* k_head = k + static_cast<long long>(bh_index) * seq_len * head_dim;
  const scalar_t* v_head = v + static_cast<long long>(bh_index) * seq_len * head_dim;
  scalar_t* out_row = out + static_cast<long long>(row) * head_dim;

  __shared__ scalar_t shared_k[kTile * kMaxHeadDim];
  __shared__ scalar_t shared_v[kTile * kMaxHeadDim];
  __shared__ float scores[kTile];
  __shared__ float out_acc[kMaxHeadDim];
  __shared__ float running_max;
  __shared__ float running_sum;

  if (threadIdx.x == 0) {
    running_max = -CUDART_INF_F;
    running_sum = 0.0f;
    for (int d = 0; d < head_dim; ++d) out_acc[d] = 0.0f;
  }
  __syncthreads();
  const float scale = rsqrtf(static_cast<float>(head_dim));

  for (int tile_start = 0; tile_start < seq_len; tile_start += kTile) {
    const int elements = kTile * head_dim;
    for (int linear = threadIdx.x; linear < elements; linear += blockDim.x) {
      const int local_key = linear / head_dim;
      const int d = linear % head_dim;
      const int key_index = tile_start + local_key;
      if (key_index < seq_len) {
        const long long offset = static_cast<long long>(key_index) * head_dim + d;
        shared_k[local_key * kMaxHeadDim + d] = k_head[offset];
        shared_v[local_key * kMaxHeadDim + d] = v_head[offset];
      } else {
        shared_k[local_key * kMaxHeadDim + d] = scalar_t(0);
        shared_v[local_key * kMaxHeadDim + d] = scalar_t(0);
      }
    }
    __syncthreads();

    for (int local_key = threadIdx.x; local_key < kTile; local_key += blockDim.x) {
      const int key_index = tile_start + local_key;
      if (key_index >= seq_len || (causal && key_index > query_index)) {
        scores[local_key] = -CUDART_INF_F;
        continue;
      }
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += static_cast<float>(q_row[d]) *
               static_cast<float>(shared_k[local_key * kMaxHeadDim + d]);
      }
      scores[local_key] = dot * scale;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float tile_max = -CUDART_INF_F;
      for (int j = 0; j < kTile; ++j) tile_max = fmaxf(tile_max, scores[j]);
      const float new_max = fmaxf(running_max, tile_max);
      const float alpha = isfinite(running_max) ? __expf(running_max - new_max) : 0.0f;
      float probabilities[kTile];
      float tile_sum = 0.0f;
      for (int j = 0; j < kTile; ++j) {
        probabilities[j] = isfinite(scores[j]) ? __expf(scores[j] - new_max) : 0.0f;
        tile_sum += probabilities[j];
      }
      for (int d = 0; d < head_dim; ++d) {
        float weighted_value = 0.0f;
        for (int j = 0; j < kTile; ++j) {
          weighted_value += probabilities[j] * static_cast<float>(shared_v[j * kMaxHeadDim + d]);
        }
        out_acc[d] = alpha * out_acc[d] + weighted_value;
      }
      running_sum = alpha * running_sum + tile_sum;
      running_max = new_max;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    for (int d = 0; d < head_dim; ++d) {
      const float value = running_sum > 0.0f ? out_acc[d] / running_sum : 0.0f;
      out_row[d] = static_cast<scalar_t>(value);
    }
  }
}

void launch_flash_attention_cuda(const torch::Tensor& q, const torch::Tensor& k,
                                 const torch::Tensor& v, torch::Tensor& out, bool causal) {
  const int batch_heads = static_cast<int>(q.size(0) * q.size(1));
  const int seq_len = static_cast<int>(q.size(2));
  const int head_dim = static_cast<int>(q.size(3));
  const dim3 grid(batch_heads * seq_len);
  const dim3 block(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "flash_attention_forward_cuda", [&] {
    flash_attention_forward_kernel<scalar_t><<<grid, block, 0, stream>>>(
        q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(), batch_heads, seq_len, head_dim, causal);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
