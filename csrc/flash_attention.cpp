#include "flash_attention.h"
#include <c10/cuda/CUDAGuard.h>

torch::Tensor flash_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, and v must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, and v must have shape [B, H, S, D]");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "this implementation requires equal Q/K/V shapes");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, and v must be on the same device");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "q, k, and v must have the same dtype");
  TORCH_CHECK(q.scalar_type() == at::kFloat || q.scalar_type() == at::kHalf, "supported dtypes are float32 and float16");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q, k, and v must be contiguous");
  TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0, "batch, heads, and sequence length must be positive");
  TORCH_CHECK(q.size(3) > 0 && q.size(3) <= 128, "head dimension must be in [1, 128]");

  const c10::cuda::CUDAGuard device_guard(q.device());
  auto out = torch::empty_like(q);
  launch_flash_attention_cuda(q, k, v, out, causal);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flash_attention_forward, "Simplified FlashAttention forward (CUDA)");
}
