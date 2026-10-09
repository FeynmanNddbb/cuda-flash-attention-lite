#pragma once
#include <torch/extension.h>

torch::Tensor flash_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);
void launch_flash_attention_cuda(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, torch::Tensor& out, bool causal);
