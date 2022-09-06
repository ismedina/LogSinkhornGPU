#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "src/logsumexp.cuh"
#include "src/utils.hpp"

template <typename Dtype>
void LogSumExpGPU(at::Tensor alpha, at::Tensor beta, Dtype dx) {
  // Check `alpha` has shape (B, N) and `beta` has shape (B, M)
  int B = alpha.size(0);
  int M = beta.size(1);
  int N = alpha.size(1);
  if (B != beta.size(0))
    throw std::invalid_argument(Formatter()
                                << "alpha and beta must have the same batch size");
  if (B*N != alpha.numel())
    throw std::invalid_argument(Formatter()
                                << "Shape mismatch: first two dimensions of alpha "
                                << "must be the only non-trivial ones");
  if (B*M != beta.numel())
    throw std::invalid_argument(Formatter()
                                << "Shape mismatch: first two dimensions of beta "
                                << "must be the only non-trivial ones");                            

  LogSumExpGPUKernel<Dtype>(B, M, N, alpha.data_ptr<Dtype>(), beta.data_ptr<Dtype>(), dx);
}

template void LogSumExpGPU<float>(at::Tensor alpha, at::Tensor beta, float dx);

// template void LogSumExpGPU<float>(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c);

// void permute(at::Tensor x)
// {
//   at::Tensor y = x.permute({0,2,1});
// }