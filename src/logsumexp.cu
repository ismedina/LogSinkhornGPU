#include "src/utils.hpp"
#include <math.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// Compiling this with --use_fast_math we get some noticeable performance improvement
template <typename Dtype>
__global__ void logsumexp(int B, int M, int N, Dtype *alpha, Dtype *beta, Dtype dx)
{
    // alpha is input, with size (B, N)
    // beta is output, with size (B, M)
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of the reduction, this is (b,i), this is of the result beta
    int b = index / M;
    int i = index % M;
    if (b >= B){ // care for bigger-than-size indices
        return;
    }
    dx = dx*dx; // for just multiplying to the square cost
    Dtype m = -1e30f; // TODO: check initialization
    for (int j = 0; j<N; j++)
    {
        m = max(m, alpha[b*N+j] - (i-j)*(i-j)*dx); // TODO: still to check the squared part
    }
    Dtype s = 0.0f;
    for (int j = 0; j<N; j++)
    {
        s += exp(alpha[b*N+j] - (i-j)*(i-j)*dx - m); // TODO: still to check the squared part
    }
    beta[index] = log(s)+m;
}

// template <typename Dtype>
// __global__ void logsumexp(int B, int M, int N, Dtype *alpha, Dtype *beta, Dtype dx)
// {
//     // alpha is input, with size (B, N)
//     // beta is output, with size (B, M)
//     int index = blockIdx.x * blockDim.x + threadIdx.x; // index of the reduction, this is (b,i), this is of the result beta
//     int b = index / M;
//     int i = index % M;
//     if (b >= B){ // care for bigger-than-size indices
//         return;
//     }
//     // dx = dx*dx; // for just multiplying to the square cost
//     Dtype m = -1e30f; // TODO: check initialization
//     Dtype diff;
//     for (int j = 0; j<N; j++)
//     {
//         diff = (i-j)*dx;
//         m = max(m, alpha[b*N+j] - diff*diff); // TODO: still to check the squared part
//     }
//     Dtype s = 0.0f; 
//     for (int j = 0; j<N; j++)
//     {
//         diff = (i-j)*dx;
//         s += exp(alpha[b*N+j] - diff*diff - m); // TODO: still to check the squared part
//     }
//     beta[index] = log(s)+m;
// }

template <typename Dtype>
void LogSumExpGPUKernel(int B, int M, int N, Dtype *alpha, Dtype *beta, Dtype dx)
{
  // number of elements to process is size of beta, this is, B*M
  int blockSize = 256;
  int numBlocks = (B*M + blockSize - 1) / blockSize; 
  logsumexp<<<numBlocks, blockSize>>>(B, M, N, alpha, beta, dx);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    throw std::runtime_error(Formatter()
                             << "CUDA kernel failed : " << std::to_string(err));
}

template void LogSumExpGPUKernel<float>(int B, int M, int N, float *alpha, float *beta, float dx);

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