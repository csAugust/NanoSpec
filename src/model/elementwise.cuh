#pragma once
#include <cuda_runtime.h>
#include "../trait.cuh"
#include "../utils.cuh"

namespace {
template <typename T2>
__global__ void batched_add_kernel(int dim, const T2* a, const T2* b, T2* c) {
    int row = blockIdx.x * dim;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < dim) {
        c[row + col] = a[row + col] + b[col];
    }
}

template <typename T2>
__global__ void elementwise_add_kernel(int dim, const T2* a, const T2* b, T2* c) {
    int row = blockIdx.x * dim;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < dim) {
        c[row + col] = a[row + col] + b[row + col];
    }
}

template <typename T>
__global__ void batched_mul_kernel(int dim, const T* a, const T* b, T* c) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    T bv = b[row];
    for (int i = col; i < dim; i += blockDim.x) {
        c[row * dim + i] = a[row * dim + i] * bv;
    }
}
} // namespace

template <typename T>
void batched_add(const Stream& stream, int num_tokens, int dim, const T* a, const T* b, T* c) {
    using T2 = typename TypeTraits<T>::half2;
    dim = dim / 2;
    batched_add_kernel<T2><<<dim3(num_tokens, CEIL_DIV(dim, 512)), 512, 0, stream.stream>>>(dim, (T2*)a, (T2*)b, (T2*)c);
}

template <typename T>
void elementwise_add(const Stream& stream, int num_tokens, int dim, const T* a, const T* b, T* c) {
    using T2 = typename TypeTraits<T>::half2;
    dim = dim / 2;
    elementwise_add_kernel<T2><<<dim3(num_tokens, CEIL_DIV(dim, 512)), 512, 0, stream.stream>>>(dim, (T2*)a, (T2*)b, (T2*)c);
}

template <typename T>
void batched_mul(const Stream& stream, int num_tokens, int dim, const T* a, const T* b, T* c) {
    batched_mul_kernel<<<num_tokens, 128, 0, stream.stream>>>(dim, (T*)a, (T*)b, (T*)c);
}

// Concatenate 2 tensors [n, dim] + [n, dim] → [n, dim*2] along last dim
namespace {
template <typename T>
__global__ void concat_2_kernel(int dim, const T* a, const T* b, T* out) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    for (int i = col; i < dim; i += blockDim.x) {
        out[row * dim * 2 + i] = a[row * dim + i];
        out[row * dim * 2 + dim + i] = b[row * dim + i];
    }
}

// Concatenate 3 tensors [n, dim] + [n, dim] + [n, dim] → [n, dim*3] along last dim
template <typename T>
__global__ void concat_3_kernel(int dim, const T* a, const T* b, const T* c, T* out) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    for (int i = col; i < dim; i += blockDim.x) {
        out[row * dim * 3 + i] = a[row * dim + i];
        out[row * dim * 3 + dim + i] = b[row * dim + i];
        out[row * dim * 3 + dim * 2 + i] = c[row * dim + i];
    }
}
} // namespace

template <typename T>
void concat_2(const Stream& stream, int num_tokens, int dim, const T* a, const T* b, T* out) {
    concat_2_kernel<<<num_tokens, 512, 0, stream.stream>>>(dim, a, b, out);
}

template <typename T>
void concat_3(const Stream& stream, int num_tokens, int dim, const T* a, const T* b, const T* c, T* out) {
    concat_3_kernel<<<num_tokens, 512, 0, stream.stream>>>(dim, a, b, c, out);
}