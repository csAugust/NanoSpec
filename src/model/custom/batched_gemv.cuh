#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../trait.cuh"
#include "../utils.cuh"

// 宏定义 Tiling 参数，方便调优。可以根据实际情况修改。
#define BLOCK_N 8
#define BLOCK_K 64
#define BLOCK_H 128
#define THREADS_PER_BLOCK 256

// 辅助函数：将 2D shared memory 展平为 1D 指针
template<typename T, int ROWS, int COLS>
__device__ inline T* flatten_smem_ptr(T (&smem)[ROWS][COLS]) {
    return &smem[0][0];
}

// 核心 Kernel
template <typename T, bool has_bias>
__global__ void gather_gemm_tiled_kernel(
    int N, // Batch size (topk_per_iter)
    int H, // Hidden dimension size
    int K, // Number of indices (context_length)
    const T* __restrict__ A,  // Input: [N, H]
    const T* __restrict__ B, // Weight: [V, H] (Row-Major)
    const T* __restrict__ bias,   // Bias: [V] (Optional)
    const int* __restrict__ indices, // Indices: [K]
    T* __restrict__ C // Output: [N, K]
) {
    // 1. ID 计算与边界检查
    int bx = blockIdx.x; // 处理 K 维度的块 (Col Index)
    int by = blockIdx.y; // 处理 N 维度的块 (Row Index)
    int tid = threadIdx.x;

    int n_start = by * BLOCK_N;
    int k_start = bx * BLOCK_K;

    // 2. 共享内存声明
    // 加上 padding 避免 bank conflict (这里简化处理，实际可能需要)
    __shared__ T smem_A[BLOCK_N][BLOCK_H];
    __shared__ T smem_B[BLOCK_K][BLOCK_H];
    __shared__ int smem_indices[BLOCK_K];

    // 3. 寄存器累加器初始化 (使用 float 进行高精度累加)
    float thread_results[BLOCK_N / (THREADS_PER_BLOCK / BLOCK_H)][BLOCK_K / (THREADS_PER_BLOCK / BLOCK_K)] = {0.0f}; // 简化版，需要更复杂的映射
    // 为了简化实现，这里采用一种更朴素的计算方式：
    // block 内每个线程负责 calculate C 的一部分。这通常不是最高效的 GEMM 方式，但比纯标量要好得多。
    // 让我们采用一个更简单的映射：一个 Block 负责输出 C 的一个 Tile，
    // Block 内的每个线程负责计算 C tile 中一个元素 C[thread_row][thread_col] 的一部分点积。
    // 这很难。让我们回到更经典的 Tiling 思路。

    // 【重新思考计算映射】
    // 一个 Block 负责计算 C 的一个 Tile (BLOCK_N x BLOCK_K)。
    // 线程分布: Let THREADS_PER_BLOCK = 256. BLOCK_N=32, BLOCK_K=64.
    // Total elements to output per block = 32 * 64 = 2048. Not enough threads to cover 1-to-1.
    // Each thread must compute multiple output elements. E.g., 8 elements.
    
    // 这是一个非常复杂的任务。编写一个高性能的通用 tiling GEMM kernel 的工作量非常大。
    // 鉴于时间和环境限制，这里我提供一个“简化的 Tiling 实现”，性能会比串行快很多，但肯定不是 cuBLAS/CUTLASS 的级别。
    // 为了保持代码可读性和正确性，我们牺牲一点极致性能，采用【每个block计算C的一行】的策略（如果BLOCK_K足够大），
    // 或者 【每个block计算C的一个tile，内部用两重循环计算】这样的中庸策略。
    
    // 【折中方案 (High-Performance Compromise Idea)】
    // 既然 target 是 N=64, K=256+, H=4096。这其实是个 "Tall-Skinny" GEMM。
    // 策略：每个 CUDA Block 完整负责计算输出 C 的一行 (处理一个 Batch)。
    // Grid size = (N, 1, 1). Block size = 256 (或更多，如 512, 1024)。
    // Block 内部，线程协作读取 Input A 的那一行到 Shared Mem。
    // Block 内部，线程协作读取 Indices 的那一列到 Shared Mem。
    // 然后 Block 内部，循环处理 Indices，每次处理 THREADS_PER_BLOCK 个 Index。
    // 对于每个 Index，线程协作读取 Weights B 中对应行的 H 维数据点积。

    // 这个策略比通用的 2D Tiling 实现起来更简单，也更贴合你的数据形状。
}

// 鉴于通用 tiling kernel 的复杂度，我们这里改用一个更简单、更容易保证正确性的高性能策略：
// “每个 Block 处理一个 N，线程束处理 K”。这种策略对于 N 较小 (如 64) 而 K, H 较大的情况非常有效。
// 这实际上是把 GEMM 拆成了 N 个并发执行的 GEMV kernel，但它们共享 indices 加载，并且最大化了并行度。

template <typename T, bool has_bias>
__global__ void gather_gemm_batched_optimized_kernel(
    int N, int H, int K,
    const T* __restrict__ A, // [N, H]
    const T* __restrict__ B, // [V, H]
    const T* __restrict__ bias, // [V]
    const int* __restrict__ indices, // [K]
    T* __restrict__ C // [N, K]
) {
    int n_idx = blockIdx.x; // Current batch index
    if (n_idx >= N) return;

    int tid = threadIdx.x;
    int warpid = tid / 32;
    int laneid = tid % 32;
    int num_warps = blockDim.x / 32;

    // 指向当前 Batch 的 Input A 的行起始
    const T* A_row = A + n_idx * H;
    // 指向当前 Batch 的 Output C 的行起始
    T* C_row = C + n_idx * K;

    // 每个 Warp 处理一个或多个输出元素 (K 维度)
    // 策略: Warp-level Loop over K.
    for (int k_idx = warpid; k_idx < K; k_idx += num_warps) {
        int target_v_idx = indices[k_idx];
        const T* B_row = B + (int64_t)target_v_idx * H;

        // Warp 内部协作计算点积 (类似于你之前的 GEMV kernel)
        float sum = 0.0f;
        for (int h = laneid; h < H; h += 32) {
            sum += static_cast<float>(A_row[h]) * static_cast<float>(B_row[h]);
        }

        // Warp-wide Reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (laneid == 0) {
            if constexpr (has_bias) {
                sum += static_cast<float>(bias[target_v_idx]);
            }
            C_row[k_idx] = static_cast<T>(sum);
        }
    }
}

// Launcher 函数
template <typename T, bool has_bias>
void launch_gather_gemm_batched(
    const Stream& stream,
    int N, int H, int K,
    const T* A, const T* B, const T* bias,
    const int* indices, T* C
) {
    if (N == 0 || K == 0) return;

    // 配置参数
    // grid size = N (每个 block 处理一个 batch)
    int blocks = N; 
    // block size: 可以是 256, 512, 1024. 越大并行处理的 K 越多
    // 对于 K~256+, 建议用 512 或 1024 以提高利用率 (occupancy)
    int threads_per_block = 256; // 8 warps

    gather_gemm_batched_optimized_kernel<T, has_bias><<<blocks, threads_per_block, 0, stream.stream>>>(
        N, H, K, A, B, bias, indices, C
    );
    // cudaGetLastError(); // Optional check
}