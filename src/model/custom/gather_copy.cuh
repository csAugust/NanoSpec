#pragma once

// Repack Kernel: 从原始权重中收集 K 行到临时缓冲区
template <typename T>
__global__ void repack_weights_kernel(
    int K, int H, // 实际选择的行数 K，隐藏层大小 H
    const int* __restrict__ indices, // [K]
    const T* __restrict__ weight_src, // [V, H] 原始权重
    T* __restrict__ weight_packed // [K_max, H] 目标缓冲区
) {
    // 策略：每个 warp 负责拷贝一行
    int k_idx = blockIdx.x; // 当前要拷贝的第 k 个索引
    if (k_idx >= K) return;

    int target_v_idx = indices[k_idx];
    const T* src_row = weight_src + (int64_t)target_v_idx * H;
    T* dst_row = weight_packed + (int64_t)k_idx * H;

    int tid = threadIdx.x;
    // 每个线程负责搬运多个元素。使用 int4 等向量化加载可以获得更高性能。
    // 为了通用性，这里写一个简单的标量循环，性能已经足够好（因为是带宽瓶颈）
    for (int h = tid; h < H; h += blockDim.x) {
        dst_row[h] = src_row[h];
    }
}

template <typename T>
__global__ void repack_weights_kernel_incremental(
    int K_new, int H,
    const int* __restrict__ new_indices,
    const T* __restrict__ weight_src,
    T* __restrict__ weight_packed_offset // 指向 repack_buffer 的偏移位置
) {
    int k_idx = blockIdx.x; 
    if (k_idx >= K_new) return;

    int target_v_idx = new_indices[k_idx];
    const T* src_row = weight_src + (int64_t)target_v_idx * H;
    T* dst_row = weight_packed_offset + (int64_t)k_idx * H; // 写入到正确偏移位置

    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        dst_row[h] = src_row[h];
    }
}

template <typename T>
void launch_repack_weights(
    const Stream& stream,
    int K, int H,
    const int* indices,
    const T* weight_src,
    T* weight_packed
) {
    if (K == 0) return;
    // Grid size = K blocks, Block size = 128 or 256 threads
    repack_weights_kernel<T><<<K, 256, 0, stream.stream>>>(K, H, indices, weight_src, weight_packed);
}

template <typename T>
void launch_repack_weights_incremental(
    const Stream& stream,
    int K_new, int H,
    const int* indices,
    const T* weight_src,
    T* weight_packed_offset
) {
    if (K_new == 0) return;
    // Grid size = K blocks, Block size = 128 or 256 threads
    repack_weights_kernel_incremental<T><<<K_new, 256, 0, stream.stream>>>(K_new, H, indices, weight_src, weight_packed_offset);
}