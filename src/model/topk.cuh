#pragma once
#include "../utils.cuh"
#include "../trait.cuh"
#include <algorithm> // 需要包含以使用 std::min

namespace functions {
namespace {
// ... [保留原有的 warpBitonicSort, warpBitonicMerge, blockBitonicReduce 实现不变] ...
template<typename T, int N>
static __device__ inline void warpBitonicSort(T& v1, int& pos, bool asc) {
    int lane_id = threadIdx.x & (N - 1);
    #pragma unroll
    for (int k = 2; k <= N; k *= 2) {
        bool desc = ((lane_id & k) == 0) ^ asc;
        #pragma unroll
        for (int j = k / 2; j > 0; j /= 2) {
            T v2 = __shfl_xor_sync(0xFFFFFFFF, v1, j);
            int pos2 = __shfl_xor_sync(0xFFFFFFFF, pos, j);
            bool upper = (lane_id & j) != 0;
            if (desc ^ (v1 > v2 || (v1 == v2 && pos < pos2)) ^ upper) {
                v1 = v2;
                pos = pos2;
            }
        }
    }
}
template<typename T, int N>
static __device__ inline void warpBitonicMerge(T& v1, int& pos1, T& v2, int& pos2) {
    if (v1 < v2 || (v1 == v2 && pos1 > pos2)) {
        v1 = v2;
        pos1 = pos2;
    }
    int lane_id = threadIdx.x & (N - 1);
    // resort
    #pragma unroll
    for (int j = N / 2; j > 0; j /= 2) {
        v2 = __shfl_xor_sync(0xFFFFFFFF, v1, j);
        int pos2 = __shfl_xor_sync(0xFFFFFFFF, pos1, j);
        bool upper = (lane_id & j) != 0;
        if ((v1 < v2 || (v1 == v2 && pos1 > pos2)) ^ upper) {
            v1 = v2;
            pos1 = pos2;
        }
    }
}
template<typename T, int N>
static __device__ inline void blockBitonicReduce(T& v, int& pos) {
    __shared__ T shared_val[1024];
    __shared__ int shared_pos[1024];
    // block reduce
    shared_val[threadIdx.x] = v;
    shared_pos[threadIdx.x] = pos;
    // inter warp reduce
    #pragma unroll
    for (int i = 512; i >= 32; i >>= 1) {
        if (blockDim.x > i) {
            __syncthreads();
            if (threadIdx.x < i) {
                int idx_next = (i << 1) - threadIdx.x - 1;
                T nw_v = (idx_next < blockDim.x) ? shared_val[idx_next] : T(-TypeTraits<T>::inf());
                int nw_pos = (idx_next < blockDim.x) ? shared_pos[idx_next] : -1;
                warpBitonicMerge<T, N>(v, pos, nw_v, nw_pos); // merge and rebuild in desc order
                shared_val[threadIdx.x] = v;
                shared_pos[threadIdx.x] = pos;
            }
        }
    }
    // intra warp reduce
    if (threadIdx.x < 32) {
        warpBitonicSort<T, 32>(v, pos, false);
    }
}
// ... [保留 kernel_bitonic_topk, kernel_bitonic_topk_multiblock, kernel_bitonic_topk_multiblock_copy] ...
template<typename T, int N>
static __global__ void kernel_bitonic_topk(
    int n, int top,
    T *inp,     // (batch, n)
    float *out,     // (batch, top)
    int *idx    // (batch, top)
) {
    int offset_inp = blockIdx.x * n;
    int offset_out = blockIdx.x * top;
    T local_v = threadIdx.x < n ? inp[offset_inp + threadIdx.x] : -TypeTraits<T>::inf();
    int local_pos = threadIdx.x;
    warpBitonicSort<T, N>(local_v, local_pos, false); // local sort in desc order
    for (int i = blockDim.x; i < n; i += blockDim.x) {
        T nw_v = (i + threadIdx.x) < n ? inp[offset_inp + i + threadIdx.x] : -TypeTraits<T>::inf();
        int nw_pos = i + threadIdx.x;
        // step.1: local sort
        warpBitonicSort<T, N>(nw_v, nw_pos, true); // local sort in asc order
        // step.2&3: merge and rebuild
        warpBitonicMerge<T, N>(local_v, local_pos, nw_v, nw_pos); // merge and rebuild in desc order
    }
    blockBitonicReduce<T, N>(local_v, local_pos);
    if (threadIdx.x < top) {
        out[offset_out + threadIdx.x] = local_v;
        idx[offset_out + threadIdx.x] = local_pos;
    }
}
// intra-block topk
// gridDim(batch, n / 1024, 1), threadDim(1024, 1, 1)
template<typename T, int N, bool ordered>
static __global__ void kernel_bitonic_topk_multiblock(
    int n,
    const T *inp,       // (batch, n)
    const int *idx_inp, // (batch, n)
    T *out,     // (batch, n / 1024 * N)
    int *idx    // (batch, n / 1024 * N)
) {
    int offset_col = blockIdx.y * blockDim.x + threadIdx.x;
    int offset_inp = blockIdx.x * n + offset_col;
    int offset_out = blockIdx.x * (gridDim.y * N) + blockIdx.y * N + threadIdx.x;
    T local_v = (offset_col < n) ? inp[offset_inp] : T(-TypeTraits<T>::inf());
    int local_pos = (idx_inp == nullptr) ? offset_col : ((offset_col < n) ? idx_inp[offset_inp] : -1);
    if (!ordered) warpBitonicSort<T, N>(local_v, local_pos, false); // local sort in desc order
    blockBitonicReduce<T, N>(local_v, local_pos);
    if (threadIdx.x < N) {
        out[offset_out] = local_v;
        idx[offset_out] = local_pos;
    }
}
// copy kernel
// gridDim(batch, 1, 1),   blockDim(top, 1, 1)
template<typename T>
static __global__ void kernel_bitonic_topk_multiblock_copy (
    int n, int top,
    const T *inp,       // (batch, n)
    const int *idx_inp, // (batch, n)
    T *out,         // (batch, top)
    int *idx            // (batch, top)
) {
    int offset_inp = blockIdx.x * n + threadIdx.x;
    int offset_out = blockIdx.x * top + threadIdx.x;
    if (threadIdx.x < top) {
        out[offset_out] = inp[offset_inp];
        idx[offset_out] = idx_inp[offset_inp];
    }
}

// ... [保留 TOPK_SIZE_DISPATCH 和 bitonic_topk] ...
#define TOPK_SIZE_DISPATCH(top, ...) \
    do { \
        const int &top_v = top; \
        if (top_v > 16) { \
            const int top_size = 32; \
            __VA_ARGS__ \
        } else if (top_v > 8) { \
            const int top_size = 16; \
            __VA_ARGS__ \
        } else if (top_v > 4) { \
            const int top_size = 8; \
            __VA_ARGS__ \
        } else if (top_v > 2) { \
            const int top_size = 4; \
            __VA_ARGS__ \
        } else if (top_v > 1) { \
            const int top_size = 2; \
            __VA_ARGS__ \
        } else { \
            const int top_size = 1; \
            __VA_ARGS__ \
        } \
    } while(0)
template <typename T>
void bitonic_topk(
    const Stream& stream,
    const int batch,
    const int n,
    const int top,
    const T* x, 
    T* out, 
    int* pos,	
    T* buf_val,
    int* buf_pos,
    T* nw_buf_val,
    int* nw_buf_pos
) {
    TOPK_SIZE_DISPATCH(top, {
        bool first = true;
        dim3 blockDim(1024, 1, 1);
        unsigned int tmp_n = n;
        do {
            dim3 gridDim(batch, CEIL_DIV(tmp_n, 1024), 1);
            if (first) {
                first = false;
                kernel_bitonic_topk_multiblock<T, top_size, false><<<gridDim, blockDim, 0, stream.stream>>>(
                    tmp_n,
                    x,
                    nullptr,
                    buf_val,
                    buf_pos
                );
            } else {
                kernel_bitonic_topk_multiblock<T, top_size, false><<<gridDim, blockDim, 0, stream.stream>>>(
                    tmp_n,
                    buf_val,
                    buf_pos,
                    nw_buf_val,
                    nw_buf_pos
                );
                buf_val = nw_buf_val;
                buf_pos = nw_buf_pos;
            }
            tmp_n = CEIL_DIV(tmp_n, 1024) * top_size;
        } while (tmp_n > top_size);
        // copy to output tensor
        {
            dim3 gridDim(batch, 1, 1);
            blockDim = dim3(top_size, 1, 1);
            kernel_bitonic_topk_multiblock_copy<T><<<gridDim, blockDim, 0, stream.stream>>>(
                top_size, top,
                buf_val,
                buf_pos,
                out,
                pos
            );
        }
    });
}
// ... [保留 set_topk_to_neg_inf_kernel 和 set_topk_to_neg_inf] ...
template<typename T>
static __global__ void set_topk_to_neg_inf_kernel(int dim, T* x, const int* topk_pos) {
    // 使用 blockIdx.x 来处理 batch 维度偏移
    x[blockIdx.x * dim + topk_pos[threadIdx.x]] = -TypeTraits<T>::inf();
}
} // namespace

template<typename T>
void set_topk_to_neg_inf(const Stream& stream, int num_tokens, int dim, int top, T* x, const int* topk_pos) {
    set_topk_to_neg_inf_kernel<<<num_tokens, top, 0, stream.stream>>>(dim, x, topk_pos);
}


template <typename T>
struct TopK {
private:
    T *buf_val, *nw_buf_val;
    int *buf_pos, *nw_buf_pos;
public:
    int dim, top;
    T* topk_val;
    int* topk_pos;
    T* tmp_x;

    TopK(const int dim, const int top) {
        this->dim = dim;
        this->top = top;
        this->tmp_x = nullptr; // 初始化指针
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        // 这里的 max_top_size 是 32，因为 TOPK_SIZE_DISPATCH 最大钳位到 32
        const int max_top_size = 32;
        offset = memory->allocate((void**)&buf_val, offset, num_tokens * CEIL_DIV(dim, 1024) * max_top_size * sizeof(T));
        offset = memory->allocate((void**)&buf_pos, offset, num_tokens * CEIL_DIV(dim, 1024) * max_top_size * sizeof(int));
        offset = memory->allocate((void**)&nw_buf_val, offset, num_tokens * CEIL_DIV(dim, 1024) * max_top_size * sizeof(T));
        offset = memory->allocate((void**)&nw_buf_pos, offset, num_tokens * CEIL_DIV(dim, 1024) * max_top_size * sizeof(int));

        // 如果需要的 top 大于基础 kernel 能处理的最大值 (32)，则需要临时 buffer 用于迭代 masking
        if (this->top > 32) { 
            offset = memory->allocate((void**)&tmp_x, offset, num_tokens * dim * sizeof(T));
        }
        offset = memory->allocate((void**)&topk_val, offset, num_tokens * this->top * sizeof(T));
        offset = memory->allocate((void**)&topk_pos, offset, num_tokens * this->top * sizeof(int));
        return offset;
    }

    void prefill(
        const Stream& stream,
        int num_tokens,
        const T* input,
        int dim = -1,
        int top = -1
    ) {
        if (dim == -1) dim = this->dim;
        if (top == -1) top = this->top;

        T* current_input = const_cast<T*>(input);
        
        // 如果 top > 32，我们需要进行多次迭代，每次迭代需要修改输入数据（mask掉已选中的），
        // 所以必须拷贝一份输入数据到 tmp_x。
        if (top > 32) {
            // 使用异步拷贝，并指定 stream
            cudaMemcpyAsync(this->tmp_x, input, num_tokens * dim * sizeof(T), cudaMemcpyDeviceToDevice, stream.stream);
            current_input = this->tmp_x;
        }

        int remaining_top = top;
        int output_offset = 0;

        // 循环迭代，直到找到所需的全部 top 个元素
        while (remaining_top > 0) {
            // 底层 bitonic_topk kernel 受限于 warp 大小，每次最多只能稳定输出 32 个结果。
            // 我们取当前剩余需求和 32 中的较小值作为本次迭代的目标。
            int current_iter_top = std::min(remaining_top, 32);

            bitonic_topk<T>(
                stream,
                num_tokens,
                dim, 
                current_iter_top, // 本次迭代只需找这么多
                current_input,    // 输入数据（源数据或 tmp_x）
                this->topk_val + output_offset, // 输出值 buffer 偏移
                this->topk_pos + output_offset, // 输出索引 buffer 偏移
                this->buf_val, this->buf_pos,
                this->nw_buf_val, this->nw_buf_pos
            );

            remaining_top -= current_iter_top;

            // 如果还需要进行下一轮迭代，则将本来找到的元素的输入值设为负无穷大（masking）
            if (remaining_top > 0) {
                set_topk_to_neg_inf(
                    stream,
                    num_tokens,
                    dim, 
                    current_iter_top, // kernel 线程数等于本次找到的元素个数
                    this->tmp_x,      // 必须修改临时 buffer
                    this->topk_pos + output_offset // 指向本次迭代找到的索引位置
                );
            }
            output_offset += current_iter_top;
        }
    }
};
} // namespace functions