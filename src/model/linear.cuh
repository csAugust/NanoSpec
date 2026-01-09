#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../trait.cuh"
#include "../utils.cuh"
#include "elementwise.cuh"
#include "custom/batched_gemv.cuh"

// Kernel implementation using Warp-level primitives
template <typename T, bool has_bias>
__global__ void gather_gemv_warp_reduction_kernel(
    int H, // Hidden dimension size
    int K, // Number of indices to select
    const T* __restrict__ input,  // Shape [H]
    const T* __restrict__ weight, // Shape [V, H], assumed row-major contiguous
    const T* __restrict__ bias,   // Shape [V], optional
    const int* __restrict__ indices, // Shape [K]
    T* __restrict__ output // Shape [K], smaller output buffer
) {
    // Calculate global warp ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // If this warp is outside the needed indices range, exit
    if (warp_id >= K) return;

    // Get the target token index for this warp
    int target_idx = indices[warp_id];

    // Pointer to the start of the specific weight row
    // Using int64_t for offset to avoid overflow with large models
    const T* weight_row = weight + (int64_t)target_idx * H;

    // Accumulate in float32 for precision
    float sum = 0.0f;

    // Collaborative loading and computation loop
    // Threads in a warp process the Hidden dimension in chunks of 32
    for (int i = lane_id; i < H; i += 32) {
        // Implicit cast from T to float assumed here.
        // Use __half2float() if T is half and implicit cast fails.
        float w_val = static_cast<float>(weight_row[i]);
        float i_val = static_cast<float>(input[i]);
        sum += w_val * i_val;
    }

    // Warp-wide reduction to sum up partial results from threads
    // Using __shfl_down_sync intrinsic
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Thread 0 within the warp holds the sum and writes final result
    if (lane_id == 0) {
        if constexpr (has_bias) {
            sum += static_cast<float>(bias[target_idx]);
        }
        // Cast back to T for storage
        output[warp_id] = static_cast<T>(sum);
    }
}

// Helper function to launch the kernel
template <typename T, bool has_bias>
void launch_gather_gemv(
    const Stream& stream,
    int H, int K,
    const T* input, const T* weight, const T* bias,
    const int* indices, T* output
) {
    if (K == 0) return;

    // Configuration: e.g., 256 threads (8 warps) per block
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    // Calculate required blocks to cover K warps
    int blocks = (K + warps_per_block - 1) / warps_per_block;

    gather_gemv_warp_reduction_kernel<T, has_bias><<<blocks, threads_per_block, 0, stream.stream>>>( // Assuming stream holds a cudaStream_t compatible handle
        H, K, input, weight, bias, indices, output
    );
    // Optional: add cudaGetLastError() check here for debugging
}


template <typename T, bool transposed=true>
void linear(const Stream& stream, int num_tokens, int dim_in, int dim_out, const T* input, const T* weight, T* output, bool inplace=false) {
    float alpha = 1.0f;
    float beta = inplace ? 1.0f : 0.0f;
    if constexpr (transposed) {
        cublasCheck(cublasGemmEx(stream.cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_out, num_tokens, dim_in,
            &alpha,
            weight, TypeTraits<T>::cublas_type(), dim_in,
            input, TypeTraits<T>::cublas_type(), dim_in,
            &beta,
            output, TypeTraits<T>::cublas_type(), dim_out,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));
    } else {
        cublasCheck(cublasGemmEx(stream.cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_out, num_tokens, dim_in,
            &alpha,
            weight, TypeTraits<T>::cublas_type(), dim_out,
            input, TypeTraits<T>::cublas_type(), dim_in,
            &beta,
            output, TypeTraits<T>::cublas_type(), dim_out,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));
    }
}

template <typename T, bool transposed=true, bool has_bias=false>
struct Linear {
    int dim_in;
    int dim_out;
    T* output;
    T* weight;
    T* bias;

    Linear(int dim_in, int dim_out) {
        this->dim_in = dim_in;
        this->dim_out = dim_out;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(dim_in * dim_out * sizeof(T));
        if constexpr (has_bias) {
            bias = (T*)memory->allocate_for_model(dim_out * sizeof(T));
        }
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * dim_out * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("weight") != std::string::npos) {
            cudaMemcpy((void*)weight, ptr, dim_in * dim_out * sizeof(T), cudaMemcpyHostToDevice);
        } else if (name.find("bias") != std::string::npos) {
            cudaMemcpy((void*)bias, ptr, dim_out * sizeof(T), cudaMemcpyHostToDevice);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* tgt=nullptr, bool inplace=false) {
        if (tgt == nullptr) tgt = this->output;
        linear<T, transposed>(stream, num_tokens, dim_in, dim_out, input, weight, tgt, inplace);
        if constexpr (has_bias) {
            batched_add<T>(stream, num_tokens, dim_out, tgt, bias, tgt);
        }
    }

    // Calculates linear projection only for selected indices.
    // num_indices: K, the number of selected tokens.
    // indices: Device pointer to an int array of size K containing token IDs.
    // tgt_output: Device pointer to output buffer of size K*sizeof(T).
    // num_tokens is assumed to be 1.
    void prefill_gathered(const Stream& stream, T* input, const int* indices, int num_indices, T* tgt_output) {
        // Safety check: This optimization relies on the layout assumption consistent with
        // transposed=true (weight stored logically row-major [Out, In]).
        if constexpr (!transposed) {
            // You might want to handle error reporting differently in your framework
            printf("Error: Gathered GEMM currently only implemented for transposed Linear layers (like lm_head).\n");
            abort();
        }

        // dim_in here is hidden_size (H)
        // dim_out here is vocab_size (V), which we don't use directly in the launch params
        launch_gather_gemv<T, has_bias>(
            stream,
            this->dim_in, // H
            num_indices,  // K
            input,
            this->weight,
            this->bias, // Will be nullptr if has_bias is false, kernel handles it
            indices,
            tgt_output
        );
    }

    void prefill_repack_sync(const Stream& stream, int32_t num_tokens, int32_t effective_dim_out, T* input, T* repack_weight, T* tgt=nullptr, bool inplace=false) {
        linear<T, transposed>(stream, num_tokens, dim_in, effective_dim_out, input, repack_weight, tgt, inplace);
    }

    // N: num_tokens (batch size)
    // K: num_indices (context length)
    // input: [N, H]
    // tgt_output: [N, K]
    void prefill_gathered_batched(const Stream& stream, int N, T* input, const int* indices, int K, T* tgt_output) {
        static_assert(transposed, "Gathered GEMM requires transposed weight layout [Out, In]");
        
        // N here is num_tokens (batch size)
        // H is dim_in
        // K is num_indices (context length)
        // V is dim_out
        
        launch_gather_gemm_batched<T, has_bias>(
            stream,
            N,           // N
            this->dim_in, // H
            K,           // K
            input,       // A
            this->weight,// B
            this->bias,  // bias
            indices,     // indices
            tgt_output   // C
        );
    }
};