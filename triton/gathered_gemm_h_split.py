# filename: kernels.py
import triton
import triton.language as tl
import torch
from typing import Optional


# --- Triton Kernel 定义 ---
# 使用 @triton.jit 装饰器
# 注意函数签名中的类型注解，AOT编译需要它们
@triton.jit
def gathered_gemm_h_split_kernel(
    # 指针 (64位)
    A_ptr: tl.pointer_type(tl.float16),
    B_ptr: tl.pointer_type(tl.float16),
    Indices_ptr: tl.pointer_type(tl.int32),
    C_accum_ptr: tl.pointer_type(tl.float32), # 累加器用 fp32，初始化为0
    # 形状参数 (32位整数)
    N: tl.int32, H: tl.int32, K_real: tl.int32, K_padded: tl.int32, V: tl.int32,
    # Stride 参数 (32位整数，单位是元素个数)
    stride_an: tl.int32, stride_ah: tl.int32,
    stride_bv: tl.int32, stride_bh: tl.int32,
    # 编译时常量 (Block大小)
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr
):
    # 1.Grid ID 与切分
    # Grid = [N/BLOCK_N, K_padded/BLOCK_K, H/BLOCK_H]
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_h = tl.program_id(2)

    # 计算当前 Block 负责的 N 和 K 范围
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # 计算当前 Block 负责的 H 范围 (partial sum)
    offs_h_base = pid_h * BLOCK_H
    offs_h = offs_h_base + tl.arange(0, BLOCK_H)

    # 边界 Mask (处理 N 和 K 不是 Block倍数的情况，虽然我们要求 K_padded 是倍数)
    mask_n = offs_n < N
    mask_k = offs_k < K_padded
    mask_h = offs_h < H # H 通常是 BLOCK_H 的倍数 (如 4096 vs 128)，这个mask可能是全true

    # --- 加载 Indices ---
    # 指针: Indices_ptr + offs_k
    idx_ptrs = Indices_ptr + offs_k
    # Mask: 只加载前 K_real 个真实的 index，padding 部分的 mask 为 false
    mask_k_real = offs_k < K_real
    # 加载 index，越界部分填充为 0 (指向 B 的第0行，计算时不影响结果因为后面会mask)
    # 必须转换为 int64 用于指针计算
    b_indices = tl.load(idx_ptrs, mask=mask_k_real, other=0).to(tl.int64)

    # --- 加载 A Tile [BLOCK_N, BLOCK_H] ---
    # 指针: A_ptr + (offs_n[:, None] * stride_an + offs_h[None, :] * stride_ah)
    a_ptrs = A_ptr + (offs_n[:, None] * stride_an + offs_h[None, :] * stride_ah)
    # Mask: 需要限制 N 和 H 范围
    a_mask = mask_n[:, None] & mask_h[None, :]
    # 加载 fp16 数据
    a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

    # --- 加载 B Tile (Gather) [BLOCK_K, BLOCK_H] ---
    # 指针: B_ptr + (b_indices[:, None] * stride_bv + offs_h[None, :] * stride_bh)
    # 这里利用了 broadcasting 机制进行 gather 寻址计算
    b_ptrs = B_ptr + (b_indices[:, None] * stride_bv + offs_h[None, :] * stride_bh)
    # Mask: 需要限制 K_real 和 H 范围。注意这里用 K_real mask，
    # 这样 padding 部分的行不会被加载，节省带宽。
    b_mask = mask_k_real[:, None] & mask_h[None, :]
    b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # --- 计算 Partial Sum [BLOCK_N, BLOCK_K] ---
    # 结果累加到 fp32。Triton 会自动利用 Tensor Core 如果可用。
    # a_tile @ b_tile.T
    partial_sum = tl.dot(a_tile, tl.trans(b_tile), out_dtype=tl.float32)

    # --- 原子累加写回 [BLOCK_N, BLOCK_K] ---
    # 指针: C_accum_ptr + (offs_n[:, None] * K_padded + offs_k[None, :]) # C是连续的
    c_ptrs = C_accum_ptr + (offs_n[:, None] * K_padded + offs_k[None, :])
    # Mask: 写入时只需要限制 N 和 K_padded 范围
    c_mask = mask_n[:, None] & mask_k[None, :]

    # 关键: 使用 atomic_add 将当前 H-split 的结果加到全局内存
    tl.atomic_add(c_ptrs, partial_sum, mask=c_mask)

# --- 2. AOT Launcher 函数 (新版 API 的核心) ---
# 这个 Python 函数定义了对外的 C++ API 签名和默认的 Grid 计算逻辑。
# 使用类型提示来定义参数类型。

# 指定目标 GPU 架构
# cuda:86 -> RTX 30/40系, A5000/6000
# cuda:80 -> A100
# cuda:90 -> H100
# target = triton.common.backend.cuda.CUDABackend(device=torch.cuda.current_device())
# 如果你有多个 GPU，确保选择了正确的目标
# target = triton.common.backend.cuda.CUDABackend(target_capability=86) 

@triton.aot_function()
def gathered_gemm_h_split(
    # 指针类型使用 torch.Tensor
    A_ptr: torch.Tensor,
    B_ptr: torch.Tensor,
    Indices_ptr: torch.Tensor,
    C_accum_ptr: torch.Tensor,
    # 标量类型使用 int
    N: int, H: int, K_real: int, K_padded: int, V: int,
    # 显式传入 Grid 维度 (如果需要的话，新版有时候能自动推断，但显式传入更安全)
    grid_n: int, grid_k: int, grid_h: int,
    # 编译时常量 (constexpr) 作为关键字参数传入，并设置默认值
    BLOCK_N: int = 32,
    BLOCK_K: int = 32,
    BLOCK_H: int = 128,
    num_warps: int = 4,
    num_stages: int = 3
):
    # 计算 strides (假设连续布局)
    stride_an = H
    stride_ah = 1
    stride_bv = H
    stride_bh = 1

    # 调用 JIT Kernel
    # 注意：Grid 的设置方式变了。
    # 我们显式传入一个 tuple 作为 grid 参数
    gathered_gemm_h_split_kernel[(grid_n, grid_k, grid_h)](
        A_ptr, B_ptr, Indices_ptr, C_accum_ptr,
        N, H, K_real, K_padded, V,
        stride_an, stride_ah, stride_bv, stride_bh,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
        num_warps=num_warps, num_stages=num_stages
    )

# --- 3. 执行生成过程 ---
if __name__ == "__main__":
    import sys
    # 指定输出文件名前缀
    output_name = "triton_kernels"
    
    # print(f"Start compiling for target: {target.capability}...")
    
    # Triton 3.x 的这步操作会触发 AOT 编译并生成 .h 和 .c 文件
    # 它会自动在当前目录下生成 output_name.h 和 output_name.c
    triton.aot.generate_c_code(
        gathered_gemm_h_split, # 传入被装饰的 launcher 函数
        output_name,
        sys.argv[1:] if len(sys.argv) > 1 else [] # 可选参数
    )
    print(f"Compilation done. Generated {output_name}.h and {output_name}.c")
    