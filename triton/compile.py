# filename: compile.py
import triton

# 1. 指定要编译的源代码文件路径
kernel_src_file = "/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/triton/gathered_gemm_h_split.py"

# 2. 定义要编译的内核及其签名
# 这部分配置保持不变，用于告诉编译器如何实例化模板
start_signature = [
    "*fp16", "*fp16", "*i32", "*fp32", # 指针参数
    "i32", "i32", "i32", "i32", "i32", # 形状参数
    "i32", "i32", "i32", "i32",        # Stride 参数
]

# 针对 H=4096 的调优配置
constexprs = {
   "BLOCK_N": 32,
   "BLOCK_K": 32,
   "BLOCK_H": 128,
   # num_warps 和 num_stages 通常作为命名参数传递给 triton.compile，而不是在这里
}

# 3. 配置 triton.compile 的入口参数
# 这告诉编译器去 "kernels.py" 文件里找一个叫 "gathered_gemm_h_split_kernel" 的函数，
# 并用给定的 signature 和 constants 进行编译。
compile_kwargs = {
    "kernels": {
        "gathered_gemm_h_split_kernel": (start_signature, constexprs)
    },
    "num_warps": 4,  # 设置每个 Block 的 Warp 数
    "num_stages": 3, # 设置软件流水线级数
}

# 4. 设置输出文件名
output_name = "triton_kernels"

# 5. 设置目标 GPU 架构 (根据你的实际GPU修改，例如 'cuda:86' for RTX4090/A5000, 'cuda:80' for A100)
target = "cuda:86"

print(f"Compiling '{kernel_src_file}' for target: {target}...")

# --- 【修正后的调用方式】 ---
triton.compile(
    kernel_src_file, # 修正点：第一个参数传源代码文件路径
    **compile_kwargs # 将 num_warps, num_stages 等作为命名参数解包传入
)

# 如果需要在生成的 .h 文件中指定一个干净的函数别名，可以使用 'pkg_name' 参数，
# 但这通常需要更复杂的配置。标准做法是直接使用生成的带哈希的长名字。

print(f"Compilation done. Generated {output_name}.h and {output_name}.c")