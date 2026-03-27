#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

#include "utils.cuh"
#include "trait.cuh"
#include "model/profiling.cuh"
#include "model/model.cuh"
#include "model/medusa.cuh"
#include "model/eagle.cuh"
#include "model/eagle3.cuh"

DraftProfiler g_profiler;

#define DTYPE_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND == 0) {                              \
      using elem_type = __half;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = __nv_bfloat16; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define BOOL_SWITCH(COND, CONST_NAME, ...)               \
  [&] {                                      \
    if (COND) {                              \
      constexpr static bool CONST_NAME = true; \
      return __VA_ARGS__();                  \
    } else {                                 \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

Model* model;

void init_base_model(
    int64_t memory_limit,
    std::uintptr_t memory_pool,
    int vocab_size,
    int num_hidden_layers,
    int hidden_size,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    int torch_dtype,
    int chunk_length,
    int attention_bias
) {
    init_resources();

    BOOL_SWITCH(attention_bias, has_attention_bias, [&] {
        DTYPE_SWITCH(torch_dtype, [&] {
            model = new ModelImpl<elem_type, has_attention_bias>(
                memory_limit,
                reinterpret_cast<void*>(memory_pool),
                vocab_size,
                num_hidden_layers,
                hidden_size,
                intermediate_size,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                rms_norm_eps,
                chunk_length
            );
        });
    });
}

void init_medusa_model(
    int num_heads,
    int num_layers,
    int topk_per_head,
    int tree_size,
    std::uintptr_t tree_indices,
    std::uintptr_t draft_position_ids,
    int V,
    int torch_dtype
) {
    DTYPE_SWITCH(torch_dtype, [&] {
        const bool attention_bias = dynamic_cast<ModelImpl<elem_type, true>*>(model) != nullptr;
        BOOL_SWITCH(attention_bias, has_attention_bias, [&] {
            model = new MedusaImpl<elem_type, has_attention_bias>(
                (ModelImpl<elem_type, has_attention_bias>*)model,
                num_heads,
                num_layers,
                topk_per_head,
                tree_size,
                V,
                reinterpret_cast<int32_t*>(tree_indices),
                reinterpret_cast<int32_t*>(draft_position_ids)
            );
        });
    });
}

void init_eagle_model(
    int num_layers,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int V,
    int torch_dtype,
    int max_context_tokens
) {
    DTYPE_SWITCH(torch_dtype, [&] {
        const bool attention_bias = dynamic_cast<ModelImpl<elem_type, true>*>(model) != nullptr;
        BOOL_SWITCH(attention_bias, has_attention_bias, [&] {
            model = new EagleImpl<elem_type, has_attention_bias>(
                (ModelImpl<elem_type, has_attention_bias>*)model,
                num_layers,
                intermediate_size,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                rms_norm_eps,
                num_iter,
                topk_per_iter,
                tree_size,
                V,
                max_context_tokens
            );
        });
    });
}

void init_eagle3_model(
    int num_layers,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int draft_vocab_size,
    int torch_dtype,
    int max_context_tokens
) {
    DTYPE_SWITCH(torch_dtype, [&] {
        const bool attention_bias = dynamic_cast<ModelImpl<elem_type, true>*>(model) != nullptr;
        BOOL_SWITCH(attention_bias, has_attention_bias, [&] {
            model = new Eagle3Impl<elem_type, has_attention_bias>(
                (ModelImpl<elem_type, has_attention_bias>*)model,
                num_layers,
                intermediate_size,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                rms_norm_eps,
                num_iter,
                topk_per_iter,
                tree_size,
                draft_vocab_size,
                max_context_tokens
            );
        });
    });
}

int init_storage() {
    return model->init_storage();
}

void load_model(std::string name, std::uintptr_t param) {
    model->load_to_storage(name, reinterpret_cast<void*>(param));
}

void prefill(int input_length, int history_length, std::uintptr_t input, std::uintptr_t position_ids, std::uintptr_t output) {
    model->prefill(input_length, history_length, reinterpret_cast<int32_t*>(input), reinterpret_cast<int32_t*>(position_ids), (void*)(output));
}

void decode(int input_length, int padded_length, std::uintptr_t input, std::uintptr_t position_ids, std::uintptr_t cache_length, std::uintptr_t mask_2d, std::uintptr_t output, bool cuda_graph) {
    if (cuda_graph) {
        if (graphCreated_padding_length != padded_length || graphCreated_input_length != input_length) {
            if (graphExec != nullptr) {
                cudaGraphExecDestroy(graphExec);
                graphExec = nullptr;
            }
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
                graph = nullptr;
            }
            cudaStreamBeginCapture(calc_stream.stream, cudaStreamCaptureModeGlobal);
            model->decode(input_length, padded_length, reinterpret_cast<int32_t*>(input), reinterpret_cast<int32_t*>(position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(mask_2d), reinterpret_cast<void*>(output));
            cudaStreamEndCapture(calc_stream.stream, &graph);
            cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
            graphCreated_padding_length = padded_length;
            graphCreated_input_length = input_length;
        }
        cudaGraphLaunch(graphExec, calc_stream.stream);
    } else {
        model->decode(input_length, padded_length, reinterpret_cast<int32_t*>(input), reinterpret_cast<int32_t*>(position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(mask_2d), reinterpret_cast<void*>(output));
    }
}

void draft(std::uintptr_t tree_draft_ids, std::uintptr_t tree_position_ids, std::uintptr_t cache_length, std::uintptr_t attn_mask, std::uintptr_t tree_parent, std::uintptr_t context_tokens, std::int32_t context_length, std::int32_t mode) {
    model->draft(reinterpret_cast<int32_t*>(tree_draft_ids), reinterpret_cast<int32_t*>(tree_position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(attn_mask), reinterpret_cast<int32_t*>(tree_parent), reinterpret_cast<int32_t*>(context_tokens), context_length, mode);
}

int verify_and_fix(int num_tokens, std::uintptr_t pred, std::uintptr_t gt, std::uintptr_t position_ids, std::uintptr_t cache_length, std::uintptr_t attn_mask, std::uintptr_t tree_parent) {
    return model->verify(num_tokens, reinterpret_cast<int32_t*>(pred), reinterpret_cast<int32_t*>(gt), reinterpret_cast<int32_t*>(position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(attn_mask), reinterpret_cast<int32_t*>(tree_parent));
}

void trigger_async_prefetch(std::uintptr_t gpu_context_buffer, std::uintptr_t cpu_new_tokens, std::int32_t num_new, std::int32_t offset) {
    if (num_new == 0) return;
    model->async_prefetch(
        reinterpret_cast<int32_t*>(gpu_context_buffer),
        reinterpret_cast<int32_t*>(cpu_new_tokens),
        num_new,
        offset
    );
}

void enable_draft_profiling(bool enabled) {
    if (enabled && g_profiler.accum[0] == 0 && g_profiler.num_calls == 0) {
        g_profiler.init();
    }
    g_profiler.enabled = enabled;
}

void reset_draft_profiling() {
    g_profiler.reset();
}

std::vector<float> get_draft_timing() {
    // Returns [backbone_ms, lmhead_ms, tree_ops_ms, gather_wait_ms, num_calls]
    return {
        g_profiler.accum[PL_BACKBONE],
        g_profiler.accum[PL_LMHEAD],
        g_profiler.accum[PL_TREE],
        g_profiler.accum[PL_GATHER_WAIT],
        (float)g_profiler.num_calls
    };
}

PYBIND11_MODULE(C, m) {
    m.def("init_base_model", &init_base_model, "Init base model");
    m.def("init_medusa_model", &init_medusa_model, "Init medusa model");
    m.def("init_eagle_model", &init_eagle_model, "Init eagle model");
    m.def("init_eagle3_model", &init_eagle3_model, "Init eagle3 model");
    m.def("init_storage", &init_storage, "Init storage");
    m.def("load_model", &load_model, "Load model");
    m.def("prefill", &prefill, "Prefill");
    m.def("decode", &decode, "Decode");
    m.def("draft", &draft, "Draft");
    m.def("verify_and_fix", &verify_and_fix, "Verify and fix");
    m.def("trigger_async_prefetch", &trigger_async_prefetch, "trigger_async_prefetch");
    m.def("enable_draft_profiling", &enable_draft_profiling, "Enable/disable draft profiling");
    m.def("reset_draft_profiling", &reset_draft_profiling, "Reset draft profiling counters");
    m.def("get_draft_timing", &get_draft_timing, "Get accumulated draft timing [backbone, lmhead, tree, gather_wait, num_calls]");
}