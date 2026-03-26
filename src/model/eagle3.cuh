#pragma once
#include "tree_drafter.cuh"
#include "model.cuh"
#include "topk.cuh"
#include "layer.cuh"
#include "kvcache.cuh"
#include "norm.cuh"
#include "elementwise.cuh"
#include "eagle.cuh"  // reuse utility kernels: log_softmax, build_dynamic_tree, remap_*, etc.

// EAGLE-3 specific kernel: convert d2t offset mapping to direct remap
// target_id[i] = i + d2t[i]  (d2t stores offsets, we precompute direct mapping)
namespace {
__global__ void d2t_to_remap_kernel(int n, const int32_t* d2t, int32_t* remap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        remap[i] = i + d2t[i];
    }
}
} // namespace

void d2t_to_remap(const Stream& stream, int n, const int32_t* d2t, int32_t* remap) {
    d2t_to_remap_kernel<<<CEIL_DIV(n, 256), 256, 0, stream.stream>>>(n, d2t, remap);
}

// Build inverse mapping: t2d_index[target_id] = draft_index (or -1 if not in draft vocab)
// token_id_remap[draft_idx] = target_id  =>  t2d_index[target_id] = draft_idx
namespace {
__global__ void build_t2d_index_kernel(int draft_vocab_size, const int32_t* token_id_remap, int32_t* t2d_index) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < draft_vocab_size) {
        t2d_index[token_id_remap[i]] = i;
    }
}

// Convert target vocab IDs to draft vocab indices using t2d_index
__global__ void target_to_draft_kernel(int n, const int32_t* target_ids, int32_t* draft_ids, const int32_t* t2d_index) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        draft_ids[i] = t2d_index[target_ids[i]];
    }
}
} // namespace

void build_t2d_index(const Stream& stream, int vocab_size, int draft_vocab_size, const int32_t* token_id_remap, int32_t* t2d_index) {
    cudaMemsetAsync(t2d_index, 0xff, vocab_size * sizeof(int32_t), stream.stream);  // fill with -1
    build_t2d_index_kernel<<<CEIL_DIV(draft_vocab_size, 256), 256, 0, stream.stream>>>(draft_vocab_size, token_id_remap, t2d_index);
}

void target_to_draft(const Stream& stream, int n, const int32_t* target_ids, int32_t* draft_ids, const int32_t* t2d_index) {
    target_to_draft_kernel<<<CEIL_DIV(n, 256), 256, 0, stream.stream>>>(n, target_ids, draft_ids, t2d_index);
}

template<typename T, bool has_attention_bias=false>
struct Eagle3Impl : Model {
    int num_layers;  // number of draft layers (typically 1)
    int num_iter;
    int topk_per_iter;
    int tree_size;
    int total_tried;
    int draft_vocab_size;

    ModelImpl<T, has_attention_bias>* model;

    // EAGLE-3 draft model components
    Linear<T>* fc;                // hidden_size*3 → hidden_size (only for first draft)
    RMSNorm<T>* hidden_norm;     // norm hidden states before attention
    RMSNorm<T>* input_layernorm; // norm embeddings before attention
    // QKV: input_dim = hidden_size*2 (concat of normed embed + normed hidden)
    Linear<T, true>* qkv_proj;
    Linear<T, true>* q_proj;     // alias into qkv_proj
    Linear<T, true>* k_proj;     // alias into qkv_proj
    Linear<T, true>* v_proj;     // alias into qkv_proj
    Linear<T>* o_proj;
    GatedFFN<T>* eagle3_ffn;     // post-attention FFN (has its own post_attention_layernorm)
    RMSNorm<T>* final_norm;      // norm before lm_head
    Linear<T>* lm_head;          // hidden_size → draft_vocab_size (EAGLE-3 built-in d2t mode)

    KVCacheManager<T>* kv_caches;
    int num_attention_heads, num_key_value_heads, head_dim;

    // Vocabulary mapping for EAGLE-3 built-in d2t mode
    int32_t* token_id_remap;  // precomputed: remap[i] = i + d2t[i]  (draft_idx → target_id)
    int32_t* t2d_index;       // inverse: t2d_index[target_id] = draft_idx (or -1)

    // ---- Modes 1 & 2: dynamic vocab pruning on EAGLE-3's own lm_head ----
    int32_t max_context_tokens;        // max context buffer size
    int32_t* context_draft_ids;        // GPU buffer: draft indices converted from context target IDs
    T* repack_buffer;                  // for mode 2 weight repacking from lm_head

    // TopK functions
    functions::TopK<T>* topk_func;   // topk over draft_vocab_size (or context_length for modes 1/2)
    functions::TopK<T>* topk_func_2; // topk over total_tried (for tree selection)

    // Capture buffers for base model intermediate hidden states
    // Indices: {2, num_base_layers/2, num_base_layers-3}
    T* captured_hidden[3];
    int capture_indices[3];

    // Working buffers
    T* concat3_buf;    // [chunk_length, hidden_size*3] for fc input
    T* concat2_buf;    // [chunk_length, hidden_size*2] for QKV input
    T* working_buf;    // [chunk_length, hidden_size] main working buffer (decoder output)
    T* prev_buf;       // [chunk_length, hidden_size] previous iteration's selected output
    T* prev_embed;     // [chunk_length, hidden_size] shifted embeddings
    int num_prev;
    int num_history_tokens;

    // Attention buffers
    T* attn_output;
    float *softmax_lse, *softmax_lse_accum, *oaccum;

    // Draft state
    int32_t *eagle_position_ids, *eagle_cache_length;
    int *eagle_original_length, eagle_padded_length;
    uint64_t *eagle_mask_2d, *tmp_mask_2d;
    T* eagle_logits;
    T* tired_history_val; int32_t* tired_history_pos;
    int32_t* tired_history_parent;
    bool is_first_draft;

    int32_t *h_best, *d_best;
    T* tmp_kvcache;

    Eagle3Impl(
        ModelImpl<T, has_attention_bias>* model,
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
        int max_context_tokens
    ) {
        this->model = model;
        this->num_layers = num_layers;
        this->num_iter = num_iter;
        this->topk_per_iter = topk_per_iter;
        this->tree_size = tree_size;
        this->total_tried = topk_per_iter * topk_per_iter * (num_iter - 1) + topk_per_iter;
        this->draft_vocab_size = draft_vocab_size;
        this->num_attention_heads = num_attention_heads;
        this->num_key_value_heads = num_key_value_heads;
        this->head_dim = head_dim;
        this->max_context_tokens = max_context_tokens;

        // Capture layer indices: {2, mid, last-3}
        int n = model->num_hidden_layers;
        capture_indices[0] = 2;
        capture_indices[1] = n / 2;
        capture_indices[2] = n - 3;
        printf("Eagle3: capture layers: %d, %d, %d\n", capture_indices[0], capture_indices[1], capture_indices[2]);

        // fc: hidden_size*3 → hidden_size
        fc = new Linear<T>(model->hidden_size * 3, model->hidden_size);

        // Decoder layer components
        hidden_norm = new RMSNorm<T>(model->hidden_size, rms_norm_eps);
        input_layernorm = new RMSNorm<T>(model->hidden_size, rms_norm_eps);

        // QKV with input dim = hidden_size*2
        int qkv_out_dim = (num_attention_heads + 2 * num_key_value_heads) * head_dim;
        qkv_proj = new Linear<T, true>(model->hidden_size * 2, qkv_out_dim);
        q_proj = new Linear<T, true>(model->hidden_size * 2, num_attention_heads * head_dim);
        k_proj = new Linear<T, true>(model->hidden_size * 2, num_key_value_heads * head_dim);
        v_proj = new Linear<T, true>(model->hidden_size * 2, num_key_value_heads * head_dim);
        o_proj = new Linear<T>(num_attention_heads * head_dim, model->hidden_size);

        eagle3_ffn = new GatedFFN<T>(model->hidden_size, intermediate_size, rms_norm_eps);
        final_norm = new RMSNorm<T>(model->hidden_size, rms_norm_eps);
        lm_head = new Linear<T>(model->hidden_size, draft_vocab_size);

        kv_caches = new KVCacheManager<T>(num_layers, num_key_value_heads, head_dim);

        printf("Eagle3: draft_vocab_size=%d, max_context_tokens=%d\n",
            draft_vocab_size, max_context_tokens);

        topk_func = new functions::TopK<T>(draft_vocab_size, topk_per_iter);
        topk_func_2 = new functions::TopK<T>(total_tried, this->tree_size - 1);
    }

    void init_weight_ptr(Memory* memory) {
        fc->init_weight_ptr(memory);
        hidden_norm->init_weight_ptr(memory);
        input_layernorm->init_weight_ptr(memory);
        qkv_proj->init_weight_ptr(memory);
        // Alias q/k/v into qkv weight
        q_proj->weight = qkv_proj->weight;
        k_proj->weight = q_proj->weight + model->hidden_size * 2 * num_attention_heads * head_dim;
        v_proj->weight = k_proj->weight + model->hidden_size * 2 * num_key_value_heads * head_dim;
        o_proj->init_weight_ptr(memory);
        eagle3_ffn->init_weight_ptr(memory);
        final_norm->init_weight_ptr(memory);
        lm_head->init_weight_ptr(memory);

        kv_caches->rotary_embedding = this->model->kv_caches->rotary_embedding;
        token_id_remap = (int32_t*)memory->allocate_for_model(draft_vocab_size * sizeof(int32_t));
        t2d_index = (int32_t*)memory->allocate_for_model(model->vocab_size * sizeof(int32_t));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        // Capture buffers (3 x [num_tokens, hidden_size])
        for (int i = 0; i < 3; i++) {
            offset = memory->allocate((void**)&captured_hidden[i], offset, num_tokens * model->hidden_size * sizeof(T));
        }

        // Concat buffers
        offset = memory->allocate((void**)&concat3_buf, offset, num_tokens * model->hidden_size * 3 * sizeof(T));
        offset = memory->allocate((void**)&concat2_buf, offset, num_tokens * model->hidden_size * 2 * sizeof(T));

        // fc output
        offset = fc->init_output_ptr(memory, num_tokens, offset);

        // Norms (must be at different offsets since both are needed simultaneously)
        offset = hidden_norm->init_output_ptr(memory, num_tokens, offset);
        offset = input_layernorm->init_output_ptr(memory, num_tokens, offset);

        // QKV
        int64_t qkv_end = qkv_proj->init_output_ptr(memory, num_tokens, offset);
        q_proj->output = qkv_proj->output;
        k_proj->output = q_proj->output + num_tokens * num_attention_heads * head_dim;
        v_proj->output = k_proj->output + num_tokens * num_key_value_heads * head_dim;

        // Attention output buffer
        memory->allocate((void**)&attn_output, offset);
        int64_t lse_end = memory->allocate((void**)&softmax_lse, qkv_end, num_tokens * num_attention_heads * sizeof(float));
        int64_t lse_accum_end = memory->allocate((void**)&softmax_lse_accum, lse_end, num_tokens * num_attention_heads * sizeof(float));
        int64_t oaccum_end = memory->allocate((void**)&oaccum, lse_accum_end, num_tokens * num_attention_heads * head_dim * sizeof(float));

        // O projection
        int64_t o_end = o_proj->init_output_ptr(memory, num_tokens, qkv_end);
        offset = std::max(oaccum_end, o_end);

        // FFN
        offset = eagle3_ffn->init_output_ptr(memory, num_tokens, offset);

        // Final norm + lm_head
        int64_t fn_end = final_norm->init_output_ptr(memory, num_tokens, offset);
        offset = fn_end;
        offset = lm_head->init_output_ptr(memory, 64, offset);

        // Working buffers
        offset = memory->allocate((void**)&working_buf, offset, num_tokens * model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&prev_buf, offset, num_tokens * model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&prev_embed, offset, num_tokens * model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&eagle_cache_length, offset, sizeof(int32_t));

        // Draft tree buffers
        offset = memory->allocate((void**)&eagle_logits, offset, 2 * topk_per_iter * draft_vocab_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_mask_2d, offset, topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tmp_mask_2d, offset, topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tired_history_val, offset, total_tried * sizeof(T));
        offset = memory->allocate((void**)&tired_history_pos, offset, total_tried * sizeof(int32_t));
        offset = memory->allocate((void**)&tired_history_parent, offset, topk_per_iter * (num_iter - 1) * sizeof(int32_t));
        cudaMallocHost(&eagle_original_length, sizeof(int32_t));

        offset = topk_func->init_output_ptr(memory, topk_per_iter, offset);
        offset = topk_func_2->init_output_ptr(memory, 1, offset);

        offset = memory->allocate((void**)&d_best, offset, 2 * sizeof(int32_t));
        cudaMallocHost(&h_best, 2 * sizeof(int32_t));
        offset = memory->allocate((void**)&tmp_kvcache, offset, 64 * model->kv_caches->num_hidden_layers * 2 * model->kv_caches->dim * sizeof(T));

        // Mode 1/2 buffers: context_draft_ids and repack_buffer
        if (max_context_tokens > 0) {
            offset = memory->allocate((void**)&context_draft_ids, offset, sizeof(int32_t) * max_context_tokens);
            offset = memory->allocate((void**)&repack_buffer, offset, sizeof(T) * max_context_tokens * model->hidden_size);
        }

        return offset;
    }

    int init_storage() {
        this->model->init_weight_ptr(this->model->memory);
        this->init_weight_ptr(this->model->memory);
        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = this->init_output_ptr(this->model->memory, this->model->chunk_length, offset);
        float ratio = float(this->model->num_hidden_layers) / (this->model->num_hidden_layers + this->num_layers);
        kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        kv_caches->init_output_ptr(this->model->memory, kv_cache_offset);
        return min(kv_caches->budget + 1, this->model->kv_caches->budget);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 6) == "eagle3") {
            if (name.substr(0, 9) == "eagle3.fc") {
                fc->load_to_storage(name, ptr);
            } else if (name.find("eagle3.hidden_norm") != std::string::npos) {
                hidden_norm->load_to_storage(name, ptr);
            } else if (name.find("eagle3.input_layernorm") != std::string::npos) {
                input_layernorm->load_to_storage(name, ptr);
            } else if (name.find("eagle3.final_norm") != std::string::npos) {
                final_norm->load_to_storage(name, ptr);
            } else if (name.find("eagle3.lm_head") != std::string::npos) {
                lm_head->load_to_storage(name, ptr);
            } else if (name.find("eagle3.token_id_remap") != std::string::npos) {
                // Direct token_id_remap loading (freq-ranked indices for FR-Spec)
                cudaMemcpy((void*)token_id_remap, ptr, draft_vocab_size * sizeof(int32_t), cudaMemcpyHostToDevice);
                printf("Eagle3: loaded token_id_remap (V=%d)\n", draft_vocab_size);
            } else if (name.find("eagle3.d2t") != std::string::npos) {
                // d2t is stored as offsets; convert to direct remap on load
                int32_t* tmp_d2t;
                cudaMalloc(&tmp_d2t, draft_vocab_size * sizeof(int32_t));
                cudaMemcpy(tmp_d2t, ptr, draft_vocab_size * sizeof(int32_t), cudaMemcpyHostToDevice);
                d2t_to_remap(calc_stream, draft_vocab_size, tmp_d2t, token_id_remap);
                // Build inverse mapping: t2d_index[target_id] = draft_idx
                build_t2d_index(calc_stream, model->vocab_size, draft_vocab_size, token_id_remap, t2d_index);
                cudaStreamSynchronize(calc_stream.stream);
                cudaFree(tmp_d2t);
                printf("Eagle3: loaded d2t, built token_id_remap and t2d_index\n");
            } else {
                // Layer weights: eagle3.layers.0.attn.*, eagle3.layers.0.mlp.*
                std::regex layer_regex("eagle3\\.layers\\.(\\d+)\\.(.*)");
                std::smatch matches;
                if (std::regex_search(name, matches, layer_regex)) {
                    std::string sub_name = matches[2];
                    if (sub_name.find("attn.q_proj") != std::string::npos) {
                        q_proj->load_to_storage(sub_name, ptr);
                    } else if (sub_name.find("attn.k_proj") != std::string::npos) {
                        k_proj->load_to_storage(sub_name, ptr);
                    } else if (sub_name.find("attn.v_proj") != std::string::npos) {
                        v_proj->load_to_storage(sub_name, ptr);
                    } else if (sub_name.find("attn.o_proj") != std::string::npos) {
                        o_proj->load_to_storage(sub_name, ptr);
                    } else if (sub_name.find("mlp") != std::string::npos || sub_name.find("post_attention_layernorm") != std::string::npos) {
                        eagle3_ffn->load_to_storage(sub_name, ptr);
                    } else {
                        throw std::invalid_argument("Eagle3: unsupported layer weight: " + name);
                    }
                } else {
                    throw std::invalid_argument("Eagle3: unsupported weight: " + name);
                }
            }
        } else {
            this->model->load_to_storage(name, ptr);
        }
    }

    // ============================================================
    // Base model forward with intermediate hidden state capture
    // ============================================================

    void prefill_base_with_capture(int32_t num_tokens, int32_t num_history_tokens, T* embed, int32_t* position_ids, void* output) {
        T* layer_output = nullptr;
        for (int i = 0; i < model->num_hidden_layers; i++) {
            // Capture hidden state BEFORE this layer modifies embed
            // At this point, if layer_output != nullptr, the hidden state entering layer i
            // is (embed + layer_output), but embed hasn't been modified yet by this layer's norm.
            for (int c = 0; c < 3; c++) {
                if (i == capture_indices[c]) {
                    if (layer_output != nullptr) {
                        elementwise_add(calc_stream, num_tokens, model->hidden_size, embed, layer_output, captured_hidden[c]);
                    } else {
                        cudaMemcpyAsync(captured_hidden[c], embed, num_tokens * model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice, calc_stream.stream);
                    }
                }
            }
            model->layers[i]->prefill(num_tokens, num_history_tokens, embed, layer_output, position_ids, model->kv_caches->caches[i]);
            layer_output = model->layers[i]->output;
        }
        model->norm->prefill(calc_stream, num_tokens, embed, layer_output);
        model->lm_head->prefill(calc_stream, num_tokens, model->norm->output, (T*)output);
    }

    void decode_base_with_capture(int32_t num_tokens, int32_t padded_length, T* embed, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        Mask mask(mask_2d, num_tokens, num_tokens);
        T* layer_output = nullptr;
        for (int i = 0; i < model->num_hidden_layers; i++) {
            for (int c = 0; c < 3; c++) {
                if (i == capture_indices[c]) {
                    if (layer_output != nullptr) {
                        elementwise_add(calc_stream, num_tokens, model->hidden_size, embed, layer_output, captured_hidden[c]);
                    } else {
                        cudaMemcpyAsync(captured_hidden[c], embed, num_tokens * model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice, calc_stream.stream);
                    }
                }
            }
            model->layers[i]->decode(num_tokens, padded_length, embed, layer_output, position_ids, cache_length, mask, model->kv_caches->caches[i]);
            layer_output = model->layers[i]->output;
        }
        model->norm->prefill(calc_stream, num_tokens, embed, layer_output);
        model->lm_head->prefill(calc_stream, num_tokens, model->norm->output, (T*)output);
    }

    // ============================================================
    // EAGLE-3 draft model forward (decoder layer)
    // ============================================================

    // Run the draft decoder layer on num_tok tokens.
    // hidden_in: [num_tok, hidden_size] — the hidden state input (fc output or prev iter output)
    // embed_in: [num_tok, hidden_size] — the token embeddings
    // output goes to working_buf
    // hidden_in is modified IN-PLACE (residual add in FFN)
    void eagle3_layer_prefill(int num_tok, T* hidden_in, T* embed_in, int32_t* pos_ids, int num_history) {
        // 1. hidden_norm(hidden_in) → hidden_norm->output
        hidden_norm->prefill(calc_stream, num_tok, hidden_in, nullptr);
        // 2. input_layernorm(embed_in) → input_layernorm->output
        input_layernorm->prefill(calc_stream, num_tok, embed_in, nullptr);

        // 3. concat [normed_embed, normed_hidden] → concat2_buf [hidden_size*2]
        concat_2(calc_stream, num_tok, model->hidden_size, input_layernorm->output, hidden_norm->output, concat2_buf);

        // 4. QKV projection
        T* k_cache = kv_caches->caches[0]->offset_k(num_history);
        T* v_cache = kv_caches->caches[0]->offset_v(num_history);
        q_proj->prefill(calc_stream, num_tok, concat2_buf);
        k_proj->prefill(calc_stream, num_tok, concat2_buf, k_cache);
        v_proj->prefill(calc_stream, num_tok, concat2_buf, v_cache);

        // 5. RoPE
        kv_caches->rotary_embedding->prefill(calc_stream, num_tok, num_attention_heads, num_key_value_heads, q_proj->output, k_cache, pos_ids);

        // 6. Flash attention (prefill mode)
        mha_fwd_kvcache(
            TypeTraits<T>::type_code() == 1,
            1, num_tok, num_history + num_tok, num_tok,
            num_attention_heads, num_key_value_heads, head_dim,
            q_proj->output, kv_caches->caches[0]->k_cache, kv_caches->caches[0]->v_cache,
            nullptr, Mask(nullptr),
            attn_output, softmax_lse, softmax_lse_accum, oaccum,
            rsqrtf(float(head_dim)), true, -1, -1, 0, calc_stream.stream
        );

        // 7. O projection
        o_proj->prefill(calc_stream, num_tok, attn_output);

        // 8. Residual + FFN: hidden_in += o_proj->output, then post_attn_norm + MLP
        eagle3_ffn->prefill(calc_stream, num_tok, hidden_in, o_proj->output);
        // hidden_in now = hidden_in + attn_output (in-place)
        // eagle3_ffn->output = mlp_output

        // 9. Final residual: working_buf = hidden_in + mlp_output
        elementwise_add(calc_stream, num_tok, model->hidden_size, hidden_in, eagle3_ffn->output, working_buf);
    }

    void eagle3_layer_decode(int num_tok, T* hidden_in, T* embed_in, int32_t* pos_ids, int32_t* cache_length, const Mask& mask) {
        // Same structure as prefill but using decode-mode attention (with KV cache)
        hidden_norm->prefill(calc_stream, num_tok, hidden_in, nullptr);
        input_layernorm->prefill(calc_stream, num_tok, embed_in, nullptr);
        concat_2(calc_stream, num_tok, model->hidden_size, input_layernorm->output, hidden_norm->output, concat2_buf);

        // QKV
        if (num_tok > 1) {
            qkv_proj->prefill(calc_stream, num_tok, concat2_buf, v_proj->output);
            permute(calc_stream, num_tok, num_attention_heads * head_dim, num_key_value_heads * head_dim, v_proj->output, qkv_proj->output);
        } else {
            qkv_proj->prefill(calc_stream, num_tok, concat2_buf);
        }
        T* q = qkv_proj->output;
        T* k = q + num_tok * num_attention_heads * head_dim;
        T* v = k + num_tok * num_key_value_heads * head_dim;
        kv_caches->rotary_embedding->prefill(calc_stream, num_tok, num_attention_heads, num_key_value_heads, q, k, pos_ids);
        copy_to_kvcache(calc_stream, num_tok, k, v, kv_caches->caches[0], cache_length);

        mha_fwd_kvcache(
            TypeTraits<T>::type_code() == 1,
            1, num_tok, eagle_padded_length, num_tok,
            num_attention_heads, num_key_value_heads, head_dim,
            q, kv_caches->caches[0]->k_cache, kv_caches->caches[0]->v_cache,
            cache_length, mask,
            attn_output, softmax_lse, softmax_lse_accum, oaccum,
            rsqrtf(float(head_dim)), true, -1, -1, 0, calc_stream.stream
        );

        o_proj->prefill(calc_stream, num_tok, attn_output);
        eagle3_ffn->prefill(calc_stream, num_tok, hidden_in, o_proj->output);
        elementwise_add(calc_stream, num_tok, model->hidden_size, hidden_in, eagle3_ffn->output, working_buf);
    }

    // Run eagle3 draft prefill: fc + decoder layer on all previous tokens
    void eagle3_draft_prefill(int num_history_tokens) {
        // prev_embed already has shifted embeddings (set up in prefill())
        // Add last token's embedding
        cudaMemcpy(prev_embed + (num_prev - 1) * model->hidden_size,
                    model->embedding->output, model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        // Concatenate 3 captured hidden states → [num_prev, hidden_size*3]
        concat_3(calc_stream, num_prev, model->hidden_size,
                 captured_hidden[0], captured_hidden[1], captured_hidden[2], concat3_buf);

        // fc: hidden_size*3 → hidden_size
        fc->prefill(calc_stream, num_prev, concat3_buf);
        // fc->output: [num_prev, hidden_size]

        // Run decoder layer (prefill mode)
        eagle3_layer_prefill(num_prev, fc->output, prev_embed, eagle_position_ids, num_history_tokens);
        // working_buf now has the decoder output [num_prev, hidden_size]
    }

    // Run eagle3 draft decode for subsequent iterations
    void eagle3_draft_decode(int32_t* cache_length) {
        // prev_buf has the selected hidden states from previous iteration
        // model->embedding->output has embeddings for current draft tokens
        eagle3_layer_decode(num_prev, prev_buf, model->embedding->output, eagle_position_ids, cache_length,
                           Mask(eagle_mask_2d, topk_per_iter, topk_per_iter * 0));  // mask will be set per iteration
    }

    // ============================================================
    // Main interface methods
    // ============================================================

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        // 1. Embed input tokens
        model->embedding->prefill(calc_stream, num_tokens, input);

        // 2. If there were previous draft tokens, run eagle3_draft_prefill on them
        if (num_history_tokens > 0) {
            eagle3_draft_prefill(this->num_history_tokens);
        }

        // 3. Store shifted embeddings: prev_embed[0..n-2] = embed[1..n-1]
        cudaMemcpy(prev_embed, model->embedding->output + model->hidden_size,
                   (num_tokens - 1) * model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        // 4. Run base model with capture
        prefill_base_with_capture(num_tokens, num_history_tokens, model->embedding->output, position_ids, output);

        // 5. Store position IDs for eagle3 draft
        cudaMemcpy(eagle_position_ids, position_ids, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->num_prev = num_tokens;
        this->num_history_tokens = num_history_tokens;
        this->is_first_draft = true;
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        // Embed and run base model decode with capture
        model->embedding->prefill(calc_stream, num_tokens, input);
        decode_base_with_capture(num_tokens, padded_length, model->embedding->output, position_ids, cache_length, mask_2d, output);
    }

    void draft(int32_t* tree_draft_ids, int32_t* tree_position_ids, int32_t* cache_length, uint64_t* tree_attn_mask, int32_t* tree_parent, int32_t* context_tokens, int32_t context_length, int32_t mode) {
        if (mode == 0) {
            // EAGLE-3 d2t mode: full draft_vocab_size lm_head (baseline = FR-Spec)
            draft_eagle3(tree_draft_ids, tree_position_ids, cache_length, tree_attn_mask, tree_parent);
        } else if (mode == 1) {
            // Dynamic vocab pruning: indexed GEMM on EAGLE-3's lm_head with context tokens
            draft_eagle3_indexed_gemm(tree_draft_ids, tree_position_ids, cache_length, tree_attn_mask, tree_parent, context_tokens, context_length);
        } else if (mode == 2) {
            // Dynamic vocab pruning: async prefetch on EAGLE-3's lm_head with context tokens
            draft_eagle3_prefetch(tree_draft_ids, tree_position_ids, cache_length, tree_attn_mask, tree_parent, context_tokens, context_length);
        }
    }

    void draft_eagle3(int32_t* tree_draft_ids, int32_t* tree_position_ids, int32_t* cache_length, uint64_t* tree_attn_mask, int32_t* tree_parent) {
        cudaMemcpy(eagle_original_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
        eagle_padded_length = (eagle_original_length[0] + 256 - 1) / 128 * 128;

        if (is_first_draft) {
            model->embedding->prefill(calc_stream, 1, tree_draft_ids);
            eagle3_draft_prefill(this->num_history_tokens);
        } else {
            // Non-first draft: process num_prev accepted tokens through draft model (decode mode)
            // 1. Embed root token and add to prev_embed
            model->embedding->prefill(calc_stream, 1, tree_draft_ids);
            cudaMemcpy(prev_embed + (num_prev - 1) * model->hidden_size,
                       model->embedding->output, model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

            // 2. Concatenate captured hidden states and apply fc
            concat_3(calc_stream, num_prev, model->hidden_size,
                     captured_hidden[0], captured_hidden[1], captured_hidden[2], concat3_buf);
            fc->prefill(calc_stream, num_prev, concat3_buf);

            // 3. Run decoder layer in decode mode (appending to eagle KV cache)
            eagle3_layer_decode(num_prev, fc->output, prev_embed, eagle_position_ids, cache_length,
                               Mask(nullptr));
        }
        cudaMemcpy(eagle_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(eagle_position_ids, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        repeat(calc_stream, topk_per_iter, 1, 0, eagle_position_ids);

        { // d = 0: topk on the last position
            final_norm->prefill(calc_stream, 1, working_buf + (num_prev - 1) * model->hidden_size, nullptr);
            lm_head->prefill(calc_stream, 1, final_norm->output, eagle_logits);
            log_softmax(calc_stream, 1, draft_vocab_size, eagle_logits);

            topk_func->prefill(calc_stream, 1, eagle_logits);
            cudaMemcpy(tired_history_val, topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(tired_history_pos, topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);

            // Map draft indices to target vocab: remap[draft_idx] = draft_idx + d2t[draft_idx]
            remap(calc_stream, topk_per_iter, topk_func->topk_pos, topk_func_2->topk_pos, token_id_remap);

            cudaMemcpy(topk_func_2->topk_val, topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            // Repeat last position's working_buf for topk_per_iter draft branches
            repeat(calc_stream, topk_per_iter, model->hidden_size, num_prev - 1, working_buf, prev_buf);
            init_tree(calc_stream, topk_per_iter, eagle_mask_2d);
        }

        for (int d = 1; d < num_iter; ++d) {
            add(calc_stream, 1, eagle_cache_length, topk_per_iter);

            // Embed the draft tokens (target vocab IDs)
            model->embedding->prefill(calc_stream, topk_per_iter, topk_func_2->topk_pos);

            // Run decoder layer (decode mode with tree mask)
            eagle3_layer_decode(topk_per_iter, prev_buf, model->embedding->output,
                               eagle_position_ids, eagle_cache_length,
                               Mask(eagle_mask_2d, topk_per_iter, topk_per_iter * d));

            add(calc_stream, topk_per_iter, eagle_position_ids, 1);

            // lm_head on working_buf
            final_norm->prefill(calc_stream, topk_per_iter, working_buf, nullptr);
            lm_head->prefill(calc_stream, topk_per_iter, final_norm->output, eagle_logits);
            log_softmax(calc_stream, topk_per_iter, draft_vocab_size, eagle_logits);

            topk_func->prefill(calc_stream, topk_per_iter, eagle_logits);
            cumsum(calc_stream, topk_per_iter, topk_per_iter, topk_func->topk_val, topk_func_2->topk_val);
            cudaMemcpy(tired_history_val + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, topk_func->topk_val, topk_per_iter * topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(tired_history_pos + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, topk_func->topk_pos, topk_per_iter * topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            topk_func_2->prefill(calc_stream, 1, topk_func->topk_val, topk_per_iter * topk_per_iter, topk_per_iter);

            cudaMemcpy(tmp_mask_2d, eagle_mask_2d, topk_per_iter * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            set_parent(calc_stream, topk_per_iter, tired_history_parent + (d - 1) * topk_per_iter, topk_func_2->topk_pos, topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter);
            update_tree(calc_stream, topk_per_iter, topk_per_iter * d, eagle_mask_2d, tmp_mask_2d, topk_func_2->topk_pos);

            // Select hidden states for next iteration
            remap_hidden(calc_stream, topk_per_iter, model->hidden_size, topk_func_2->topk_pos, working_buf, prev_buf, topk_per_iter);
            // Map draft indices → target vocab IDs
            remap_id(calc_stream, topk_per_iter, topk_func_2->topk_pos, topk_func->topk_pos, token_id_remap);
        }

        // Final tree selection
        topk_func_2->prefill(calc_stream, 1, tired_history_val);

        // Build tree
        build_dynamic_tree(calc_stream, tree_size, eagle_original_length[0], topk_per_iter, tired_history_parent, topk_func_2->topk_pos, tree_position_ids, tree_attn_mask, tree_parent);
        remap_id(calc_stream, tree_size - 1, topk_func_2->topk_pos, tired_history_pos, token_id_remap, tree_draft_ids + 1);

        is_first_draft = false;
    }

    // Helper: run eagle3 draft hidden computation (shared by all FR-Spec modes)
    // After this, working_buf has the decoder output, eagle_cache_length and eagle_position_ids are set
    void prepare_eagle3_draft_hidden(int32_t* tree_draft_ids, int32_t* cache_length) {
        cudaMemcpy(eagle_original_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
        eagle_padded_length = (eagle_original_length[0] + 256 - 1) / 128 * 128;

        if (is_first_draft) {
            model->embedding->prefill(calc_stream, 1, tree_draft_ids);
            eagle3_draft_prefill(this->num_history_tokens);
        } else {
            model->embedding->prefill(calc_stream, 1, tree_draft_ids);
            cudaMemcpy(prev_embed + (num_prev - 1) * model->hidden_size,
                       model->embedding->output, model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
            concat_3(calc_stream, num_prev, model->hidden_size,
                     captured_hidden[0], captured_hidden[1], captured_hidden[2], concat3_buf);
            fc->prefill(calc_stream, num_prev, concat3_buf);
            eagle3_layer_decode(num_prev, fc->output, prev_embed, eagle_position_ids, cache_length, Mask(nullptr));
        }
        cudaMemcpy(eagle_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(eagle_position_ids, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        repeat(calc_stream, topk_per_iter, 1, 0, eagle_position_ids);
    }

    // Helper: run eagle3 draft layer for depth iteration d (shared by all modes)
    void eagle3_draft_depth_iter(int d) {
        add(calc_stream, 1, eagle_cache_length, topk_per_iter);
        model->embedding->prefill(calc_stream, topk_per_iter, topk_func_2->topk_pos);
        eagle3_layer_decode(topk_per_iter, prev_buf, model->embedding->output,
                           eagle_position_ids, eagle_cache_length,
                           Mask(eagle_mask_2d, topk_per_iter, topk_per_iter * d));
        add(calc_stream, topk_per_iter, eagle_position_ids, 1);
    }

    // ---- Mode 1: indexed GEMM on EAGLE-3's own lm_head with context tokens ----
    // context_tokens: target vocab IDs (filtered in Python to only draft-vocab-valid tokens)
    // Internally converts to draft indices via t2d_index for lm_head indexing
    void draft_eagle3_indexed_gemm(int32_t* tree_draft_ids, int32_t* tree_position_ids, int32_t* cache_length, uint64_t* tree_attn_mask, int32_t* tree_parent, int32_t* context_tokens, int32_t context_length) {
        prepare_eagle3_draft_hidden(tree_draft_ids, cache_length);

        // Convert target IDs → draft indices for lm_head indexing
        target_to_draft(calc_stream, context_length, context_tokens, context_draft_ids, t2d_index);

        { // d = 0
            final_norm->prefill(calc_stream, 1, working_buf + (num_prev - 1) * model->hidden_size, nullptr);
            lm_head->prefill_gathered(calc_stream, final_norm->output, context_draft_ids, context_length, eagle_logits);
            log_softmax(calc_stream, 1, context_length, eagle_logits);

            topk_func->prefill(calc_stream, 1, eagle_logits, context_length);
            cudaMemcpy(tired_history_val, topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(tired_history_pos, topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);

            // Remap: index-in-context → target vocab ID (for embedding and tree)
            remap(calc_stream, topk_per_iter, topk_func->topk_pos, topk_func_2->topk_pos, context_tokens);
            cudaMemcpy(topk_func_2->topk_val, topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            repeat(calc_stream, topk_per_iter, model->hidden_size, num_prev - 1, working_buf, prev_buf);
            init_tree(calc_stream, topk_per_iter, eagle_mask_2d);
        }

        for (int d = 1; d < num_iter; ++d) {
            eagle3_draft_depth_iter(d);

            final_norm->prefill(calc_stream, topk_per_iter, working_buf, nullptr);
            for (int i = 0; i < topk_per_iter; i++) {
                lm_head->prefill_gathered(calc_stream,
                    final_norm->output + i * model->hidden_size,
                    context_draft_ids, context_length,
                    eagle_logits + i * context_length);
            }
            log_softmax(calc_stream, topk_per_iter, context_length, eagle_logits);

            topk_func->prefill(calc_stream, topk_per_iter, eagle_logits, context_length);
            cumsum(calc_stream, topk_per_iter, topk_per_iter, topk_func->topk_val, topk_func_2->topk_val);
            cudaMemcpy(tired_history_val + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, topk_func->topk_val, topk_per_iter * topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(tired_history_pos + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, topk_func->topk_pos, topk_per_iter * topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            topk_func_2->prefill(calc_stream, 1, topk_func->topk_val, topk_per_iter * topk_per_iter, topk_per_iter);

            cudaMemcpy(tmp_mask_2d, eagle_mask_2d, topk_per_iter * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            set_parent(calc_stream, topk_per_iter, tired_history_parent + (d - 1) * topk_per_iter, topk_func_2->topk_pos, topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter);
            update_tree(calc_stream, topk_per_iter, topk_per_iter * d, eagle_mask_2d, tmp_mask_2d, topk_func_2->topk_pos);
            remap_hidden(calc_stream, topk_per_iter, model->hidden_size, topk_func_2->topk_pos, working_buf, prev_buf, topk_per_iter);
            remap_id(calc_stream, topk_per_iter, topk_func_2->topk_pos, topk_func->topk_pos, context_tokens);
        }

        topk_func_2->prefill(calc_stream, 1, tired_history_val);
        build_dynamic_tree(calc_stream, tree_size, eagle_original_length[0], topk_per_iter, tired_history_parent, topk_func_2->topk_pos, tree_position_ids, tree_attn_mask, tree_parent);
        remap_id(calc_stream, tree_size - 1, topk_func_2->topk_pos, tired_history_pos, context_tokens, tree_draft_ids + 1);
        is_first_draft = false;
    }

    // ---- Mode 2: async prefetch on EAGLE-3's own lm_head with context tokens ----
    // Repack buffer filled by async_prefetch; context_tokens are target IDs for remap
    void draft_eagle3_prefetch(int32_t* tree_draft_ids, int32_t* tree_position_ids, int32_t* cache_length, uint64_t* tree_attn_mask, int32_t* tree_parent, int32_t* context_tokens, int32_t context_length) {
        prepare_eagle3_draft_hidden(tree_draft_ids, cache_length);

        // Wait for async repack to complete
        cudaStreamWaitEvent(calc_stream.stream, copy_event, 0);

        { // d = 0
            final_norm->prefill(calc_stream, 1, working_buf + (num_prev - 1) * model->hidden_size, nullptr);
            lm_head->prefill_repack_sync(calc_stream, 1, context_length, final_norm->output, repack_buffer, eagle_logits);
            log_softmax(calc_stream, 1, context_length, eagle_logits);

            topk_func->prefill(calc_stream, 1, eagle_logits, context_length);
            cudaMemcpy(tired_history_val, topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(tired_history_pos, topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);

            // Remap: index-in-context → target vocab ID
            remap(calc_stream, topk_per_iter, topk_func->topk_pos, topk_func_2->topk_pos, context_tokens);
            cudaMemcpy(topk_func_2->topk_val, topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            repeat(calc_stream, topk_per_iter, model->hidden_size, num_prev - 1, working_buf, prev_buf);
            init_tree(calc_stream, topk_per_iter, eagle_mask_2d);
        }

        for (int d = 1; d < num_iter; ++d) {
            eagle3_draft_depth_iter(d);

            final_norm->prefill(calc_stream, topk_per_iter, working_buf, nullptr);
            lm_head->prefill_repack_sync(calc_stream, topk_per_iter, context_length, final_norm->output, repack_buffer, eagle_logits);
            log_softmax(calc_stream, topk_per_iter, context_length, eagle_logits);

            topk_func->prefill(calc_stream, topk_per_iter, eagle_logits, context_length);
            cumsum(calc_stream, topk_per_iter, topk_per_iter, topk_func->topk_val, topk_func_2->topk_val);
            cudaMemcpy(tired_history_val + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, topk_func->topk_val, topk_per_iter * topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(tired_history_pos + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, topk_func->topk_pos, topk_per_iter * topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            topk_func_2->prefill(calc_stream, 1, topk_func->topk_val, topk_per_iter * topk_per_iter, topk_per_iter);

            cudaMemcpy(tmp_mask_2d, eagle_mask_2d, topk_per_iter * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            set_parent(calc_stream, topk_per_iter, tired_history_parent + (d - 1) * topk_per_iter, topk_func_2->topk_pos, topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter);
            update_tree(calc_stream, topk_per_iter, topk_per_iter * d, eagle_mask_2d, tmp_mask_2d, topk_func_2->topk_pos);
            remap_hidden(calc_stream, topk_per_iter, model->hidden_size, topk_func_2->topk_pos, working_buf, prev_buf, topk_per_iter);
            remap_id(calc_stream, topk_per_iter, topk_func_2->topk_pos, topk_func->topk_pos, context_tokens);
        }

        topk_func_2->prefill(calc_stream, 1, tired_history_val);
        build_dynamic_tree(calc_stream, tree_size, eagle_original_length[0], topk_per_iter, tired_history_parent, topk_func_2->topk_pos, tree_position_ids, tree_attn_mask, tree_parent);
        remap_id(calc_stream, tree_size - 1, topk_func_2->topk_pos, tired_history_pos, context_tokens, tree_draft_ids + 1);
        is_first_draft = false;
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, int32_t* tree_parent) {
        verify_draft(calc_stream, num_tokens, pred, gt, position_ids, cache_length, mask_2d, tree_parent, d_best);
        cudaMemcpyAsync(h_best, d_best, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);

        this->num_prev = h_best[0];

        // Select accepted positions from captured_hidden using pred indices.
        // Must use temp buffer to avoid in-place aliasing (remap reads and writes same buffer).
        for (int c = 0; c < 3; c++) {
            remap_hidden(calc_stream, num_prev, model->hidden_size, pred, captured_hidden[c], working_buf);
            cudaMemcpyAsync(captured_hidden[c], working_buf, num_prev * model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice, calc_stream.stream);
        }

        // Fix base model KV cache
        fix_kv_cache(calc_stream, h_best[0], model->kv_caches->num_hidden_layers * 2, model->kv_caches->dim, pred, gt, cache_length, model->kv_caches->d_flat_caches, tmp_kvcache);

        // Re-embed accepted tokens for next draft cycle
        model->embedding->prefill(calc_stream, num_prev, pred);
        cudaMemcpy(prev_embed, model->embedding->output, num_prev * model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        make_arange(calc_stream, num_prev, cache_length, eagle_position_ids);

        return h_best[0];
    }

    void async_prefetch(int32_t* gpu_context_buffer, int32_t* cpu_new_tokens, int32_t num_new, int32_t offset) {
        // 1. Copy new target IDs to GPU context buffer
        cudaMemcpyAsync(
            gpu_context_buffer + offset, cpu_new_tokens,
            num_new * sizeof(int32_t), cudaMemcpyHostToDevice, copy_stream.stream
        );
        // 2. Convert target IDs → draft IDs on GPU
        target_to_draft_kernel<<<CEIL_DIV(num_new, 256), 256, 0, copy_stream.stream>>>(
            num_new, gpu_context_buffer + offset, context_draft_ids + offset, t2d_index
        );
        // 3. Repack weights from EAGLE-3's own lm_head using draft IDs
        launch_repack_weights_incremental(
            copy_stream, num_new, this->model->hidden_size,
            context_draft_ids + offset, this->lm_head->weight,
            this->repack_buffer + (int64_t)offset * this->model->hidden_size
        );
        cudaEventRecord(copy_event, copy_stream.stream);
    }
};
