"""
Profiling generate method for ablation study.
Wraps the existing tree_drafter generate loop, adding per-step logging of:
  - coverage: fraction of accepted tokens that were in the active vocabulary
  - vocab_size: number of tokens in the active vocabulary at each step
  - accept_length: per-step acceptance length

Ablation modes (passed as ablation_mode string):
  - "full"     : Ctx+Ext (= NanoSpec mode 2, both prefill init and decode update)
  - "ctx_only" : prefill init only, no decode-time vocab update
  - "ext_only" : no prefill init, only decode-time vocab update
"""

import time
import torch
import numpy as np
from llamacu import C


def generate_with_profiling(model, input_ids, generation_length=100, teminators=[],
                            tokenizer=None, is_warmup=False, ablation_mode="full"):
    """
    Returns:
        tokens, accept_lengths, model_step, draft_time_list, step_stats
    where step_stats is a list of dicts per decode step:
        {
            "vocab_size": int,
            "accept_length": int,
            "coverage": float,   # fraction of accepted tokens in active vocab
            "accepted_tokens": list[int],
        }
    """
    assert input_ids.dtype == torch.int32
    assert ablation_mode in ("full", "ctx_only", "ext_only")

    prefix_length = input_ids.numel()
    position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
    logits = model.prefill(input_ids, position_ids)

    MODE = 2  # always use mode 2 (prefetch) for the underlying C++ draft
    context_length = 0

    if is_warmup:
        MODE = 0

    # --- prefill-time vocab init (Ctx) ---
    if not is_warmup and ablation_mode in ("full", "ctx_only"):
        topk_indices = torch.topk(logits, k=3, dim=-1).indices
        combined_candidates_gpu = torch.cat([input_ids.view(-1), topk_indices.view(-1)])
        unique_candidates_gpu = torch.unique(combined_candidates_gpu)
        new_tokens_set = set(unique_candidates_gpu.tolist())

        new_tokens_tensor_cpu, new_context_length = model.update_context_new(new_tokens_set, MODE)
        if new_tokens_tensor_cpu is not None:
            C.trigger_async_prefetch(
                model.context_tokens_tensor.data_ptr(),
                new_tokens_tensor_cpu.data_ptr(),
                len(new_tokens_tensor_cpu),
                context_length,
            )
            context_length = new_context_length
    elif not is_warmup and ablation_mode == "ext_only":
        pass  # no prefill init

    model.tree_draft_ids[:1].copy_(logits[prefix_length - 1].argmax(dim=-1))

    tokens = torch.empty((generation_length), dtype=torch.int32, device="cuda")
    tokens[0].copy_(model.tree_draft_ids[0])
    accept_lengths = []
    i = 0
    model_step = 0
    terminal = False
    step_stats = []

    step_draft_times = []

    torch.cuda.synchronize()
    decoding_start_time = time.time()

    while i < generation_length - 1 and not terminal:
        model.cache_length[0] = prefix_length + i

        # When context is empty (e.g., ext_only first steps), fall back to mode 0
        effective_mode = MODE if context_length > 0 else 0
        C.draft(model.tree_draft_ids.data_ptr(), model.tree_position_ids.data_ptr(),
                model.cache_length.data_ptr(), model.tree_attn_mask.data_ptr(),
                model.tree_parent.data_ptr(),
                model.context_tokens_tensor.data_ptr(), context_length, effective_mode)

        logits = model.decode(model.tree_draft_ids, model.tree_position_ids,
                              model.cache_length, mask_2d=model.tree_attn_mask)
        model.tree_gt_ids.copy_(logits.argmax(dim=-1))

        accept_length = C.verify_and_fix(
            model.tree_draft_ids.numel(), model.tree_draft_ids.data_ptr(),
            model.tree_gt_ids.data_ptr(), model.tree_position_ids.data_ptr(),
            model.cache_length.data_ptr(), model.tree_attn_mask.data_ptr(),
            model.tree_parent.data_ptr()
        )

        model_step += 1
        accept_lengths.append(accept_length)

        for temin in teminators:
            if temin in model.tree_draft_ids[:accept_length]:
                terminal = True
        append_length = min(accept_length, generation_length - 1 - i)

        # --- per-step profiling ---
        accepted_token_ids = model.tree_draft_ids[:append_length].tolist()
        current_vocab = model.context_tokens_set
        current_vocab_size = len(current_vocab)

        if current_vocab_size > 0 and len(accepted_token_ids) > 0:
            hits = sum(1 for t in accepted_token_ids if t in current_vocab)
            coverage = hits / len(accepted_token_ids)
        else:
            coverage = 0.0

        step_stats.append({
            "vocab_size": current_vocab_size,
            "accept_length": accept_length,
            "coverage": coverage,
            "accepted_tokens": accepted_token_ids,
        })

        # --- decode-time vocab update (Ext) ---
        if not is_warmup and ablation_mode in ("full", "ext_only"):
            topk_indices = logits.topk(k=3, dim=-1).indices
            combined_ids_gpu = torch.cat([model.tree_draft_ids.view(-1), model.tree_gt_ids.view(-1), topk_indices.view(-1)])
            unique_ids_gpu = torch.unique(combined_ids_gpu)
            new_tokens_set = set(unique_ids_gpu.tolist())

            new_tokens_tensor_cpu, new_context_length = model.update_context_new(new_tokens_set, MODE)
            if new_tokens_tensor_cpu is not None:
                C.trigger_async_prefetch(
                    model.context_tokens_tensor.data_ptr(),
                    new_tokens_tensor_cpu.data_ptr(),
                    len(new_tokens_tensor_cpu),
                    context_length % model.max_context_tokens,
                )
                context_length = new_context_length
        elif not is_warmup and ablation_mode == "ctx_only":
            pass  # no decode update

        tokens[1 + i:1 + i + append_length].copy_(model.tree_draft_ids[:append_length])
        model.tree_draft_ids[0] = model.tree_draft_ids[accept_length - 1]
        i += accept_length

    torch.cuda.synchronize()
    decoding_total_time = time.time() - decoding_start_time
    step_draft_times.append(decoding_total_time)

    tokens = tokens[:1 + i].tolist()

    model.context_tokens_set.clear()
    model.context_tokens_tensor = torch.empty((model.max_context_tokens), dtype=torch.int32, device="cuda")

    return tokens, accept_lengths, model_step, step_draft_times, step_stats
