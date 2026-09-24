# Copyright (c) ModelScope Contributors. All rights reserved.
"""FA3 attention forward on CUDA: leaf behind the 'sdpa' (and 'flash_attention_3')
registry keys.

Fast path delegates to transformers' own ``flash_attention_forward`` (which
dispatches ``flash_attention_3`` to ``flash_attn_interface`` on Hopper) -- it is
contract-compatible with the sdpa leaf: same (B, H, S, D) inputs (it transposes
internally), same dropout/scaling/is_causal kwargs.

The one contract difference is the mask: the sdpa mask factory may hand the leaf
a 4D additive float mask (padded batches), which flash kernels cannot consume.
Those calls fall back to transformers' stock sdpa forward, so anything the stock
'sdpa' path could run keeps running identically; the FA3 fast path applies
whenever the mask is None or a 2D padding mask.
"""
from __future__ import annotations


def cuda_fa3_attention_forward(module, query, key, value, attention_mask, *args, **kwargs):
    if attention_mask is not None and attention_mask.dim() == 4:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        return sdpa_attention_forward(module, query, key, value, attention_mask, *args, **kwargs)

    from transformers.integrations.flash_attention import flash_attention_forward

    return flash_attention_forward(module, query, key, value, attention_mask, *args, **kwargs)
