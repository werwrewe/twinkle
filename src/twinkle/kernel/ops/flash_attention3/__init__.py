# Copyright (c) ModelScope Contributors. All rights reserved.
"""flash_attention_3 op registration: installs the FA3 attention forward into the
transformers global attention registry (not a plain class/attr replacement,
hence the custom installer) -- exactly the same shape as sdpa_attention.

The default config references this op via the logical target 'flash_attention_3';
the logical target is only a mapping label passed to install_fa3 and is never
resolved by the generic replacer.

Pure leaf replacement, no selection change: on CUDA a default-built model has
``config._attn_implementation == 'sdpa'`` (PyTorch ``F.scaled_dot_product_attention``),
so the installer puts the FA3 leaf behind the **'sdpa'** registry key -- every
default model then runs FA3 with zero config flips and zero kwargs, mirroring how
the NPU sdpa op puts its fused leaf behind the same key. The 'flash_attention_3'
key is filled too, so an explicit ``attn_implementation='flash_attention_3'``
from_pretrained kwarg and the sequence-parallel FA wrapper land on the same leaf.

Compatibility with the sdpa entry and with explicit choices:
- ``backends=('cuda',)``: on NPU resolve_impl finds no backend, the installer is
  never called, and the 'sdpa' key keeps whatever the NPU sdpa op installed --
  exactly one attention entry is active per platform.
- an explicit config ('flash_attention_2', 'eager', ...) dispatches to a different
  registry key and is never touched.
"""
from __future__ import annotations

from twinkle import get_logger
from twinkle.utils.device_mesh import Platform
from ...registry import KernelImpl, lazy_import, register_op

logger = get_logger()


def _is_fa3_cuda_available() -> tuple[bool, str | None]:
    """CUDA platform + transformers' own FA3 availability check.

    The transformers check is distribution-level (``flash-attn-3`` must appear in
    ``packages_distributions()``), so a hand-copied build tree correctly reports
    unavailable here instead of silently substituting a Hub kernel.
    """
    if Platform.device_prefix() != 'cuda':
        return False, f"platform is '{Platform.device_prefix()}', not 'cuda'"
    # transformers' own check is distribution-level ('flash-attn-3' must appear in
    # packages_distributions()), so a hand-copied build tree correctly reports unavailable
    # here instead of silently substituting a Hub kernel later.
    from transformers.utils import import_utils
    if not import_utils.is_flash_attn_3_available():
        return False, 'flash-attn-3 not available per transformers (distribution check failed)'
    return True, None


def install_fa3(model, target, impl) -> None:
    """One-shot install of the FA3 attention forward (global modeling_utils dict).

    ``AttentionInterface._global_mapping`` is a private transformers attribute;
    guard against its removal so an upstream change can't take down the rest
    of kernelize.

    Each key is installed independently and skipped when it already holds a
    wrapper that is not the stock transformers function (e.g. the
    sequence-parallel ``local_flash_attn``/``local_sdpa_attn`` partials):
    replacing such a wrapper here would silently drop sequence parallelism.
    """
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
        from transformers.integrations.flash_attention import flash_attention_forward as stock_fa_forward
        from transformers.integrations.sdpa_attention import sdpa_attention_forward as stock_sdpa_forward
    except ImportError:
        return
    for key, stock in (('sdpa', stock_sdpa_forward), ('flash_attention_3', stock_fa_forward)):
        current = ALL_ATTENTION_FUNCTIONS.get(key)
        if current is not None and current is not impl and current is not stock:
            logger.info('[FA3] %s already holds %r; keeping the existing wrapper', key, current)
            continue
        try:
            AttentionInterface._global_mapping[key] = impl
        except AttributeError:
            logger.warning('[FA3] AttentionInterface._global_mapping unavailable; skipping')
        ALL_ATTENTION_FUNCTIONS[key] = impl


register_op(
    'flash_attention_3',
    implementations={
        'cuda':
        KernelImpl(
            load=lazy_import('twinkle.kernel.ops.flash_attention3.cuda:cuda_fa3_attention_forward'),
            available=_is_fa3_cuda_available,
        ),
    },
    installer=install_fa3,
)
