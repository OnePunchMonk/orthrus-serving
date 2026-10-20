# SPDX-License-Identifier: Apache-2.0
"""Checkpoint weight filtering for Orthrus' autoregressive path.

Kept free of SGLang imports so it can be unit-tested on CPU.
"""


def is_diffusion_weight(name: str) -> bool:
    """Whether ``name`` belongs to Orthrus' diffusion-only attention view.

    Orthrus adds ``{q,k,v,o}_proj_diff`` and ``{q,k}_norm_diff`` next to each
    layer's Qwen3 attention. The autoregressive forward never touches them
    (``is_diffusion_pass=False`` in the reference ``modeling_orthrus.py``),
    and every other tensor in the checkpoint is a plain Qwen3 weight.

    Matching is on whole module segments: Qwen3's own loader does substring
    matching, so ``q_proj_diff`` would otherwise be routed into ``qkv_proj``.
    """
    return any(part.endswith("_diff") for part in name.split("."))
