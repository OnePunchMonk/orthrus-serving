# SPDX-License-Identifier: Apache-2.0
"""CPU-only: no SGLang or GPU needed."""

from orthrus_sglang.weights import is_diffusion_weight

# Every per-layer tensor name in chiennv/Orthrus-Qwen3-1.7B's safetensors.
_ATTN = "model.layers.3.self_attn"
_DIFFUSION = [
    f"{_ATTN}.{m}.weight"
    for m in (
        "q_proj_diff",
        "k_proj_diff",
        "v_proj_diff",
        "o_proj_diff",
        "q_norm_diff",
        "k_norm_diff",
    )
]
_AR = [
    f"{_ATTN}.{m}.weight"
    for m in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
] + [
    "model.layers.3.mlp.gate_proj.weight",
    "model.layers.3.mlp.up_proj.weight",
    "model.layers.3.mlp.down_proj.weight",
    "model.layers.3.input_layernorm.weight",
    "model.layers.3.post_attention_layernorm.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
]


def test_diffusion_weights_are_dropped():
    assert all(is_diffusion_weight(n) for n in _DIFFUSION)


def test_ar_weights_are_kept():
    assert not any(is_diffusion_weight(n) for n in _AR)
