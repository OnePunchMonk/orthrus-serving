# SPDX-License-Identifier: Apache-2.0
"""Inference-only Orthrus model for SGLang (autoregressive path)."""

from collections.abc import Iterable

import torch

from sglang.srt.models.qwen3 import Qwen3ForCausalLM

from orthrus_sglang.weights import is_diffusion_weight


class OrthrusLM(Qwen3ForCausalLM):
    """Orthrus served autoregressively.

    Orthrus' AR view is Qwen3 unchanged (the released checkpoints even keep
    the base Qwen3 weights bit-for-bit), so this reuses SGLang's Qwen3 model
    and its kernels, CUDA graphs and radix cache as-is. The diffusion-view
    weights are dropped at load time rather than allocated, which saves
    ~350M params of GPU memory on the 1.7B checkpoint.
    """

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return super().load_weights(
            (name, tensor)
            for name, tensor in weights
            if not is_diffusion_weight(name)
        )


class OrthrusForCausalLM(OrthrusLM):
    """Alias: SGLang keys architectures by class ``__name__``."""


EntryClass = [OrthrusLM, OrthrusForCausalLM]
