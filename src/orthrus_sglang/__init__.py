# SPDX-License-Identifier: Apache-2.0
"""SGLang external model package: native Orthrus model support.

Point SGLang at this package and ``chiennv/Orthrus-*`` checkpoints
(``architectures: ["OrthrusLM"]``) load without ``--trust-remote-code``::

    SGLANG_EXTERNAL_MODEL_PACKAGE=orthrus_sglang \\
        python -m sglang.launch_server --model-path chiennv/Orthrus-Qwen3-1.7B

SGLang imports every module in this package and registers each module's
``EntryClass`` under its class name (``sglang.srt.models.registry``).

Like the vLLM plugin, this serves the autoregressive path only. See
``docs/DIFFUSION_MODE.md``.
"""
