# Diffusion-mode decoding: why it's reference code, not an active feature

This plugin activates Orthrus's **autoregressive serving path only**. The diffusion-mode, self-speculative decoding path (`src/orthrus_vllm/speculative/orthrus_proposer.py`) is included as real, tested reference code — it is not wired up, and importing this package does not enable it. This is a deliberate, honest limitation, not an oversight, and it's worth explaining why.

## What the code actually does (verified on GPU, see the original PR threads)

The proposer implements Orthrus's propose→verify→accept loop against vLLM's `SpecDecodeBaseProposer` interface. Per the original PR (vllm-project/vllm#53753) comment history:

- End-to-end generation works via `speculative_config={"method": "orthrus", ...}` on a patched vLLM build.
- A real bug was found and fixed: `OrthrusProposer` sets `kv_sharing_target_layer_name` so its diffusion attention *reads* the target's paged KV cache, but vLLM's `Attention.forward` also uses that field to decide whether to *write* this layer's own KV into the cache — correct for the one existing KV-sharing pattern (Gemma4 MTP, which is Q-only), wrong for Orthrus (which has real K/V projections that do need writing). Fixing this took the measured acceptance rate from ~3% to ~53% (100-prompt benchmark, A10G, `chiennv/Orthrus-Qwen3-1.7B`, `num_speculative_tokens=4`), in line with the reference implementation's own reported 39-82%.
- Honestly reported result: even after the fix, end-to-end throughput was still a net slowdown (0.65x vs. plain autoregressive) on that hardware/model/config. The reference implementation's own benchmark (run directly, not through vLLM) shows a real 2.39x speedup with byte-identical output vs. AR — so the approach is sound, but vLLM's specific integration overhead wasn't yet fully closed against the reference's numbers when this work stopped.

None of this is fabricated or aspirational — it's the literal comment history on the closed upstream PR, linked in the main README.

## Why it can't be a clean OOT plugin today

vLLM's plugin system (`vllm.general_plugins`) officially supports registering **models** via `ModelRegistry.register_model` — that's exactly what this package does for the AR path, with zero patches to vLLM's core. There is no equivalent, documented extension point for registering a **custom speculative-decoding drafter method**. The drafter is selected by a hardcoded `if/elif` chain on `speculative_config.method` inside `GPUModelRunner.__init__` (`vllm/v1/worker/gpu_model_runner.py`), which vLLM's engine constructs internally — there's no supported hook to intercept that construction from outside the vllm-project/vllm codebase, and vLLM's `SpeculativeConfig.method` validation independently whitelists a fixed set of method-name strings in `vllm/config/speculative.py`.

Monkey-patching vLLM internals at plugin import time (rewriting those methods at runtime) was considered and deliberately rejected for this repo: it would silently drift out of sync with any upstream vLLM changes to `GPUModelRunner`, and a plugin that patches vLLM's own dispatch logic is exactly the kind of fragile, hard-to-debug integration vLLM's plugin system was designed to avoid.

## The honest path to activating this

The actual fix is a small, generic, non-model-specific upstream change to vLLM: an extensibility point for OOT speculative-decoding drafters, analogous to `ModelRegistry.register_model` — e.g. a `SpecDecodeRegistry` that plugins can register a `(method_name, proposer_class)` pair against, with `GPUModelRunner` consulting the registry as a fallback in its existing `if/elif` chain. This is a fundamentally different kind of PR than the closed #44792/#53753: it's infrastructure that benefits every future OOT model needing custom draft logic, not a single niche model's feature, and it isn't subject to the same "not enough adoption to justify in-tree maintenance" objection that closed the model-support PR (see the main README for the exact maintainer feedback).

That upstream PR has not been opened yet. Until it lands (or an equivalent extension point exists), this package ships the tested proposer code for reference and reuse, but diffusion-mode decoding is not something `pip install orthrus-vllm` alone will get you.

## SGLang already has this extension point

SGLang (checked at v0.5.21) ships the registry described above: `SpeculativeAlgorithm.register("NAME")` in `sglang/srt/speculative/spec_info.py` lets a plugin register a custom speculative worker factory, which `--speculative-algorithm NAME` then selects. So on SGLang, diffusion-mode decoding could ship from this repo without an upstream change. It would still mean writing an SGLang speculative worker (propose a block through the `*_diff` view over the target's KV cache, verify with the AR view), not porting `OrthrusProposer` as-is; `orthrus_sglang` currently drops the `*_diff` weights and serves the AR path only.
