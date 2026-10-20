# orthrus-serving

Native serving support for `chiennv/Orthrus-*` checkpoints in **vLLM** and **SGLang**, with no remote code, no engine fork and no monkeypatching:

- `src/orthrus_vllm`: a vLLM **out-of-tree plugin**, installed the same way any other vLLM model plugin is.
- `src/orthrus_sglang`: an SGLang **external model package**, loaded via `SGLANG_EXTERNAL_MODEL_PACKAGE` (see [SGLang](#sglang)).

Formerly `orthrus-vllm`; the old GitHub URL redirects here.

This exists because the equivalent in-tree vLLM PRs were closed (linked below). See [Why this is a plugin, not a vLLM PR](#why-this-is-a-plugin-not-a-vllm-pr) for the PR history and why an OOT plugin is vLLM's own documented path for this situation.

## Credit

This package integrates **Orthrus**, the original architecture and checkpoints, into vLLM. All of the model design, training, and released weights are the original authors' work:

> **Orthrus: Memory-Efficient Parallel Token Generation via Dual-View Diffusion**
> [chiennv2000/orthrus](https://github.com/chiennv2000/orthrus) · [arXiv:2605.12825](https://arxiv.org/abs/2605.12825) · Chien Nguyen
> Checkpoints: [`chiennv/Orthrus-Qwen3-1.7B`](https://huggingface.co/chiennv/Orthrus-Qwen3-1.7B), [`chiennv/Orthrus-Qwen3-4B`](https://huggingface.co/chiennv/Orthrus-Qwen3-4B), [`chiennv/Orthrus-Qwen3-8B`](https://huggingface.co/chiennv/Orthrus-Qwen3-8B)

This package contains no original model architecture or training contribution — it is an integration layer only.

## What this does

- Registers `OrthrusConfig` so `model_type="orthrus"` checkpoints load without remote code (`AutoConfig.register`).
- Registers `OrthrusForCausalLM` / `OrthrusLM` as a vLLM text-generation architecture (`ModelRegistry.register_model`).
- Serves Orthrus checkpoints through vLLM's **standard autoregressive path** — this is the fully working, supported part of this package.

## What this does *not* do (yet)

Diffusion-mode self-speculative decoding (Orthrus's actual headline feature — up to 2.39-7.8x speedup, strictly lossless, per the original repo) is **not** activated by this plugin. The proposer code is included as tested reference code, not a working feature. Read [`docs/DIFFUSION_MODE.md`](docs/DIFFUSION_MODE.md) for exactly what was validated, what the real, non-toy blocker is (vLLM's spec-decode drafter dispatch has no OOT extension point, unlike model registration), and what upstream change would actually fix it.

## Verification status

The vLLM autoregressive path is a port of what was validated end-to-end on real GPUs (A10G and A100-80GB) in the [#44792 PR thread](https://github.com/vllm-project/vllm/pull/44792). Both the vLLM plugin (vllm 0.31.0) and the SGLang package (sglang 0.5.21) have since been served end-to-end on an A10G with `chiennv/Orthrus-Qwen3-1.7B`; see [Benchmark](#benchmark). Larger checkpoints and multi-GPU setups have not been tested.

## Status: work in progress [incomplete]

This project is paused. Done and open items:

- [x] vLLM plugin: autoregressive serving
- [x] SGLang external model package: autoregressive serving, diffusion weights not loaded
- [x] Serving benchmark on A10G: SGLang vs vLLM vs base Qwen3 (1.7B)
- [x] SGLang greedy parity with base Qwen3: 8/8 identical
- [ ] Confirm vLLM greedy parity against vLLM + plain Qwen3 (currently 6/8 vs the SGLang reference; `vllm-qwen3` is in `benchmarks/modal_bench.py` but the run was aborted)
- [ ] vLLM plugin: skip allocating the `*_diff` weights when serving AR-only (3.87 vs 3.28 GiB of weights today)
- [ ] Benchmark the 4B / 8B checkpoints and tensor parallelism
- [ ] Compare KV cache capacity at an equal memory fraction across engines
- [ ] Diffusion-mode decoding on SGLang via `SpeculativeAlgorithm.register` (see [`docs/DIFFUSION_MODE.md`](docs/DIFFUSION_MODE.md))
- [ ] Diffusion-mode decoding on vLLM (blocked on an upstream drafter extension point)

## Install

```bash
pip install vllm>=0.11.0
pip install -e .
```

The `register_orthrus` entry point loads automatically whenever vLLM starts, no explicit import needed. To load only this plugin among others, set `VLLM_PLUGINS=register_orthrus`.

## Usage

```python
from vllm import LLM, SamplingParams

llm = LLM(model="chiennv/Orthrus-Qwen3-1.7B", trust_remote_code=False)
outputs = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=32))
print(outputs[0].outputs[0].text)
```

## SGLang

Orthrus' autoregressive view is plain Qwen3: the released checkpoints even keep the `Qwen/Qwen3-*` backbone weights bit-for-bit, and the extra `*_diff` attention weights are only read in diffusion mode. So `orthrus_sglang.OrthrusLM` reuses SGLang's own `Qwen3ForCausalLM` (and with it SGLang's kernels, CUDA graphs and radix cache) and drops the diffusion-only weights at load time, saving ~350M params of GPU memory on the 1.7B checkpoint.

```bash
pip install sglang
SGLANG_EXTERNAL_MODEL_PACKAGE=orthrus_sglang PYTHONPATH=src \
    python -m sglang.launch_server --model-path chiennv/Orthrus-Qwen3-1.7B
```

`SGLANG_EXTERNAL_MODEL_PACKAGE` makes SGLang import every module in the package and register each `EntryClass` by class name (`OrthrusLM`, which is what the checkpoints' `architectures` field names). The server speaks the usual OpenAI API on port 30000. As with vLLM, this is the autoregressive path only.

## Benchmark

`benchmarks/modal_bench.py` serves `chiennv/Orthrus-Qwen3-1.7B` on SGLang and on vLLM (this repo's plugin), plus `Qwen/Qwen3-1.7B` on SGLang as a reference, one after another on the same Modal GPU, all driven by the same client (`sglang.bench_serving`, OpenAI `/v1/completions`, greedy, fixed output lengths). It also checks greedy output parity against the base Qwen3 model.

```bash
modal run benchmarks/modal_bench.py            # full matrix
modal run benchmarks/modal_bench.py --quick    # smoke test
```

Results on one A10G (sglang 0.5.21, vllm 0.31.0, context length 4096, each engine's default memory settings; raw numbers in `benchmarks/results/a10g-20261020-112501.json`):

| server | out tok/s c=1 | out tok/s c=8 | out tok/s c=32 | ShareGPT c=32 out tok/s | TPOT p50 c=32 (ms) | TTFT p50 c=32 (ms) |
|---|---|---|---|---|---|---|
| SGLang + Orthrus | 111 | 653 | 1321 | 1849 | 20.6 | 905 |
| vLLM + Orthrus | 108 | 650 | 1324 | 1838 | 22.7 | 365 |
| SGLang + Qwen3-1.7B (reference) | 111 | 651 | 1323 | 1849 | 20.6 | 905 |

(c = max concurrency; the c=1/8/32 columns are random 1024-in / 256-out prompts.)

- **SGLang serves Orthrus at zero overhead**: every metric matches base Qwen3-1.7B on SGLang within noise, and greedy outputs are identical on 8/8 prompts (expected, since the AR weights are the same).
- **Throughput is a tie between engines.** SGLang has ~10% lower per-token decode latency at c=32. vLLM has much lower median TTFT on the random 1k-prompt runs (365 vs 905 ms at c=32), which is a scheduling-policy difference rather than anything Orthrus-specific; on ShareGPT, TTFT is close (43 vs 51 ms).
- **Memory**: SGLang loads 3.28 GiB of weights (diffusion weights dropped) vs 3.87 GiB for the vLLM plugin, which keeps the `*_diff` weights resident. The KV cache sizes (108k vs 142k tokens) are not comparable, because the two engines' default memory fractions differ.
- **vLLM greedy parity** vs the SGLang reference: 6/8 identical. The two misses diverge at a near-tie token (positions 23 and 4) and continue fluently. That looks like cross-engine bf16 kernel numerics, but this run didn't include vLLM + plain Qwen3, which is the run that would confirm it.

## Why this is a plugin, not a vLLM PR

This started as two PRs directly against `vllm-project/vllm`:

- **[#44792](https://github.com/vllm-project/vllm/pull/44792)** — `[Model] Add Orthrus model support (autoregressive path)`.
- **[#53753](https://github.com/vllm-project/vllm/pull/53753)** — `[Model][Spec Decode] Orthrus diffusion-mode decoding (WIP)`, an explicit follow-up building on #44792.

Both were closed — see the linked threads for the full discussion — and this repository packages that same work as a standard OOT plugin instead, which doesn't require vllm-project/vllm's approval to ship.

## License

Apache-2.0 for the integration code in this repository (see [`LICENSE`](LICENSE)). The Orthrus architecture and checkpoints themselves are MIT-licensed by their original author — see [`NOTICE`](NOTICE) and the [original repository](https://github.com/chiennv2000/orthrus) for full attribution.
