# SPDX-License-Identifier: Apache-2.0
"""Serving benchmark: Orthrus on SGLang vs. vLLM, plus base Qwen3 on SGLang.

All three servers run one after another in a single container, on the same
physical GPU, driven by the same client (``sglang.bench_serving``, OpenAI
``/v1/completions`` path for every engine, greedy, ``ignore_eos``). SGLang
lives in the system Python; vLLM + this repo's plugin live in /opt/vllm.

    modal run benchmarks/modal_bench.py            # full matrix
    modal run benchmarks/modal_bench.py --quick    # smoke test, ~1 min of load

Results land in benchmarks/results/<gpu>-<timestamp>.json.
"""

import json
import os
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import modal

SGLANG_VERSION = "0.5.21"
VLLM_VERSION = "0.31.0"
GPU = "A10G"
ORTHRUS = "chiennv/Orthrus-Qwen3-1.7B"
QWEN3 = "Qwen/Qwen3-1.7B"
PORT = 30000
CONTEXT_LEN = 4096

REPO = Path(__file__).resolve().parents[1]

image = (
    modal.Image.from_registry(f"lmsysorg/sglang:v{SGLANG_VERSION}")
    .entrypoint([])
    .run_commands(
        "pip install uv",
        "uv venv /opt/vllm --python 3.12",
        f"uv pip install --python /opt/vllm/bin/python vllm=={VLLM_VERSION}",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(
        REPO,
        "/repo",
        copy=True,
        ignore=[".git", "**/__pycache__", "benchmarks/results", "*.egg-info"],
    )
    .run_commands("uv pip install --python /opt/vllm/bin/python --no-deps /repo")
)

app = modal.App("orthrus-serving-bench", image=image)
cache = modal.Volume.from_name("orthrus-serving-cache", create_if_missing=True)

SERVERS = {
    "sglang-orthrus": {
        "model": ORTHRUS,
        "backend": "sglang-oai",
        "cmd": [
            "python", "-m", "sglang.launch_server",
            "--model-path", ORTHRUS,
            "--context-length", str(CONTEXT_LEN),
        ],
        "env": {
            "SGLANG_EXTERNAL_MODEL_PACKAGE": "orthrus_sglang",
            "PYTHONPATH": "/repo/src",
        },
    },
    "vllm-orthrus": {
        "model": ORTHRUS,
        "backend": "vllm",
        "cmd": [
            "/opt/vllm/bin/vllm", "serve", ORTHRUS,
            "--max-model-len", str(CONTEXT_LEN),
        ],
        "env": {"VLLM_PLUGINS": "register_orthrus"},
    },
    "sglang-qwen3": {
        "model": QWEN3,
        "backend": "sglang-oai",
        "cmd": [
            "python", "-m", "sglang.launch_server",
            "--model-path", QWEN3,
            "--context-length", str(CONTEXT_LEN),
        ],
        "env": {},
    },
}

# (name, bench_serving args)
BENCHES = [
    ("random-1k-256-c1", ["--dataset-name", "random", "--random-input-len", "1024",
                          "--random-output-len", "256", "--random-range-ratio", "1",
                          "--max-concurrency", "1", "--num-prompts", "32"]),
    ("random-1k-256-c8", ["--dataset-name", "random", "--random-input-len", "1024",
                          "--random-output-len", "256", "--random-range-ratio", "1",
                          "--max-concurrency", "8", "--num-prompts", "128"]),
    ("random-1k-256-c32", ["--dataset-name", "random", "--random-input-len", "1024",
                           "--random-output-len", "256", "--random-range-ratio", "1",
                           "--max-concurrency", "32", "--num-prompts", "256"]),
    ("sharegpt-c32", ["--dataset-name", "sharegpt", "--max-concurrency", "32",
                      "--num-prompts", "400"]),
]
QUICK_BENCHES = [
    ("random-1k-256-c8", ["--dataset-name", "random", "--random-input-len", "1024",
                          "--random-output-len", "256", "--random-range-ratio", "1",
                          "--max-concurrency", "8", "--num-prompts", "16"]),
]

PARITY_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n",
    "Explain why the sky is blue in two sentences.",
    "The three laws of thermodynamics are",
    "Translate to German: The weather is nice today.",
    "Write a haiku about GPUs.",
    "Q: What is 17 * 23?\nA:",
    "Once upon a time, in a small village by the sea,",
]
PARITY_MAX_TOKENS = 64

METRIC_KEYS = [
    "completed", "request_throughput", "output_throughput", "total_throughput",
    "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "median_itl_ms",
    "median_e2e_latency_ms",
]

# Startup facts both engines print, for the memory comparison.
LOG_PATTERNS = {
    "kv_cache_tokens": [
        r"max_total_num_tokens=(\d+)",  # SGLang
        r"GPU KV cache size: ([\d,]+) tokens",  # vLLM
    ],
    "weights_gib": [
        r"Load weight end\..*?mem usage=([\d.]+) GB",  # SGLang
        r"Model loading took ([\d.]+) GiB",  # vLLM
    ],
}


def _scrape(log: str) -> dict:
    out = {}
    for key, patterns in LOG_PATTERNS.items():
        for pattern in patterns:
            if m := re.search(pattern, log):
                out[key] = float(m.group(1).replace(",", ""))
                break
    return out


def _wait_ready(proc: subprocess.Popen, log_path: str, timeout: float = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited ({proc.returncode}):\n{Path(log_path).read_text()[-4000:]}"
            )
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5):
                return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"server not ready:\n{Path(log_path).read_text()[-4000:]}")


def _complete(model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": PARITY_MAX_TOKENS,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", body,
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)["choices"][0]["text"]


def _bench(server: dict, args: list[str], out_file: str) -> dict:
    subprocess.run(
        ["python", "-m", "sglang.bench_serving",
         "--backend", server["backend"],
         "--base-url", f"http://127.0.0.1:{PORT}",
         "--model", server["model"],
         "--warmup-requests", "4", "--seed", "0", "--disable-tqdm",
         "--output-file", out_file, *args],
        check=True,
    )
    # One JSON line per run; this file only ever holds one run.
    record = json.loads(Path(out_file).read_text().strip().splitlines()[-1])
    return {k: record.get(k) for k in METRIC_KEYS}


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=2 * 3600)
def run(servers: list[str], quick: bool) -> dict:
    results = {
        "gpu": GPU, "sglang": SGLANG_VERSION, "vllm": VLLM_VERSION,
        "context_len": CONTEXT_LEN, "servers": {},
    }
    for name in servers:
        server = SERVERS[name]
        log_path = f"/tmp/{name}.log"
        print(f"=== {name}: starting", flush=True)
        t0 = time.time()
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                [*server["cmd"], "--port", str(PORT)],
                env={**os.environ, **server["env"]},
                stdout=log, stderr=subprocess.STDOUT,
            )
        try:
            _wait_ready(proc, log_path)
            entry = {"startup_s": round(time.time() - t0, 1)}
            entry.update(_scrape(Path(log_path).read_text()))
            entry["parity"] = [_complete(server["model"], p) for p in PARITY_PROMPTS]
            entry["benches"] = {}
            for bench_name, args in QUICK_BENCHES if quick else BENCHES:
                print(f"=== {name}: {bench_name}", flush=True)
                entry["benches"][bench_name] = _bench(
                    server, args, f"/tmp/{name}-{bench_name}.jsonl"
                )
            results["servers"][name] = entry
        except Exception as e:
            results["servers"][name] = {"error": str(e)[-4000:]}
            print(f"=== {name}: FAILED\n{e}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(5)  # let the GPU memory drain before the next server
        cache.commit()
    return results


def _report(results: dict) -> str:
    servers = {k: v for k, v in results["servers"].items() if "error" not in v}
    lines = [f"GPU {results['gpu']}, sglang {results['sglang']}, vllm {results['vllm']}", ""]

    lines.append("| server | startup s | weights GiB | KV cache tokens |")
    lines.append("|---|---|---|---|")
    for name, s in servers.items():
        lines.append(f"| {name} | {s.get('startup_s')} | {s.get('weights_gib')} "
                     f"| {s.get('kv_cache_tokens')} |")

    bench_names = sorted({b for s in servers.values() for b in s["benches"]})
    for bench in bench_names:
        lines += ["", f"**{bench}**", "",
                  "| server | out tok/s | req/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | ITL p50 ms |",
                  "|---|---|---|---|---|---|---|"]
        for name, s in servers.items():
            m = s["benches"].get(bench)
            if m:
                lines.append(
                    f"| {name} | {m['output_throughput']:.0f} | {m['request_throughput']:.2f} "
                    f"| {m['median_ttft_ms']:.1f} | {m['p99_ttft_ms']:.1f} "
                    f"| {m['median_tpot_ms']:.2f} | {m['median_itl_ms']:.2f} |"
                )

    ref = servers.get("sglang-qwen3", {}).get("parity")
    if ref:
        lines += ["", f"Greedy parity vs sglang-qwen3 ({len(ref)} prompts, "
                      f"{PARITY_MAX_TOKENS} tokens):"]
        for name, s in servers.items():
            if name != "sglang-qwen3":
                same = sum(a == b for a, b in zip(s["parity"], ref))
                lines.append(f"- {name}: {same}/{len(ref)} identical")

    for name, s in results["servers"].items():
        if "error" in s:
            lines += ["", f"{name} FAILED: {s['error'][-1500:]}"]
    return "\n".join(lines)


@app.local_entrypoint()
def main(servers: str = ",".join(SERVERS), quick: bool = False):
    results = run.remote(servers.split(","), quick)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = REPO / "benchmarks" / "results" / f"{GPU.lower()}-{stamp}{'-quick' if quick else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(_report(results))
    print(f"\nwrote {out}")
