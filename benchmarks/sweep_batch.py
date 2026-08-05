"""Portable decode-throughput sweep: steps/s vs batch size and KV length.

Run this first on any new machine. It needs no audio and no reference text: it
drives the real decoder with random embeddings, so it measures exactly the
quantity that sets wall clock (decode steps per second) without waiting for a
full transcription.

Two things it is designed to expose:

1. The small-batch valley. Decode reads every model weight once per step, so
   steps/s should be nearly flat in B. On a bandwidth-starved machine it is
   not: B=2..8 can be several times slower per step than B=1, and recover
   above B=12. A default of "batch 8" lands in the worst case.
2. KV growth. steps/s decays as the cache fills, so the sweep reports early
   and late blocks separately instead of one average.

    uv run python benchmarks/sweep_batch.py \
        --batches 1,2,4,8,16,32,64 --steps 400 --json sweep.json

Output is safe to attach to an issue: it records chip, core count, memory and
mlx version, but deliberately no hostname or paths.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_audio.stt.utils import load as load_model
from mlx_asr.decode import batched_decoder_forward

PROMPT_LEN = 39
DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def machine_info():
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "?"

    try:
        mem = mx.device_info()["max_recommended_working_set_size"] / 1e9
    except Exception:
        mem = None
    return {
        "chip": sh("sysctl -n machdep.cpu.brand_string"),
        "ram_gb": round(int(sh("sysctl -n hw.memsize") or 0) / 1e9),
        "gpu_cores": sh("system_profiler SPDisplaysDataType | awk '/Total Number of Cores/{print $NF; exit}'"),
        "gpu_working_set_gb": round(mem, 1) if mem else None,
        "macos": platform.mac_ver()[0],
        "mlx": mx.__version__,
    }


def run_one(model, B, n_steps, dim, kv_bits=None, block=None):
    """Decode n_steps at batch B. Returns (overall_sps, per_block_sps, peak_gb)."""
    model._ensure_ada_scales(2400)
    mx.reset_peak_memory()
    caches = [
        QuantizedKVCache(group_size=64, bits=kv_bits) if kv_bits else KVCache()
        for _ in model.decoder.layers
    ]
    prefix = mx.random.normal((B, PROMPT_LEN, dim)).astype(mx.bfloat16)
    h = batched_decoder_forward(model.decoder, prefix, 0, caches, "causal")
    tokens = mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1)
    mx.eval(tokens)

    block = block or n_steps
    blocks = []
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    for i in range(n_steps):
        tok_emb = model.decoder.embed_tokens(tokens)
        h = batched_decoder_forward(model.decoder, tok_emb[:, None, :],
                                    PROMPT_LEN + i, caches, None)
        tokens = mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1)
        mx.async_eval(tokens)
        if (i + 1) % block == 0:
            mx.eval(tokens)
            blocks.append(round(block / (time.perf_counter() - t0), 2))
            t0 = time.perf_counter()
    mx.eval(tokens)
    overall = n_steps / (time.perf_counter() - t_all)
    peak = mx.get_peak_memory() / 1e9
    del caches
    mx.clear_cache()
    return overall, blocks, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batches", default="1,2,4,8,12,16,24,32,48,64")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--block", type=int, default=100)
    p.add_argument("--kv-bits", type=int, default=None)
    p.add_argument("--json", help="write results here")
    a = p.parse_args()

    info = machine_info()
    print(json.dumps(info, indent=2))
    model = load_model(a.model)
    dim = model.config.decoder.dim

    rows = []
    print(f"\n{'B':>4} {'steps/s':>8} {'ms/step':>8} {'tok/s':>7} {'x rt':>7} "
          f"{'peak GB':>8}  per-100-step blocks")
    for B in [int(x) for x in a.batches.split(",")]:
        try:
            run_one(model, B, 20, dim, a.kv_bits)  # warm up this shape
            sps, blocks, peak = run_one(model, B, a.steps, dim, a.kv_bits, a.block)
        except Exception as e:
            print(f"{B:>4} FAILED {type(e).__name__}: {str(e)[:50]}")
            rows.append({"batch": B, "error": f"{type(e).__name__}: {e}"})
            continue
        # each decode step consumes 80ms of audio per row
        xrt = sps * B * 0.08
        rows.append({"batch": B, "steps_per_s": round(sps, 2),
                     "ms_per_step": round(1e3 / sps, 1), "tok_per_s": round(sps * B),
                     "x_realtime": round(xrt, 1), "peak_gb": round(peak, 2),
                     "blocks": blocks})
        print(f"{B:>4} {sps:>8.2f} {1e3 / sps:>8.1f} {sps * B:>7.0f} {xrt:>7.1f} "
              f"{peak:>8.2f}  {blocks}")

    ok = [r for r in rows if "x_realtime" in r]
    if ok:
        best = max(ok, key=lambda r: r["x_realtime"])
        print(f"\nbest: B={best['batch']} at {best['x_realtime']}x realtime "
              f"({best['peak_gb']}GB peak)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"machine": info, "model": a.model, "kv_bits": a.kv_bits,
                       "steps": a.steps, "results": rows}, f, indent=2)
        print(f"[saved] {a.json}")


if __name__ == "__main__":
    main()
