"""`mlx-asr-bench`: find the best batch size on THIS machine.

Prints a ready-to-paste GitHub issue body so a measured profile for your
hardware can be added to profiles.json. No audio file and no reference
transcript needed: it drives the real decoder with random embeddings, which
measures the quantity that actually sets wall clock (decode steps per second).

    mlx-asr-bench                    # ~3 min, sensible batch ladder
    mlx-asr-bench --quick            # ~1 min
    mlx-asr-bench --json out.json    # also save raw numbers
"""

import argparse
import json
import sys
import time

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from mlx_audio.stt.utils import load as load_model

from .decode import batched_decoder_forward
from .hardware import machine_info, resolve_profile

PROMPT_LEN = 39
DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def ladder(gpu_gb: float, quick: bool) -> list:
    """Batch sizes to try, scaled to available GPU memory.

    Always includes 1 (the bandwidth-limit reference) and skips nothing in
    2..8, because that valley is the whole reason this bench exists.
    """
    if quick:
        base = [1, 8, 16, 32, 64, 128, 192]
    else:
        base = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    # KV is roughly 0.11GB per row per 400 steps at this model size; keep a
    # wide margin so the bench itself never swaps.
    cap = max(8, int(gpu_gb / 0.13))
    return [b for b in base if b <= cap]


def run_one(model, B, n_steps, dim, kv_bits=None):
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

    t0 = time.perf_counter()
    for i in range(n_steps):
        tok_emb = model.decoder.embed_tokens(tokens)
        h = batched_decoder_forward(
            model.decoder, tok_emb[:, None, :], PROMPT_LEN + i, caches, None
        )
        tokens = mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1)
        mx.async_eval(tokens)
    mx.eval(tokens)
    sps = n_steps / (time.perf_counter() - t0)
    peak = mx.get_peak_memory() / 1e9
    del caches
    mx.clear_cache()
    return sps, peak


def main(argv=None):
    p = argparse.ArgumentParser(prog="mlx-asr-bench", description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--batches", help="override the ladder, e.g. 1,16,32,64")
    p.add_argument("--kv-bits", type=int, default=None, choices=[4, 8])
    p.add_argument("--json", help="write raw results here")
    a = p.parse_args(argv)

    info = machine_info()
    current = resolve_profile(info)
    print(f"mlx-asr bench: {info['chip']}, {info['ram_gb']}GB RAM, "
          f"{info['gpu_cores']} GPU cores, {info['gpu_working_set_gb']}GB GPU working set")
    print(f"current profile: batch {current['batch']} ({current['matched']})\n")

    model = load_model(a.model)
    dim = model.config.decoder.dim
    batches = ([int(x) for x in a.batches.split(",")] if a.batches
               else ladder(info["gpu_working_set_gb"] or 10.0, a.quick))
    print(f"testing batches {batches} at {a.steps} steps each\n")
    print(f"{'batch':>6} {'steps/s':>8} {'ms/step':>8} {'x realtime':>11} {'peak GB':>8}")

    rows = []
    for B in batches:
        try:
            run_one(model, B, 20, dim, a.kv_bits)  # warm up this shape
            sps, peak = run_one(model, B, a.steps, dim, a.kv_bits)
        except Exception as e:
            print(f"{B:>6}  failed: {type(e).__name__}")
            rows.append({"batch": B, "error": type(e).__name__})
            continue
        xrt = sps * B * 0.08  # each step consumes 80ms of audio per row
        rows.append({"batch": B, "steps_per_s": round(sps, 2),
                     "ms_per_step": round(1e3 / sps, 1),
                     "x_realtime": round(xrt, 1), "peak_gb": round(peak, 2)})
        print(f"{B:>6} {sps:>8.2f} {1e3 / sps:>8.1f} {xrt:>11.1f} {peak:>8.2f}")

    ok = [r for r in rows if "x_realtime" in r]
    if not ok:
        print("\nno batch size completed; please open an issue with the error above")
        return 1

    # Prefer the fastest, but drop configs whose peak leaves no room for the
    # encoder and the OS. The bench decodes only; a real run also holds
    # adapter outputs and the audio.
    budget = (info["gpu_working_set_gb"] or 10.0) * 0.6
    safe = [r for r in ok if r["peak_gb"] <= budget] or ok
    best = max(safe, key=lambda r: r["x_realtime"])
    chunk = 15 if best["batch"] >= 96 else 20 if best["batch"] >= 48 else 30

    print(f"\nbest: batch {best['batch']} -> {best['x_realtime']}x realtime decode "
          f"({best['peak_gb']}GB peak)")
    if best["batch"] != current["batch"]:
        print(f"this beats your current profile (batch {current['batch']}). "
              f"Run with:  --max-batch {best['batch']} --chunk-seconds {chunk}")

    issue = {
        "machine": info,
        "model": a.model,
        "kv_bits": a.kv_bits,
        "steps_per_measurement": a.steps,
        "suggested_profile": {
            "match": {"chip": info["chip"],
                      "ram_gb_min": max(0, (info["ram_gb"] or 0) - 2),
                      "ram_gb_max": (info["ram_gb"] or 0) + 2},
            "batch": best["batch"],
            "chunk_seconds": chunk,
            "peak_gb": best["peak_gb"],
            "model_id": info.get("model_id"),
            "source": f"{info['chip']}, {info['ram_gb']}GB, "
                      f"{info['gpu_cores']} GPU cores"
                      + (f" ({info['model_id']})" if info.get("model_id") else "")
                      + f", mlx {info['mlx']}",
        },
        "results": rows,
    }

    print("\n" + "=" * 72)
    print("Copy everything below into a new issue titled:")
    print(f"  Profile: {info['chip']} {info['ram_gb']}GB"
          + (f" ({info['model_id']})" if info.get("model_id") else ""))
    print("=" * 72)
    print(f"\n**Machine:** {info['chip']}, {info['ram_gb']}GB RAM, "
          f"{info['gpu_cores']} GPU cores"
          + (f", {info['model_id']}" if info.get("model_id") else "")
          + f", macOS {info['macos']}, mlx {info['mlx']}")
    print(f"**Model:** `{a.model}`"
          + (f" (kv-bits {a.kv_bits})" if a.kv_bits else ""))
    print(f"**Best:** batch {best['batch']}, {best['x_realtime']}x realtime decode, "
          f"{best['peak_gb']}GB peak\n")
    print("| batch | steps/s | ms/step | x realtime | peak GB |")
    print("|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['batch']} | failed ({r['error']}) | | | |")
        else:
            print(f"| {r['batch']} | {r['steps_per_s']} | {r['ms_per_step']} | "
                  f"{r['x_realtime']} | {r['peak_gb']} |")
    print("\n<details><summary>raw json</summary>\n\n```json")
    print(json.dumps(issue, indent=2))
    print("```\n</details>")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(issue, f, indent=2)
        print(f"\n[saved] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
