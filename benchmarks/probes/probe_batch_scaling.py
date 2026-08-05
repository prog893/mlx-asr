"""Measure decode steps/s as a function of batch size B.

Why this matters: with everything in one batch, total decode wall clock is
(steps of the LONGEST chunk) / (steps per second), and steps/s is nearly
independent of B while decode is weight-bandwidth-bound. So halving chunk
length while doubling B should nearly halve wall clock. This probe finds where
that stops holding (bandwidth saturation or memory pressure).

Runs synthetic decode only (random adapter embeddings, no real audio), so it
measures the decode kernel cost and KV growth, not transcription quality.
"""

import argparse
import time

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from mlx_audio.stt.utils import load as load_model
from mlx_asr.decode import batched_decoder_forward


DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def probe(model, B, n_steps, dim, quant_kv=None):
    model._ensure_ada_scales(2400)
    caches = [
        QuantizedKVCache(group_size=64, bits=quant_kv) if quant_kv else KVCache()
        for _ in model.decoder.layers
    ]
    prompt_len = 39
    prefix = mx.random.normal((B, prompt_len, dim)).astype(mx.bfloat16)
    h = batched_decoder_forward(model.decoder, prefix, 0, caches, "causal")
    tokens = mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1)
    mx.eval(tokens)

    t0 = time.perf_counter()
    for pos in range(prompt_len, prompt_len + n_steps):
        tok_emb = model.decoder.embed_tokens(tokens)
        h = batched_decoder_forward(model.decoder, tok_emb[:, None, :], pos, caches, None)
        tokens = mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1)
        mx.async_eval(tokens)
    mx.eval(tokens)
    dt = time.perf_counter() - t0
    del caches
    mx.clear_cache()
    return n_steps / dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batches", default="1,8,16,24,32,48,64")
    p.add_argument("--quant-kv", type=int, default=None, help="bits for QuantizedKVCache")
    a = p.parse_args()

    model = load_model(a.model)
    dim = model.config.decoder.dim
    print(f"model={a.model} steps={a.steps} quant_kv={a.quant_kv}")
    print(f"{'B':>4} {'steps/s':>9} {'tok/s':>8} {'x realtime':>11} {'peak GB':>8}")
    for B in [int(x) for x in a.batches.split(",")]:
        mx.reset_peak_memory()
        try:
            sps = probe(model, B, a.steps, dim, a.quant_kv)
        except Exception as e:
            print(f"{B:>4} FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        peak = mx.get_peak_memory() / 1e9
        # each step consumes 80ms of audio per row
        print(f"{B:>4} {sps:>9.2f} {sps * B:>8.0f} {sps * B * 0.08:>11.1f} {peak:>8.2f}")


if __name__ == "__main__":
    main()
