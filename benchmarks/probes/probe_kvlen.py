"""Does steps/s hold as the KV cache grows to real chunk lengths?

The batch sweep ran only 100 steps, so KV stayed tiny. Real chunks are 60s =
750 steps, and the whole point of the short-chunk strategy is that wall clock
is (steps of longest chunk) / (steps per second). That trade only pays if
steps/s does not degrade as KV grows.

Reports ms/step averaged over each successive block of steps, so cache-growth
drift is visible rather than averaged away.
"""

import argparse
import time

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from mlx_audio.stt.utils import load as load_model
from mlx_asr.decode import batched_decoder_forward

PROMPT_LEN = 39


DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--block", type=int, default=100)
    p.add_argument("--quant-kv", type=int, default=None)
    a = p.parse_args()

    model = load_model(a.model)
    model._ensure_ada_scales(2400)
    dim = model.config.decoder.dim
    B = a.batch

    mx.reset_peak_memory()
    caches = [
        QuantizedKVCache(group_size=64, bits=a.quant_kv) if a.quant_kv else KVCache()
        for _ in model.decoder.layers
    ]
    prefix = mx.random.normal((B, PROMPT_LEN, dim)).astype(mx.bfloat16)
    h = batched_decoder_forward(model.decoder, prefix, 0, caches, "causal")
    tokens = mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1)
    mx.eval(tokens)

    print(f"model={a.model} B={B} quant_kv={a.quant_kv}")
    print(f"{'kv_len':>8} {'ms/step':>9} {'steps/s':>8} {'x rt':>7} {'peak GB':>8}")
    t0 = time.perf_counter()
    for i in range(a.steps):
        pos = PROMPT_LEN + i
        tok_emb = model.decoder.embed_tokens(tokens)
        h = batched_decoder_forward(model.decoder, tok_emb[:, None, :], pos, caches, None)
        tokens = mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1)
        mx.async_eval(tokens)
        if (i + 1) % a.block == 0:
            mx.eval(tokens)
            dt = time.perf_counter() - t0
            sps = a.block / dt
            print(
                f"{pos:>8} {1e3 / sps:>9.1f} {sps:>8.2f} "
                f"{sps * B * 0.08:>7.1f} {mx.get_peak_memory() / 1e9:>8.2f}"
            )
            t0 = time.perf_counter()


if __name__ == "__main__":
    main()
