"""Isolate the cost of the per-step host sync in the decode loop.

The batched driver does ``tokens = np.array(tokens_lazy)`` at the top of every
step so it can test for EOS and record token ids. That is a full GPU->host
barrier once per 80ms of audio per row: the GPU cannot start step N+1 until
Python has finished step N's bookkeeping.

Variants measured, all with identical math:
  sync_every_step  what the driver does today
  sync_every_k     defer the barrier, keep token ids as lazy arrays, sync in blocks
  no_sync          upper bound (never reads tokens on host until the end)

Every variant is warmed up first and run twice, because the first call after a
batch-shape change pays Metal kernel specialization.
"""

import argparse
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

from mlx_audio.stt.utils import load as load_model
from mlx_asr.decode import batched_decoder_forward

PROMPT_LEN = 39


DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def run(model, B, n_steps, dim, mode, k=32):
    model._ensure_ada_scales(2400)
    caches = [KVCache() for _ in model.decoder.layers]
    prefix = mx.random.normal((B, PROMPT_LEN, dim)).astype(mx.bfloat16)
    h = batched_decoder_forward(model.decoder, prefix, 0, caches, "causal")
    tokens = mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1)
    mx.eval(tokens)

    collected = []
    pending = []
    t0 = time.perf_counter()
    for pos in range(PROMPT_LEN, PROMPT_LEN + n_steps):
        if mode == "sync_every_step":
            collected.append(np.array(tokens))
        elif mode == "sync_every_k":
            pending.append(tokens)
            if len(pending) == k:
                collected.append(np.array(mx.stack(pending)))
                pending = []
        tok_emb = model.decoder.embed_tokens(tokens)
        h = batched_decoder_forward(model.decoder, tok_emb[:, None, :], pos, caches, None)
        tokens = mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1)
        mx.async_eval(tokens)
    if pending:
        collected.append(np.array(mx.stack(pending)))
    mx.eval(tokens)
    dt = time.perf_counter() - t0
    del caches
    mx.clear_cache()
    return n_steps / dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batches", default="1,8,16,32")
    a = p.parse_args()

    model = load_model(a.model)
    dim = model.config.decoder.dim
    modes = ["sync_every_step", "sync_every_k", "no_sync"]
    print(f"model={a.model} steps={a.steps}  (steps/s, best of 2 after warmup)")
    print(f"{'B':>4} " + " ".join(f"{m:>16}" for m in modes) + f" {'x rt (best)':>12}")
    for B in [int(x) for x in a.batches.split(",")]:
        row = []
        for mode in modes:
            run(model, B, 24, dim, mode)  # warmup at this shape
            best = max(run(model, B, a.steps, dim, mode) for _ in range(2))
            row.append(best)
        xrt = max(row) * B * 0.08
        print(f"{B:>4} " + " ".join(f"{v:>16.2f}" for v in row) + f" {xrt:>12.1f}")


if __name__ == "__main__":
    main()
