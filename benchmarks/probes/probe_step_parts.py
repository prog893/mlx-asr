"""Find which op in the decode step blows up when B > 1.

Decode reads all model weights once per step, so in a bandwidth-bound regime
steps/s should be almost flat in B. Measured, B=1 runs at the bandwidth limit
(~21ms/step) while B=8 takes ~100ms and B=16 is *faster* than B=8. A fixed
penalty that appears only for B>1 points at one op taking a different kernel
path, not at bandwidth.

Method: run the real decode loop with one component ablated at a time. The
ablations change the numerics (garbage tokens), which is fine; only timing is
read. Ablating an op that is the bottleneck collapses step time.
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from mlx_audio.stt.utils import load as load_model

PROMPT_LEN = 39


DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def step_forward(decoder, x, start_pos, caches, ablate):
    """One decoder forward over x: [B, S, dim], optionally skipping parts."""
    B, S, _ = x.shape
    for i, layer in enumerate(decoder.layers):
        attn = layer.attention
        h = layer.attention_norm(x)
        if ablate == "attn_proj":
            o = h
        else:
            q = attn.wq(h).reshape(B, S, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
            k = attn.wk(h).reshape(B, S, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
            v = attn.wv(h).reshape(B, S, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
            if ablate != "rope":
                q = mx.fast.rope(q, attn.head_dim, traditional=True,
                                 base=attn.rope_theta, scale=1.0, offset=start_pos)
                k = mx.fast.rope(k, attn.head_dim, traditional=True,
                                 base=attn.rope_theta, scale=1.0, offset=start_pos)
            if ablate == "kvcache":
                pass  # attend to the current step only, cache never grows
            else:
                k, v = caches[i].update_and_fetch(k, v)
            if ablate == "sdpa":
                o = q.transpose(0, 2, 1, 3).reshape(B, S, attn.n_heads * attn.head_dim)
            else:
                o = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=attn.scale, mask=None if S == 1 else "causal"
                )
                o = o.transpose(0, 2, 1, 3).reshape(B, S, attn.n_heads * attn.head_dim)
            o = attn.wo(o)
        x = x + o

        if ablate == "ffn":
            continue
        h = layer.ffn_norm(x)
        ada = decoder._ada_scales[i]
        if layer.ada_rms_norm_t_cond is not None and ada is not None:
            h = h * (1.0 + ada)
        gate = nn.silu(layer.feed_forward_w1(h))
        x = x + layer.feed_forward_w2(gate * layer.feed_forward_w3(h))
    return decoder.norm(x)


def run(model, B, n_steps, dim, ablate=None):
    model._ensure_ada_scales(2400)
    caches = [KVCache() for _ in model.decoder.layers]
    prefix = mx.random.normal((B, PROMPT_LEN, dim)).astype(mx.bfloat16)
    h = step_forward(model.decoder, prefix, 0, caches, ablate)
    if ablate == "logits":
        tokens = mx.zeros((B,), dtype=mx.int32)
    else:
        tokens = mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1)
    mx.eval(tokens, h)

    t0 = time.perf_counter()
    for pos in range(PROMPT_LEN, PROMPT_LEN + n_steps):
        tok_emb = model.decoder.embed_tokens(tokens)
        h = step_forward(model.decoder, tok_emb[:, None, :], pos, caches, ablate)
        if ablate == "logits":
            tokens = mx.zeros((B,), dtype=mx.int32) + int(h.sum().item() == 0)
        else:
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
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batches", default="1,8,16")
    a = p.parse_args()

    model = load_model(a.model)
    dim = model.config.decoder.dim
    ablations = [None, "sdpa", "kvcache", "rope", "logits", "ffn", "attn_proj"]
    print(f"model={a.model} steps={a.steps}   values are ms/step")
    hdr = " ".join(f"{str(x or 'full'):>10}" for x in ablations)
    print(f"{'B':>4} {hdr}")
    for B in [int(x) for x in a.batches.split(",")]:
        row = []
        for ab in ablations:
            run(model, B, 20, dim, ab)  # warm up this shape+path
            row.append(1e3 / run(model, B, a.steps, dim, ab))
        print(f"{B:>4} " + " ".join(f"{v:>10.1f}" for v in row))


if __name__ == "__main__":
    main()
