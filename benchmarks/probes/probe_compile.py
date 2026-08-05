"""Does mx.compile help the decode step?

One decode step runs 26 layers x ~10 ops = a few hundred small kernel launches,
each with dispatch overhead. mx.compile fuses elementwise chains and caches the
graph, which is the standard fix for launch-bound loops.

Complication: the KV cache mutates state and grows every step, so the whole step
cannot be naively compiled (the shapes change and the caches are captured
state). What CAN be compiled is the pure part of each layer: the SwiGLU FFN
(norm -> ada scale -> w1/w3 -> silu -> multiply -> w2) has no cache interaction.

Measured here:
  plain            current implementation
  compiled_ffn     FFN body wrapped in mx.compile, shapeless where possible
  compiled_norm    just the norm + ada-scale elementwise chain compiled
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlx_lm.models.cache import KVCache
from mlx_audio.stt.utils import load as load_model

PROMPT_LEN = 39


DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def make_ffn(layer, ada, compiled):
    w1, w2, w3 = layer.feed_forward_w1, layer.feed_forward_w2, layer.feed_forward_w3
    norm = layer.ffn_norm
    use_ada = layer.ada_rms_norm_t_cond is not None and ada is not None

    def ffn(x):
        h = norm(x)
        if use_ada:
            h = h * (1.0 + ada)
        return x + w2(nn.silu(w1(h)) * w3(h))

    return mx.compile(ffn) if compiled else ffn


def make_norm(layer, ada, compiled):
    norm = layer.ffn_norm
    use_ada = layer.ada_rms_norm_t_cond is not None and ada is not None

    def f(x):
        h = norm(x)
        if use_ada:
            h = h * (1.0 + ada)
        return h

    return mx.compile(f) if compiled else f


def step(decoder, x, pos, caches, ffns=None, norms=None):
    B, S, _ = x.shape
    for i, layer in enumerate(decoder.layers):
        attn = layer.attention
        h = layer.attention_norm(x)
        q = attn.wq(h).reshape(B, S, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        k = attn.wk(h).reshape(B, S, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        v = attn.wv(h).reshape(B, S, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, attn.head_dim, traditional=True, base=attn.rope_theta,
                         scale=1.0, offset=pos)
        k = mx.fast.rope(k, attn.head_dim, traditional=True, base=attn.rope_theta,
                         scale=1.0, offset=pos)
        k, v = caches[i].update_and_fetch(k, v)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=None)
        o = o.transpose(0, 2, 1, 3).reshape(B, S, attn.n_heads * attn.head_dim)
        x = x + attn.wo(o)

        if ffns is not None:
            x = ffns[i](x)
        else:
            h = norms[i](x) if norms is not None else layer.ffn_norm(x)
            if norms is None:
                ada = decoder._ada_scales[i]
                if layer.ada_rms_norm_t_cond is not None and ada is not None:
                    h = h * (1.0 + ada)
            gate = nn.silu(layer.feed_forward_w1(h))
            x = x + layer.feed_forward_w2(gate * layer.feed_forward_w3(h))
    return decoder.norm(x)


def run(model, B, n_steps, dim, mode):
    d = model.decoder
    model._ensure_ada_scales(2400)
    ffns = norms = None
    if mode == "compiled_ffn":
        ffns = [make_ffn(l, d._ada_scales[i], True) for i, l in enumerate(d.layers)]
    elif mode == "plain_ffn_fn":
        ffns = [make_ffn(l, d._ada_scales[i], False) for i, l in enumerate(d.layers)]
    elif mode == "compiled_norm":
        norms = [make_norm(l, d._ada_scales[i], True) for i, l in enumerate(d.layers)]

    caches = [KVCache() for _ in d.layers]
    prefix = mx.random.normal((B, PROMPT_LEN, dim)).astype(mx.bfloat16)
    h = step(d, prefix, 0, caches, ffns, norms)
    tok = mx.argmax(d.logits(h[:, -1, :]), axis=-1)
    mx.eval(tok)

    t0 = time.perf_counter()
    for i in range(n_steps):
        emb = d.embed_tokens(tok)[:, None, :]
        h = step(d, emb, PROMPT_LEN + i, caches, ffns, norms)
        tok = mx.argmax(d.logits(h[:, -1, :]), axis=-1)
        mx.async_eval(tok)
    mx.eval(tok)
    dt = time.perf_counter() - t0
    del caches
    mx.clear_cache()
    return n_steps / dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batches", default="1,16,32")
    a = p.parse_args()

    model = load_model(a.model)
    dim = model.config.decoder.dim
    modes = ["plain", "plain_ffn_fn", "compiled_ffn", "compiled_norm"]
    print(f"model={a.model} steps={a.steps}  ms/step")
    print(f"{'B':>4} " + " ".join(f"{m:>14}" for m in modes))
    for B in [int(x) for x in a.batches.split(",")]:
        row = []
        for m in modes:
            try:
                run(model, B, 15, dim, m)
                row.append(1e3 / run(model, B, a.steps, dim, m))
            except Exception as e:
                print(f"  B={B} {m}: {type(e).__name__}: {str(e)[:60]}")
                row.append(float("nan"))
        print(f"{B:>4} " + " ".join(f"{v:>14.1f}" for v in row))
        base = row[0]
        best = min(v for v in row if v == v)
        if best < base * 0.97:
            i = row.index(best)
            print(f"     -> {modes[i]} is {base / best:.2f}x faster at B={B}")


if __name__ == "__main__":
    main()
