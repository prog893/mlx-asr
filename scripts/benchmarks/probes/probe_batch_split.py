"""Can the qmv/qmm valley be dodged by reshaping the decode batch?

Upstream MLX confirms the valley is a kernel-dispatch artifact, not bandwidth:
"The drop from 4 to 8 is that we switch from batched qmv to the qmm"
(ml-explore/mlx discussion #1593). The crossover point (`vector_limit`, from
get_qmv_batch_limit) is compile-time and machine-specific, so it cannot be tuned
from Python.

But the dispatch decision is made per matmul on the leading dim, so if B rows in
one call land in the slow region, running the SAME work as several calls whose
leading dim sits in the fast region might beat it. Two variants tested here:

  fold   [B,1,dim] -> [1,B,dim]: one call whose M is B, so a >=vector_limit B
         dispatches to qmm instead of the batched-qmv path.
  split  run the layer stack on N sub-batches of size B/N and concatenate.

Both compute identical math to the plain path; only kernel dispatch differs.
Correctness is checked against the plain path, not assumed.
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# probes sit three levels down: scripts/ for metrics, root for mlx_asr.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlx_audio.stt.utils import load as load_model


DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def layer_stack(decoder, x, start_pos, caches, fold):
    """One decode step over x: [B, 1, dim].

    fold=True reshapes the token axis so the quantized matmuls see M=B with a
    leading batch of 1, which is a different dispatch than B x M=1.
    Attention still runs per row, so the KV cache semantics are unchanged.
    """
    B = x.shape[0]
    for i, layer in enumerate(decoder.layers):
        attn = layer.attention
        h = layer.attention_norm(x)
        hp = h.reshape(1, B, -1) if fold else h
        q = attn.wq(hp).reshape(B, 1, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        k = attn.wk(hp).reshape(B, 1, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        v = attn.wv(hp).reshape(B, 1, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, attn.head_dim, traditional=True, base=attn.rope_theta,
                         scale=1.0, offset=start_pos)
        k = mx.fast.rope(k, attn.head_dim, traditional=True, base=attn.rope_theta,
                         scale=1.0, offset=start_pos)
        k, v = caches[i].update_and_fetch(k, v)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=None)
        o = o.transpose(0, 2, 1, 3).reshape(B, 1, attn.n_heads * attn.head_dim)
        op = o.reshape(1, B, -1) if fold else o
        x = x + attn.wo(op).reshape(B, 1, -1)

        h = layer.ffn_norm(x)
        ada = decoder._ada_scales[i]
        if layer.ada_rms_norm_t_cond is not None and ada is not None:
            h = h * (1.0 + ada)
        hp = h.reshape(1, B, -1) if fold else h
        gate = nn.silu(layer.feed_forward_w1(hp))
        up = layer.feed_forward_w3(hp)
        x = x + layer.feed_forward_w2(gate * up).reshape(B, 1, -1)
    return decoder.norm(x)


def run(model, B, n_steps, dim, mode, n_split=1):
    from mlx_lm.models.cache import KVCache

    model._ensure_ada_scales(2400)
    prompt_len = 39
    fold = mode == "fold"
    groups = n_split if mode == "split" else 1
    per = B // groups
    caches = [[KVCache() for _ in model.decoder.layers] for _ in range(groups)]

    # Prefill each group with the same [B_g, P, dim] prefix shape.
    toks = []
    for g in range(groups):
        prefix = mx.random.normal((per, prompt_len, dim)).astype(mx.bfloat16)
        h = _prefill(model.decoder, prefix, caches[g])
        toks.append(mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1))
    mx.eval(toks)

    t0 = time.perf_counter()
    for i in range(n_steps):
        pos = prompt_len + i
        outs = []
        for g in range(groups):
            emb = model.decoder.embed_tokens(toks[g])[:, None, :]
            h = layer_stack(model.decoder, emb, pos, caches[g], fold)
            outs.append(mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1))
        toks = outs
        for t in toks:
            mx.async_eval(t)
    mx.eval(toks)
    dt = time.perf_counter() - t0
    del caches
    mx.clear_cache()
    return n_steps / dt


def _prefill(decoder, prefix, caches):
    from mlx_asr.decode import batched_decoder_forward

    return batched_decoder_forward(decoder, prefix, 0, caches, "causal")


def check_equiv(model, dim):
    """fold must produce the same numbers as plain for one step."""
    from mlx_lm.models.cache import KVCache

    model._ensure_ada_scales(2400)
    B = 8
    x = mx.random.normal((B, 1, dim)).astype(mx.bfloat16)
    mx.eval(x)
    outs = []
    for fold in (False, True):
        caches = [KVCache() for _ in model.decoder.layers]
        outs.append(layer_stack(model.decoder, x, 0, caches, fold))
        mx.eval(outs[-1])
    err = float(mx.abs(outs[0] - outs[1]).max())
    print(f"fold vs plain max|diff| = {err:.6f} "
          f"({'equivalent' if err < 1e-2 else 'NOT equivalent'})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batches", default="4,8,12,16,24,32")
    a = p.parse_args()

    model = load_model(a.model)
    dim = model.config.decoder.dim
    check_equiv(model, dim)

    print(f"\n{'B':>4} {'plain':>9} {'fold':>9} {'split2':>9} {'split4':>9}  "
          f"best (x realtime)")
    for B in [int(x) for x in a.batches.split(",")]:
        res = {}
        for mode, ns in (("plain", 1), ("fold", 1), ("split", 2), ("split", 4)):
            if mode == "split" and (B % ns or B // ns < 1):
                res[f"{mode}{ns}"] = None
                continue
            key = mode if mode != "split" else f"split{ns}"
            try:
                run(model, B, 15, dim, mode, ns)  # warm up
                res[key] = 1e3 / run(model, B, a.steps, dim, mode, ns)
            except Exception as e:
                print(f"  B={B} {key} failed: {type(e).__name__}")
                res[key] = None
        cells = [res.get(k) for k in ("plain", "fold", "split2", "split4")]
        best_k = min((k for k in res if res[k]), key=lambda k: res[k])
        sps = 1e3 / res[best_k]
        print(f"{B:>4} " + " ".join(f"{c:>9.1f}" if c else f"{'-':>9}" for c in cells)
              + f"  {best_k} {sps * B * 0.08:>5.1f}x")


if __name__ == "__main__":
    main()
