"""Can the encoder be batched, and is it worth it?

The encoder is ~35% of wall clock on an M4 and runs one chunk at a time,
because mlx_audio's EncoderAttention reshapes to [1, seq, heads, dim]. Chunks
are independent, so in principle N chunks of equal length can go through as
[N, seq, dim] and share the weight reads, exactly like the decoder batching.

This probe reimplements the encoder layer stack with a batch axis and compares
against the stock per-chunk loop for both speed and numerical agreement. It
only handles the encode_full path (chunk shorter than the 750-frame sliding
window, i.e. audio under ~60s), which is the case that matters for the chunk
lengths this tool uses.
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlx_audio.stt.utils import load as load_model

from mlx_asr.audio import load_audio_16k, split_at_silences
from mlx_asr.decode import encode_chunk


DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def _layer_stack(enc, x, offset, mask, caches=None):
    """All 32 encoder layers over x: [B, S, dim] at rope offset ``offset``."""
    B, S, _ = x.shape
    for i, layer in enumerate(enc.transformer_layers):
        attn = layer.attention
        h = layer.attention_norm(x)
        q = attn.wq(h).reshape(B, S, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        k = attn.wk(h).reshape(B, S, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        v = attn.wv(h).reshape(B, S, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, attn.head_dim, traditional=True,
                         base=attn.rope_theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, attn.head_dim, traditional=True,
                         base=attn.rope_theta, scale=1.0, offset=offset)
        if caches is not None:
            k, v = caches[i].update_and_fetch(k, v)
        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / (attn.head_dim ** 0.5), mask=mask
        )
        o = o.transpose(0, 2, 1, 3).reshape(B, S, attn.n_heads * attn.head_dim)
        x = x + attn.wo(o)

        h = layer.ffn_norm(x)
        gate = nn.silu(layer.feed_forward_w1(h))
        x = x + layer.feed_forward_w2(gate * layer.feed_forward_w3(h))
    return x


def _downsample_project(enc, x):
    B = x.shape[0]
    cfg = enc.config
    ds = cfg.downsample_factor
    ds_len = x.shape[1] // ds
    x = x[:, : ds_len * ds, :].reshape(B, ds_len, cfg.dim * ds)
    x = nn.gelu(enc.audio_language_projection_0(x))
    return enc.audio_language_projection_2(x)


def batched_encoder_forward(enc, conv_out_b):
    """Batched equivalent of AudioEncoder.__call__ over [B, S, dim].

    Mirrors encode_chunks exactly: walk sliding-window-sized blocks, carrying a
    per-layer RotatingKVCache so each block attends to the previous window.
    mlx-lm's caches are already batch-shaped ([B, heads, S, dim]), so the only
    thing the stock code was missing is the batch axis in the reshapes.
    """
    from mlx_lm.models.cache import RotatingKVCache

    sw = enc.config.sliding_window
    S = conv_out_b.shape[1]
    if S <= sw:
        return _downsample_project(
            enc, enc.transformer_norm(_layer_stack(enc, conv_out_b, 0, "causal"))
        )

    caches = [RotatingKVCache(max_size=sw, keep=0) for _ in enc.transformer_layers]
    outs = []
    for start in range(0, S, sw):
        blk = conv_out_b[:, start : min(start + sw, S), :]
        mask = caches[0].make_mask(blk.shape[1], window_size=sw)
        x = _layer_stack(enc, blk, start, mask, caches)
        outs.append(enc.transformer_norm(x))
    return _downsample_project(enc, mx.concatenate(outs, axis=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--chunk-seconds", type=float, default=30.0)
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--delay-ms", type=int, default=2400)
    a = p.parse_args()

    model = load_model(a.model)
    enc = model.encoder
    audio = load_audio_16k(a.audio)
    chunks = split_at_silences(audio, target_s=a.chunk_seconds)
    print(f"{len(chunks)} chunks of ~{a.chunk_seconds}s from {a.audio}")

    # Reference: stock per-chunk path.
    n = min(8, len(chunks))
    t0 = time.perf_counter()
    ref = [encode_chunk(model, c, a.delay_ms)[0] for c in chunks[:n]]
    t_ref = time.perf_counter() - t0
    print(f"[per-chunk] {n} chunks in {t_ref:.2f}s ({t_ref / n:.3f}s/chunk)")

    # Batched path needs equal-length input, so pad audio to the longest.
    # (Unlike the decoder this padding is unavoidable: one shared seq axis.)
    maxlen = max(len(c) for c in chunks[:n])
    padded = [np.pad(c, (0, maxlen - len(c))) for c in chunks[:n]]
    mels = [model._prepare_mel(c, a.delay_ms)[0] for c in padded]
    convs = [enc.conv_stem(m) for m in mels]
    mx.eval(convs)
    S = convs[0].shape[0]
    print(f"conv out seq={S}, sliding_window={enc.config.sliding_window}, "
          f"path={'full' if S <= enc.config.sliding_window else 'chunked'}")

    for B in [int(x) for x in a.batches.split(",")]:
        if B > n:
            continue
        cb = mx.stack(convs[:B])
        mx.eval(cb)
        out = batched_encoder_forward(enc, cb)  # warm up
        mx.eval(out)
        t0 = time.perf_counter()
        out = batched_encoder_forward(enc, cb)
        mx.eval(out)
        dt = time.perf_counter() - t0
        # agreement against the stock path for row 0
        err = float(mx.abs(out[0, : ref[0].shape[0]] - ref[0]).max())
        print(f"[batched B={B}] {dt:.2f}s total, {dt / B:.3f}s/chunk "
              f"(speedup {t_ref / n / (dt / B):.2f}x)  max|diff| vs stock = {err:.4f}")


if __name__ == "__main__":
    main()
