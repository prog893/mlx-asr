"""Batched encoder + decoder for Voxtral Realtime.

The mlx-audio implementation hardcodes batch=1 in its attention reshapes, so
the decoder forward here reuses the loaded module weights with [B,S,dim]
shapes. Chunks are far shorter than the 8192 sliding window, so a plain
KVCache replaces the RotatingKVCache.
"""

import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import quantized_scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from mlx_audio.stt.models.voxtral_realtime.config import _num_delay_tokens

from .audio import TOKEN_S  # noqa: F401  (re-exported; see audio.py)


def encode_chunk(model, audio_np, delay_ms=None):
    """Run mel + conv stem + encoder for one chunk. Returns (adapter, n_audio)."""
    mel, _ = model._prepare_mel(audio_np, delay_ms)
    enc = model.encoder
    conv_out = enc.conv_stem(mel)
    n_audio = conv_out.shape[0] // enc.config.downsample_factor
    if conv_out.shape[0] <= enc.config.sliding_window:
        adapter = enc.encode_full(conv_out)
    else:
        parts = [enc.downsample_and_project(c) for c in enc.encode_chunks(conv_out)]
        adapter = mx.concatenate(parts, axis=0)
    mx.eval(adapter)
    return adapter, n_audio


def batched_decoder_forward(decoder, x, start_pos, caches, mask):
    """Decoder forward over x: [B, S, dim] using the loaded layer weights."""
    B, S, _ = x.shape
    for i, layer in enumerate(decoder.layers):
        attn = layer.attention
        h = layer.attention_norm(x)
        q = attn.wq(h).reshape(B, S, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        k = attn.wk(h).reshape(B, S, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        v = attn.wv(h).reshape(B, S, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, attn.head_dim, traditional=True, base=attn.rope_theta,
                         scale=1.0, offset=start_pos)
        k = mx.fast.rope(k, attn.head_dim, traditional=True, base=attn.rope_theta,
                         scale=1.0, offset=start_pos)
        k, v = caches[i].update_and_fetch(k, v)
        if isinstance(k, tuple):
            # QuantizedKVCache hands back (values, scales, biases) triples.
            # mlx.core.fast has no quantized SDPA in 0.32, so use mlx-lm's
            # wrapper, which dequantizes and falls back to the dense kernel.
            o = quantized_scaled_dot_product_attention(
                q, k, v, scale=attn.scale, mask=mask,
                group_size=caches[i].group_size, bits=caches[i].bits,
            )
        else:
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, S, attn.n_heads * attn.head_dim)
        x = x + attn.wo(o)

        h = layer.ffn_norm(x)
        ada = decoder._ada_scales[i]
        if layer.ada_rms_norm_t_cond is not None and ada is not None:
            h = h * (1.0 + ada)
        gate = nn.silu(layer.feed_forward_w1(h))
        x = x + layer.feed_forward_w2(gate * layer.feed_forward_w3(h))
    return decoder.norm(x)


def make_prompt_ids(cfg, n_delay, extra=None):
    """Build the decoder prompt: [BOS] + STREAMING_PAD * (n_left_pad + n_delay).

    ``extra`` token ids overwrite the tail of the left-pad region, so the
    decoder sees them as text it already emitted during the leading silence.
    That biases vocabulary at no per-step cost. The prompt length is fixed by
    the architecture, so only the last (n_left_pad_tokens - 1) ids survive;
    callers should put the most important keywords last.
    """
    ids = [cfg.bos_token_id] + [cfg.streaming_pad_token_id] * (
        cfg.n_left_pad_tokens + n_delay
    )
    if extra:
        extra = extra[-(cfg.n_left_pad_tokens - 1):]
        start = 1 + cfg.n_left_pad_tokens - len(extra)
        ids[start : start + len(extra)] = extra
    return ids


def prompt_capacity(cfg) -> int:
    """How many bias tokens actually fit in the prompt."""
    return cfg.n_left_pad_tokens - 1


def transcribe_batch(
    model, chunks, delay_ms=None, prompt_ids_extra=None, kv_bits=None, log=print,
    warmup_tokens=None,
):
    """Lockstep batched decode.

    Returns a per-chunk list of (token_id, time_s) where time_s is the speech
    time within that chunk's *kept* region.

    warmup_tokens: optional per-chunk count of leading decoder positions to
    discard, for chunks that carry prefix overlap (see split_with_overlap).
    Those positions transcribe audio the previous chunk already covered; the
    model still decodes them, which is the point, since that is what gives the
    kept region left context. Times are rebased so 0 is the start of the kept
    region.
    """
    cfg = model.config
    delay_ms = delay_ms or cfg.transcription_delay_ms
    model._ensure_ada_scales(delay_ms)
    B = len(chunks)

    # Encode every chunk at its own length, then right-pad the *adapter output*
    # to a common position axis. Padding the audio first would make short
    # chunks pay the longest chunk's encoder cost: the encoder is causal and
    # runs per chunk, so only the shared decode axis needs alignment.
    t0 = time.perf_counter()
    adapters = []
    row_n_audio = []
    for c in chunks:
        adapter, n = encode_chunk(model, c, delay_ms)
        adapters.append(adapter)
        row_n_audio.append(n)
    n_audio = max(row_n_audio)
    pad_to = max(a.shape[0] for a in adapters)
    adapter_batch = mx.stack([
        a if a.shape[0] == pad_to
        else mx.concatenate([a, mx.zeros((pad_to - a.shape[0], a.shape[1]), a.dtype)])
        for a in adapters
    ])
    adapter_len = adapter_batch.shape[1]
    mx.eval(adapter_batch)
    t_encode = time.perf_counter() - t0
    log(f"[encode] {B} chunks in {t_encode:.1f}s")

    n_delay = _num_delay_tokens(delay_ms)
    prompt_len = 1 + cfg.n_left_pad_tokens + n_delay

    # prompt_ids_extra is either one shared token-id list, or a per-chunk list
    # whose entries may be None (chunk 0 has no preceding context to carry).
    # Test for a nested sequence rather than for list-ness of element 0, which
    # misfires when the first per-chunk entry is None.
    if prompt_ids_extra and any(
        isinstance(e, (list, tuple)) or e is None for e in prompt_ids_extra
    ):
        if len(prompt_ids_extra) != B:
            raise ValueError(
                f"per-chunk prompt list has {len(prompt_ids_extra)} entries, "
                f"expected {B}"
            )
        per_chunk_extra = list(prompt_ids_extra)
    else:
        per_chunk_extra = [prompt_ids_extra] * B

    text_emb = mx.stack([
        model.decoder.embed_tokens(mx.array(make_prompt_ids(cfg, n_delay, e)))
        for e in per_chunk_extra
    ])
    prefix = adapter_batch[:, :prompt_len, :] + text_emb

    caches = [
        QuantizedKVCache(group_size=64, bits=kv_bits) if kv_bits else KVCache()
        for _ in model.decoder.layers
    ]
    h = batched_decoder_forward(model.decoder, prefix, 0, caches, "causal")
    tokens_lazy = mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1)
    mx.async_eval(tokens_lazy)

    eos = cfg.eos_token_id
    warm = list(warmup_tokens) if warmup_tokens else [0] * B
    timed = [[] for _ in range(B)]
    done = [False] * B
    t0 = time.perf_counter()
    n_steps = 0

    for pos in range(prompt_len, n_audio):
        tokens = np.array(tokens_lazy)
        step = pos - prompt_len + 1
        for b in range(B):
            if done[b]:
                continue
            # Past its own audio a row reads zero-padded adapter positions, so
            # anything it emits there is not transcription.
            if pos >= row_n_audio[b]:
                done[b] = True
                continue
            t = int(tokens[b])
            if t == eos:
                # A row carrying prefix overlap can emit EOS while still inside
                # the warm-up region (the overlap is a complete utterance as far
                # as the model is concerned). Ending the row there would drop the
                # whole kept region, so only EOS past the warm-up counts.
                if step > warm[b]:
                    done[b] = True
                continue
            if step > warm[b]:
                timed[b].append((t, (step - warm[b]) * TOKEN_S))
        if all(done):
            break

        tok_emb = model.decoder.embed_tokens(mx.array(tokens))
        x = adapter_batch[:, pos, :] + tok_emb if pos < adapter_len else tok_emb
        h = batched_decoder_forward(model.decoder, x[:, None, :], pos, caches, None)
        tokens_lazy = mx.argmax(model.decoder.logits(h[:, -1, :]), axis=-1)
        mx.async_eval(tokens_lazy)

        n_steps += 1
        if n_steps % 256 == 0:
            mx.clear_cache()

    t_decode = time.perf_counter() - t0
    log(
        f"[decode] {n_steps} steps in {t_decode:.1f}s "
        f"({n_steps / max(t_decode, 1e-9):.1f} steps/s, "
        f"{B * n_steps / max(t_decode, 1e-9):.0f} tok/s)"
    )
    mx.clear_cache()
    return timed, {"encode_s": t_encode, "decode_s": t_decode, "steps": n_steps}
