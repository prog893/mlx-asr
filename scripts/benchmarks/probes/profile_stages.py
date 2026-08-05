"""Wall-clock breakdown per stage: audio load, mel, conv stem, encoder, decode.

The batched driver's [encode]/[decode] logs lump mel + conv + transformer into
one number. Knowing the split decides whether batching the encoder is worth it:
the encoder currently runs one chunk at a time (mlx_audio shapes are seq-only),
so if it is a large share of wall clock there is a second batching win available.
"""

import argparse
import time

import mlx.core as mx
import numpy as np

from mlx_audio.stt.utils import load as load_model
from mlx_asr.decode import SAMPLE_RATE, load_audio_16k, split_at_silences


DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    mx.eval(out)
    return out, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--chunk-seconds", type=float, default=60.0)
    p.add_argument("--delay-ms", type=int, default=2400)
    p.add_argument("--limit-chunks", type=int, default=None)
    a = p.parse_args()

    t0 = time.perf_counter()
    audio = load_audio_16k(a.audio)
    t_load = time.perf_counter() - t0
    dur = len(audio) / SAMPLE_RATE

    chunks = split_at_silences(audio, target_s=a.chunk_seconds)
    if a.limit_chunks:
        chunks = chunks[: a.limit_chunks]
    print(f"[audio] {dur:.1f}s load={t_load:.1f}s -> {len(chunks)} chunks")

    t0 = time.perf_counter()
    model = load_model(a.model)
    print(f"[model] {time.perf_counter() - t0:.1f}s")

    model._ensure_ada_scales(a.delay_ms)
    enc = model.encoder
    max_len = max(len(c) for c in chunks)
    padded = [np.pad(c, (0, max_len - len(c))) for c in chunks]

    t_mel = t_conv = t_tf = 0.0
    n_audio = None
    for c in padded:
        mel, tm = timed(model._prepare_mel, c, a.delay_ms)
        t_mel += tm
        conv_out, tc = timed(enc.conv_stem, mel[0])
        t_conv += tc
        n_audio = conv_out.shape[0] // enc.config.downsample_factor

        t1 = time.perf_counter()
        if conv_out.shape[0] <= enc.config.sliding_window:
            adapter = enc.encode_full(conv_out)
        else:
            adapter = mx.concatenate(
                [enc.downsample_and_project(ch) for ch in enc.encode_chunks(conv_out)],
                axis=0,
            )
        mx.eval(adapter)
        t_tf += time.perf_counter() - t1

    print(f"[mel]     {t_mel:6.1f}s  ({t_mel / len(chunks):.2f}s/chunk)")
    print(f"[conv]    {t_conv:6.1f}s  ({t_conv / len(chunks):.2f}s/chunk)")
    print(f"[encoder] {t_tf:6.1f}s  ({t_tf / len(chunks):.2f}s/chunk)")
    print(f"[encode total] {t_mel + t_conv + t_tf:.1f}s for {len(chunks)} chunks")
    print(f"[decode would be] {n_audio} steps")


if __name__ == "__main__":
    main()
