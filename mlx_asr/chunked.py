"""Chunked long-form decoding on MLX.

`mlx-whisper` ships only Whisper's *sequential* long-form driver: 30s windows,
each conditioned on the last one's text and timestamps. That is correct for full
Whisper and wrong for distil models, which keep 2 decoder layers instead of 32 and
cannot maintain state across windows. kotoba-whisper measured 94.23% coverage CER
through the sequential driver versus 26.16% through transformers' *chunked* driver,
on identical weights (RESULTS.md).

The important part: chunked long-form is a property of the **driver**, not the
model or the framework. Slice the audio, transcribe each slice independently,
offset the timestamps, stitch. So there is no reason a distil model has to leave
MLX to be decoded correctly, and this module is that driver.

Why it matters beyond correctness: running kotoba through torch/MPS made its
throughput incomparable to every other row in the benchmark, so it always carried
an asterisk. On MLX it is measured the same way as everything else.

Deliberately simple. No cross-window conditioning (that is the whole point), no
overlap between windows (Whisper's own chunked implementation does not use it
either, and this project's overlap experiments did not generalize; see
RESULTS.md), and independent windows mean an error cannot propagate.
"""

import json
from pathlib import Path

import numpy as np

# The keys an MLX Whisper conversion's config.json has, and a transformers one
# does not. mlx-whisper splats config.json straight into a dataclass, so handing
# it a transformers repo raises `TypeError: unexpected keyword argument
# '_name_or_path'` from library internals. Detecting the format here is what lets
# such a repo be converted instead (convert.py).
MLX_CONFIG_KEYS = ("n_mels", "n_audio_state", "n_text_layer")


def _is_mlx_format(cfg_path):
    """True if this config.json describes an MLX Whisper model.

    An MLX config is the small ModelDimensions set; a transformers one carries
    `_name_or_path`, `architectures` and dozens more. Returns True when it cannot
    tell, so an unreadable or absent config falls through to the loader's own
    error rather than being guessed at.
    """
    try:
        cfg = json.loads(Path(cfg_path).read_text())
    except (OSError, ValueError):
        return True
    return any(k in cfg for k in MLX_CONFIG_KEYS)


def resolve_mlx_weights(model_repo, log=print):
    """Return a path or repo id whose weights the MLX loader can read.

    Transformers-format weights are converted rather than rejected. The formats
    hold the same numbers, so this is a key rename plus one conv axis swap (see
    convert.py), it needs no torch, and the result is cached, so it happens once
    per machine. Before this, `--model kotoba` printed conversion instructions and
    stopped, which made the alias unusable out of the box.
    """
    path = Path(model_repo)
    if path.is_dir():
        cfg = path / "config.json"
        if not cfg.exists() or _is_mlx_format(cfg):
            return str(path)
    else:
        # config.json alone, which is a few hundred bytes, so the format can be
        # decided before pulling 1.5GB of weights. Fetched rather than read from
        # the cache because on a fresh machine nothing is cached yet, and that is
        # exactly the case that has to work: the default `kotoba` repo is
        # transformers-format and would otherwise fail on first run.
        try:
            from huggingface_hub import hf_hub_download

            cfg = hf_hub_download(model_repo, "config.json")
        except Exception:
            return model_repo      # offline, gated, or no such file: let the
        if _is_mlx_format(cfg):    # loader report it
            return model_repo

    from .convert import cached_mlx_copy

    return cached_mlx_copy(str(model_repo), log=log)


def transcribe_chunked(audio, model_repo, chunk_length_s=10.0, language=None,
                       sample_rate=16000, log=print, **decode_opts):
    """Decode long audio as independent windows. Returns (cues, text, meta).

    ``audio`` is a float32 numpy array at ``sample_rate``. ``cues`` is
    ``[(start_s, end_s, text)]`` on the original timeline, matching what
    ``output.build_cues`` returns, so every writer works unchanged.

    ``chunk_length_s`` is the one knob that matters here and it is
    material-dependent: on our corpus kotoba scored 26.16% at 10s versus 49.57% at
    30s, but on clean narration 20s beat both. Sweep it on your own audio.
    """
    from mlx_whisper.transcribe import transcribe

    model_repo = resolve_mlx_weights(model_repo, log=log)
    n = int(chunk_length_s * sample_rate)
    if n <= 0:
        raise ValueError("chunk_length_s must be positive")
    # A window shorter than this carries no usable speech and Whisper tends to
    # hallucinate on it, so the tail is dropped rather than decoded.
    min_tail = sample_rate // 2

    cues, parts = [], []
    n_windows = max(1, (len(audio) + n - 1) // n)
    log(f"[chunked] {len(audio)/sample_rate:.0f}s in {n_windows} independent "
        f"{chunk_length_s:g}s windows (no cross-window conditioning)")
    for i in range(0, len(audio), n):
        piece = np.asarray(audio[i : i + n], dtype=np.float32)
        if len(piece) < min_tail:
            continue
        offset = i / sample_rate
        r = transcribe(piece, path_or_hf_repo=model_repo, language=language,
                       verbose=False,
                       # never condition: this is what the sequential driver gets
                       # wrong for distil models
                       condition_on_previous_text=False, **decode_opts)
        for s in r.get("segments", []):
            text = (s.get("text") or "").strip()
            if not text:
                continue
            start = offset + float(s["start"])
            end = offset + float(s["end"])
            # Whisper occasionally emits an end at or before its start; clamp so
            # downstream writers and the timing metric see a valid interval.
            cues.append((start, max(end, start + 0.05), text))
            parts.append(text)

    # Windows are decoded independently, so a late window can in principle report
    # a start before an earlier one's end. Sorting keeps the cue list monotonic,
    # which the SRT/VTT writers assume.
    cues.sort(key=lambda c: c[0])
    return cues, "".join(parts), {"windows": n_windows, "segments": len(cues)}
