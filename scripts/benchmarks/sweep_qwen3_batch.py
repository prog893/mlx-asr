"""Does batching whole chunks help the Qwen3-ASR decoder? (issue #1)

`--max-batch` exits 2 on `qwen3-asr` because `Qwen3ASR.generate(batch_size=)` exists
upstream but no value had been measured for this decoder. The only batch-size finding in
this project ("never use 2-8", docs/benchmarks/decode-throughput.md) was measured on
Voxtral, which shares no decode path with it.

This sweeps batch size against BOTH throughput and accuracy, because upstream's batched
path is not a pure speed knob:

  * `max_tokens` is decremented across the whole batch (`remaining_tokens -= group_tokens`
    in `_transcribe_batched`), so one runaway chunk can starve its batch-mates and
    truncate their text. The per-window budget that mlx_asr's own loop exists to enforce
    (see backends.TOKENS_PER_SECOND) does not survive batching.
  * chunks in a group are zero-padded to the longest member before the encoder runs, so a
    batch mixing a 30s and a 3s chunk pays encoder cost for 30s on both.

Either effect changes the transcript, so a throughput-only sweep would report a win that
costs accuracy without noticing.

    uv run python scripts/benchmarks/sweep_qwen3_batch.py \\
        --corpus DIR --batches 1,2,4,8,16 --chunk-seconds 15 --json out.json

Window length matters: at the shipped 30s a 13-minute file yields ~26 chunks, so batch 32
would be one group and the knob would be untestable above that. Pass a shorter window to
get enough chunks to fill several groups, and note that the accuracy baseline then shifts
with it ([qwen3-asr.md](../../docs/benchmarks/qwen3-asr.md) measured 15s as a tie with 30s).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx
import numpy as np
from mlx_audio.stt.utils import load as load_model

from benchmarks.machine_state import machine_state, warn_if_busy
from metrics.eval_coverage import load_reference, score_pair
from mlx_asr.audio import SAMPLE_RATE, load_audio_16k
from mlx_asr.backends import _split_for_qwen3, qwen3_decode
from mlx_asr.models import resolve

AUDIO_EXT = {".wav", ".m4a", ".mp3", ".flac", ".mp4", ".mov"}


def find_pairs(corpus: Path):
    """Match each audio file to its reference by stem, as run_corpus.py does."""
    pairs = []
    for audio in sorted(corpus.iterdir()):
        if audio.suffix.lower() not in AUDIO_EXT:
            continue
        for cand in (f"{audio.stem}_transcript.txt", f"{audio.stem}.srt",
                     f"{audio.stem}.vtt"):
            ref = corpus / cand
            if ref.exists():
                pairs.append((audio, ref))
                break
    return pairs


def decode_batched(loaded, audio, language, chunk_len, batch, log,
                   group_budget=False):
    """One file at one batch size, through the shipped path where possible.

    batch == 1 calls `qwen3_decode`, i.e. exactly what the CLI runs today, so the
    baseline arm is not a reimplementation. batch > 1 has to reach upstream's
    `generate(batch_size=)` directly, because mlx_asr's own loop is per-chunk by
    construction; that asymmetry IS the thing being measured, so it is recorded rather
    than hidden.
    """
    if batch == 1:
        return qwen3_decode(loaded, audio, language, chunk_len, log=log)

    windows = _split_for_qwen3(audio, chunk_len)
    duration = len(audio) / SAMPLE_RATE
    from mlx_asr.backends import MIN_CHUNK_MAX_TOKENS, TOKENS_PER_SECOND
    per_window = [max(MIN_CHUNK_MAX_TOKENS, int(len(c) / SAMPLE_RATE * TOKENS_PER_SECOND))
                  for c, _ in windows]

    if group_budget:
        # Feed upstream one GROUP at a time with a budget sized for that group only, so
        # a runaway chunk can overrun its own group and no further. This is the closest
        # a batched call can get to the per-window cap mlx_asr enforces at batch 1, and
        # it is the arm that answers "would batching be worth shipping if the budget
        # were fixed" rather than "is upstream's budget accounting broken".
        texts = []
        for i in range(0, len(windows), batch):
            group = windows[i:i + batch]
            joined = np.concatenate([c for c, _ in group])
            r = loaded.generate(joined, language=language, batch_size=batch,
                                chunk_duration=chunk_len,
                                max_tokens=sum(per_window[i:i + batch]),
                                verbose=False)
            texts.append((getattr(r, "text", "") or "").strip())
        text = " ".join(t for t in texts if t)
    else:
        # Upstream's own accounting: one budget for the whole file, decremented across
        # every group (`remaining_tokens -= group_tokens`).
        r = loaded.generate(audio, language=language, batch_size=batch,
                            chunk_duration=chunk_len, max_tokens=sum(per_window),
                            verbose=False)
        text = (getattr(r, "text", "") or "").strip()
    # Upstream returns one joined string, so per-window cues are not recoverable here.
    # Accuracy is scored on the full text either way (coverage CER over the file), so
    # this loses nothing that is being measured.
    return [(0.0, duration, text)], text, {"segments": len(windows),
                                           "budget_total": sum(per_window)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--batches", default="1,2,4,8,16")
    p.add_argument("--chunk-seconds", type=float, default=15.0)
    p.add_argument("--size", default="1.7B")
    p.add_argument("--quantization", default=None)
    p.add_argument("--language", default="ja")
    p.add_argument("--limit", type=int)
    p.add_argument("--group-budget", action="store_true",
                   help="cap tokens per GROUP instead of per file, which is "
                        "the closest a batched call gets to the per-window "
                        "cap the shipped batch-1 loop enforces")
    p.add_argument("--json", required=True)
    a = p.parse_args()

    state = machine_state()
    warn_if_busy(state)
    pairs = find_pairs(Path(a.corpus).expanduser())
    if a.limit:
        pairs = pairs[:a.limit]
    batches = [int(b) for b in a.batches.split(",")]

    spec = resolve("qwen3-asr", a.size)
    repo = spec.repo_for(a.quantization) if a.quantization else spec.repo
    print(f"{repo}, window {a.chunk_seconds:g}s, batches {batches}, "
          f"{len(pairs)} files", file=sys.stderr, flush=True)
    loaded = load_model(repo)

    results = {}
    for batch in batches:
        per_file, t0 = [], time.time()
        # Per arm, because this is a shared machine: a neighbour arriving between arms
        # would otherwise leave one x-realtime figure quietly incomparable to the rest.
        arm_state = machine_state()
        warn_if_busy(arm_state)
        mx.reset_peak_memory()
        for audio_path, ref_path in pairs:
            audio = load_audio_16k(str(audio_path))
            t = time.time()
            cues, text, meta = decode_batched(
                loaded, audio, a.language, a.chunk_seconds, batch,
                log=lambda *_: None, group_budget=a.group_budget)
            wall = time.time() - t
            dur = len(audio) / SAMPLE_RATE
            unit, sc = score_pair(load_reference(str(ref_path)), text)
            per_file.append({
                "file": audio_path.name, "duration_s": round(dur, 1),
                "wall_s": round(wall, 2), "x_realtime": round(dur / wall, 2),
                "unit": unit, "coverage_cer": sc["coverage_cer"],
                "ref_chars": sc["ref_chars"],
                "chars": len(text), "segments": meta.get("segments"),
                "runaway_segments": meta.get("runaway_segments"),
            })
            print(f"  B={batch} {audio_path.name}: {dur / wall:.1f}x "
                  f"{unit} {sc['coverage_cer']:.2%}",
                  file=sys.stderr, flush=True)

        # Length-weighted aggregate per unit, matching run_corpus.py: total charged
        # edits over total reference units, NOT a mean of per-file rates.
        agg = {}
        for unit in {r["unit"] for r in per_file}:
            rows = [r for r in per_file if r["unit"] == unit]
            agg[unit] = {
                "error_rate": round(
                    sum(r["coverage_cer"] * r["ref_chars"] for r in rows)
                    / max(sum(r["ref_chars"] for r in rows), 1), 6),
                "files": len(rows),
            }
        total_audio = sum(r["duration_s"] for r in per_file)
        total_wall = sum(r["wall_s"] for r in per_file)
        results[str(batch)] = {
            "batch": batch, "aggregate": agg,
            "x_realtime": round(total_audio / total_wall, 2),
            "wall_s": round(total_wall, 1),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            "total_chars": sum(r["chars"] for r in per_file),
            "machine_at_arm_start": {k: arm_state[k] for k in
                                     ("busy", "busy_reasons", "load_1min",
                                      "swapout_pages_per_s", "gpu_in_use_gb")},
            "per_file": per_file,
        }
        print(f"B={batch}: {results[str(batch)]['x_realtime']}x, "
              f"{agg}, peak {results[str(batch)]['peak_gb']}GB",
              file=sys.stderr, flush=True)
        Path(a.json).write_text(json.dumps(
            {"machine": state, "model": repo, "chunk_seconds": a.chunk_seconds,
             "language": a.language, "arms": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
