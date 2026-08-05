"""Evaluate one config across a whole corpus of audio/transcript pairs.

This is what lifts the project off n=1. It expects a directory of pairs named
``<stem>.<audio-ext>`` plus ``<stem>_transcript.txt`` (or ``<stem>.srt``), and it
reports per-file and aggregate numbers.

Three things it handles that the single-clip scripts did not:

1. **Edited references.** Some transcripts deliberately omit audio (off-topic
   chatter, other-language side conversation, music in breaks). Plain CER is
   meaningless there -- it reads as a huge insertion rate for a correct
   transcription -- so the coverage-aware metric is used instead. See
   eval_coverage.py.
2. **Speaker labels.** Multi-speaker references carry "Name:" turn markers, which
   are diarization metadata rather than speech and must not be scored.
3. **Mixed languages.** Unit is chosen per file from the reference: WER for
   space-delimited scripts, CER for CJK. Char and word rates are aggregated
   separately, since they are not comparable quantities.

Aggregates are length-weighted (total charged edits / total reference units), not
a mean of per-file rates, so a 13-minute file does not swing the result as much as
a 93-minute one.

    uv run python scripts/benchmarks/run_corpus.py --corpus DIR --json out.json \\
        [--chunk-seconds 60] [--max-batch 16] [--overlap-seconds 0] [--limit N]
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Both the repo root (for mlx_asr) and scripts/ (for metrics) have to be
# importable; this file lives two levels down from the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx
from mlx_audio.stt.utils import load as load_model

from benchmarks.machine_state import machine_state, warn_if_busy
from metrics.eval_coverage import load_reference, score_pair
from mlx_asr.audio import SAMPLE_RATE, load_audio_16k, split_with_overlap
from mlx_asr.decode import transcribe_batch
from mlx_asr.text import transcript_text

DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
AUDIO_EXT = {".wav", ".WAV", ".m4a", ".mp3", ".flac", ".mp4", ".mov"}


def find_pairs(corpus: Path):
    """Match each audio file to its transcript, by stem."""
    pairs = []
    for audio in sorted(corpus.iterdir()):
        if audio.suffix not in AUDIO_EXT:
            continue
        stem = audio.stem
        for cand in (corpus / f"{stem}_transcript.txt", corpus / f"{stem}.srt",
                     corpus / f"{stem}.txt"):
            if cand.exists():
                pairs.append((audio, cand))
                break
    return pairs


def to_16k_mono(path: Path, tmpdir: Path) -> Path:
    """Downmix/resample via ffmpeg when the source is not already 16kHz mono.

    The loader can resample in-process, but these sources are 96kHz multi-GB
    WAVs; letting ffmpeg stream the conversion avoids holding the raw file in
    memory.
    """
    out = tmpdir / (path.stem + ".16k.wav")
    if out.exists():
        return out
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar",
         str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out), "-y"],
        check=True,
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--chunk-seconds", type=float, default=60.0)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--delay-ms", type=int, default=2400)
    p.add_argument("--kv-bits", type=int, default=8)
    p.add_argument("--overlap-seconds", type=float, default=0.0)
    p.add_argument("--min-cut", type=int, default=30,
                   help="char-unit insertion run length treated as cut audio")
    p.add_argument("--min-cut-words", type=int, default=6,
                   help="same threshold for the word-unit path")
    p.add_argument("--limit", type=int, help="only the N shortest files")
    p.add_argument("--prompt", default="",
                   help="vocabulary/topic bias applied to every file; also used to "
                        "test that an INSTRUCTION here is harmful")
    p.add_argument("--label", default="", help="name for this config in the output")
    p.add_argument("--keep-hyp", help="directory to write hypotheses into")
    p.add_argument("--json")
    a = p.parse_args()

    corpus = Path(a.corpus).expanduser()
    pairs = find_pairs(corpus)
    if not pairs:
        print(f"no audio/transcript pairs found in {corpus}", file=sys.stderr)
        return 2

    tmpdir = Path(tempfile.gettempdir()) / "voxtral_corpus_16k"
    tmpdir.mkdir(exist_ok=True)
    hypdir = Path(a.keep_hyp) if a.keep_hyp else None
    if hypdir:
        hypdir.mkdir(parents=True, exist_ok=True)

    # Convert first so the ordering by duration is known before loading a model.
    prepared = []
    for audio, ref in pairs:
        wav = to_16k_mono(audio, tmpdir)
        dur = wav.stat().st_size / (SAMPLE_RATE * 2)
        prepared.append((audio.stem, wav, ref, dur))
    prepared.sort(key=lambda x: x[3])
    if a.limit:
        prepared = prepared[: a.limit]

    label = a.label or (f"c{a.chunk_seconds:.0f}b{a.max_batch}"
                        f"{'_ov%g' % a.overlap_seconds if a.overlap_seconds else ''}"
                        f"{'_kv%d' % a.kv_bits if a.kv_bits else ''}")
    total_audio = sum(d for _, _, _, d in prepared)
    # Read before the model loads, so any GPU memory reported belongs to something
    # else. Warned about here and recorded in the JSON below: an x-realtime figure is
    # a property of a machine in a state, and this project has already had to void a
    # whole session because the state was not on the page.
    state = machine_state()
    print(f"machine: {state['label']}, load {state['load_1min']}, "
          f"{state['gpu_in_use_gb']}GB GPU already in use")
    warn_if_busy(state)
    print(f"corpus: {len(prepared)} files, {total_audio/3600:.2f}h audio")
    print(f"config: {label} (delay {a.delay_ms}ms)\n")
    print(f"{'file':<26} {'u':>1} {'dur':>6} {'ref':>6} {'x rt':>6} {'cov':>6} "
          f"{'covERR':>7} {'plain':>7} {'extra':>6}")

    model = load_model(a.model)
    prompt_ids = None
    if a.prompt:
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
        from mlx_audio.stt.utils import get_model_path

        tek = Tekkenizer.from_file(str(get_model_path(a.model) / "tekken.json"))
        prompt_ids = tek.encode(a.prompt, bos=False, eos=False)
    rows = []
    tot_wall = tot_dur = 0
    # Char and word error rates are different units; aggregate them separately
    # and never average across them.
    agg_ref = {"char": 0, "word": 0}
    agg_charged = {"char": 0.0, "word": 0.0}
    for stem, wav, refpath, dur in prepared:
        audio = load_audio_16k(str(wav))
        dur = len(audio) / SAMPLE_RATE
        chunks, offsets, warmup = split_with_overlap(
            audio, target_s=a.chunk_seconds, overlap_s=a.overlap_seconds
        )
        t0 = time.perf_counter()
        timed_all = []
        try:
            for i in range(0, len(chunks), a.max_batch):
                sl = slice(i, i + a.max_batch)
                timed, _ = transcribe_batch(
                    model, chunks[sl], a.delay_ms, prompt_ids, a.kv_bits,
                    log=lambda *x: None, warmup_tokens=warmup[sl],
                )
                timed_all.extend(timed)
        except Exception as e:
            print(f"{stem:<26} FAILED {type(e).__name__}: {str(e)[:40]}")
            rows.append({"file": stem, "error": f"{type(e).__name__}: {e}"[:200]})
            mx.clear_cache()
            continue
        wall = time.perf_counter() - t0

        hyp_text = transcript_text(model, timed_all)
        if hypdir:
            (hypdir / f"{stem}.{label}.txt").write_text(hyp_text + "\n",
                                                        encoding="utf-8")
        # Unit is chosen per file from the reference script: WER for
        # space-delimited languages, CER for CJK. A corpus can mix both.
        unit, s = score_pair(load_reference(str(refpath)), hyp_text,
                             a.min_cut, a.min_cut_words)

        agg_ref[unit] += s["ref_chars"]
        agg_charged[unit] += s["coverage_cer"] * s["ref_chars"]
        tot_wall += wall
        tot_dur += dur
        rows.append({"file": stem, "duration_s": round(dur, 1), "unit": unit,
                     "x_realtime": round(dur / wall, 1), "chunks": len(chunks),
                     **{k: v for k, v in s.items() if k != "excused_runs"},
                     "excused_run_count": len(s["excused_runs"])})
        print(f"{stem:<26} {unit[0]} {dur:>6.0f} {s['ref_chars']:>6} "
              f"{dur/wall:>6.1f} {s['coverage']*100:>5.1f}% "
              f"{s['coverage_cer']*100:>6.2f}% {s['cer']*100:>6.1f}% "
              f"{s['extra_ratio']:>6.2f}")
        mx.clear_cache()

    summary = {}
    print()
    for unit in ("char", "word"):
        if agg_ref[unit]:
            v = agg_charged[unit] / agg_ref[unit]
            summary[unit] = round(v, 5)
            m = "coverageCER" if unit == "char" else "coverageWER"
            n = sum(1 for r in rows if r.get("unit") == unit)
            print(f"aggregate {m} over {n} {unit}-unit files "
                  f"({agg_ref[unit]} ref {unit}s, length-weighted): {v*100:.2f}%")
    print(f"throughput: {tot_dur/max(tot_wall,1e-9):.1f}x realtime "
          f"({tot_dur/3600:.2f}h audio in {tot_wall/60:.1f} min of decode)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"config": vars(a), "label": label,
                       "machine": state,
                       "aggregate": summary,
                       "ref_units": agg_ref,
                       "x_realtime": round(tot_dur / max(tot_wall, 1e-9), 2),
                       "results": rows}, f, indent=2, ensure_ascii=False)
        print(f"[saved] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
