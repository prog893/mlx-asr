"""Baseline the same corpus with OpenAI Whisper (and Japanese-tuned variants).

The point of this script is comparability, not Whisper tuning. Every number it
produces is scored by the *same* functions `run_corpus.py` uses, on the *same*
16kHz mono audio, so a row here can be put next to a Voxtral row without an
asterisk:

  * coverage CER/WER from `eval_coverage.py`, since these references
    deliberately omit audio (see the module docstring there);
  * unit chosen per file from the reference script (CER for CJK, WER for
    space-delimited), never averaged across units;
  * length-weighted aggregation (total charged edits / total reference units);
  * x-realtime measured with the model already loaded, matching `run_corpus.py`,
    which loads before its loop. Model load and HF download are excluded.

Whisper is run at its library defaults (temperature fallback 0.0..1.0,
`condition_on_previous_text=True`), because that is what a user gets from
`mlx_whisper.transcribe`, and a default-vs-default comparison is the honest one.
`--greedy` pins temperature 0 for a determinism check.

Language handling is the one place this harness must make a real choice, because
Whisper takes a language token that Voxtral does not, and all three obvious
options are wrong on this corpus:

  force one code      `--language ja` pins the output language, so the two
                      English files score ~100% WER. Refused outright below.
  autodetect          Whisper decides from the first 30s. Measured on tiny: it
                      returned *Russian* for two Japanese files and produced
                      102-106% CER with extra_ratio 7.5-8.5 (hallucination
                      loops). This corpus really does contain Russian side
                      conversation, so a 30s window is not a reliable sample.
  per-file, from the  The default here, and what this sweep uses. Each file gets
  reference script    the language its own reference is written in: `ja` for CJK,
                      `en` for space-delimited. This is what a user who knows
                      what they recorded would pass, and it keeps a
                      language-detection failure from being scored as an ASR
                      error, which is not what is being compared here.

The per-file choice reuses `is_space_delimited` from eval_coverage.py, the same
function that picks the scoring unit, so language and metric can never disagree.
`--autodetect` restores Whisper's own detection for a zero-config data point.
The Japanese-only models (kotoba-*) are always forced to `ja` and, unless
`--all-files` is given, skip the English files instead of scoring gibberish.

    uv run --python .venv-whisper python scripts/benchmarks/run_whisper.py \\
        --corpus DIR --model mlx-community/whisper-large-v3-mlx --json out.json
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
from mlx_whisper.transcribe import transcribe

from benchmarks.machine_state import machine_state, warn_if_busy
from metrics.eval_coverage import is_space_delimited, load_reference, score_pair

SAMPLE_RATE = 16000
AUDIO_EXT = {".wav", ".WAV", ".m4a", ".mp3", ".flac", ".mp4", ".mov"}
# Models with a Japanese-only decoder: forcing another language yields garbage,
# so English files are skipped rather than scored.
JA_ONLY = ("kotoba",)


def find_pairs(corpus: Path):
    """Match each audio file to its transcript by stem (as run_corpus.py does)."""
    pairs = []
    for audio in sorted(corpus.iterdir()):
        if audio.suffix not in AUDIO_EXT:
            continue
        for cand in (corpus / f"{audio.stem}_transcript.txt",
                     corpus / f"{audio.stem}.srt", corpus / f"{audio.stem}.txt"):
            if cand.exists():
                pairs.append((audio, cand))
                break
    return pairs


def to_16k_mono(path: Path, tmpdir: Path) -> Path:
    """Reuse the exact cache directory run_corpus.py writes, so both benchmarks
    read byte-identical audio and no conversion difference can leak into CER."""
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
    p.add_argument("--corpus", help="directory of <stem>.wav + <stem>_transcript.txt")
    p.add_argument("--pair", nargs=2, action="append", metavar=("AUDIO", "REF"),
                   help="score a single explicit audio/reference pair; repeatable")
    p.add_argument("--model", required=True, help="HF repo of an mlx-whisper model")
    p.add_argument("--language", default=None,
                   help="force one language code for every file; refused on a "
                        "mixed-script set. Default: per-file from the reference")
    p.add_argument("--autodetect", action="store_true",
                   help="let Whisper detect from the first 30s. Unreliable on "
                        "this corpus (returned Russian for Japanese files)")
    p.add_argument("--greedy", action="store_true",
                   help="temperature 0.0 only, disabling the fallback ladder")
    p.add_argument("--no-condition", action="store_true",
                   help="condition_on_previous_text=False (loop mitigation)")
    p.add_argument("--min-cut", type=int, default=30)
    p.add_argument("--min-cut-words", type=int, default=6)
    p.add_argument("--limit", type=int, help="only the N shortest files")
    p.add_argument("--all-files", action="store_true",
                   help="run Japanese-only models on the English files too")
    p.add_argument("--label", default="")
    p.add_argument("--keep-hyp", help="directory to write hypotheses into")
    p.add_argument("--json")
    a = p.parse_args()

    pairs = []
    if a.corpus:
        corpus = Path(a.corpus).expanduser()
        pairs += find_pairs(corpus)
        if not pairs:
            print(f"no pairs found in {corpus}", file=sys.stderr)
            return 2
    for audio, ref in a.pair or []:
        pairs.append((Path(audio).expanduser(), Path(ref).expanduser()))

    tmpdir = Path(tempfile.gettempdir()) / "voxtral_corpus_16k"
    tmpdir.mkdir(exist_ok=True)
    hypdir = Path(a.keep_hyp) if a.keep_hyp else None
    if hypdir:
        hypdir.mkdir(parents=True, exist_ok=True)

    ja_only = any(k in a.model.lower() for k in JA_ONLY)

    prepared = []
    for audio, ref in pairs:
        wav = to_16k_mono(audio, tmpdir)
        dur = wav.stat().st_size / (SAMPLE_RATE * 2)
        # Language comes from the reference script, by the same test that picks
        # the scoring unit, so the two can never disagree.
        ref_is_en = is_space_delimited(load_reference(str(ref)))
        if a.autodetect:
            lang = None
        elif ja_only:
            lang = "ja"
        else:
            lang = a.language or ("en" if ref_is_en else "ja")
        prepared.append((audio.stem, wav, ref, dur, lang, ref_is_en))
    prepared.sort(key=lambda x: x[3])
    if a.limit:
        prepared = prepared[: a.limit]

    skipped = []
    if ja_only and not a.all_files:
        keep = []
        for row in prepared:
            if row[5]:  # space-delimited reference: not this model's language
                skipped.append(row[0])
            else:
                keep.append(row)
        prepared = keep

    # Guard the mixed-language footgun: forcing one code across a mixed set pins
    # the output language, so the other-language files score ~100% and the
    # aggregate reads as a model failure rather than a harness mistake.
    if a.language and not ja_only and len({r[5] for r in prepared}) > 1:
        print(f"refusing --language {a.language}: this set mixes CJK and "
              f"space-delimited references, so one forced code scores the "
              f"others near 100%. Omit --language to take each file's language "
              f"from its reference, or pass --autodetect.", file=sys.stderr)
        return 2

    label = a.label or a.model.split("/")[-1]
    opts = {"verbose": False}
    if a.greedy:
        opts["temperature"] = 0.0
    if a.no_condition:
        opts["condition_on_previous_text"] = False

    langs = sorted({r[4] or "auto" for r in prepared})
    total_audio = sum(d for _, _, _, d, _, _ in prepared)
    print(f"model:  {a.model}")
    print(f"corpus: {len(prepared)} files, {total_audio/3600:.2f}h audio"
          f"{'  (skipped non-CJK: %s)' % ', '.join(skipped) if skipped else ''}")
    print(f"opts:   language={'/'.join(langs)} "
          f"temperature={'0.0' if a.greedy else 'fallback'} "
          f"condition={not a.no_condition}\n")
    # Read before the weights load, so any GPU memory reported belongs to another
    # process. Recorded in every _dump below, because an x-realtime figure without the
    # machine's state is the one number this project has already had to throw away.
    state = machine_state()
    print(f"machine: {state['label']}, load {state['load_1min']}, "
          f"{state['gpu_in_use_gb']}GB GPU already in use")
    warn_if_busy(state)
    print(f"{'file':<26} {'u':>1} {'dur':>6} {'ref':>6} {'x rt':>6} {'cov':>6} "
          f"{'covERR':>7} {'plain':>7} {'extra':>6}")

    # Load once, outside the timing loop, so x-realtime is decode-only and
    # matches run_corpus.py. mlx-whisper has no separate load call, so a 1s
    # silent probe warms the weight cache instead.
    probe = tmpdir / "_warm.wav"
    if not probe.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        f"anullsrc=r={SAMPLE_RATE}:cl=mono", "-t", "1",
                        "-c:a", "pcm_s16le", str(probe), "-y"], check=True)
    t_load = time.perf_counter()
    transcribe(str(probe), path_or_hf_repo=a.model, language="ja", **opts)
    load_s = time.perf_counter() - t_load

    rows = []
    tot_wall = tot_dur = 0.0
    agg_ref = {"char": 0, "word": 0}
    agg_charged = {"char": 0.0, "word": 0.0}

    for stem, wav, refpath, dur, lang, _ in prepared:
        t0 = time.perf_counter()
        try:
            r = transcribe(str(wav), path_or_hf_repo=a.model, language=lang,
                           **opts)
        except Exception as e:
            print(f"{stem:<26} FAILED {type(e).__name__}: {str(e)[:40]}")
            rows.append({"file": stem, "error": f"{type(e).__name__}: {e}"[:200]})
            mx.clear_cache()
            continue
        wall = time.perf_counter() - t0

        hyp_text = r["text"]
        if hypdir:
            (hypdir / f"{stem}.{label}.txt").write_text(hyp_text + "\n",
                                                        encoding="utf-8")
        unit, s = score_pair(load_reference(str(refpath)), hyp_text,
                             a.min_cut, a.min_cut_words)

        agg_ref[unit] += s["ref_chars"]
        agg_charged[unit] += s["coverage_cer"] * s["ref_chars"]
        tot_wall += wall
        tot_dur += dur
        rows.append({"file": stem, "duration_s": round(dur, 1), "unit": unit,
                     "x_realtime": round(dur / wall, 1),
                     "requested_language": lang or "auto",
                     "detected_language": r.get("language"),
                     "segments": len(r.get("segments", [])),
                     **{k: v for k, v in s.items() if k != "excused_runs"},
                     "excused_run_count": len(s["excused_runs"])})
        print(f"{stem:<26} {unit[0]} {dur:>6.0f} {s['ref_chars']:>6} "
              f"{dur/wall:>6.1f} {s['coverage']*100:>5.1f}% "
              f"{s['coverage_cer']*100:>6.2f}% {s['cer']*100:>6.1f}% "
              f"{s['extra_ratio']:>6.2f}")
        mx.clear_cache()

        if a.json:  # write incrementally; these runs are long enough to interrupt
            _dump(a, label, agg_ref, agg_charged, rows, tot_dur, tot_wall, load_s,
                  langs, skipped, complete=False, expected=len(prepared),
                  machine=state)

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
          f"({tot_dur/3600:.2f}h audio in {tot_wall/60:.1f} min of decode; "
          f"model load {load_s:.0f}s excluded)")
    if a.json:
        _dump(a, label, agg_ref, agg_charged, rows, tot_dur, tot_wall, load_s,
              langs, skipped, complete=True, expected=len(prepared),
              machine=state)
        print(f"[saved] {a.json}")
    return 0


def _dump(a, label, agg_ref, agg_charged, rows, tot_dur, tot_wall, load_s,
          langs, skipped, complete=False, expected=None, machine=None):
    """Write the results JSON.

    Called after every file, so an in-progress file on disk looks exactly like a
    finished one apart from ``complete``. That flag is not cosmetic: a partial
    aggregate is a *plausible* number over a subset of the corpus, so without it a
    reader (or a script waiting on the file to appear) can quote a 7-file mean as
    if it were the 20-file result. Anything consuming these must check it.
    """
    summary = {u: round(agg_charged[u] / agg_ref[u], 5)
               for u in ("char", "word") if agg_ref[u]}
    with open(a.json, "w") as f:
        json.dump({"engine": "mlx-whisper", "config": vars(a), "label": label,
                   "complete": complete,
                   "machine": machine,
                   "files_scored": len(rows), "files_expected": expected,
                   "languages": langs, "skipped_files": skipped,
                   "aggregate": summary, "ref_units": agg_ref,
                   "x_realtime": round(tot_dur / max(tot_wall, 1e-9), 2),
                   "model_load_s": round(load_s, 1),
                   "results": rows}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
