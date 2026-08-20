"""Evaluate Qwen3-ASR across a corpus, scored by the same functions as every other row.

The point is comparability, not tuning. Every number here comes from
`eval_coverage.py`, on the same 16kHz mono audio `run_corpus.py` and `run_whisper.py`
read, with the unit chosen per file from the reference and aggregates length-weighted,
so a row from this script can sit next to a Voxtral or Whisper row without an asterisk.

Three things specific to this engine:

**One run is its score.** It decodes greedily (`temperature=0.0` becomes `mx.argmax`,
no fallback ladder), so repeating a config adds no information, exactly as with
Voxtral and unlike Whisper. `--repeat` exists only to *verify* that claim.

**Language is always explicit.** Autodetect corrupts multi-chunk transcripts upstream
(`language` is reassigned inside the chunk loop, leaving `language X<asr_text>`
embedded in later chunks' text), and an unmapped code is interpolated into the prompt
verbatim. Both are silent. Language is taken per file from the reference script, by
the same `is_space_delimited` test that picks the scoring unit, so the two can never
disagree; this matches what `run_whisper.py` does and for the same reason.

**Kana and lenient CER are reported beside coverage CER.** This model's Japanese
finetune advertises inverse text normalisation, so it will prefer `2018年` where a
reference typist may have written the reading out. Plain coverage CER charges each of
those as an error, and the gap to the kana figure is the size of that confound.

Peak GPU memory is recorded per file and overall. The audio encoder is not quantized
at any precision level and its attention mask is materialised densely, so this
engine's memory profile does not follow from its weight size the way the others do.

    uv run python scripts/benchmarks/run_qwen3.py --corpus DIR --json out.json \\
        [--model qwen3-asr] [--chunk-seconds 60] [--limit N]
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Both the repo root (for mlx_asr) and scripts/ (for metrics) have to be importable;
# this file lives two levels down from the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx
from mlx_audio.stt.utils import load as load_model

from benchmarks.machine_state import machine_state, warn_if_busy
from metrics.eval_coverage import is_space_delimited, load_reference, score_pair
from metrics.eval_coverage_kana import coverage_kana, coverage_lenient
from mlx_asr.audio import SAMPLE_RATE, load_audio_16k
from mlx_asr.backends import qwen3_decode, qwen3_language
from mlx_asr.models import resolve as resolve_model

AUDIO_EXT = {".wav", ".WAV", ".m4a", ".mp3", ".flac", ".mp4", ".mov"}


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
    """Reuse the exact cache directory run_corpus.py writes, so every engine here
    reads byte-identical audio and no conversion difference can leak into CER."""
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
    p.add_argument("--model", default="qwen3-asr",
                   help="registry alias or HF repo id of a Qwen3-ASR build")
    p.add_argument("--chunk-seconds", type=float, default=None,
                   help="decode window. Default: the registry's (30s, measured best "
                        "on this corpus), NOT the library's 1200s, at which a "
                        "sub-20-minute file is one chunk")
    p.add_argument("--min-cut", type=int, default=30)
    p.add_argument("--min-cut-words", type=int, default=6)
    p.add_argument("--limit", type=int, help="only the N shortest files")
    p.add_argument("--repeat", type=int, default=1,
                   help="decode each file N times and report whether the text is "
                        "identical. Greedy, so >1 is a determinism CHECK, not a "
                        "distribution")
    p.add_argument("--label", default="")
    p.add_argument("--keep-hyp", help="directory to write hypotheses into")
    p.add_argument("--json")
    p.add_argument("--require-idle", action="store_true",
                   help="refuse to start unless machine_state reports busy=false, "
                        "instead of warning and running anyway")
    p.add_argument("--wait-for-idle", type=float, default=0.0, metavar="MIN",
                   help="with --require-idle, poll for up to this many minutes "
                        "waiting for the host to go quiet before giving up")
    a = p.parse_args()

    spec = resolve_model(a.model)
    if spec.backend != "mlx-qwen3":
        print(f"--model {a.model} resolves to backend {spec.backend}, not "
              f"mlx-qwen3. Use run_corpus.py (voxtral) or run_whisper.py.",
              file=sys.stderr)
        return 2
    chunk_len = a.chunk_seconds or spec.opts.get("chunk_length_s", 30.0)

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

    # Convert first, so ordering by duration is known before a model is loaded.
    prepared = []
    for audio, ref in pairs:
        wav = to_16k_mono(audio, tmpdir)
        dur = wav.stat().st_size / (SAMPLE_RATE * 2)
        ref_is_en = is_space_delimited(load_reference(str(ref)))
        prepared.append((audio.stem, wav, ref, dur, ref_is_en))
    prepared.sort(key=lambda x: x[3])
    if a.limit:
        prepared = prepared[: a.limit]

    label = a.label or f"{spec.alias}_c{chunk_len:.0f}"
    total_audio = sum(r[3] for r in prepared)
    # Read before the weights load, so any GPU memory reported belongs to another
    # process. Recorded in the JSON too: an x-realtime figure without the machine's
    # state is the one number this project has already had to throw away.
    state = machine_state()
    print(f"model:  {spec.label} ({spec.repo})")
    print(f"corpus: {len(prepared)} files, {total_audio/3600:.2f}h audio")
    print(f"config: {label}, window {chunk_len:g}s, "
          f"{a.repeat} run{'s' if a.repeat > 1 else ''} per file")
    print(f"machine: {state['label']}, load {state['load_1min']}, "
          f"{state['gpu_in_use_gb']}GB GPU already in use")
    warn_if_busy(state)
    # A sweep's whole output is a comparison of throughput across configs, so one
    # contaminated arm does not just add noise, it reverses the ranking. Refusing is
    # opt-in because the accuracy numbers in a busy run are still valid (greedy
    # decode), so the call belongs to whoever launched it.
    if a.require_idle and state["busy"]:
        deadline = time.monotonic() + a.wait_for_idle * 60
        while state["busy"] and time.monotonic() < deadline:
            # flush: this runs under nohup with a redirected stdout, where print is
            # block-buffered, and a wait with no visible progress is indistinguishable
            # from a hang.
            print(f"waiting for an idle host ({'; '.join(state['busy_reasons'])})",
                  flush=True)
            time.sleep(30)
            state = machine_state()
        if state["busy"]:
            print(f"refusing to start: {'; '.join(state['busy_reasons'])}",
                  file=sys.stderr)
            return 2
        print(f"host went idle (load {state['load_1min']}, "
              f"{state['gpu_in_use_gb']}GB GPU)")

    t_load = time.perf_counter()
    model = load_model(spec.repo)
    load_s = time.perf_counter() - t_load
    print(f"loaded in {load_s:.1f}s\n")

    print(f"{'file':<26} {'u':>1} {'dur':>6} {'ref':>6} {'x rt':>6} {'cov':>6} "
          f"{'covERR':>7} {'kana':>7} {'lenient':>8} {'seg':>4} {'peakGB':>7} "
          f"{'aud':>5} {'loop':>4}")

    rows = []
    tot_wall = tot_dur = 0.0
    agg_ref = {"char": 0, "word": 0}
    agg_charged = {"char": 0.0, "word": 0.0}
    # Kana units are a different denominator again, so they aggregate separately.
    agg_kana_ref = agg_kana_charged = 0.0
    agg_len_charged = 0.0
    agg_len_ref = 0

    for stem, wav, refpath, dur, ref_is_en in prepared:
        audio = load_audio_16k(str(wav))
        dur = len(audio) / SAMPLE_RATE
        # Per file from the reference script, by the same function that picks the
        # scoring unit, so a language-detection failure is never scored as an ASR
        # error and the two choices cannot disagree.
        lang = qwen3_language(model, spec, "en" if ref_is_en else "ja",
                             log=lambda *x: None)

        mx.reset_peak_memory()
        texts, walls = [], []
        failed = None
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            try:
                _, text, meta = qwen3_decode(model, audio, lang, chunk_len,
                                             log=lambda *x: None)
            except Exception as e:                     # noqa: BLE001
                failed = f"{type(e).__name__}: {e}"[:200]
                break
            walls.append(time.perf_counter() - t0)
            texts.append(text)
        peak_gb = mx.get_peak_memory() / 1e9
        if failed:
            print(f"{stem:<26} FAILED {failed[:48]}")
            rows.append({"file": stem, "error": failed})
            mx.clear_cache()
            continue

        wall = walls[0]
        hyp_text = texts[0]
        if hypdir:
            (hypdir / f"{stem}.{label}.txt").write_text(hyp_text + "\n",
                                                        encoding="utf-8")
        ref_text = load_reference(str(refpath))
        unit, s = score_pair(ref_text, hyp_text, a.min_cut, a.min_cut_words)
        # The two reading-based metrics are SUPPLEMENTARY, so a failure in one must not
        # discard a decode that already cost minutes of GPU time. This is not a
        # hypothetical: a 25-digit run in a real hypothesis raised IndexError inside the
        # number reader and killed a sweep arm four files in. The coverage CER above is
        # deliberately outside the guard, because that one IS the result.
        try:
            kana = coverage_kana(ref_text, hyp_text, a.min_cut)
            lenient = coverage_lenient(ref_text, hyp_text, a.min_cut)
        except Exception as e:                        # noqa: BLE001
            kana = lenient = None
            print(f"  [{stem}] kana/lenient scoring failed "
                  f"({type(e).__name__}: {e}); coverage CER above is unaffected",
                  file=sys.stderr, flush=True)

        agg_ref[unit] += s["ref_chars"]
        agg_charged[unit] += s["coverage_cer"] * s["ref_chars"]
        if kana:
            agg_kana_ref += kana["ref_chars"]
            agg_kana_charged += kana["coverage_cer"] * kana["ref_chars"]
        if lenient:
            agg_len_ref += lenient["ref_chars"]
            agg_len_charged += lenient["lenient_cer"] * lenient["ref_chars"]
        tot_wall += wall
        tot_dur += dur

        row = {"file": stem, "duration_s": round(dur, 1), "unit": unit,
               "x_realtime": round(dur / wall, 1),
               "requested_language": lang,
               # "forced", never "detected": this engine's autodetect is unusable
               # through this library, so nothing here detected a language.
               "language_source": meta["language_source"],
               "segments": meta["segments"],
               "cue_source": meta["cue_source"],
               "peak_memory_gb": round(peak_gb, 2),
               # Both are decode-health facts, not accuracy: a file whose budget ran
               # out has a CER dominated by the missing tail, and one with a
               # repetition loop has a CER dominated by the loop. Recorded per row so
               # such a file can be excluded from an aggregate rather than quietly
               # inflating it. See backends.qwen3_decode.
               "audio_coverage": meta["audio_coverage"],
               "runaway_segments": meta["runaway_segments"],
               "empty_segments": meta["empty_segments"],
               "max_tokens_per_chunk": meta["max_tokens_per_chunk"],
               **{k: v for k, v in s.items() if k != "excused_runs"},
               "excused_run_count": len(s["excused_runs"])}
        if kana:
            row["kana_coverage_cer"] = round(kana["coverage_cer"], 5)
            row["kana_ref_chars"] = kana["ref_chars"]
        if lenient:
            row["lenient_coverage_cer"] = round(lenient["lenient_cer"], 5)
            row["lenient_forgiven"] = lenient["forgiven"]
        if a.repeat > 1:
            # The determinism check. Greedy decoding should give byte-identical
            # text; if it does not, every single-run number in this project's
            # Qwen3 rows would need a distribution instead.
            row["repeat_runs"] = len(texts)
            row["repeat_identical"] = all(t == texts[0] for t in texts)
            row["repeat_x_realtime"] = [round(dur / w, 1) for w in walls]
        rows.append(row)

        print(f"{stem:<26} {unit[0]} {dur:>6.0f} {s['ref_chars']:>6} "
              f"{dur/wall:>6.1f} {s['coverage']*100:>5.1f}% "
              f"{s['coverage_cer']*100:>6.2f}% "
              f"{(kana['coverage_cer']*100 if kana else float('nan')):>6.2f}% "
              f"{(lenient['lenient_cer']*100 if lenient else float('nan')):>7.2f}% "
              f"{meta['segments']:>4} {peak_gb:>7.2f} "
              f"{meta['audio_coverage']*100:>4.0f}% {meta['runaway_segments']:>4}"
              + ("" if a.repeat == 1
                 else ("  identical" if row["repeat_identical"] else "  DIFFERS")),
              flush=True)
        mx.clear_cache()

        if a.json:   # write incrementally; these runs are long enough to interrupt
            _dump(a, spec, label, chunk_len, agg_ref, agg_charged, agg_kana_ref,
                  agg_kana_charged, agg_len_ref, agg_len_charged, rows, tot_dur,
                  tot_wall, load_s, state, complete=False, expected=len(prepared))

    print()
    for unit in ("char", "word"):
        if agg_ref[unit]:
            v = agg_charged[unit] / agg_ref[unit]
            m = "coverageCER" if unit == "char" else "coverageWER"
            n = sum(1 for r in rows if r.get("unit") == unit)
            print(f"aggregate {m} over {n} {unit}-unit files "
                  f"({agg_ref[unit]} ref {unit}s, length-weighted): {v*100:.2f}%")
    if agg_kana_ref:
        print(f"aggregate coverage kanaCER    ({agg_kana_ref:.0f} kana): "
              f"{100 * agg_kana_charged / agg_kana_ref:.2f}%")
    if agg_len_ref:
        print(f"aggregate coverage lenientCER ({agg_len_ref} ref chars): "
              f"{100 * agg_len_charged / agg_len_ref:.2f}%")
    # Named, not buried. An aggregate containing a file that stopped at 2% of its
    # audio is not an accuracy figure for this engine; it is mostly a measure of the
    # missing tail. Whoever reads the number has to be told which files are in it.
    truncated = [r["file"] for r in rows if r.get("audio_coverage", 1) < 0.99]
    looped = [r["file"] for r in rows if r.get("runaway_segments")]
    if truncated:
        print(f"\nWARNING: {len(truncated)} of {len(rows)} files were TRUNCATED "
              f"(the model's token budget ran out mid-file): "
              f"{', '.join(truncated)}. Their CER is dominated by the missing tail, "
              f"so the aggregates above are not accuracy figures for this engine.")
    if looped:
        print(f"WARNING: {len(looped)} of {len(rows)} files contain repetition "
              f"loops: {', '.join(looped)}.")
    peaks = [r["peak_memory_gb"] for r in rows if "peak_memory_gb" in r]
    if peaks:
        print(f"peak GPU memory: {max(peaks):.2f}GB")
    print(f"throughput: {tot_dur/max(tot_wall,1e-9):.1f}x realtime "
          f"({tot_dur/3600:.2f}h audio in {tot_wall/60:.1f} min of decode; "
          f"model load {load_s:.0f}s excluded)")
    if a.json:
        _dump(a, spec, label, chunk_len, agg_ref, agg_charged, agg_kana_ref,
              agg_kana_charged, agg_len_ref, agg_len_charged, rows, tot_dur,
              tot_wall, load_s, state, complete=True, expected=len(prepared))
        print(f"[saved] {a.json}")
    return 0


def _dump(a, spec, label, chunk_len, agg_ref, agg_charged, kana_ref, kana_charged,
          len_ref, len_charged, rows, tot_dur, tot_wall, load_s, machine,
          complete=False, expected=None):
    """Write the results JSON.

    Called after every file, so an in-progress file on disk looks exactly like a
    finished one apart from ``complete``. That flag is not cosmetic: a partial
    aggregate is a plausible number over a subset of the corpus, so without it a
    reader can quote a 7-file mean as the 20-file result. Anything consuming these
    must check it; `summarize_engines.py` and `compare_engines.py` do.
    """
    summary = {u: round(agg_charged[u] / agg_ref[u], 5)
               for u in ("char", "word") if agg_ref[u]}
    if kana_ref:
        summary["char_kana"] = round(kana_charged / kana_ref, 5)
    if len_ref:
        summary["char_lenient"] = round(len_charged / len_ref, 5)
    peaks = [r["peak_memory_gb"] for r in rows if "peak_memory_gb" in r]
    with open(a.json, "w") as f:
        json.dump({"engine": "mlx-qwen3", "config": vars(a), "label": label,
                   "model": spec.repo, "chunk_seconds": chunk_len,
                   "complete": complete,
                   "machine": machine,
                   "files_scored": len(rows), "files_expected": expected,
                   "aggregate": summary, "ref_units": agg_ref,
                   "kana_ref_chars": round(kana_ref), "lenient_ref_chars": len_ref,
                   "x_realtime": round(tot_dur / max(tot_wall, 1e-9), 2),
                   "peak_memory_gb": round(max(peaks), 2) if peaks else None,
                   # Anything reading `aggregate` has to be able to see that some
                   # files did not finish; the number looks perfectly ordinary.
                   "truncated_files": [r["file"] for r in rows
                                       if r.get("audio_coverage", 1) < 0.99],
                   "looped_files": [r["file"] for r in rows
                                    if r.get("runaway_segments")],
                   "model_load_s": round(load_s, 1),
                   # Not a subtitle-capable engine. Recorded in every result file so
                   # a timing figure can never be derived from these cues by mistake.
                   "cue_source": "chunk_boundaries",
                   "results": rows}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
