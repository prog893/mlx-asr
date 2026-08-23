"""Evaluate ReazonSpeech k2-v2 across a corpus, scored by the same functions as every other row.

Same contract as the other runners: `eval_coverage.py`, same cached 16kHz mono
audio, unit chosen per file from the reference, length-weighted aggregates. The
decode goes through ``backends.reazon_k2_decode``, the exact function the CLI
runs.

Engine facts this script relies on:

**One run is its score.** The transducer decodes greedily; a repeat adds nothing.

**CPU engine.** sherpa-onnx runs the authors' ONNX files on CPU, so there is no
GPU peak to record and the x-realtime figure is a CPU figure. It is comparable
to nothing else in this project's tables and is reported for completeness only.
Peak process RSS is recorded at the end instead of GPU memory.

**No language input exists.** Japanese-only weights with no language token;
English-reference files are skipped and named (the kotoba rule).

    uv run python scripts/benchmarks/run_reazon.py --corpus DIR --json out.json \
        [--chunk-seconds 30] [--precision int8] [--limit N]
"""

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.machine_state import machine_state, warn_if_busy
from metrics.eval_coverage import is_space_delimited, load_reference, score_pair
from metrics.eval_coverage_kana import coverage_kana, coverage_lenient
from mlx_asr.audio import SAMPLE_RATE, load_audio_16k
from mlx_asr.backends import reazon_k2_decode, reazon_k2_load
from mlx_asr.models import resolve as resolve_model

AUDIO_EXT = {".wav", ".WAV", ".m4a", ".mp3", ".flac", ".mp4", ".mov"}


def find_pairs(corpus: Path):
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
    p.add_argument("--model", default="reazon",
                   help="the family name; the k2 build is its only entry")
    p.add_argument("--precision", choices=("int8", "fp32"), default="fp32",
                   help="which ONNX build inside the repo. fp32 is the default: "
                        "measured on this corpus, int8 drops whole phrases "
                        "(296 vs 376 chars on one file)")
    p.add_argument("--chunk-seconds", type=float, default=None,
                   help="window length. Default: the registry's 30s, unswept")
    p.add_argument("--min-cut", type=int, default=30)
    p.add_argument("--min-cut-words", type=int, default=6)
    p.add_argument("--limit", type=int)
    p.add_argument("--label", default="")
    p.add_argument("--keep-hyp", help="directory to write hypotheses into")
    p.add_argument("--json")
    p.add_argument("--require-idle", action="store_true",
                   help="refuse to start unless machine_state reports busy=false. "
                        "A CPU decode contends differently from a GPU one but a "
                        "busy host still corrupts throughput")
    p.add_argument("--wait-for-idle", type=float, default=0.0, metavar="MIN")
    a = p.parse_args()

    spec = resolve_model(a.model)
    if spec.backend != "sherpa-onnx":
        print(f"--model {a.model} resolves to backend {spec.backend}, not "
              f"sherpa-onnx.", file=sys.stderr)
        return 2
    chunk_len = a.chunk_seconds or spec.opts.get("chunk_length_s", 30.0)

    try:
        import sherpa_onnx
    except ImportError:
        print("reazon-k2 needs sherpa-onnx: uv sync --extra reazon",
              file=sys.stderr)
        return 2

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

    prepared = []
    skipped = []
    for audio, ref in pairs:
        wav = to_16k_mono(audio, tmpdir)
        dur = wav.stat().st_size / (SAMPLE_RATE * 2)
        if is_space_delimited(load_reference(str(ref))):
            skipped.append(audio.stem)
            continue
        prepared.append((audio.stem, wav, ref, dur))
    prepared.sort(key=lambda x: x[3])
    if a.limit:
        prepared = prepared[: a.limit]

    label = a.label or f"{spec.alias}_{a.precision}_c{chunk_len:.0f}"
    total_audio = sum(r[3] for r in prepared)
    state = machine_state()
    print(f"model:  {spec.label} ({spec.repo}, {a.precision})")
    print(f"corpus: {len(prepared)} files, {total_audio/3600:.2f}h audio"
          f"{'  (skipped non-JP: %s)' % ', '.join(skipped) if skipped else ''}")
    print(f"config: {label}, window {chunk_len:g}s, CPU decode")
    print(f"machine: {state['label']}, load {state['load_1min']}, "
          f"{state['gpu_in_use_gb']}GB GPU already in use")
    warn_if_busy(state)
    if a.require_idle and state["busy"]:
        deadline = time.monotonic() + a.wait_for_idle * 60
        while state["busy"] and time.monotonic() < deadline:
            print(f"waiting for an idle host ({'; '.join(state['busy_reasons'])})",
                  flush=True)
            time.sleep(30)
            state = machine_state()
        if state["busy"]:
            print(f"refusing to start: {'; '.join(state['busy_reasons'])}",
                  file=sys.stderr)
            return 2
        print(f"host went idle (load {state['load_1min']})")

    t_load = time.perf_counter()
    weights_dir = reazon_k2_load(spec.repo, a.precision)
    suffix = ".int8" if a.precision == "int8" else ""
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(Path(weights_dir) / f"encoder-epoch-99-avg-1{suffix}.onnx"),
        decoder=str(Path(weights_dir) / f"decoder-epoch-99-avg-1{suffix}.onnx"),
        joiner=str(Path(weights_dir) / f"joiner-epoch-99-avg-1{suffix}.onnx"),
        tokens=str(Path(weights_dir) / "tokens.txt"),
        num_threads=4,
    )
    load_s = time.perf_counter() - t_load
    print(f"loaded in {load_s:.1f}s\n")

    print(f"{'file':<26} {'dur':>6} {'ref':>6} {'x rt':>6} {'cov':>6} "
          f"{'covERR':>7} {'kana':>7} {'lenient':>8} {'seg':>4} {'win':>4}")

    rows = []
    tot_wall = tot_dur = 0.0
    agg_ref = {"char": 0, "word": 0}
    agg_charged = {"char": 0.0, "word": 0.0}
    agg_kana_ref = agg_kana_charged = 0.0
    agg_len_charged = 0.0
    agg_len_ref = 0

    for stem, wav, refpath, dur in prepared:
        audio = load_audio_16k(str(wav))
        dur = len(audio) / SAMPLE_RATE
        t0 = time.perf_counter()
        try:
            cues, hyp_text, meta = reazon_k2_decode(recognizer, audio, chunk_len,
                                                    log=lambda *x: None)
        except Exception as e:                     # noqa: BLE001
            print(f"{stem:<26} FAILED {type(e).__name__}: {str(e)[:48]}",
                  flush=True)
            rows.append({"file": stem, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        wall = time.perf_counter() - t0

        if hypdir:
            (hypdir / f"{stem}.{label}.txt").write_text(hyp_text + "\n",
                                                        encoding="utf-8")
        ref_text = load_reference(str(refpath))
        unit, s = score_pair(ref_text, hyp_text, a.min_cut, a.min_cut_words)
        try:
            kana = coverage_kana(ref_text, hyp_text, a.min_cut)
            lenient = coverage_lenient(ref_text, hyp_text, a.min_cut)
        except Exception as e:                    # noqa: BLE001
            kana = lenient = None
            print(f"  [{stem}] kana/lenient scoring failed ({e}); coverage "
                  f"above is unaffected", file=sys.stderr, flush=True)

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
               "language_source": meta["language_source"],
               "segments": meta["segments"],
               "cue_source": meta["cue_source"],
               "token_count": meta["token_count"],
               "windows": meta["windows"],
               "empty_windows": meta["empty_windows"],
               **{k: v for k, v in s.items() if k != "excused_runs"},
               "excused_run_count": len(s["excused_runs"])}
        if kana:
            row["kana_coverage_cer"] = round(kana["coverage_cer"], 5)
            row["kana_ref_chars"] = kana["ref_chars"]
        if lenient:
            row["lenient_coverage_cer"] = round(lenient["lenient_cer"], 5)
            row["lenient_forgiven"] = lenient["forgiven"]
        rows.append(row)

        print(f"{stem:<26} {dur:>6.0f} {s['ref_chars']:>6} "
              f"{dur/wall:>6.1f} {s['coverage']*100:>5.1f}% "
              f"{s['coverage_cer']*100:>6.2f}% "
              f"{(kana['coverage_cer']*100 if kana else float('nan')):>6.2f}% "
              f"{(lenient['lenient_cer']*100 if lenient else float('nan')):>7.2f}% "
              f"{meta['segments']:>4} {meta['windows']:>4}", flush=True)

        if a.json:
            _dump(a, spec, label, chunk_len, agg_ref, agg_charged, agg_kana_ref,
                  agg_kana_charged, agg_len_ref, agg_len_charged, rows, tot_dur,
                  tot_wall, load_s, state, complete=False,
                  expected=len(prepared), skipped=skipped)

    print()
    for u in ("char", "word"):
        if agg_ref[u]:
            v = agg_charged[u] / agg_ref[u]
            m = "coverageCER" if u == "char" else "coverageWER"
            n = sum(1 for r in rows if r.get("unit") == u)
            print(f"aggregate {m} over {n} {u}-unit files "
                  f"({agg_ref[u]} ref {u}s, length-weighted): {v*100:.2f}%")
    if agg_kana_ref:
        print(f"aggregate coverage kanaCER    ({agg_kana_ref:.0f} kana): "
              f"{100 * agg_kana_charged / agg_kana_ref:.2f}%")
    if agg_len_ref:
        print(f"aggregate coverage lenientCER ({agg_len_ref} ref chars): "
              f"{100 * agg_len_charged / agg_len_ref:.2f}%")
    # macOS ru_maxrss is bytes (Linux reports KB); this engine is mac-only.
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"peak process RSS: {rss_gb:.2f}GB (CPU engine; no GPU memory)")
    print(f"throughput: {tot_dur/max(tot_wall,1e-9):.1f}x realtime "
          f"({tot_dur/3600:.2f}h audio in {tot_wall/60:.1f} min of decode; "
          f"model load {load_s:.0f}s excluded)")
    if a.json:
        _dump(a, spec, label, chunk_len, agg_ref, agg_charged, agg_kana_ref,
              agg_kana_charged, agg_len_ref, agg_len_charged, rows, tot_dur,
              tot_wall, load_s, state, complete=True, expected=len(prepared),
              skipped=skipped, rss_gb=rss_gb)
        print(f"[saved] {a.json}")
    return 0


def _dump(a, spec, label, chunk_len, agg_ref, agg_charged, kana_ref, kana_charged,
          len_ref, len_charged, rows, tot_dur, tot_wall, load_s, machine,
          complete=False, expected=None, skipped=None, rss_gb=None):
    summary = {u: round(agg_charged[u] / agg_ref[u], 5)
               for u in ("char", "word") if agg_ref[u]}
    if kana_ref:
        summary["char_kana"] = round(kana_charged / kana_ref, 5)
    if len_ref:
        summary["char_lenient"] = round(len_charged / len_ref, 5)
    with open(a.json, "w") as f:
        json.dump({"engine": "sherpa-onnx", "config": vars(a), "label": label,
                   "model": spec.repo, "chunk_seconds": chunk_len,
                   "complete": complete,
                   "machine": machine,
                   "files_scored": len(rows), "files_expected": expected,
                   "skipped_files": skipped or [],
                   "aggregate": summary, "ref_units": agg_ref,
                   "kana_ref_chars": round(kana_ref), "lenient_ref_chars": len_ref,
                   "x_realtime": round(tot_dur / max(tot_wall, 1e-9), 2),
                   "peak_rss_gb": round(rss_gb, 2) if rss_gb else None,
                   "model_load_s": round(load_s, 1),
                   "cue_source": "token_times",
                   "results": rows}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
