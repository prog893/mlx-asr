"""Timing quality across engines, on the files that have timed references.

Only some references carry real timings (authored SRTs); the plain-text ones
cannot support this at all. So this is a separate, smaller sweep than the accuracy
one, and it reports per-file rather than pooling: with a handful of files an
aggregate would hide more than it shows.

Both engines are asked for SRT output on the same audio, then scored by
eval_timing. Voxtral has per-token timestamps and builds cues with its own
heuristic (mlx_asr.output.build_cues); Whisper emits segment-level
timestamps. That is a real difference in what the two produce, not a harness
artifact, and it is exactly what the break metrics are measuring.

    uv run python benchmarks/run_timing_sweep.py --corpus DIR --json out.json
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.eval_timing import score

SAMPLE_RATE = 16000


def find_timed_pairs(corpus: Path):
    """Audio paired with a *timed* reference only (SRT/VTT, never plain text)."""
    pairs = []
    for audio in sorted(corpus.iterdir()):
        if audio.suffix.lower() not in (".wav", ".flac", ".m4a", ".mp3"):
            continue
        for ext in (".srt", ".vtt"):
            cand = corpus / (audio.stem + ext)
            if cand.exists():
                pairs.append((audio, cand))
                break
    return pairs


def voxtral_srt(audio: Path, out: Path, chunk=30.0, batch=32):
    """Decode with the CLI. Note `-o` is taken literally: given an explicit
    output path the CLI does NOT append `.srt`, so pass the full filename."""
    subprocess.run(
        ["uv", "run", "mlx-asr", str(audio), "-f", "srt",
         "-o", str(out), "--chunk-seconds", str(chunk),
         "--max-batch", str(batch)],
        check=True, capture_output=True,
    )
    return out


def whisper_srt(audio: Path, out: Path, model: str, python: str,
                no_condition=True):
    """Write an SRT from mlx-whisper via a child process (its own venv)."""
    code = (
        "import sys, json\n"
        "from mlx_whisper.transcribe import transcribe\n"
        "from mlx_whisper.writers import get_writer\n"
        "audio, model, outdir, stem, nocond = sys.argv[1:6]\n"
        "r = transcribe(audio, path_or_hf_repo=model, language='ja',\n"
        "               verbose=False,\n"
        "               condition_on_previous_text=(nocond != '1'))\n"
        "get_writer('srt', outdir)(r, stem + '.x')\n"
    )
    subprocess.run(
        [python, "-c", code, str(audio), model, str(out.parent), out.stem,
         "1" if no_condition else "0"],
        check=True, capture_output=True,
    )
    produced = out.parent / (out.stem + ".x.srt")
    if produced.exists():
        produced.replace(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--whisper-python", default=".venv-whisper/bin/python")
    p.add_argument("--whisper-model",
                   default="mlx-community/whisper-large-v3-turbo")
    p.add_argument("--srt-dir", default="bench_out/timing_srt")
    p.add_argument("--min-anchor", type=int, default=8)
    p.add_argument("--break-tol", type=float, default=0.5)
    p.add_argument("--skip-decode", action="store_true",
                   help="reuse SRTs already in --srt-dir")
    p.add_argument("--json")
    a = p.parse_args()

    corpus = Path(a.corpus).expanduser()
    pairs = find_timed_pairs(corpus)
    if not pairs:
        print(f"no audio with a timed (.srt/.vtt) reference in {corpus}",
              file=sys.stderr)
        return 2
    srtdir = Path(a.srt_dir)
    srtdir.mkdir(parents=True, exist_ok=True)

    print(f"{len(pairs)} file(s) with timed references\n")
    engines = [("voxtral", None), ("whisper-turbo-nocond", a.whisper_model)]
    rows = []
    for audio, ref in pairs:
        for name, model in engines:
            out = srtdir / f"{audio.stem}.{name}.srt"
            if not (a.skip_decode and out.exists()):
                try:
                    if model is None:
                        voxtral_srt(audio, out)
                    else:
                        whisper_srt(audio, out, model, a.whisper_python)
                except subprocess.CalledProcessError as e:
                    tail = (e.stderr or b"").decode()[-160:]
                    print(f"{audio.stem:<22} {name:<22} DECODE FAILED {tail}")
                    continue
            if not out.exists():
                print(f"{audio.stem:<22} {name:<22} no SRT produced")
                continue
            s = score(str(ref), str(out), a.min_anchor, a.break_tol)
            d, b = s["drift"], s["breaks"]
            if not d or not d.get("anchors"):
                print(f"{audio.stem:<22} {name:<22} no anchors")
                continue
            rows.append({"file": audio.stem, "engine": name,
                         "anchor_coverage": round(d["anchor_coverage"], 4),
                         "median_abs_ms": round(d["median_abs_ms"], 1),
                         "p95_abs_ms": round(d["p95_abs_ms"], 1),
                         "constant_offset_ms": round(d["constant_offset_ms"], 1),
                         "drift_slope_ms_per_min":
                             None if d["drift_slope_ms_per_min"] is None
                             else round(d["drift_slope_ms_per_min"], 2),
                         "break_f1": round(b["break_f1"], 4),
                         "mid_phrase_rate": round(b["mid_phrase_rate"], 4),
                         "hyp_cues": b["hyp_cues"], "ref_cues": b["ref_cues"],
                         "hyp_median_dur_s": b["hyp_median_dur_s"],
                         "ref_median_dur_s": b["ref_median_dur_s"]})
            print(f"{audio.stem[:20]:<22} {name:<22} "
                  f"anch={d['anchor_coverage']*100:>5.1f}% "
                  f"med={d['median_abs_ms']:>6.0f}ms "
                  f"p95={d['p95_abs_ms']:>6.0f}ms "
                  f"slope={(d['drift_slope_ms_per_min'] or 0):>+6.1f}/min "
                  f"F1={b['break_f1']*100:>5.1f}% "
                  f"mid={b['mid_phrase_rate']*100:>5.1f}%")

    print()
    for name, _ in engines:
        sub = [r for r in rows if r["engine"] == name]
        if not sub:
            continue
        med = sorted(r["median_abs_ms"] for r in sub)[len(sub) // 2]
        f1 = sum(r["break_f1"] for r in sub) / len(sub)
        mid = sum(r["mid_phrase_rate"] for r in sub) / len(sub)
        sl = [abs(r["drift_slope_ms_per_min"] or 0) for r in sub]
        print(f"{name:<22} n={len(sub)}  median-of-medians {med:.0f}ms  "
              f"mean break F1 {f1*100:.1f}%  mean mid-phrase {mid*100:.1f}%  "
              f"max |slope| {max(sl):.1f} ms/min")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"corpus": str(corpus), "min_anchor": a.min_anchor,
             "break_tol": a.break_tol, "results": rows},
            indent=2, ensure_ascii=False))
        print(f"\n[saved] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
