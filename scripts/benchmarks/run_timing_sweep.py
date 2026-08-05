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

    uv run python scripts/benchmarks/run_timing_sweep.py --corpus DIR --json out.json

By default this measures the cue grouping a user actually gets. `--gap-seconds` /
`--max-chars` measure a different one, which is only for comparing configs: a
published break figure has to describe the shipped default, and once did not.

Whisper's language comes from each file's reference script, not from a constant.
Hardcoding `ja` here previously drove the one English-audio file into a repetition
loop with degenerate timestamps, and the metric then discarded it as untrustworthy
rather than reporting a harness error. See ``reference_language``.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Both the repo root (for mlx_asr) and scripts/ (for metrics) have to be
# importable; this file lives two levels down from the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metrics.eval_coverage import is_space_delimited, load_reference
from metrics.eval_timing import score

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


def reference_language(ref: Path) -> str:
    """"en" or "ja", decided from the reference script rather than assumed.

    This exists because hardcoding one language here silently destroyed a row.
    The corpus contains the same video dubbed twice, one track Japanese and one
    English, each with a matching subtitle file; forcing `ja` on the English one
    drove Whisper into a repetition loop with degenerate timestamps (identical
    start and end, one cue ending before it began), giving 9.8% anchor coverage.
    The metric then discarded the row as untrustworthy, which read as "Whisper
    struggled on hard audio" rather than "the harness asked for the wrong
    language". Detecting from the reference costs nothing and cannot misfire the
    way Whisper's own 30s autodetect does (it returned Russian for Japanese files
    elsewhere in this corpus).
    """
    return "en" if is_space_delimited(load_reference(str(ref))) else "ja"


def voxtral_srt(audio: Path, out: Path, chunk=30.0, batch=32, cue_opts=()):
    """Decode with the CLI. Note `-o` is taken literally: given an explicit
    output path the CLI does NOT append `.srt`, so pass the full filename.

    ``cue_opts`` is extra argv for the cue-grouping flags, so this script can
    measure a cue config other than the shipped default without editing library
    defaults. Passing nothing means the shipped default, which is what a user gets
    and therefore what published break figures should describe.
    """
    subprocess.run(
        ["uv", "run", "mlx-asr", str(audio), "-f", "srt",
         "-o", str(out), "--chunk-seconds", str(chunk),
         "--max-batch", str(batch), *cue_opts],
        check=True, capture_output=True,
    )
    return out


def whisper_srt(audio: Path, out: Path, model: str, python: str, language,
                no_condition=True):
    """Write an SRT from mlx-whisper via a child process (its own venv).

    The output filename is not predictable enough to construct: mlx-whisper's
    writer strips the extension at the *last* dot, so a stem of `x.y` yields
    `x.srt`, not `x.y.srt`. Guessing wrong meant the rename silently never fired
    and the file was reported missing. So it writes into a private directory and
    takes whatever single `.srt` appears.
    """
    code = (
        "import sys\n"
        "from mlx_whisper.transcribe import transcribe\n"
        "from mlx_whisper.writers import get_writer\n"
        "audio, model, outdir, lang, nocond = sys.argv[1:6]\n"
        "r = transcribe(audio, path_or_hf_repo=model, language=lang,\n"
        "               verbose=False,\n"
        "               condition_on_previous_text=(nocond != '1'))\n"
        "get_writer('srt', outdir)(r, 'hyp')\n"
    )
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [python, "-c", code, str(audio), model, td, language,
             "1" if no_condition else "0"],
            check=True, capture_output=True,
        )
        produced = list(Path(td).glob("*.srt"))
        if len(produced) != 1:
            raise RuntimeError(
                f"expected one SRT from mlx-whisper, got {len(produced)}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(produced[0].read_bytes())
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
    p.add_argument("--language",
                   help="force one language for Whisper. Default and recommended: "
                        "detect per file from its reference script")
    # Cue grouping, passed through to the CLI. Omit both to measure what a user
    # actually gets, which is the only thing a published break figure may claim.
    p.add_argument("--gap-seconds", type=float)
    p.add_argument("--max-chars", type=int)
    p.add_argument("--json")
    a = p.parse_args()

    cue_opts = []
    for flag, val in (("--gap-seconds", a.gap_seconds),
                      ("--max-chars", a.max_chars)):
        if val is not None:
            cue_opts += [flag, str(val)]

    corpus = Path(a.corpus).expanduser()
    pairs = find_timed_pairs(corpus)
    if not pairs:
        print(f"no audio with a timed (.srt/.vtt) reference in {corpus}",
              file=sys.stderr)
        return 2
    srtdir = Path(a.srt_dir)
    srtdir.mkdir(parents=True, exist_ok=True)

    print(f"{len(pairs)} file(s) with timed references"
          + (f", cue overrides {' '.join(cue_opts)}" if cue_opts else "")
          + "\n")
    engines = [("voxtral", None), ("whisper-turbo-nocond", a.whisper_model)]
    rows = []
    for audio, ref in pairs:
        lang = a.language or reference_language(ref)
        for name, model in engines:
            out = srtdir / f"{audio.stem}.{name}.srt"
            if not (a.skip_decode and out.exists()):
                try:
                    if model is None:
                        voxtral_srt(audio, out, cue_opts=cue_opts)
                    else:
                        whisper_srt(audio, out, model, a.whisper_python,
                                    language=lang)
                except (subprocess.CalledProcessError, RuntimeError) as e:
                    tail = (getattr(e, "stderr", b"") or b"").decode()[-160:]
                    print(f"{audio.stem:<22} {name:<22} DECODE FAILED "
                          f"{tail or e}")
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
                         # Recorded per row because it is per file, and because a
                         # wrong value here is what produced the degenerate row.
                         "language": lang if model else None,
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
             "break_tol": a.break_tol,
             # Without this a break-F1 number cannot be attributed to a cue
             # config, which is exactly how a published figure came to describe
             # settings the CLI does not ship.
             "cue_overrides": {"gap_seconds": a.gap_seconds,
                               "max_chars": a.max_chars},
             "language_forced": a.language,
             "results": rows},
            indent=2, ensure_ascii=False))
        print(f"\n[saved] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
