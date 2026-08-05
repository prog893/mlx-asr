"""Tune cue grouping against the break metrics, without re-decoding.

`build_cues` turns per-token timestamps into subtitle cues using four knobs
(`gap_s`, `max_chars`, `max_dur_s`, and sentence punctuation). The timing sweep
showed the defaults split mid-phrase on well over half of cues against
human-authored references, against a materially lower rate for Whisper, so the
knobs are worth a sweep.

Note what this cannot tell you. Scoring against these references measures agreement
with one editor's subtitle conventions, not good segmentation, which is why neither
sweep result was ever adopted as a default (see mlx_asr/output.py). Treat the output
as "what matches this reference set", and sweep on your own references instead.

The decode is the expensive part and the knobs do not affect it, so this reuses
saved token timings: run the CLI once with `--stats-json`, or point at the JSON
output format which carries per-cue times. Here we go one level deeper and
re-run only `build_cues` over cached `(token, time)` pairs, which makes a full
grid cost seconds rather than hours.

Scored with eval_timing's break metrics (boundary F1 against the author's cue
ends, plus mid-phrase rate). Drift is NOT scored here: regrouping cues cannot
move a token's timestamp, so drift is invariant to these knobs by construction.

    uv run python scripts/benchmarks/sweep_cues.py --corpus corpus_timed --json out.json
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

# Both the repo root (for mlx_asr) and scripts/ (for metrics) have to be
# importable; this file lives two levels down from the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx
from mlx_audio.stt.utils import load as load_model

from metrics.eval_timing import break_stats, parse_cues
from mlx_asr.audio import load_audio_16k, split_with_overlap
from mlx_asr.decode import transcribe_batch
from mlx_asr.output import build_cues

DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def find_timed(corpus: Path):
    out = []
    for a in sorted(corpus.iterdir()):
        if a.suffix.lower() not in (".wav", ".flac", ".m4a", ".mp3"):
            continue
        for ext in (".srt", ".vtt"):
            r = corpus / (a.stem + ext)
            if r.exists():
                out.append((a, r))
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--chunk-seconds", type=float, default=30.0)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--delay-ms", type=int, default=2400)
    p.add_argument("--kv-bits", type=int, default=8)
    p.add_argument("--gaps", default="0.4,0.7,1.0,1.5")
    p.add_argument("--max-chars", default="20,28,32,42,56,72")
    p.add_argument("--max-durs", default="5.0,7.0,9.0")
    p.add_argument("--break-tol", type=float, default=0.5)
    p.add_argument("--limit", type=int)
    p.add_argument("--json")
    a = p.parse_args()

    pairs = find_timed(Path(a.corpus).expanduser())
    if a.limit:
        pairs = pairs[: a.limit]
    if not pairs:
        print("no audio with a timed reference", file=sys.stderr)
        return 2

    gaps = [float(x) for x in a.gaps.split(",")]
    chars = [int(x) for x in a.max_chars.split(",")]
    durs = [float(x) for x in a.max_durs.split(",")]
    print(f"{len(pairs)} timed file(s); grid = "
          f"{len(gaps)}x{len(chars)}x{len(durs)} = "
          f"{len(gaps)*len(chars)*len(durs)} configs\n")

    model = load_model(a.model)

    # Decode once per file; the grid only re-groups the cached token timings.
    cached = []
    for audio, ref in pairs:
        wav = load_audio_16k(str(audio))
        # keyword args: the third positional is search_s, not a sample rate,
        # and passing 16000 there silently collapses everything into one chunk
        chunks, offsets, warmup = split_with_overlap(
            wav, target_s=a.chunk_seconds, overlap_s=0.0)
        timed_all = []
        for i in range(0, len(chunks), a.max_batch):
            sl = slice(i, i + a.max_batch)
            timed, _ = transcribe_batch(model, chunks[sl], a.delay_ms, None,
                                        a.kv_bits, log=lambda *x: None,
                                        warmup_tokens=warmup[sl])
            timed_all.extend(timed)
        cached.append((audio.stem, ref, timed_all, offsets))
        print(f"[decoded] {audio.stem} ({len(chunks)} chunks)")
        mx.clear_cache()

    rows = []
    for gap in gaps:
        for mc in chars:
            for md in durs:
                f1s, mids, ncues, per_file = [], [], [], {}
                for stem, ref, timed_all, offsets in cached:
                    cues = build_cues(model, timed_all, offsets, gap_s=gap,
                                      max_chars=mc, max_dur_s=md)
                    b = break_stats(parse_cues(str(ref)),
                                    [(s, e, t) for s, e, t in cues],
                                    a.break_tol)
                    if b:
                        f1s.append(b["break_f1"])
                        mids.append(b["mid_phrase_rate"])
                        ncues.append(b["hyp_cues"] / max(b["ref_cues"], 1))
                        per_file[stem] = {
                            "break_f1": round(b["break_f1"], 4),
                            "mid_phrase_rate": round(b["mid_phrase_rate"], 4),
                            "cue_ratio": round(
                                b["hyp_cues"] / max(b["ref_cues"], 1), 3)}
                if not f1s:
                    continue
                row = {"gap_s": gap, "max_chars": mc, "max_dur_s": md,
                       "break_f1": sum(f1s) / len(f1s),
                       "mid_phrase_rate": sum(mids) / len(mids),
                       "cue_ratio": sum(ncues) / len(ncues), "n_files": len(f1s),
                       # kept so a win can be checked for being carried by one
                       # recording rather than holding across the set
                       "per_file": per_file}
                rows.append(row)

    rows.sort(key=lambda r: -r["break_f1"])
    print(f"\n{'gap':>5} {'chars':>6} {'dur':>5} {'break F1':>9} "
          f"{'mid-phrase':>11} {'cues/ref':>9}")
    for r in rows[:20]:
        print(f"{r['gap_s']:>5.1f} {r['max_chars']:>6} {r['max_dur_s']:>5.1f} "
              f"{r['break_f1']*100:>8.1f}% {r['mid_phrase_rate']*100:>10.1f}% "
              f"{r['cue_ratio']:>9.2f}")
    if not rows:
        # Every grid point scored None, which means no reference parsed to any
        # cues. Reported rather than raised on rows[0], since by here the whole
        # corpus has already been decoded and the cause is the references.
        print("\nno scorable grid points: do the references parse as SRT/VTT?",
              file=sys.stderr)
        return 2
    best = rows[0]
    print(f"\nbest: gap_s={best['gap_s']} max_chars={best['max_chars']} "
          f"max_dur_s={best['max_dur_s']}  "
          f"F1 {best['break_f1']*100:.1f}% mid {best['mid_phrase_rate']*100:.1f}%")
    # Read the shipped values from the function rather than restating them: this
    # comparison row named 1.0/32 as "current" long after the default became
    # 1.2/28, so it was silently measuring against the wrong baseline.
    sig = inspect.signature(build_cues).parameters
    dg, dc, dd = (sig["gap_s"].default, sig["max_chars"].default,
                  sig["max_dur_s"].default)
    cur = next((r for r in rows if r["gap_s"] == dg and r["max_chars"] == dc
                and r["max_dur_s"] == dd), None)
    if cur:
        print(f"shipped defaults ({dg}/{dc}/{dd}): F1 {cur['break_f1']*100:.1f}% "
              f"mid {cur['mid_phrase_rate']*100:.1f}%  "
              f"=> {(best['break_f1']-cur['break_f1'])*100:+.1f} points available")
    else:
        print(f"shipped defaults ({dg}/{dc}/{dd}) are not in this grid, so there "
              f"is no baseline to compare against")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"corpus": a.corpus, "break_tol": a.break_tol,
             "shipped_defaults": {"gap_s": dg, "max_chars": dc, "max_dur_s": dd},
             "files": [c[0] for c in cached], "results": rows},
            indent=2, ensure_ascii=False))
        print(f"[saved] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
