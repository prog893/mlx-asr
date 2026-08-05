"""How much does cue layout affect CER, versus the transcript content itself?

Two sources of noise get conflated when you score a concatenated SRT:

1. **Split characters.** Cue text used to be assembled by appending raw token
   bytes and decoding at each flush. Tekken is a byte-level BPE, so a flush
   landing mid-character split it and both halves became U+FFFD. That was 217
   stray characters across this project's saved SRTs, each counting as a CER
   error while saying nothing about recognition. Fixed in output.py by only
   breaking on a complete-character boundary.

2. **Layout itself.** Cue segmentation is underdetermined: many groupings of the
   same tokens are equally valid. Concatenating cues drops nothing, but the
   `.strip()` per cue removes leading/trailing whitespace at every boundary, so
   where the boundaries fall can change the joined string.

This script scores the same run three ways to separate them:

    token text   the model's output, decoded once per chunk (mlx_asr.text)
    srt joined   cue texts concatenated, i.e. what earlier results measured
    srt raw      as above but without normalization, to expose U+FFFD

A nonzero token-vs-srt gap means the reported CER was partly a property of the
cue heuristics rather than of the transcription.

    uv run python benchmarks/compare_text_vs_srt.py --reference ref.srt \
        --pairs bench_out/run.txt:bench_out/run.srt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz.distance import Levenshtein

from evals.eval_cer import load_text, normalize
from evals.eval_cer_lenient import lenient_ops, to_kana

FFFD = "�"


def score(ref, ref_kana, text):
    t = normalize(text)
    d = Levenshtein.distance(ref, t)
    kd = Levenshtein.distance(ref_kana, to_kana(t))
    n_err, forgiven, _ = lenient_ops(ref, t)
    return {
        "chars": len(t),
        "cer": d / max(len(ref), 1),
        "kana_cer": kd / max(len(ref_kana), 1),
        "lenient_cer": n_err / max(len(ref), 1),
        "fffd": t.count(FFFD),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--pairs", nargs="+", required=True,
                   help="one or more txt:srt pairs from the SAME run")
    p.add_argument("--json")
    a = p.parse_args()

    ref = normalize(load_text(a.reference))
    ref_kana = to_kana(ref)
    print(f"reference: {len(ref)} chars\n")
    print(f"{'run':>26} {'source':>10} {'chars':>6} {'CER':>8} {'lenient':>9} "
          f"{'kana':>8} {'U+FFFD':>7}")

    rows = []
    for pair in a.pairs:
        txt_path, srt_path = pair.split(":")
        label = Path(txt_path).stem
        try:
            txt = load_text(txt_path)
            srt = load_text(srt_path)
        except Exception as e:
            print(f"{label:>26}  skipped: {type(e).__name__}")
            continue
        s_txt = score(ref, ref_kana, txt)
        s_srt = score(ref, ref_kana, srt)
        for src, s in (("token text", s_txt), ("srt joined", s_srt)):
            print(f"{label:>26} {src:>10} {s['chars']:>6} {s['cer']*100:>7.2f}% "
                  f"{s['lenient_cer']*100:>8.2f}% {s['kana_cer']*100:>7.2f}% "
                  f"{s['fffd']:>7}")
        gap = (s_srt["cer"] - s_txt["cer"]) * 100
        print(f"{'':>26} {'gap':>10} {s_srt['chars']-s_txt['chars']:>+6} "
              f"{gap:>+7.2f}%")
        rows.append({"run": label, "token_text": s_txt, "srt_joined": s_srt,
                     "cer_gap_points": round(gap, 3)})

    if rows:
        gaps = [r["cer_gap_points"] for r in rows]
        print(f"\nSRT-vs-token CER gap: min {min(gaps):+.2f}, max {max(gaps):+.2f}, "
              f"mean {sum(gaps)/len(gaps):+.2f} points")
        print("A gap near zero means cue layout is not distorting the metric.")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"reference": a.reference, "results": rows}, f, indent=2)
        print(f"[saved] {a.json}")


if __name__ == "__main__":
    main()
