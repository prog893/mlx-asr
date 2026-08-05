"""Rescore every saved hypothesis under all three metrics.

Needed because the metric set grew after most experiments had already run. The
model outputs are on disk, so nothing has to be re-decoded: plain CER is
unchanged (eval_cer.py was never edited), and lenient CER is simply applied to
the existing SRTs.

Also reports a bootstrap confidence interval, because every experiment in this
project is a single deterministic run over ONE clip. Repeating a run is
pointless (greedy decoding gives byte-identical output); the uncertainty that
matters is "how much would this number move on different audio?". Resampling the
reference/hypothesis alignment by segment approximates that, and it is what
decides whether a 0.3-point difference between two configs means anything.

    uv run python benchmarks/rescore_all.py --reference ref.srt bench_out/*.srt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rapidfuzz.distance import Levenshtein

from evals.eval_cer import load_text, normalize
from evals.eval_cer_lenient import lenient_ops, to_kana


def bootstrap_ci(ref: str, hyp: str, n_boot: int = 400, block: int = 200,
                 seed: int = 0):
    """Moving-block bootstrap CI for CER.

    Characters within an utterance are highly correlated, so resampling single
    characters would badly understate the spread. Blocks of ~200 characters keep
    local structure intact while still varying which material is counted.
    Returns (lo, hi) at 95%.
    """
    rng = np.random.default_rng(seed)
    n = len(ref)
    if n < block * 2:
        return (float("nan"), float("nan"))
    # Align once, then resample reference positions in blocks and count the
    # edits whose source position falls inside the sampled blocks.
    ops = Levenshtein.editops(ref, hyp)
    per_pos = np.zeros(n + 1, dtype=np.float32)
    for o in ops:
        per_pos[min(o.src_pos, n)] += 1

    n_blocks = n // block
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        errs = sum(float(per_pos[s : s + block].sum()) for s in starts)
        vals.append(errs / (n_blocks * block))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("hypotheses", nargs="+")
    p.add_argument("--reference", required=True)
    p.add_argument("--no-ci", action="store_true")
    p.add_argument("--json")
    a = p.parse_args()

    ref = normalize(load_text(a.reference))
    ref_kana = to_kana(ref)
    print(f"reference: {len(ref)} chars, {len(ref_kana)} kana\n")
    hdr = f"{'hypothesis':>28} {'CER':>8} {'kanaCER':>9} {'lenientCER':>11}"
    if not a.no_ci:
        hdr += f" {'95% CI on CER':>18}"
    print(hdr)

    rows = []
    for h in a.hypotheses:
        try:
            hyp = normalize(load_text(h))
        except Exception as e:
            print(f"{Path(h).name:>28}  unreadable: {type(e).__name__}")
            continue
        d = Levenshtein.distance(ref, hyp)
        cer = d / max(len(ref), 1)
        kd = Levenshtein.distance(ref_kana, to_kana(hyp))
        kana = kd / max(len(ref_kana), 1)
        n_err, forgiven, _ = lenient_ops(ref, hyp)
        len_cer = n_err / max(len(ref), 1)
        row = {"file": Path(h).name, "cer": round(cer, 4),
               "kana_cer": round(kana, 4), "lenient_cer": round(len_cer, 4),
               "forgiven": forgiven}
        line = (f"{Path(h).name:>28} {cer*100:>7.2f}% {kana*100:>8.2f}% "
                f"{len_cer*100:>10.2f}%")
        if not a.no_ci:
            lo, hi = bootstrap_ci(ref, hyp)
            row["cer_ci95"] = [round(lo, 4), round(hi, 4)]
            line += f"   [{lo*100:>5.2f}, {hi*100:>5.2f}]"
        rows.append(row)
        print(line)

    if len(rows) > 1 and not a.no_ci:
        best = min(rows, key=lambda r: r["cer"])
        print(f"\nbest plain CER: {best['file']} at {best['cer']*100:.2f}%")
        lo, hi = best["cer_ci95"]
        half = (hi - lo) / 2
        print(f"its 95% CI half-width is {half*100:.2f} points, so differences "
              f"smaller than ~{half*100:.1f} points between configs are not "
              f"resolvable on this single clip.")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"reference": a.reference, "results": rows}, f, indent=2)
        print(f"[saved] {a.json}")


if __name__ == "__main__":
    main()
