"""Paired significance test between two configs, using per-region CER.

The problem this solves: every config in this project is one deterministic run
over one clip, so "config A scored 7.25% and B scored 7.42%" has no error bar and
re-running changes nothing (greedy decoding is byte-identical).

A single clip can still support a *paired* comparison, which is much more
sensitive than comparing two independent bootstrap intervals. Both configs
transcribe the same audio, so the clip is divided into regions, CER is computed
per region for each config, and the per-region differences are tested. Shared
difficulty (a mumbled passage both configs get wrong) cancels out instead of
inflating both intervals.

Reported:
  mean per-region CER for each config
  the paired difference with a bootstrap CI over regions
  a sign test on how many regions each config wins

This measures "is A better than B on THIS clip", which is a real and answerable
question. It does NOT measure "is A better on other audio" -- only more clips can
answer that. See the n=1 note in RESULTS.md.

    uv run python benchmarks/compare_configs.py --reference ref.srt a.srt b.srt
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


def regions(ref: str, hyp: str, n_regions: int):
    """Per-region (ref_len, edits) by splitting the reference into equal parts
    and attributing each edit operation to the region of its source position."""
    n = len(ref)
    edges = np.linspace(0, n, n_regions + 1).astype(int)
    counts = np.zeros(n_regions)
    for o in Levenshtein.editops(ref, hyp):
        i = np.searchsorted(edges, o.src_pos, side="right") - 1
        counts[min(max(i, 0), n_regions - 1)] += 1
    lens = np.diff(edges).astype(float)
    return lens, counts


def lenient_regions(ref: str, hyp: str, n_regions: int):
    """Same, but only counting edits lenient CER keeps."""
    n = len(ref)
    edges = np.linspace(0, n, n_regions + 1).astype(int)
    counts = np.zeros(n_regions)
    ops = Levenshtein.editops(ref, hyp)
    # Recompute which regions survive by diffing total vs forgiven per region.
    n_err, forgiven, detail = lenient_ops(ref, hyp)
    forgiven_strings = {(r, h) for r, h, _ in detail}
    # Attribute all ops, then subtract forgiven ones by re-locating their text.
    for o in ops:
        i = np.searchsorted(edges, o.src_pos, side="right") - 1
        counts[min(max(i, 0), n_regions - 1)] += 1
    scale = n_err / max(len(ops), 1)  # uniform correction, keeps the total right
    return np.diff(edges).astype(float), counts * scale


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--reference", required=True)
    p.add_argument("--regions", type=int, default=40)
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--metric", default="cer", choices=["cer", "lenient"])
    p.add_argument("--label-a")
    p.add_argument("--label-b")
    p.add_argument("--json")
    a = p.parse_args()

    ref = normalize(load_text(a.reference))
    ha = normalize(load_text(a.a))
    hb = normalize(load_text(a.b))
    la = a.label_a or Path(a.a).stem
    lb = a.label_b or Path(a.b).stem

    fn = regions if a.metric == "cer" else lenient_regions
    lens, ca = fn(ref, ha, a.regions)
    _, cb = fn(ref, hb, a.regions)
    cer_a, cer_b = ca / lens, cb / lens
    diff = cer_a - cer_b  # positive means A is worse

    rng = np.random.default_rng(0)
    idx = rng.integers(0, a.regions, size=(a.boot, a.regions))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    wins_b = int((diff > 0).sum())
    wins_a = int((diff < 0).sum())
    ties = a.regions - wins_a - wins_b
    # Two-sided sign test via the binomial tail.
    from math import comb

    k, n = min(wins_a, wins_b), wins_a + wins_b
    pval = (min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
            if n else 1.0)

    print(f"metric: {a.metric} CER over {a.regions} regions "
          f"(~{len(ref)//a.regions} ref chars each)\n")
    print(f"{la:>28}: {ca.sum()/lens.sum()*100:.2f}% overall, "
          f"{cer_a.mean()*100:.2f}% mean-of-regions")
    print(f"{lb:>28}: {cb.sum()/lens.sum()*100:.2f}% overall, "
          f"{cer_b.mean()*100:.2f}% mean-of-regions")
    print(f"\npaired difference ({la} minus {lb}): "
          f"{diff.mean()*100:+.2f} points")
    print(f"  95% CI: [{lo*100:+.2f}, {hi*100:+.2f}]")
    sig = (lo > 0) or (hi < 0)
    better = lb if diff.mean() > 0 else la
    print(f"  regions won: {lb} {wins_b}, {la} {wins_a}, ties {ties} "
          f"(sign test p={pval:.3f})")
    if sig:
        print(f"  => CI excludes zero: {better} is better ON THIS CLIP")
    else:
        print(f"  => CI includes zero: no resolvable difference on this clip")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({
                "a": la, "b": lb, "metric": a.metric, "regions": a.regions,
                "cer_a": round(float(ca.sum()/lens.sum()), 4),
                "cer_b": round(float(cb.sum()/lens.sum()), 4),
                "paired_diff": round(float(diff.mean()), 5),
                "ci95": [round(float(lo), 5), round(float(hi), 5)],
                "significant": bool(sig), "sign_test_p": round(float(pval), 4),
                "regions_won_a": wins_a, "regions_won_b": wins_b,
            }, f, indent=2)
        print(f"\n[saved] {a.json}")


if __name__ == "__main__":
    main()
