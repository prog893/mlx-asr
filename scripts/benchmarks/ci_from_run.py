"""95% confidence intervals for the aggregates of one or more corpus runs.

    uv run python scripts/benchmarks/ci_from_run.py RUN.json [RUN.json ...]

For each run and each scoring unit (char = coverage CER, word = coverage WER), prints the
length-weighted aggregate and its 95% interval from a bootstrap over FILES, the same
resampling `compare_engines.py` uses for paired differences. Also prints x realtime with
its interval (total audio over total wall clock, resampled the same way).

The interval answers "how much would this number move on a different draw of files like
these", which is what an error bar on a doc chart should show. It is not a run-to-run
spread: the greedy engines reproduce byte-identically, so there is none to show.
"""

import argparse
import json
import sys

import numpy as np


def rows(path, allow_partial=False):
    """Per-file rows of a run, refusing a partial one unless asked.

    An interval over the files that happened to finish looks as plausible as one over
    the whole corpus, so a run that is marked incomplete, has error rows, or scored
    fewer files than it expected is an error by default.
    """
    d = json.load(open(path))
    results = d.get("results", [])
    errors = [r.get("file") for r in results if "error" in r]
    problems = []
    if d.get("complete") is False:
        problems.append("marked incomplete")
    if errors:
        problems.append(f"{len(errors)} error rows ({', '.join(map(str, errors[:3]))})")
    if d.get("files_expected") and d.get("files_scored") != d.get("files_expected"):
        problems.append(f"scored {d.get('files_scored')} of {d['files_expected']} files")
    if problems:
        msg = f"{path}: " + "; ".join(problems)
        if not allow_partial:
            sys.exit(f"ERROR: {msg}. Pass --allow-partial to compute intervals anyway.")
        print(f"WARNING: {msg}", file=sys.stderr)
    return d.get("label") or path, [r for r in results if "error" not in r]


def interval(values, weights, n_boot, rng):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    point = float((values * weights).sum() / weights.sum())
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    w = weights[idx]
    boot = (values[idx] * w).sum(1) / w.sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-partial", action="store_true",
                   help="compute intervals for an incomplete run instead of refusing")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    for path in a.runs:
        label, rs = rows(path, a.allow_partial)
        print(f"== {label} ({len(rs)} files)")
        for unit, name in (("char", "JP coverage CER"), ("word", "EN coverage WER")):
            sel = [r for r in rs if r.get("unit") == unit and "coverage_cer" in r]
            if not sel:
                continue
            ref = [r["ref_chars"] for r in sel]
            pt, lo, hi = interval([r["coverage_cer"] for r in sel], ref, a.boot, rng)
            print(f"  {name}: {pt*100:.2f}% [{lo*100:.2f}, {hi*100:.2f}]  (n={len(sel)})")
        timed = [r for r in rs if r.get("x_realtime") and r.get("duration_s")]
        if timed:
            dur = np.array([r["duration_s"] for r in timed], float)
            wall = dur / np.array([r["x_realtime"] for r in timed], float)
            idx = rng.integers(0, len(dur), size=(a.boot, len(dur)))
            boot = dur[idx].sum(1) / wall[idx].sum(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"  x realtime: {dur.sum()/wall.sum():.1f}x [{lo:.1f}, {hi:.1f}]  (n={len(timed)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
