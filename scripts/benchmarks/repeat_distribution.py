"""Test a deterministic engine against a nondeterministic one, properly.

The situation this exists for: Voxtral decodes greedily and reproduces
byte-identically, so one run IS its score. Whisper's temperature-fallback ladder
samples, so one run is a draw from a distribution. Comparing "Whisper's 14.93%"
to "Voxtral's 16.44%" as if both were fixed points is the mistake, and taking two
Whisper runs and calling the difference noise is the over-correction.

The right question is: **what fraction of Whisper's run distribution beats
Voxtral's fixed value?** With N repeat runs of the same Whisper config this
script reports:

  * the run-to-run spread (min/max/mean/SD) per scoring unit
  * how many individual runs beat the deterministic baseline
  * a one-sample t-interval on the mean, tested against the baseline as a
    constant, which is the correct test when only one side has sampling error
  * the same thing per file, since a config can win on aggregate while losing on
    most files

No bootstrap over files here: that answers "would this hold on other audio",
which `compare_engines.py` already does. This answers "does this hold on a rerun
of the same audio", which is a different and prior question.

    uv run python scripts/benchmarks/repeat_distribution.py \\
        --runs whisper_r1.json whisper_r2.json whisper_r3.json \\
        --baseline voxtral.json
"""

import argparse
import json
import statistics as stats
import sys
from math import sqrt
from pathlib import Path


def load(path):
    d = json.load(open(path))
    # Keyed on (duration, unit) rather than duration alone: the corpus contains one
    # recording dubbed into two languages, identical in length to the sample, and a
    # duration-only key silently kept just one of them. Duration is still the basis
    # because an anonymized run and a local one share duration but not filenames.
    rows = {(round(float(r["duration_s"])), r.get("unit")): r
            for r in d.get("results", []) if "duration_s" in r}
    return d.get("label") or Path(path).stem, d, rows


def t_crit(df, conf=0.95):
    """Two-sided t critical value; small table avoids a scipy dependency."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
             20: 2.086, 30: 2.042}
    if df in table:
        return table[df]
    keys = sorted(table)
    if df < keys[0]:
        return table[keys[0]]
    if df > keys[-1]:
        return 1.96
    lo = max(k for k in keys if k <= df)
    return table[lo]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="repeat-run JSONs of ONE nondeterministic config")
    p.add_argument("--baseline", required=True,
                   help="run JSON of the deterministic engine")
    p.add_argument("--json")
    a = p.parse_args()

    blabel, bdoc, brows = load(a.baseline)
    runs = [load(f) for f in a.runs]
    # Only complete runs: a partially-written JSON would silently bias the mean.
    # Two checks, because the row-count one is relative. If every run happened to be
    # half-written they would all agree on a wrong count and all look complete, so
    # the explicit flag is authoritative where present.
    flagged = [r[0] for r in runs if r[1].get("complete") is False]
    runs = [r for r in runs if r[1].get("complete") is not False]
    if flagged:
        print(f"note: dropping {len(flagged)} run(s) still in progress: "
              f"{', '.join(flagged)}\n")
    if not runs:
        print("no complete repeat runs given", file=sys.stderr)
        return 2
    n_files = max(len(r[2]) for r in runs)
    full = [r for r in runs if len(r[2]) == n_files]
    dropped = [r[0] for r in runs if len(r[2]) != n_files]
    if dropped:
        print(f"note: dropping {len(dropped)} short run(s): "
              f"{', '.join(dropped)}\n")
    if len(full) < 2:
        print("need at least 2 complete repeat runs", file=sys.stderr)
        return 2

    out = {"baseline": blabel, "n_runs": len(full),
           "runs": [r[0] for r in full], "units": []}
    print(f"{len(full)} complete repeat runs of a nondeterministic engine "
          f"vs deterministic baseline '{blabel}'\n")

    for unit, metric in (("char", "coverageCER"), ("word", "coverageWER")):
        base = bdoc["aggregate"].get(unit)
        vals = [r[1]["aggregate"].get(unit) for r in full]
        if base is None or any(v is None for v in vals):
            continue
        n = len(vals)
        mean, sd = stats.mean(vals), (stats.stdev(vals) if n > 1 else 0.0)
        se = sd / sqrt(n)
        t = t_crit(n - 1)
        lo, hi = mean - t * se, mean + t * se
        wins = sum(1 for v in vals if v < base)

        print(f"=== {metric} ===")
        print(f"  baseline ({blabel}):  {base*100:.2f}%  (deterministic, "
              f"no error bar)")
        print(f"  repeat runs:          "
              f"{', '.join(f'{v*100:.2f}%' for v in sorted(vals))}")
        print(f"  mean {mean*100:.2f}%  SD {sd*100:.2f}  "
              f"range {min(vals)*100:.2f}-{max(vals)*100:.2f}  "
              f"spread {(max(vals)-min(vals))*100:.2f} points")
        print(f"  95% t-interval on the mean: "
              f"[{lo*100:.2f}%, {hi*100:.2f}%]")
        print(f"  runs beating the baseline: {wins}/{n}")
        if hi < base:
            verdict = "whisper is better: interval entirely below baseline"
        elif lo > base:
            verdict = "baseline is better: interval entirely above"
        else:
            verdict = ("NOT resolved: the run distribution straddles the "
                       "baseline")
        print(f"  => {verdict}\n")
        out["units"].append({
            "unit": unit, "metric": metric, "baseline": round(base, 5),
            "runs": [round(v, 5) for v in vals], "mean": round(mean, 5),
            "sd": round(sd, 5), "ci95_mean": [round(lo, 5), round(hi, 5)],
            "runs_beating_baseline": wins, "n_runs": n, "verdict": verdict})

    # Per-file: an aggregate win can hide a per-file loss, and vice versa.
    print("=== per file (coverage error) ===")
    print(f"  {'file':<10} {'unit':<5} {'baseline':>9} {'whisper runs':>28} "
          f"{'wins':>6}")
    per_file = []
    for dur in sorted(brows, key=lambda k: (k[0], str(k[1]))):
        b = brows[dur]
        vs = [r[2][dur]["coverage_cer"] for r in full if dur in r[2]]
        if not vs:
            continue
        wins = sum(1 for v in vs if v < b["coverage_cer"])
        print(f"  {(b.get('file') or f'{dur[0]}s')[:10]:<10} {b['unit']:<5} "
              f"{b['coverage_cer']*100:>8.2f}% "
              f"{' '.join(f'{v*100:5.2f}' for v in vs):>28} "
              f"{wins:>3}/{len(vs)}")
        per_file.append({"file": b.get("file"), "unit": b["unit"],
                         "baseline": round(b["coverage_cer"], 5),
                         "runs": [round(v, 5) for v in vs],
                         "wins": wins, "n": len(vs)})
    out["per_file"] = per_file

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n[saved] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
