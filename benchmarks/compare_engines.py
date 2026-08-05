"""Paired per-file comparison between two corpus runs (any engine, any config).

`compare_configs.py` answers "is A better on THIS clip" by splitting one
recording into regions. This answers the harder question, "is A better on this
kind of audio", by pairing over *files*: both runs transcribed the same 7
recordings, so each file contributes one difference and shared difficulty
cancels. It is the method behind the corpus verification table in RESULTS.md,
extracted into a script now that engines (not just configs) are being compared.

Reads the JSON that run_corpus.py and run_whisper.py both emit, so a Voxtral run
and a Whisper run compare directly. Files are matched by duration rather than by
name, because the public corpus JSONs have scrubbed stems (rec-01) while local
runs carry the original ones; durations are unique to the second here.

Char-unit (CER) and word-unit (WER) files are reported separately and never
averaged, since they are different quantities.

Bootstrap detail: resampling is length-weighted over files, matching how the
aggregate is computed, so a 93-minute recording moves the interval more than a
13-minute one. With n=7 the interval is wide by construction; RESULTS.md puts the
resolution limit at about 3.2 points, and a difference inside that is reported as
unresolved rather than as a tie.

    uv run python benchmarks/compare_engines.py A.json B.json [--json out.json]
"""

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np


def load_run(path: str):
    """Return (label, {duration: row}) for a corpus run JSON."""
    d = json.load(open(path))
    label = d.get("label") or Path(path).stem
    by_dur = {}
    for r in d.get("results", []):
        if "error" in r or "duration_s" not in r:
            continue
        by_dur[round(float(r["duration_s"]))] = r
    return label, by_dur, d


def paired(a_rows, b_rows, unit):
    """Aligned (ref_units, err_a, err_b, names) for files of one unit type."""
    ref, ea, eb, names = [], [], [], []
    for dur in sorted(set(a_rows) & set(b_rows)):
        ra, rb = a_rows[dur], b_rows[dur]
        if ra.get("unit") != unit or rb.get("unit") != unit:
            continue
        # Same audio must yield the same reference length under the same metric;
        # if it does not, the two runs are not comparable and silence would hide
        # that.
        if ra["ref_chars"] != rb["ref_chars"]:
            print(f"  WARNING: {dur}s file has {ra['ref_chars']} vs "
                  f"{rb['ref_chars']} reference units; skipping", file=sys.stderr)
            continue
        ref.append(ra["ref_chars"])
        ea.append(ra["coverage_cer"])
        eb.append(rb["coverage_cer"])
        names.append(ra.get("file", str(dur)))
    return (np.array(ref, float), np.array(ea), np.array(eb), names)


def report(unit, la, lb, ref, ea, eb, names, n_boot, rng):
    m = "coverageCER" if unit == "char" else "coverageWER"
    n = len(ref)
    if n == 0:
        return None
    agg_a = float((ea * ref).sum() / ref.sum())
    agg_b = float((eb * ref).sum() / ref.sum())

    # Length-weighted paired bootstrap over files, matching the aggregate.
    idx = rng.integers(0, n, size=(n_boot, n))
    w = ref[idx]
    d_a = (ea[idx] * w).sum(1) / w.sum(1)
    d_b = (eb[idx] * w).sum(1) / w.sum(1)
    boot = d_a - d_b
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    diff = agg_a - agg_b  # positive means A is worse

    per_file = ea - eb
    wins_b = int((per_file > 0).sum())
    wins_a = int((per_file < 0).sum())
    k, tot = min(wins_a, wins_b), wins_a + wins_b
    pval = (min(1.0, 2 * sum(comb(tot, i) for i in range(k + 1)) / 2 ** tot)
            if tot else 1.0)

    print(f"\n=== {m} over {n} file(s) "
          f"({int(ref.sum())} reference {unit}s) ===")
    print(f"{la:>34}: {agg_a*100:6.2f}%")
    print(f"{lb:>34}: {agg_b*100:6.2f}%")
    print(f"\n  paired difference ({la} minus {lb}): {diff*100:+.2f} points")
    print(f"  95% CI (length-weighted over files): "
          f"[{lo*100:+.2f}, {hi*100:+.2f}]")
    print(f"  files won: {lb} {wins_b}, {la} {wins_a} (sign test p={pval:.3f})")
    sig = (lo > 0) or (hi < 0)
    if sig:
        print(f"  => CI excludes zero: "
              f"{lb if diff > 0 else la} is better on this corpus")
    else:
        print(f"  => CI includes zero: not resolvable at n={n}")
    print(f"\n  {'file':<26} {la[:14]:>14} {lb[:14]:>14} {'diff':>8}")
    for nm, x, y in zip(names, ea, eb):
        print(f"  {nm[:26]:<26} {x*100:>13.2f}% {y*100:>13.2f}% "
              f"{(x-y)*100:>+7.2f}")
    return {"unit": unit, "metric": m, "n_files": n,
            "ref_units": int(ref.sum()),
            "aggregate_a": round(agg_a, 5), "aggregate_b": round(agg_b, 5),
            "paired_diff": round(diff, 5),
            "ci95": [round(lo, 5), round(hi, 5)],
            "significant": bool(sig), "sign_test_p": round(pval, 4),
            "files_won_a": wins_a, "files_won_b": wins_b,
            "per_file": [{"file": nm, "a": round(float(x), 5),
                          "b": round(float(y), 5)}
                         for nm, x, y in zip(names, ea, eb)]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", help="corpus-run JSON (the candidate)")
    p.add_argument("b", help="corpus-run JSON (the baseline)")
    p.add_argument("--boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label-a")
    p.add_argument("--label-b")
    p.add_argument("--json")
    args = p.parse_args()

    la, a_rows, da = load_run(args.a)
    lb, b_rows, db = load_run(args.b)
    la, lb = args.label_a or la, args.label_b or lb

    shared = set(a_rows) & set(b_rows)
    if not shared:
        print("no files in common (matched by duration)", file=sys.stderr)
        return 2
    only_a = sorted(set(a_rows) - shared)
    only_b = sorted(set(b_rows) - shared)
    if only_a or only_b:
        print(f"note: {len(shared)} file(s) in common; "
              f"{la} has {len(only_a)} extra, {lb} has {len(only_b)} extra. "
              f"Only shared files are compared.")

    rng = np.random.default_rng(args.seed)
    out = {"a": la, "b": lb, "a_json": args.a, "b_json": args.b,
           "a_x_realtime": da.get("x_realtime"),
           "b_x_realtime": db.get("x_realtime"),
           "boot": args.boot, "units": []}
    for unit in ("char", "word"):
        res = report(unit, la, lb, *paired(a_rows, b_rows, unit),
                     args.boot, rng)
        if res:
            out["units"].append(res)

    if da.get("x_realtime") and db.get("x_realtime"):
        print(f"\nthroughput: {la} {da['x_realtime']}x vs "
              f"{lb} {db['x_realtime']}x realtime "
              f"({da['x_realtime']/db['x_realtime']:.2f}x relative)")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n[saved] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
