"""Compute published figures from the saved run-groups (benchmarks/results/).

    .venv/bin/python scripts/metrics/archive_query.py GROUP [RUN ...] [--unit char|word]
    .venv/bin/python scripts/metrics/archive_query.py GROUP --paired RUN_A RUN_B [--unit char]

A paired side may be `RUN+RUN+...` (repeat runs averaged per file first, as for the sampling
engines) and may name another group as `GROUP:RUN+...`.

The one place a doc number is derived from raw rows, so every page computes it the same way:

- a run's aggregate is the length-weighted coverage error over its VALID rows: a row marked
  `invalid` (superseded audio, see revisions.json) is dropped, and a supplement row for the
  same file (`supplement` key) takes its place;
- several runs of one config are reported as mean and range;
- a paired difference is the length-weighted difference over the files both runs scored,
  with a 95% interval from a bootstrap over files (20000 resamples, seed 0), the method of
  scripts/benchmarks/compare_engines.py and ci_from_run.py.

GROUP is a run-group name (`ultra-voxtral-headline`) or a path to its JSON. RUN matches a
run's `run`, or `run#arm` for a sweep arm.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "benchmarks" / "results"


def load_group(group: str) -> dict:
    p = Path(group)
    if not p.exists():
        hits = sorted(RESULTS.glob(f"*/{group}.json"))
        if len(hits) != 1:
            sys.exit(f"run-group {group!r}: {len(hits)} matches under {RESULTS}")
        p = hits[0]
    return json.loads(p.read_text(encoding="utf-8"))


def run_key(run: dict) -> str:
    return f"{run['run']}#{run['arm']}" if run.get("arm") else run["run"]


def select(group: dict, names) -> list[dict]:
    if not names:
        return group["runs"]
    out = []
    for n in names:
        hit = [r for r in group["runs"] if run_key(r) == n or (r["run"] == n and not r.get("arm"))]
        if len(hit) != 1:
            sys.exit(f"{group['group']}: run {n!r} matches {len(hit)} runs")
        out += hit
    return out


def valid_rows(run: dict, unit: str = "char") -> dict:
    """{file id: row} over valid scored rows of one unit, supplements in place of invalid.
    unit "both" pools Japanese CER and English WER rows, each weighted by its own reference
    length, which is how the 7-file paired tables were computed."""
    rows = {}
    for r in run["files"]:
        if (unit != "both" and r.get("unit") != unit) or r.get("coverage_cer") is None \
                or not r.get("ref_chars"):
            continue
        if "invalid" in r:
            continue
        rows[r["file"]] = r          # a supplement row is appended after the original
    return rows


def aggregate(run: dict, unit: str = "char"):
    rows = list(valid_rows(run, unit).values())
    if not rows:
        return None
    ref = np.array([r["ref_chars"] for r in rows], float)
    err = np.array([r["coverage_cer"] for r in rows], float)
    return float((err * ref).sum() / ref.sum())


def paired(a: dict, b: dict, unit: str = "char", n_boot: int = 20000, seed: int = 0):
    ra, rb = valid_rows(a, unit), valid_rows(b, unit)
    common = sorted(set(ra) & set(rb))
    for f in common:
        if ra[f]["ref_chars"] != rb[f]["ref_chars"]:
            sys.exit(f"file {f}: reference lengths differ ({ra[f]['ref_chars']} vs "
                     f"{rb[f]['ref_chars']}); the runs are not comparable")
    ref = np.array([ra[f]["ref_chars"] for f in common], float)
    ea = np.array([ra[f]["coverage_cer"] for f in common], float)
    eb = np.array([rb[f]["coverage_cer"] for f in common], float)
    diff = float((ea * ref).sum() / ref.sum() - (eb * ref).sum() / ref.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(n_boot, len(common)))
    w = ref[idx]
    boot = (ea[idx] * w).sum(1) / w.sum(1) - (eb[idx] * w).sum(1) / w.sum(1)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {"n_files": len(common), "diff": diff, "ci95": (lo, hi)}


def side(spec: str, default_group: dict) -> dict:
    """One side of a paired comparison: `RUN`, `RUN+RUN+...` (repeat runs averaged per file,
    over the files every run scored validly) or `GROUP:RUN+...` for a run in another group."""
    group = default_group
    if ":" in spec:
        gname, spec = spec.split(":", 1)
        group = load_group(gname)
    runs = select(group, spec.split("+"))
    if len(runs) == 1:
        return runs[0]
    per_unit = {}
    for unit in ("char", "word"):
        rows = [valid_rows(r, unit) for r in runs]
        common = set.intersection(*(set(x) for x in rows)) if rows else set()
        for f in common:
            lengths = {x[f]["ref_chars"] for x in rows}
            if len(lengths) != 1:
                sys.exit(f"file {f}: reference lengths differ across the averaged runs")
            per_unit[(unit, f)] = {"file": f, "unit": unit, "ref_chars": lengths.pop(),
                                   "coverage_cer": float(np.mean([x[f]["coverage_cer"]
                                                                   for x in rows]))}
    return {"run": spec, "files": list(per_unit.values())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("group")
    p.add_argument("runs", nargs="*")
    p.add_argument("--unit", default="char", choices=["char", "word", "both"])
    p.add_argument("--paired", nargs=2, metavar=("RUN_A", "RUN_B"))
    a = p.parse_args()
    g = load_group(a.group)
    if a.paired:
        ra, rb = side(a.paired[0], g), side(a.paired[1], g)
        r = paired(ra, rb, a.unit)
        print(f"{a.paired[0]} minus {a.paired[1]} ({a.unit}, {r['n_files']} files): "
              f"{r['diff'] * 100:+.2f} points, 95% CI [{r['ci95'][0] * 100:+.2f}, "
              f"{r['ci95'][1] * 100:+.2f}]")
        return 0
    runs = select(g, a.runs)
    vals, xs, peaks = [], [], []
    for r in runs:
        v = aggregate(r, a.unit)
        m = r.get("measures") or {}
        x, pk = m.get("x_realtime"), m.get("peak_memory_gb", m.get("peak_gb"))
        vals.append(v); xs.append(x); peaks.append(pk)
        flag = " (supplemented)" if r.get("supplements") else ""
        print(f"{run_key(r):<44} {a.unit} {v * 100 if v is not None else float('nan'):6.2f}%  "
              f"x {x}  peak {pk}{flag}")
    if len(runs) > 1:
        for name, xs_ in (("error", [v for v in vals if v is not None]),
                          ("x_realtime", [x for x in xs if x is not None])):
            if xs_:
                scale = 100 if name == "error" else 1
                print(f"{name}: mean {np.mean(xs_) * scale:.2f}, range "
                      f"{min(xs_) * scale:.2f} to {max(xs_) * scale:.2f} over {len(xs_)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
