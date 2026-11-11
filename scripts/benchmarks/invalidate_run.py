"""Mark a saved run invalid: the only edit an existing run-group file accepts.

    uv run python scripts/benchmarks/invalidate_run.py --corpus DIR \\
        benchmarks/results/2026-10/GROUP.json RUN --reason TEXT

Adds one line, `"invalid": REASON`, right after the run's `"run"` line, and writes nothing
unless that is the whole change: the file must parse back to the same group with only that
key added, and must re-serialise to the same bytes. Takes `RUN#ARM` for one arm of a sweep.
Refuses a run that is already invalid, a name that matches no run or more than one, and a
reason that holds a corpus stem or an absolute path, the same check archive_results.py
applies.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_results import Scrubber  # noqa: E402


def invalidate(text: str, run: str, reason: str) -> str:
    """`run` is a run name, or `run#arm` for one arm of a sweep saved as several records."""
    group = json.loads(text)
    name, _, arm = run.partition("#")
    same = [r for r in group["runs"] if r["run"] == name]
    hits = [r for r in same if (r.get("arm") == arm if arm else True)]
    if len(hits) != 1:
        raise SystemExit(f"run {run!r} matches {len(hits)} runs in {group['group']}")
    if "invalid" in hits[0]:
        raise SystemExit(f"run {run!r} is already invalid: {hits[0]['invalid']}")
    # anchored at a line start, so a deeper-indented "run" key cannot match; records sharing
    # a run name (a sweep's arms) appear in file order, so the k-th match is the k-th record
    marker = f'\n   "run": {json.dumps(name, ensure_ascii=False)},\n'
    k = next(i for i, r in enumerate(same) if r is hits[0])
    at = -1
    for _ in range(k + 1):
        at = text.find(marker, at + 1)
        if at < 0:
            raise SystemExit(f"run {run!r}: its line was not found in the file")
    line = f'   "invalid": {json.dumps(reason, ensure_ascii=False)},\n'
    out = text[:at + len(marker)] + line + text[at + len(marker):]
    target = hits[0]
    marked = {"run": name, "invalid": reason} | {key: v for key, v in target.items() if key != "run"}
    expected = {**group, "runs": [marked if r is target else r for r in group["runs"]]}
    if json.dumps(expected, indent=1, ensure_ascii=False) + "\n" != out:
        raise SystemExit("REFUSED: the edit would change more than the one added line")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", action="append", required=True,
                   help="local corpus dir(s); their stems may not appear in the reason")
    p.add_argument("group", type=Path)
    p.add_argument("run")
    p.add_argument("--reason", required=True)
    a = p.parse_args()
    stems = {f.stem for d in a.corpus for f in Path(d).iterdir() if f.is_file()}
    if Scrubber(stems).leaks(a.reason):
        print("REFUSED: the reason holds a corpus stem or an absolute path", file=sys.stderr)
        return 1
    out = invalidate(a.group.read_text(encoding="utf-8"), a.run, a.reason)
    a.group.write_text(out, encoding="utf-8")
    print(f"{a.group.name}: {a.run} marked invalid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
