"""Mark a saved run invalid: the only edit an existing run-group file accepts.

    uv run python scripts/benchmarks/invalidate_run.py --corpus DIR \\
        benchmarks/results/2026-10/GROUP.json RUN --reason TEXT

Adds one line, `"invalid": REASON`, right after the run's `"run"` line, and writes nothing
unless that is the whole change: the file must parse back to the same group with only that
key added, and must re-serialise to the same bytes. Refuses a run that is already invalid,
a run name that matches more than one run (a sweep's arms), and a reason that holds a corpus
stem or an absolute path, the same check archive_results.py applies.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_results import Scrubber  # noqa: E402


def invalidate(text: str, run: str, reason: str) -> str:
    group = json.loads(text)
    hits = [r for r in group["runs"] if r["run"] == run]
    if len(hits) != 1:
        raise SystemExit(f"run {run!r} matches {len(hits)} runs in {group['group']}")
    if "invalid" in hits[0]:
        raise SystemExit(f"run {run!r} is already invalid: {hits[0]['invalid']}")
    # anchored at a line start, so a deeper-indented "run" key cannot match
    marker = f'\n   "run": {json.dumps(run, ensure_ascii=False)},\n'
    if text.count(marker) != 1:
        raise SystemExit(f"run {run!r}: its line is not unique in the file")
    line = f'   "invalid": {json.dumps(reason, ensure_ascii=False)},\n'
    out = text.replace(marker, marker + line)
    target = hits[0]
    marked = {"run": run, "invalid": reason} | {k: v for k, v in target.items() if k != "run"}
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
