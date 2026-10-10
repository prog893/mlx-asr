"""Copy run JSONs into benchmarks/results/ with every private name replaced.

    uv run python scripts/benchmarks/archive_results.py --ids MAP.json --out DIR RUN.json [...]

The corpus is private recordings, so a result file may only name a file by its stable
`rec-NN` id. MAP.json maps each corpus stem to its id and lives with the corpus, outside
this repo (publishing it would publish the names). Every string in the run that equals or
contains a mapped stem is rewritten to the id, and an absolute path is cut to its last
component. Per-file scores and timings are kept: they are what intervals and paired
tests are recomputed from, and they say nothing about the audio's content. Transcripts
are never archived.

The output is checked before it is written: if any mapped stem or absolute path
survives, the file is refused rather than written.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ABS_PATH = re.compile(r"^(/Users/|/private/|/tmp/|/home/|/var/folders/)")


def scrub(obj, ids: dict):
    if isinstance(obj, dict):
        return {scrub(k, ids): scrub(v, ids) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v, ids) for v in obj]
    if isinstance(obj, str):
        s = obj
        if ABS_PATH.match(s):
            s = s.rstrip("/").rsplit("/", 1)[-1]
        for stem in sorted(ids, key=len, reverse=True):   # longest first: no partial hits
            if stem in s:
                s = s.replace(stem, ids[stem])
        return s
    return obj


def leaks(text: str, ids: dict) -> list:
    found = [stem for stem in ids if stem in text]
    found += [m for m in re.findall(r'"(/(?:Users|private|tmp|home|var/folders)/[^"]*)"', text)]
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True, help="private stem -> rec-NN map (outside the repo)")
    p.add_argument("--out", required=True)
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    ids = json.load(open(a.ids))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for run in a.runs:
        clean = json.dumps(scrub(json.load(open(run)), ids), indent=1, ensure_ascii=False)
        found = leaks(clean, ids)
        if found:
            print(f"REFUSED {run}: {len(found)} private strings survived", file=sys.stderr)
            bad += 1
            continue
        (out / Path(run).name).write_text(clean + "\n", encoding="utf-8")
    print(f"wrote {len(a.runs) - bad} of {len(a.runs)} to {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
