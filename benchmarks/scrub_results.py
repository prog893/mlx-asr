"""Strip private identifiers out of saved benchmark results.

Benchmark JSON is committed so published numbers are reproducible and auditable,
but it is generated from real recordings, so it picks up things that should not be
public: machine hostnames, absolute paths containing a username, the filenames of
private recordings, and the actual domain vocabulary used in prompt experiments.

Run this before committing new results:

    uv run python benchmarks/scrub_results.py benchmarks/*.json
    uv run python benchmarks/scrub_results.py --check benchmarks/*.json   # CI-style

Recording filenames become stable anonymous labels (rec-01, rec-02, ...) assigned
in sorted order, so per-file tables stay meaningful and line up across configs that
saw the same file set. Non-ASCII runs are replaced wholesale rather than
word-by-word so nothing leaks through a partial match, and that applies to dict
keys as well as values (per-term hit counts are keyed by the vocabulary itself).
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Anything matching these must never appear in a committed file.
PRIVATE = [
    (re.compile(r"/(?:Users|home)/[^/\"'\s]+"), "<HOME>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<HOST>"),
    (re.compile(r"[A-Za-z0-9_-]+-Mac(?:book|-Studio|-Pro|-mini)?\b"), "<HOST>"),
]
# Session-stamped recording ids: YYMMDD_NNN[_NNNN]_TrN
RECORDING = re.compile(r"\b\d{6}_\d{3}(?:_\d{4})?_Tr\d\b")
# CJK / non-ASCII runs are transcript or vocabulary content, not code.
NON_ASCII = re.compile(r"[^\x00-\x7F]{2,}")

DROP_KEYS = {"host", "hostname"}
PATH_KEYS = {"corpus", "reference", "audio", "hyp_dir", "keep_hyp", "json",
             "output", "model_path"}


def label_map(names) -> dict:
    """Stable anonymous label per recording id, assigned in sorted order.

    Sorted rather than first-seen so the same corpus yields the same labels in
    every output file, which is what lets per-file numbers be compared across
    configs after scrubbing.
    """
    return {n: f"rec-{i:02d}" for i, n in enumerate(sorted(names), 1)}


def collect_recordings(obj, acc):
    if isinstance(obj, dict):
        for v in obj.values():
            collect_recordings(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            collect_recordings(v, acc)
    elif isinstance(obj, str):
        acc.update(RECORDING.findall(obj))


def scrub_str(s: str, labels: dict) -> str:
    for rec, lab in labels.items():
        s = s.replace(rec, lab)
    for pat, repl in PRIVATE:
        s = pat.sub(repl, s)
    if NON_ASCII.search(s):
        # Prompt/vocabulary or transcript text: replace wholesale rather than
        # trying to redact words, so nothing leaks through partial matching.
        s = NON_ASCII.sub("<TEXT>", s)
    return s


def scrub(obj, labels):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in DROP_KEYS:
                continue
            # Keys can carry content too: per-term hit counts are keyed by the
            # actual vocabulary. Rewrite the key, keeping entries distinct.
            if isinstance(k, str) and NON_ASCII.search(k):
                out[f"term-{len(out) + 1:02d}"] = scrub(v, labels)
                continue
            if k in PATH_KEYS and isinstance(v, str):
                out[k] = f"<PATH>/{Path(v).name}" if v else v
                out[k] = scrub_str(out[k], labels)
                continue
            out[k] = scrub(v, labels)
        return out
    if isinstance(obj, list):
        return [scrub(v, labels) for v in obj]
    if isinstance(obj, str):
        return scrub_str(obj, labels)
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--check", action="store_true",
                   help="exit nonzero if any file still contains private data")
    a = p.parse_args()

    dirty = []
    for f in a.files:
        path = Path(f)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"{path}: unreadable ({type(e).__name__})", file=sys.stderr)
            continue
        recs = set()
        collect_recordings(data, recs)
        labels = label_map(recs)
        cleaned = scrub(data, labels)
        before = json.dumps(data, ensure_ascii=False, sort_keys=True)
        after = json.dumps(cleaned, ensure_ascii=False, sort_keys=True)
        if before == after:
            continue
        dirty.append(str(path))
        if a.check:
            print(f"DIRTY {path}")
        else:
            path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
            print(f"scrubbed {path}"
                  + (f" ({len(labels)} recording ids)" if labels else ""))

    if a.check:
        if dirty:
            print(f"\n{len(dirty)} file(s) contain private data; run without "
                  f"--check to scrub", file=sys.stderr)
            return 1
        print("clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
