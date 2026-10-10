"""Archived results name corpus files only by hash, carry no local paths, and stay complete.

benchmarks/results/ is published while the corpus is private, so a file appears only as
`sha256(stem)[:12]`, and corpus.json holds the only metadata that may be published. The
real names never enter the repo, so this test checks shape: every file id is a hash that
corpus.json lists (or a derived input it lists), no string holds an absolute path or the
date-stamped form the recordings use, every run has the fields recomputation needs, and
every run-group is described in its README. scripts/benchmarks/archive_results.py does the
exact check against the real names when a group is written.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
CORPUS = json.loads((RESULTS / "corpus.json").read_text(encoding="utf-8"))
GROUPS = sorted(p for p in RESULTS.rglob("*.json") if p.name != "corpus.json")
NAME_KEYS = {"file", "files", "looped_files", "skipped_files", "truncated_files"}
FILE_ID = re.compile(r"^(?P<id>[0-9a-f]{12}|worst\d+)(\.16k)?(\.wav)?$")
ABS = re.compile(r"/(?:Users|private|tmp|home|var/folders)/")
DATE_STAMPED = re.compile(r"\d{6}_\d{3}")
RUN_KEYS = {"run", "command", "params", "machine", "aggregate", "measures", "files", "missing"}
ALLOWED_META = {"language", "duration_s", "codec", "sample_rate", "bit_depth", "bitrate",
                "reference_length", "reference_unit"}


def _strings(obj, key=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v, key)
    elif isinstance(obj, str):
        yield key, obj


def test_corpus_metadata_holds_only_allowed_fields():
    for h, meta in CORPUS["files"].items():
        assert re.fullmatch(r"[0-9a-f]{12}", h), h
        assert set(meta) <= ALLOWED_META, set(meta) - ALLOWED_META
    for name, meta in CORPUS["derived"].items():
        assert meta["derived_from"] in CORPUS["files"], name
    assert list(CORPUS["files"]) == sorted(CORPUS["files"]), "order would leak names"


@pytest.mark.parametrize("path", GROUPS, ids=[p.stem for p in GROUPS])
def test_run_group_is_scrubbed_and_complete(path):
    group = json.loads(path.read_text(encoding="utf-8"))
    assert {"group", "question", "docs", "runs"} <= set(group)
    for run in group["runs"]:
        assert RUN_KEYS <= set(run), RUN_KEYS - set(run)
    known = set(CORPUS["files"]) | set(CORPUS["derived"])
    for key, s in _strings(group):
        if key in NAME_KEYS:
            m = FILE_ID.match(s)
            assert m and m.group("id") in known, f"{key}={s!r} is not a known file id"
        assert not ABS.search(s), f"absolute path in {key}"
        assert not DATE_STAMPED.search(s), f"date-stamped name in {key}"


def test_every_run_group_is_described():
    for path in GROUPS:
        readme = path.parent / "README.md"
        assert readme.exists(), f"{path.parent} has no README.md"
        assert path.stem in readme.read_text(encoding="utf-8"), f"{path.stem} not in README"
