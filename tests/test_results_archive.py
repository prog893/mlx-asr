"""Archived run results name files only by id and carry no local paths.

benchmarks/results/ is published, and the corpus is private recordings. The stem-to-id
map lives outside the repo, so this test cannot look for the real names; it checks the
shape instead: every file-name field is an id, and no string is an absolute path or has
the date-stamped form the private recordings use. scripts/benchmarks/archive_results.py
does the exact check against the map when a result is archived.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = sorted((ROOT / "benchmarks" / "results").rglob("*.json"))
NAME_KEYS = {"file", "files", "looped_files", "skipped_files", "truncated_files"}
ID = re.compile(r"^(rec-\d{2}|worst\d+)(\.wav)?$")
ABS = re.compile(r"^(/Users/|/private/|/tmp/|/home/|/var/folders/)")
DATE_STAMPED = re.compile(r"\d{6}_\d{3}")


def _walk(obj, key=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)
    elif isinstance(obj, str):
        yield key, obj


@pytest.mark.parametrize("path", RESULTS, ids=[str(p.relative_to(ROOT)) for p in RESULTS])
def test_archived_result_is_scrubbed(path):
    for key, s in _walk(json.loads(path.read_text(encoding="utf-8"))):
        if key in NAME_KEYS:
            assert ID.match(s), f"{key}={s!r} is not a file id"
        assert not ABS.match(s), f"absolute path in {key}"
        assert not DATE_STAMPED.search(s), f"date-stamped name in {key}"


def test_every_archive_folder_is_described():
    for folder in {p.parent for p in RESULTS}:
        readme = folder.parent / "README.md"
        assert readme.exists(), f"{folder.parent} has no README.md"
        assert folder.name in readme.read_text(encoding="utf-8"), \
            f"{folder.name} is not described in {readme.relative_to(ROOT)}"
