"""Every published number cites the saved run data it is computed from.

The convention is in AGENTS.md and scripts/metrics/number_scan.py. Two contracts:

- every source marker anywhere on the published surface resolves: the linked file exists
  under benchmarks/results/, and every run it names exists in that run-group;
- every page in DONE is fully sourced: each table holding a measured number has a `Source:`
  line under it, each prose line holding one carries a footnote that is a marker, each chart
  spec has a `"source"`, each measured figure in code has a `source:` comment, and each
  profile has a `"runs"` field.

DONE grows page by page until it equals SURFACES.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from metrics.number_scan import (CODE_SOURCE, FOOTDEF, FOOTREF, NOT_MEASURED,  # noqa: E402
                                 SOURCE_LINE,
                                 group_runs, marker_links, md_blocks, measured, spec_links)

RESULTS = ROOT / "benchmarks" / "results"
NON_GROUP = {"corpus.json", "revisions.json"}       # catalog files: cited without runs
SURFACES = sorted(
    {str(p.relative_to(ROOT)) for p in (ROOT / "docs").rglob("*.md")}
    | {"README.md", "RESULTS.md", "CONTRIBUTING.md", "mlx_asr/profiles.json",
       "scripts/docs/chart_data.py"}
    | {str(p.relative_to(ROOT)) for p in (ROOT / "mlx_asr").glob("*.py")}
    | {str(p.relative_to(ROOT)) for p in (ROOT / "scripts/docs/sweeps").glob("*.py")})
# Pages move here once every number on them is sourced; the doc pass that sources them
# lands separately.
DONE: list[str] = []


def _group_file(name: str) -> Path | None:
    hits = sorted(RESULTS.glob(f"*/{name}")) + ([RESULTS / name] if (RESULTS / name).exists() else [])
    return hits[0] if len(hits) == 1 else None


def _check_link(path: Path, runs: list[str], where: str):
    assert path.exists(), f"{where}: {path} does not exist"
    assert RESULTS in path.resolve().parents, f"{where}: {path} is not under benchmarks/results"
    if path.name in NON_GROUP:
        return
    known = group_runs(path)
    missing = [r for r in runs if r not in known]
    assert not missing, f"{where}: runs {missing} are not in {path.name}"


def _md_markers(page: Path):
    """[(where, body)] for every Source line and footnote definition on a page."""
    out = []
    for i, line in enumerate(page.read_text(encoding="utf-8").split("\n"), 1):
        m = FOOTDEF.match(line)
        if SOURCE_LINE.match(line) or (m and ".json" in line):
            out.append((f"{page.relative_to(ROOT)}:{i}", line))
    return out


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts" / "docs"))
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("surface", SURFACES)
def test_every_marker_resolves(surface):
    path = ROOT / surface
    if surface.endswith(".md"):
        for where, body in _md_markers(path):
            links = marker_links(body)
            assert links, f"{where}: a source marker with no run-group link"
            for target, runs in links:
                _check_link((path.parent / target), runs, where)
    elif surface.endswith(".json"):
        for i, prof in enumerate(json.loads(path.read_text())["profiles"]):
            for name, runs in spec_links(prof.get("runs", "")):
                f = _group_file(name)
                assert f, f"{surface} profile {i}: {name} not found"
                _check_link(f, runs, f"{surface} profile {i}")
    else:
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r'"source":\s*"([^"]+\.json[^"]*)"', text):
            for name, runs in spec_links(m.group(1)):
                f = _group_file(name)
                assert f, f"{surface}: {name} not found"
                _check_link(f, runs, surface)
        for m in CODE_SOURCE.finditer(text):
            f = _group_file(m.group(1))
            assert f, f"{surface}: {m.group(1)} not found"
            _check_link(f, [r for r in (m.group(2) or "").split(",") if r], surface)


def _md_unsourced(page: Path) -> list[str]:
    text = page.read_text(encoding="utf-8")
    defs = {m.group(1) for line in text.split("\n") if (m := FOOTDEF.match(line)) and ".json" in line}
    bad = []
    for kind, *rest in md_blocks(text):
        if kind == "table":
            block, after = rest
            if any(measured(ln) for _, ln in block) and not SOURCE_LINE.match(after) \
                    and not NOT_MEASURED.search(after):
                bad.append(f"table at line {block[0][0]} has no Source: line under it")
        else:
            lineno, line = rest
            if SOURCE_LINE.match(line) or FOOTDEF.match(line) or not measured(line) \
                    or NOT_MEASURED.search(line):
                continue
            refs = set(FOOTREF.findall(line))
            if not refs & defs:
                bad.append(f"line {lineno}: {measured(line)} without a source footnote")
    return bad


def _py_unsourced(path: Path) -> list[str]:
    bad = []
    if path.parts[-2] == "mlx_asr":
        lines = path.read_text(encoding="utf-8").split("\n")
        in_doc = False
        for i, line in enumerate(lines):
            s = line.strip()
            quotes = s.count('"""') + s.count("'''")
            published = in_doc or quotes or s.startswith("#") or '"' in s or "'" in s
            if quotes % 2:                       # a docstring opens or closes on this line
                in_doc = not in_doc
            if not published or not measured(line):
                continue
            window = "\n".join(lines[max(0, i - 3):i + 4])
            if not CODE_SOURCE.search(window) and not NOT_MEASURED.search(window):
                bad.append(f"line {i + 1}: {measured(line)} without a source: comment")
        return bad
    mod = _load(path)
    sources = getattr(mod, "SOURCES", {})
    for name, obj in vars(mod).items():
        if name.startswith("_"):
            continue
        specs = obj.items() if name == "SWEEPS" and isinstance(obj, dict) else [(name, obj)]
        for key, spec in specs:
            if isinstance(spec, dict) and "rows" in spec and "source" not in spec:
                bad.append(f"spec {key} has no \"runs\"")
            if isinstance(spec, list) and spec and isinstance(spec[0], tuple) and \
                    any(measured(str(v)) for row in spec for v in row) and key not in sources:
                bad.append(f"{key} has no entry in SOURCES")
    return bad


def _profiles_unsourced(path: Path) -> list[str]:
    return [f"profile {i} has no \"runs\""
            for i, p in enumerate(json.loads(path.read_text())["profiles"]) if not p.get("runs")]


def test_done_is_part_of_the_surface():
    assert set(DONE) <= set(SURFACES), set(DONE) - set(SURFACES)


@pytest.mark.parametrize("page", DONE)
def test_done_page_is_fully_sourced(page):
    path = ROOT / page
    bad = (_md_unsourced(path) if page.endswith(".md")
           else _profiles_unsourced(path) if page.endswith(".json") else _py_unsourced(path))
    assert not bad, f"{page}:\n  " + "\n  ".join(bad[:40])
