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
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
from archive_results import ABS_PATH, Scrubber, command  # noqa: E402
RESULTS = ROOT / "benchmarks" / "results"
CORPUS = json.loads((RESULTS / "corpus.json").read_text(encoding="utf-8"))
GROUPS = sorted(p for p in RESULTS.rglob("*.json") if p.name not in ("corpus.json", "revisions.json"))
NAME_KEYS = {"file", "files", "looped_files", "skipped_files", "truncated_files"}
FILE_ID = re.compile(r"^(?P<id>[0-9a-f]{12}|worst\d+)$")
DATE_STAMPED = re.compile(r"\d{6}_\d{3}")
RUN_KEYS = {"run", "command", "params", "machine", "aggregate", "measures", "files", "missing"}
ALLOWED_META = {"language", "duration_s", "codec", "sample_rate", "bit_depth", "bitrate",
                "reference_length", "reference_unit", "revisions"}
REVISIONS = {k: v for k, v in json.loads((RESULTS / "revisions.json").read_text(
    encoding="utf-8")).items() if not k.startswith("_")}


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
        assert not ABS_PATH.search(s), f"absolute path in {key}"
        assert not DATE_STAMPED.search(s), f"date-stamped name in {key}"


def test_every_run_group_is_described():
    for path in GROUPS:
        readme = path.parent / "README.md"
        assert readme.exists(), f"{path.parent} has no README.md"
        assert path.stem in readme.read_text(encoding="utf-8"), f"{path.stem} not in README"


def _rows(run, unit="char"):
    return [f for f in run["files"] if f.get("unit") == unit
            and f.get("coverage_cer") is not None and f.get("ref_chars")]


def _weighted(rows):
    return sum(f["coverage_cer"] * f["ref_chars"] for f in rows) / sum(f["ref_chars"] for f in rows)


@pytest.mark.parametrize("path", GROUPS, ids=[p.stem for p in GROUPS])
def test_superseded_audio_rows_are_flagged(path):
    """Every row of a revised file whose duration matches a superseded copy is invalid."""
    for run in json.loads(path.read_text(encoding="utf-8"))["runs"]:
        for f in run["files"]:
            fid = f.get("file", "").split(".")[0]
            for rev in REVISIONS.get(fid, []):
                d = f.get("duration_s")
                if d is None or abs(d - rev["previous"]["duration_s"]) < 1.0:
                    assert "invalid" in f, (path.stem, run["run"])
                    assert fid in run.get("invalid_files", []), (path.stem, run["run"])


@pytest.mark.parametrize("path", GROUPS, ids=[p.stem for p in GROUPS])
def test_published_aggregate_recomputes_from_rows(path):
    """The raw rows are enough: each run's own JP aggregate is reproduced from them."""
    for run in json.loads(path.read_text(encoding="utf-8"))["runs"]:
        agg = (run.get("aggregate") or {}).get("char")
        if isinstance(agg, dict):              # sweep_qwen3_batch.py: {"error_rate": ...}
            agg = agg.get("error_rate")
        rows = _rows(run)
        if agg is None or not rows or len(rows) != len([f for f in run["files"]
                                                         if f.get("unit") == "char"]):
            continue
        assert abs(_weighted(rows) - agg) < 5e-4, (path.stem, run["run"], _weighted(rows), agg)


def test_excluding_invalid_rows_matches_the_rescore():
    """Voxtral at the default config: 16.29% with the truncated file, 15.84% when that file
    is rescored against a reference cut to its audio. Dropping the file entirely must land
    below the published figure."""
    g = json.loads((RESULTS / "2026-10" / "ultra-voxtral-headline.json").read_text(encoding="utf-8"))
    run = next(r for r in g["runs"] if r["run"] == "vox_default_c30b128_kv8")
    rows = _rows(run)
    assert abs(_weighted(rows) - 0.16287) < 5e-4
    valid = [f for f in rows if "invalid" not in f]
    assert len(valid) == len(rows) - 1 and _weighted(valid) < _weighted(rows)


@pytest.mark.parametrize("path", ["/mnt/private/input.wav", "/Volumes/disk/a/b.wav",
                                  "/Users/x/corpus_all", "/opt/data/set/rec.wav"])
def test_any_absolute_path_is_scrubbed_and_detected(path):
    """Not only a fixed list of top-level directories: any absolute path counts."""
    scrub = Scrubber([])
    out = scrub(f"run.py --corpus {path} --model mlx-community/x https://h/y bench_out/z")
    assert not ABS_PATH.search(out), out
    assert "mlx-community/x" in out and "https://h/y" in out and "bench_out/z" in out
    assert scrub.leaks(f'"{path}"') == 1


def test_rebuilt_commands_are_runnable_shapes():
    """A sweep arm is selected by its own argument, never by a label appended to the
    command; a negative value is attached to its flag so argparse does not read an option."""
    cmd = command("sweep_gain.py", {"modes": "-12", "max_batch": 32, "vad": False, "x": None})
    assert cmd == "scripts/benchmarks/sweep_gain.py --modes=-12 --max-batch 32"
    for path in GROUPS:
        for run in json.loads(path.read_text(encoding="utf-8"))["runs"]:
            assert "(" not in run["command"] and ")" not in run["command"], run["command"]


def test_short_stem_does_not_rewrite_ordinary_text():
    scrub = Scrubber(["a", "rec"])
    out = scrub({"files": [{"file": "a", "unit": "char"}], "params": {"label": "data"}})
    assert out["files"][0]["file"] != "a" and out["files"][0]["unit"] == "char"
    assert "files" in out and out["params"]["label"] == "data"
    assert scrub.leaks('{"data": "char"}') == 0


def test_values_with_spaces_stay_one_argument():
    assert command("run_corpus.py", {"prompt": "two words"}) == \
        "scripts/benchmarks/run_corpus.py --prompt 'two words'"


def _archiver():
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
    import archive_results
    return archive_results


def test_parameter_path_with_spaces_is_cut_whole_and_detected():
    ar = _archiver()
    params = {"corpus": "/mnt/private data/recordings", "max_batch": 32}
    cleaned = {k: ar.last_component(v) for k, v in params.items()}
    assert cleaned["corpus"] == "recordings"
    cmd = ar.command("run_corpus.py", cleaned, "x.json")
    assert "private data" not in cmd and cmd.endswith("--json x.json")
    assert ar.Scrubber([]).leaks('"--corpus /mnt/private data/x"') > 0


def test_every_rebuilt_command_names_a_relative_json_output():
    for path in GROUPS:
        for run in json.loads(path.read_text(encoding="utf-8"))["runs"]:
            assert "--json " in run["command"], run["run"]
            assert not _archiver().ABS_PATH.search(run["command"].split("--json ", 1)[1])


def test_duplicate_file_ids_are_refused():
    ar = _archiver()
    with pytest.raises(SystemExit):
        ar.Scrubber(["a_drive", "a"], {"a_drive": "a"})
