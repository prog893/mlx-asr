"""Saved results name corpus files only by hash, carry no local paths, and stay complete.

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
FILE_ID = re.compile(r"^(?P<id>[0-9a-f]{12}|[a-z][a-z0-9_-]{0,31})$")   # a hash, or a derived label listed in corpus.json
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


def _rows(run, unit="char", supplements=False):
    """Scored rows; the runner's own unless `supplements` (rows added later by a rerun)."""
    return [f for f in run["files"] if f.get("unit") == unit
            and f.get("coverage_cer") is not None and f.get("ref_chars")
            and (supplements or "supplement" not in f)]


def _weighted(rows):
    return sum(f["coverage_cer"] * f["ref_chars"] for f in rows) / sum(f["ref_chars"] for f in rows)


@pytest.mark.parametrize("path", GROUPS, ids=[p.stem for p in GROUPS])
def test_superseded_audio_rows_are_flagged(path):
    """Every row of a revised file whose duration matches a superseded copy is invalid, and
    so is one with no duration, unless the group records that it ran after every revision."""
    group = json.loads(path.read_text(encoding="utf-8"))
    current = "made after every audio revision" in (group.get("note") or "")
    for run in group["runs"]:
        for f in run["files"]:
            fid = f.get("file", "").split(".")[0]
            for rev in REVISIONS.get(fid, []):
                d = f.get("duration_s")
                if (d is None and not current) or (
                        d is not None and abs(d - rev["previous"]["duration_s"]) < 1.0):
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
                                                         if f.get("unit") == "char"
                                                         and "supplement" not in f]):
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
    # with the replaced file's supplement row, every file is scored once on valid audio
    fixed = [f for f in _rows(run, supplements=True) if "invalid" not in f]
    assert sorted(f["file"] for f in fixed) == sorted(f["file"] for f in rows)


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
            if "output" in run:   # archive_text.py: the queue line verbatim (may be `python -c`)
                continue
            assert "(" not in run["command"] and ")" not in run["command"], run["command"]
            assert "[" not in run["command"], run["command"]   # a list printed as a repr


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
    assert cleaned["corpus"] == "<local>/recordings"
    assert ar.last_component("/data") == "<local>/data"
    assert ar.last_component("/data/") == "<local>/data"
    assert ar.last_component("bench_out/corpus_all") == "bench_out/corpus_all"
    cmd = ar.command("run_corpus.py", cleaned, "x.json")
    assert "private data" not in cmd and cmd.endswith("--json x.json")
    assert ar.Scrubber([]).leaks('"--corpus /mnt/private data/x"') > 0


def test_every_rebuilt_command_names_a_relative_json_output():
    for path in GROUPS:
        for run in json.loads(path.read_text(encoding="utf-8"))["runs"]:
            if "output" in run:     # archive_text.py: probe stdout or CLI --stats-json
                continue
            assert "--json " in run["command"], run["run"]
            assert not _archiver().ABS_PATH.search(run["command"].split("--json ", 1)[1])


def test_duplicate_file_ids_are_refused():
    ar = _archiver()
    with pytest.raises(SystemExit):
        ar.Scrubber(["a_drive", "a"], {"a_drive": "a"})


def _runner_json(path, rows, **cfg):
    config = {"corpus": "/x/corpus_all", "max_batch": 32, "label": "c60b32", **cfg}
    path.write_text(json.dumps({"config": config, "machine": {"chip": "M"},
                                "aggregate": {}, "results": rows}))
    return path


def _row(file, dur, cer):
    return {"file": file, "duration_s": dur, "unit": "char", "x_realtime": 5.0,
            "chunks": 3, "ref_chars": 100, "hyp_chars": 90, "cer": cer, "coverage_cer": cer,
            "subs": 1, "dels": 2, "insertions_counted": 0}


def test_supplement_adds_the_new_row_after_an_exact_canary(tmp_path):
    ar = _archiver()
    orig = _runner_json(tmp_path / "o.json", [_row("can", 112.0, 0.1), _row("new", 558.2, 0.3)])
    supp = _runner_json(tmp_path / "s.json", [_row("can", 112.0, 0.1), _row("new", 783.5, 0.2)],
                        files=["can", "new"])
    [run] = ar.build_run(orig, True, [supp])
    assert [(f["file"], f["duration_s"]) for f in run["files"]] == \
        [("can", 112.0), ("new", 558.2), ("new", 783.5)]
    assert run["files"][2]["supplement"] == 0 and "supplement" not in run["files"][1]
    s = run["supplements"][0]
    assert s["canary"] == ["can"] and s["added"] == ["new"]
    assert "--files can new" in s["command"] and s["command"].endswith("--json s.json")
    ar.mark_invalid(run, {"new": [{"superseded": "d", "reason": "r",
                                   "previous": {"duration_s": 558.2}}]})
    assert ["invalid" in f for f in run["files"]] == [False, True, False]


def test_supplement_refuses_a_canary_mismatch_or_another_config(tmp_path):
    ar = _archiver()
    orig = _runner_json(tmp_path / "o.json", [_row("can", 112.0, 0.1)])
    drift = _runner_json(tmp_path / "s1.json", [_row("can", 112.0, 0.11), _row("new", 783.5, 0.2)])
    other = _runner_json(tmp_path / "s2.json", [_row("can", 112.0, 0.1), _row("new", 783.5, 0.2)],
                         max_batch=64)
    no_canary = _runner_json(tmp_path / "s3.json", [_row("new", 783.5, 0.2)])
    for supp in (drift, other, no_canary):
        with pytest.raises(SystemExit):
            ar.build_run(orig, True, [supp])


def test_files_option_selects_stems_and_refuses_unknown_ones():
    sys.path.insert(0, str(ROOT / "scripts"))
    from benchmarks.corpus_files import only_files
    pairs = [(Path("c/a.wav"), Path("c/a.srt")), (Path("c/b.wav"), Path("c/b.srt"))]
    assert only_files(pairs, None) == pairs
    assert only_files(pairs, ["b"]) == pairs[1:]
    with pytest.raises(SystemExit):
        only_files(pairs, ["b", "zz"])


def test_whisper_canary_drift_is_recorded_not_refused(tmp_path):
    ar = _archiver()
    orig = tmp_path / "o.json"
    supp = tmp_path / "s.json"
    for p, rows in ((orig, [_row("can", 112.0, 0.1)]),
                    (supp, [_row("can", 112.0, 0.12), _row("new", 783.5, 0.2)])):
        p.write_text(json.dumps({"engine": "mlx-whisper", "config": {"model": "w"},
                                 "machine": {}, "aggregate": {}, "results": rows}))
    [run] = ar.build_run(orig, True, [supp])
    assert run["supplements"][0]["canary_drift"] == {"can": {"cer": [0.1, 0.12],
                                                             "coverage_cer": [0.1, 0.12]}}
    assert run["files"][-1]["file"] == "new"


def test_derived_prompt_texts_are_never_archived(tmp_path):
    """Topic and terms prompts name corpus content: config and arms keep only placeholders."""
    ar = _archiver()
    secret = "PRIVATETERM"
    run = {"config": {"corpus": "c", "topic_ja": secret, "terms_en": secret, "terms_ja": ""},
           "machine": {}, "arms_expected": 2, "arms": [
               {"content": "terms", "lang": "en", "text": secret, "aggregate": {},
                "results": []},
               {"content": "instruction", "lang": "en", "text": "Transcribe.",
                "aggregate": {}, "results": []}]}
    (tmp_path / "p.json").write_text(json.dumps(run))
    out = json.dumps(ar.build_run(tmp_path / "p.json", True))
    assert secret not in out and "Transcribe." in out
    assert ar.PROMPT_PLACEHOLDER in out


def test_pair_arguments_repeat_their_flag():
    ar = _archiver()
    cmd = ar.command("run_whisper.py", {"pair": [["a.wav", "a.srt"], ["b.wav", "b srt.txt"]]})
    assert cmd == "scripts/benchmarks/run_whisper.py --pair a.wav a.srt --pair b.wav 'b srt.txt'"
    assert ar.last_component([["/x/y/a.wav", "rel/a.srt"]]) == [["<local>/a.wav", "rel/a.srt"]]


@pytest.mark.parametrize("path", GROUPS, ids=[p.stem for p in GROUPS])
def test_run_names_are_unique_in_a_group(path):
    """A run is cited by name (and arm), so two runs may not share one."""
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    keys = [(r["run"], r.get("arm")) for r in runs]
    assert len(keys) == len(set(keys)), [k for k in keys if keys.count(k) > 1]


def test_set_replaces_a_parameter_the_runner_did_not_record(tmp_path):
    """sweep_qwen3_batch.py records neither corpus nor group budget; the queue line does."""
    ar = _archiver()
    d = {"model": "mlx-community/Qwen3-ASR-1.7B-8bit", "chunk_seconds": 15.0, "language": "ja",
         "arms": {"4": {"aggregate": {}, "per_file": [], "x_realtime": 1.9}}}
    (tmp_path / "b.json").write_text(json.dumps(d))
    [run] = ar.build_run(tmp_path / "b.json", True,
                         overrides={"corpus": "bench_out/one", "group_budget": False})
    assert run["params"]["corpus"] == "bench_out/one" and run["params"]["group_budget"] is False
    assert "--corpus bench_out/one " in run["command"] and "--group-budget" not in run["command"]
    [default] = ar.build_run(tmp_path / "b.json", True)
    assert "--group-budget" in default["command"]


def _archive_text(tmp_path, *extra):
    import subprocess
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "rec_a.wav").write_bytes(b"")
    queue = tmp_path / "queue.zsh"
    queue.write_text("run probe_x $PY probe.py $D/clipvar/rec_a_quiet.wav\n"
                     "arm fw_r1 /usr/bin/time -p $PY run.py --audio rec_a.wav\n")
    (tmp_path / "probe_x.txt").write_text("Fetching 4 files: 100%|x|\nrec_a_quiet: 1.0\n")
    (tmp_path / "fw_r1.log").write_text("fw_r1: 3.6x\nreal 10.0\nuser 40.0\nsys 2.0\n")
    out = tmp_path / "g.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "benchmarks" / "archive_text.py"),
                        "--corpus", str(tmp_path / "corpus"), "--queue", str(queue),
                        "--group", "g", "--question", "q", "--out", str(out), *extra,
                        str(tmp_path / "probe_x.txt"), str(tmp_path / "fw_r1.log")],
                       capture_output=True, text=True, cwd=ROOT / "scripts" / "benchmarks")
    return r, out


def test_archive_text_keeps_arm_logs_and_aliases_derived_inputs(tmp_path):
    r, out = _archive_text(tmp_path, "--alias", "rec_a_quiet=clipquiet")
    assert r.returncode == 0, r.stderr
    runs = {x["run"]: x for x in json.loads(out.read_text())["runs"]}
    assert runs["probe_x"]["stdout"] == "clipquiet: 1.0"
    assert "clipquiet.wav" in runs["probe_x"]["command"]
    assert runs["fw_r1"]["output"] == "log" and "user 40.0" in runs["fw_r1"]["log"]
    assert runs["fw_r1"]["command"].startswith("time -p $PY run.py")   # absolute path cut
    assert "rec_a" not in out.read_text()


def test_filtered_input_is_cataloged_by_its_filter_not_its_name(tmp_path):
    import subprocess
    import wave
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for path in (corpus / "rec_a.wav", tmp_path / "rec_a_quiet.wav"):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
            w.writeframes(b"\0\0" * 16000)
    runj = tmp_path / "run.json"
    runj.write_text(json.dumps({"results": [{"file": "rec_a.wav", "unit": "char",
                                             "ref_chars": 10}]}))
    out = tmp_path / "corpus.json"
    script = ROOT / "scripts" / "benchmarks" / "archive_corpus.py"
    base = [sys.executable, str(script), "--corpus", str(corpus), "--languages", str(runj),
            "--out", str(out)]
    ok = subprocess.run(base + ["--filtered",
                                f"clipquiet=rec_a:{tmp_path / 'rec_a_quiet.wav'}:volume=-14dB"],
                        capture_output=True, text=True, cwd=script.parent)
    assert ok.returncode == 0, ok.stderr
    entry = json.loads(out.read_text())["derived"]["clipquiet"]
    assert entry["ffmpeg_af"] == "volume=-14dB" and entry["duration_s"] == 1.0
    assert "rec_a" not in out.read_text()
    bad = subprocess.run(base + ["--filtered", f"rec_a=rec_a:{tmp_path / 'rec_a_quiet.wav'}:x"],
                         capture_output=True, text=True, cwd=script.parent)
    assert bad.returncode != 0


def test_redacted_parameter_leaves_params_and_command(tmp_path, monkeypatch):
    ar = _archiver()
    run = {"config": {"corpus": "c", "prompt": "SECRETTERMS", "label": "x"}, "machine": {},
           "aggregate": {}, "results": [_row("a", 10.0, 0.1)]}
    (tmp_path / "x.json").write_text(json.dumps(run))
    out = tmp_path / "g.json"
    monkeypatch.setattr(sys, "argv", ["archive_results.py", "--corpus", str(tmp_path),
                                      "--group", "g", "--question", "q", "--redact", "x=prompt",
                                      "--out", str(out), str(tmp_path / "x.json")])
    assert ar.main() == 0
    text = out.read_text()
    assert "SECRETTERMS" not in text and ar.PROMPT_PLACEHOLDER in text
