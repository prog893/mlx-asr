"""Tests for the benchmark harness itself, not for the CLI it measures.

A harness bug is worse than a code bug, because it does not look like a failure:
it looks like a result. Every property here corresponds to a bug that shipped a
wrong published number.

- **Forced language.** `run_timing_sweep.py` hardcoded `language='ja'` for
  Whisper. The corpus contains the same video dubbed into two languages, so the
  English file was transcribed as Japanese, looped, and produced degenerate
  timestamps. The timing metric then discarded the row as untrustworthy, which
  read as a model weakness rather than a harness error.
- **Cue config was unrecordable.** Break-F1 was published at a cue grouping the
  CLI does not ship, and nothing in the output said which grouping produced it.
- **The shipped defaults were restated by hand** in `sweep_cues.py`, so its
  "current defaults" baseline kept comparing against a stale pair.
"""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mlx_asr.output import build_cues

pytest.importorskip("rapidfuzz", reason="benchmark harness needs the eval extra")

from benchmarks.run_timing_sweep import reference_language  # noqa: E402

CLI = [sys.executable, "-m", "mlx_asr.cli"]

JA_SRT = """1
00:00:01,000 --> 00:00:03,000
日本語のテキストです

2
00:00:03,000 --> 00:00:05,000
これは字幕の参照です
"""

EN_SRT = """1
00:00:01,000 --> 00:00:03,000
This is an English reference

2
00:00:03,000 --> 00:00:05,000
with spaces between the words
"""


def test_reference_language_detects_japanese(tmp_path):
    p = tmp_path / "ja.srt"
    p.write_text(JA_SRT, encoding="utf-8")
    assert reference_language(p) == "ja"


def test_reference_language_detects_english(tmp_path):
    """The bug in one line: this file used to be transcribed as Japanese."""
    p = tmp_path / "en.srt"
    p.write_text(EN_SRT, encoding="utf-8")
    assert reference_language(p) == "en"


def test_reference_language_is_decided_per_file_not_globally(tmp_path):
    """Two references in one corpus may disagree, which is the whole point.

    The timed corpus really does contain a Japanese and an English file of
    identical duration, so a single corpus-wide language is always wrong for one
    of them.
    """
    ja, en = tmp_path / "a.srt", tmp_path / "b.srt"
    ja.write_text(JA_SRT, encoding="utf-8")
    en.write_text(EN_SRT, encoding="utf-8")
    assert {reference_language(ja), reference_language(en)} == {"ja", "en"}


def test_cli_exposes_the_cue_knobs_the_docs_tell_users_to_sweep():
    """`mlx_asr/output.py` advises sweeping `gap_s`; that needs to be reachable.

    It was documented advice with no flag behind it, which also forced the
    break-F1 re-measurement to go through a code edit.
    """
    r = subprocess.run(CLI + ["--help"], capture_output=True, text=True, cwd=ROOT)
    for flag in ("--gap-seconds", "--max-chars", "--max-dur-seconds"):
        assert flag in r.stdout, flag


def test_resolved_cue_config_reports_shipped_defaults_when_unset():
    """An empty override set must report what actually ships, not None."""
    from mlx_asr.cli import _resolved_cue_config

    sig = inspect.signature(build_cues).parameters
    assert _resolved_cue_config({}) == {
        "gap_s": sig["gap_s"].default,
        "max_chars": sig["max_chars"].default,
        "max_dur_s": sig["max_dur_s"].default,
    }


def test_resolved_cue_config_reports_overrides():
    from mlx_asr.cli import _resolved_cue_config

    got = _resolved_cue_config({"gap_s": 0.7, "max_chars": 32})
    assert got["gap_s"] == 0.7 and got["max_chars"] == 32
    # untouched knob still reports its shipped value, not None
    assert got["max_dur_s"] == inspect.signature(build_cues).parameters[
        "max_dur_s"].default


def test_resolved_cue_config_is_not_a_hardcoded_copy():
    """It must track `build_cues`, so the two cannot drift.

    `sweep_cues.py` drifted exactly this way: it named 1.0/32 as the current
    default for long after the shipped pair became 1.2/28.
    """
    from mlx_asr.cli import _resolved_cue_config

    src = inspect.getsource(_resolved_cue_config)
    assert "signature" in src, "should read the defaults, not restate them"


def test_sweep_cues_reads_shipped_defaults_rather_than_restating_them():
    src = (ROOT / "scripts" / "benchmarks" / "sweep_cues.py").read_text(encoding="utf-8")
    assert "inspect.signature(build_cues)" in src
    # the stale literal comparison must be gone
    assert 'r["gap_s"] == 1.0' not in src


def test_run_matrix_invokes_the_metrics_package_not_bare_filenames():
    """`run_matrix.sh` called `$PY eval_cer.py`, which is not a path that exists."""
    src = (ROOT / "scripts" / "benchmarks" / "run_matrix.sh").read_text(encoding="utf-8")
    assert "-m metrics.eval_cer" in src
    assert "$PY eval_cer.py" not in src


def _corpus_run(label, n, unit="char", dur0=100.0):
    """A minimal run JSON of the shape summarize_engines/compare_engines read."""
    rows = [{"file": f"rec-{i:02d}", "duration_s": dur0 + i * 60, "unit": unit,
             "ref_chars": 100, "hyp_chars": 100, "extra_ratio": 1.0,
             "coverage": 0.9, "cer": 0.1, "coverage_cer": 0.1,
             "x_realtime": 20.0}
            for i in range(1, n + 1)]
    return {"label": label, "complete": True,
            "aggregate": {unit: 0.1}, "ref_units": {unit: 100 * n, "word": 0},
            "x_realtime": 20.0, "results": rows}


def test_summarize_engines_header_reflects_the_actual_input(tmp_path):
    """The header used to state "7 recordings, 5.18h" for any input whatsoever.

    Asserted on the rendered output from synthetic input, so it tests the behaviour
    rather than the wording of the comment explaining it, and does not depend on
    result files, which this repo deliberately does not carry.
    """
    p = tmp_path / "run.json"
    p.write_text(json.dumps(_corpus_run("synthetic", 11)))
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmarks" / "summarize_engines.py"), str(p)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    header = next((ln for ln in r.stdout.splitlines()
                   if ln.startswith("### Corpus:")), "")
    assert header, r.stdout[:400]
    assert "11 recordings" in header, header
    # and the duration is summed, not assumed
    assert "5.18h" not in header, header


def test_timing_sweep_records_cue_overrides_and_language(tmp_path):
    """The output must say which cue config and language produced it.

    Without these fields a break-F1 number cannot be attributed to a
    configuration, which is how the published figure came to describe settings the
    CLI does not ship.
    """
    src = (ROOT / "scripts" / "benchmarks" / "run_timing_sweep.py").read_text(encoding="utf-8")
    assert '"cue_overrides"' in src
    assert '"language_forced"' in src
    assert '"language"' in src          # per-row, since it varies per file
    assert "language='ja'" not in src   # the hardcoded value is gone


def test_compare_engines_does_not_collapse_equal_duration_files(tmp_path):
    """Two recordings of identical length must both survive the paired join.

    `compare_engines.py` matches files across runs by rounded duration rather than
    by name, so that a run whose filenames were anonymized before sharing still
    compares against a local one. The corpus contains one recording dubbed into two
    languages, identical in length to the sample, which that key silently collapsed
    to one row.
    """
    sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
    from benchmarks.compare_engines import load_run

    run = {"label": "x", "results": [
        {"file": "rec-14", "duration_s": 778.3, "unit": "char",
         "ref_chars": 100, "coverage_cer": 0.1},
        {"file": "rec-20", "duration_s": 778.3, "unit": "word",
         "ref_chars": 100, "coverage_cer": 0.2},
    ]}
    p = tmp_path / "run.json"
    p.write_text(json.dumps(run))
    _, rows, _ = load_run(str(p))
    assert len(rows) == 2, f"a file was dropped: {rows}"
    assert {r["file"] for r in rows.values()} == {"rec-14", "rec-20"}


INCOMPLETE = {
    "engine": "mlx-whisper", "label": "partial", "complete": False,
    "files_scored": 2, "files_expected": 20,
    "aggregate": {"char": 0.05}, "ref_units": {"char": 100, "word": 0},
    "results": [
        {"file": "rec-01", "duration_s": 10.0, "unit": "char",
         "ref_chars": 50, "coverage_cer": 0.05},
        {"file": "rec-02", "duration_s": 20.0, "unit": "char",
         "ref_chars": 50, "coverage_cer": 0.05},
    ],
}


def test_summarize_engines_drops_an_incomplete_run(tmp_path):
    """A partial run's aggregate is a real mean over a subset, so it would sit in
    the table looking like a result. `run_whisper.py` rewrites its JSON after every
    file, so half-written runs on disk are the normal case, not an edge case."""
    p = tmp_path / "partial.json"
    p.write_text(json.dumps(INCOMPLETE))
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmarks" / "summarize_engines.py"), str(p)],
        capture_output=True, text=True, cwd=ROOT)
    assert "skipping incomplete run" in r.stderr
    assert "partial" not in r.stdout


def test_compare_engines_warns_on_an_incomplete_run(tmp_path):
    p = tmp_path / "partial.json"
    p.write_text(json.dumps(INCOMPLETE))
    full = dict(INCOMPLETE, label="full", complete=True)
    q = tmp_path / "full.json"
    q.write_text(json.dumps(full))
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmarks" / "compare_engines.py"),
         str(p), str(q)], capture_output=True, text=True, cwd=ROOT)
    assert "INCOMPLETE" in r.stderr, r.stderr


def test_repeat_distribution_refuses_when_every_run_is_incomplete(tmp_path):
    """The row-count guard is relative: if all runs are equally half-written they
    agree on a wrong count and all look complete. The flag has to be authoritative."""
    paths = []
    for i in (1, 2):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(dict(INCOMPLETE, label=f"r{i}")))
        paths.append(str(p))
    base = tmp_path / "base.json"
    base.write_text(json.dumps(dict(INCOMPLETE, label="base", complete=True)))
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmarks" / "repeat_distribution.py"),
         "--runs", *paths, "--baseline", str(base)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2
    assert "no complete repeat runs" in r.stderr


UNSUPPORTED_ON_WHISPER = [
    "--prompt=x", "--max-batch=8", "--vad", "--compact-silence", "--fast",
    "--overlap-seconds=4", "--kv-bits=8", "--no-kv-quant", "--delay-ms=960",
    "--gain=peak", "--peak-dbfs=-3", "--rms-dbfs=-20",
    "--gap-seconds=0.7", "--max-chars=32", "--max-dur-seconds=5",
]


@pytest.mark.parametrize("flag", UNSUPPORTED_ON_WHISPER)
def test_voxtral_only_flags_error_on_other_engines(flag, tmp_path):
    """An unhonourable flag must exit nonzero, not warn and continue.

    A flag that looks accepted and then does nothing yields output the user reads as
    having been produced with it. That is how this project once published a break-F1
    figure describing a cue config the CLI never applied. The check runs before any
    audio is read, so a nonexistent input still produces the flag error.
    """
    r = subprocess.run(
        CLI + [str(tmp_path / "nope.wav"), "--model", "whisper-turbo", flag],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert "not supported by --model whisper-turbo" in r.stderr
    assert flag.split("=")[0] in r.stderr


def test_language_errors_on_voxtral():
    """The mirror case: --language applies to every engine except voxtral."""
    r = subprocess.run(CLI + ["x.wav", "--language", "ja"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "--language" in r.stderr
    assert "takes no language token" in r.stderr


def test_defaults_do_not_count_as_passed(tmp_path):
    """--delay-ms and --gain have non-None defaults; leaving them alone is not a
    request, so a whisper run with no flags must not trip the check."""
    r = subprocess.run(
        CLI + [str(tmp_path / "nope.wav"), "--model", "whisper-turbo"],
        capture_output=True, text=True, cwd=ROOT)
    assert "not supported" not in r.stderr, r.stderr


# Flags the non-Voxtral engines genuinely honour, plus the ones that are not engine
# config at all. Everything else the parser accepts has to be refused there.
PORTABLE_FLAGS = {
    "--help", "--model", "--list-models", "--language", "--output-format",
    "--output", "--chunk-seconds",   # chunked drivers only; whisper-* refuse it
    "--quiet", "--stats-json",
}


def test_every_voxtral_only_flag_is_covered_by_the_list():
    """The list above must name every Voxtral-only flag the parser accepts.

    Derived from the parser rather than restated, because the two ways this check
    fails are the same one: a flag added to `main()` and to neither the rejection
    list in `_run_other_backend` nor to this test then reaches whisper, looks
    honoured and does nothing. `--peak-dbfs` and `--rms-dbfs` did exactly that;
    they were absent from both, so the suite passed while the flags were accepted
    and ignored.
    """
    from mlx_asr import cli

    parser_flags = set()
    for action in cli.build_parser()._actions:
        parser_flags.update(o for o in action.option_strings if o.startswith("--"))
    listed = {f.split("=")[0] for f in UNSUPPORTED_ON_WHISPER}
    assert parser_flags - PORTABLE_FLAGS - listed == set()


def test_machine_state_records_identity_and_load():
    """Every throughput number needs the machine and its state attached.

    The 20-file headline run was made on a host generating video on the same GPU and
    measured ~2.6x realtime where the config reaches ~31x idle. Nothing in the output
    said so, so the file was indistinguishable from a clean run and the session had to
    be voided from memory rather than from the record.
    """
    from benchmarks.machine_state import machine_state

    s = machine_state()
    for key in ("label", "chip", "ram_gb", "load_1min", "gpu_in_use_gb", "busy"):
        assert key in s, key
    assert isinstance(s["busy"], bool)
    # The label is what lands next to a published number, so it must name the chip
    # and, where the enclosure is known, the cooling class.
    assert s["chip"] is None or s["chip"] in s["label"]


def test_machine_label_uses_nameplate_gib_and_enclosure():
    """`M2 Ultra 128GB (Mac14,14)`, never a decimal-GB rendering of GiB.

    ram_gb divided by 1e9 once reported a 16GB Mac as 17 and a 128GB one as 137, and
    a run labelled `137GB` is a run whose hardware nobody can look up.
    """
    from benchmarks.machine_state import machine_label

    assert machine_label({"chip": "Apple M2 Ultra", "ram_gb": 128,
                          "model_id": "Mac14,14"}) == \
        "Apple M2 Ultra 128GB (Mac14,14)"
    # Missing pieces degrade rather than crash: an unknown enclosure still labels.
    assert machine_label({"chip": "Apple M4", "ram_gb": 16, "model_id": None}) == \
        "Apple M4 16GB"
    assert machine_label({}) == "unknown chip"


def test_busy_is_load_or_gpu_but_never_resident_swap():
    """Swap must not set `busy`.

    macOS never reclaims swap, so a host that paged days ago still reports gigabytes
    while completely idle. Gating on it would flag nearly every run, and a warning
    that always fires is one nobody reads.
    """
    from benchmarks import machine_state as ms

    quiet = {"load_1min": 0.1, "cpu_count": 10, "swap_used_mb": 18000.0,
             "gpu_in_use_gb": 0.2}
    reasons = []
    if quiet["load_1min"] / quiet["cpu_count"] > ms.LOAD_PER_CORE_WARN:
        reasons.append("load")
    if quiet["gpu_in_use_gb"] > ms.GPU_IN_USE_WARN_GB:
        reasons.append("gpu")
    assert reasons == [], "17GB of resident swap alone must not mark a host busy"
    assert not hasattr(ms, "SWAP_WARN_MB"), \
        "swap must not be a busy threshold; see the module comment"


def test_warn_if_busy_says_accuracy_survives_but_timing_does_not():
    """The warning has to state which half of a busy run is still usable.

    Greedy decoding is unaffected by contention, so the accuracy columns of a loaded
    run are valid and its timings are not. A warning that just says "busy" invites
    throwing away good accuracy data or quoting bad throughput.
    """
    from benchmarks.machine_state import warn_if_busy

    lines = []
    busy = warn_if_busy({"label": "Apple M2 Ultra 128GB (Mac14,14)", "busy": True,
                         "busy_reasons": ["50.0GB GPU memory already in use"]},
                        log=lines.append)
    assert busy is True
    msg = " ".join(lines)
    assert "x-realtime" in msg and "Accuracy is unaffected" in msg
    assert "50.0GB" in msg
    assert warn_if_busy({"busy": False}, log=lines.append) is False


def test_plot_evidence_reuses_the_one_t_table():
    """The plotting tool must not carry its own copy of the t distribution.

    It did, briefly, keyed on n instead of degrees of freedom, so a 3-run interval came
    out as +/-1.77 where the printed figure said +/-0.60. Both were describing the same
    three numbers. A plot that disagrees with the table it illustrates is worse than no
    plot, because the reader has no way to tell which one is wrong.
    """
    src = (ROOT / "scripts" / "benchmarks" / "plot_evidence.py").read_text()
    assert "from benchmarks.repeat_distribution import t_crit" in src
    # No second hardcoded t table anywhere in the file.
    assert "12.706" not in src, "plot_evidence.py has its own t table again"


def test_t_crit_is_keyed_on_degrees_of_freedom():
    """t_crit(2) is the value for 3 samples, not for 2.

    Pinned because the off-by-one is invisible in the output: both keys return a
    plausible-looking number and the interval is simply too wide or too narrow.
    """
    from benchmarks.repeat_distribution import t_crit

    assert abs(t_crit(2) - 4.303) < 0.001      # n=3
    assert abs(t_crit(1) - 12.706) < 0.001     # n=2
    assert abs(t_crit(5) - 2.571) < 0.001      # n=6


def test_bootstrap_at_n3_is_reported_as_granular():
    """A bootstrap over 3 files has only 10 distinct resamples, so its CI is not a
    smooth interval and must not be read as one.

    This is arithmetic, not a measurement: multisets of size n from n items is
    C(2n-1, n). At n=3 that is 10, so the 2.5th and 97.5th percentiles land on
    specific compositions. The English side of this corpus is n=3, and every English
    conclusion here rests on it.
    """
    from math import comb

    assert comb(2 * 3 - 1, 3) == 10
    assert comb(2 * 7 - 1, 7) == 1716
    # By n=17 the resample space is large enough that percentiles are meaningful.
    assert comb(2 * 17 - 1, 17) > 10 ** 9


def test_machine_state_records_power_policy():
    """A benchmark on battery or in Low Power Mode is not comparable to one on mains.

    Both cap sustained clocks, so a speed figure taken that way describes a power policy
    rather than a config, and nothing else in the result file would reveal it. Recorded
    for every run because a laptop is a legitimate benchmark host, just not a
    comparable one when it is unplugged.
    """
    from benchmarks.machine_state import machine_state

    s = machine_state()
    for key in ("ac_power", "battery_percent", "low_power_mode"):
        assert key in s, key
    # A Mac Studio has no battery; a laptop does. Either is valid, None is not a crash.
    assert s["ac_power"] in (True, False, None)
    assert s["low_power_mode"] in (True, False, None)


def test_swapout_RATE_is_recorded_not_just_the_total():
    """Resident swap is a property of uptime; the swapout rate is a property of now.

    The distinction is not academic. This project dismissed 17GB of resident swap as
    harmless, correctly, and then missed a host that was actively swapping out ~1.4GB/s
    while a benchmark ran on it. The total looked the same in both cases.
    """
    from benchmarks.machine_state import machine_state

    s = machine_state()
    assert "swapout_pages_per_s" in s
    assert s["swapout_pages_per_s"] is None or s["swapout_pages_per_s"] >= 0


def test_battery_and_thrashing_are_busy_reasons():
    """The warning must name the specific condition, so the reader knows the remedy.

    "not idle" is useless advice; "connect to mains" and "swapping out 1445MB/s" are
    actionable, and they call for different actions.
    """
    from benchmarks.machine_state import warn_if_busy

    for reason, expect in (
        ("on BATTERY power at 60%, connect to mains before benchmarking", "mains"),
        ("Low Power Mode is ON, which caps clocks deliberately", "Low Power Mode"),
        ("swapping out 1445MB/s (92484 pages/s)", "swapping out"),
    ):
        lines = []
        assert warn_if_busy({"label": "host", "busy": True,
                             "busy_reasons": [reason]}, log=lines.append) is True
        assert expect in " ".join(lines)


@pytest.mark.parametrize("form", ["ja", "ja_JP", "JA", "jpn", "Japanese", "japanese"])
def test_cli_accepts_every_spelling_of_a_language(form, tmp_path):
    """A user should not have to know which spelling an engine wants.

    All six reach the audio-loading stage, so they exit 1 on the missing file rather
    than 2 on a bad argument. Before normalisation, mlx-whisper would have silently
    treated "Japanese" and "jpn" as decoder prompt hints instead of language selections.
    """
    r = subprocess.run(
        CLI + [str(tmp_path / "nope.wav"), "--model", "whisper-turbo",
               "--language", form],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1, (form, r.returncode, r.stderr)
    assert "not recognised" not in r.stderr, form


@pytest.mark.parametrize("form", ["Klingon", "xx", "notalanguage"])
def test_cli_rejects_an_unknown_language_before_reading_the_audio(form, tmp_path):
    """Exit 2, and before the decode.

    Validating after the audio read would make a typo cost a full file decode, which on a
    93-minute recording is slow enough to matter. Exit 2 matches the convention for a bad
    invocation used by the unsupported-flag path.
    """
    r = subprocess.run(
        CLI + [str(tmp_path / "nope.wav"), "--model", "whisper-turbo",
               "--language", form],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2, (form, r.returncode, r.stdout, r.stderr)
    assert "not recognised" in r.stderr
    # The audio was never opened, so the file error must NOT appear.
    assert "no such file" not in r.stderr, form
