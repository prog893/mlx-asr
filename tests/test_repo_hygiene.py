"""Nothing gitignored may be tracked.

`.gitignore` is where this repo keeps what must not be published: audio, model
conversions, `bench_out/` (local run outputs and notes that name the benchmark
machine), PROGRESS.md. `git add -f` gets past it silently, and once did: a
session handoff note under `bench_out/` was force-added, pushed to a PR, and had
to be scrubbed from history. This catches a force-added path before it is pushed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_no_tracked_file_matches_gitignore():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--ignored", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert not out, f"tracked but gitignored (force-added?): {out}"
