"""Rerun saved run_corpus.py runs on a few corpus files, to add or replace one file.

    .venv/bin/python scripts/benchmarks/replay_supplement.py --corpus DIR \\
        --files CANARY_STEM NEW_STEM --out-dir DIR RUN.json [RUN.json ...]

Each RUN.json is a runner's own output. Its recorded config is replayed with `--files`,
one run at a time, into `<out-dir>/<parent>__<run>.json` (skipped if that exists), so an
interrupted replay resumes. Accuracy is not deterministic across machines, so a run
recorded on a different machine model, macOS or mlx version is refused. Each output is
then saved with `archive_results.py --supplement RUN.json=OUT.json`, which checks the
canary row against the original. The 16k conversions go to `<out-dir>/tmp16k` (TMPDIR),
because macOS cleans the system temp directory during long runs.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.machine_state import machine_state  # noqa: E402

SAME_SOFTWARE = ("model_id", "macos", "mlx")
NOT_REPLAYED = {"json", "keep_hyp", "files", "limit", "corpus"}


def argv(cfg: dict) -> list[str]:
    """The recorded config as runner arguments, `--flag=value` so a negative value parses."""
    out = []
    for k, v in cfg.items():
        if k in NOT_REPLAYED or v is None or v is False or v == "":
            continue
        flag = "--" + k.replace("_", "-")
        out.append(flag if v is True else f"{flag}={v}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--files", nargs="+", required=True,
                   help="corpus stems: at least one canary already in every run, plus new files")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("runs", nargs="+")
    a = p.parse_args()

    out_dir = Path(a.out_dir)
    tmp = out_dir / "tmp16k"
    tmp.mkdir(parents=True, exist_ok=True)
    here = machine_state()
    runner = Path(__file__).with_name("run_corpus.py")
    for run in map(Path, a.runs):
        d = json.loads(run.read_text())
        if not isinstance(d.get("results"), list) or d.get("engine"):
            sys.exit(f"{run}: only run_corpus.py runs are supported")
        diff = [k for k in SAME_SOFTWARE if d.get("machine", {}).get(k) != here.get(k)]
        if diff:
            sys.exit(f"{run}: recorded on a different {diff}, refused")
        out = out_dir / f"{run.parent.name}__{run.stem}.json"
        cmd = [sys.executable, str(runner), "--corpus", str(Path(a.corpus).resolve()),
               *argv(d["config"]), "--files", *a.files, "--json", str(out.resolve())]
        if out.exists():
            print(f"skip {out.name} (exists)")
            continue
        print(f"run  {out.name}", flush=True)
        if a.dry_run:
            continue
        log = out.with_suffix(".log").open("w")
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT,
                       env={**os.environ, "TMPDIR": str(tmp.resolve())},
                       cwd=Path(__file__).resolve().parents[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
