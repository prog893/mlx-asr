"""Run every registry alias through the real CLI and report whether it works.

The point is not accuracy (docs/benchmarks/engines.md has that on a proper corpus) but coverage:
does each alias actually load, decode, and write a file, on this machine, with no
torch and no MPS. That question was answered wrong before, twice: `--model kotoba`
pointed at weights with no MLX build, and the whisper aliases were only ever
spot-checked on `tiny`.

Scores plain CER when a complete reference is available, which is meaningful on
the narration clip and nowhere else in this project (see eval_coverage.py).

    uv run --extra whisper --extra eval python scripts/benchmarks/matrix_models.py \\
        --audio narration.wav --reference narration.txt --json matrix.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Both the repo root (for mlx_asr) and scripts/ (for metrics) have to be
# importable; this file lives two levels down from the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlx_asr.models import REGISTRY


def srt_to_text(path):
    """Concatenate cue text, dropping indices and timing lines."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or "-->" in line or line.strip().isdigit():
            continue
        out.append(line.strip())
    return "".join(out)


def run_one(alias, spec, audio, outdir, extra_args=()):
    """Invoke the CLI exactly as a user would, and report what happened."""
    out = outdir / f"{alias}.srt"
    cmd = [sys.executable, "-m", "mlx_asr.cli", str(audio),
           "--model", alias, "-o", str(out)]
    # Whisper autodetects and misfires on this material (it has returned Russian
    # for Japanese), so the matrix passes the hint every engine that takes one
    # needs. Voxtral takes none and reports the flag as ignored.
    if spec.needs_language:
        cmd += ["--language", "ja"]
    cmd += list(extra_args)

    t = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t

    row = {"alias": alias, "backend": spec.backend, "repo": spec.repo,
           "cmd": " ".join(cmd[2:]), "returncode": proc.returncode,
           "wall_s": round(elapsed, 1)}
    if proc.returncode != 0:
        row["ok"] = False
        tail = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
        row["error"] = tail[-1] if tail else "(no stderr)"
        return row
    if not out.exists() or out.stat().st_size == 0:
        row["ok"] = False
        row["error"] = "exited 0 but wrote no output"
        return row

    row["ok"] = True
    row["cues"] = Path(out).read_text(encoding="utf-8").count(" --> ")
    row["out"] = str(out)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--reference", help="complete transcript, for plain CER")
    p.add_argument("--outdir", default="bench_out/matrix")
    p.add_argument("--json")
    p.add_argument("--only", help="comma-separated aliases (default: all)")
    a = p.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    aliases = a.only.split(",") if a.only else list(REGISTRY)

    duration = None
    ref = Path(a.reference).read_text(encoding="utf-8") if a.reference else None
    scorer = None
    if ref:
        from metrics.eval_cer import cer
        scorer = cer

    rows = []
    for alias in aliases:
        spec = REGISTRY[alias]
        print(f"[{alias}] {spec.repo} ({spec.backend}) ...", flush=True)
        row = run_one(alias, spec, a.audio, outdir)
        if row["ok"] and scorer:
            row["cer"] = round(100 * scorer(ref, srt_to_text(row["out"])), 2)
        if duration is None and row["ok"]:
            from mlx_asr.audio import SAMPLE_RATE, load_audio_16k
            duration = len(load_audio_16k(a.audio)) / SAMPLE_RATE
        if row["ok"] and duration:
            row["x_realtime"] = round(duration / row["wall_s"], 1)
        rows.append(row)
        status = "ok" if row["ok"] else f"FAILED: {row['error']}"
        extra = (f"  CER {row['cer']}%" if "cer" in row else "")
        print(f"  -> {status}  {row['wall_s']}s{extra}", flush=True)

    # Sanity check, because the first version of this script silently produced a
    # perfect all-green matrix by forgetting to pass --model: every row ran the
    # default engine. Two different engines agreeing byte-for-byte on 3 minutes of
    # spontaneous speech does not happen, so treat it as a harness bug.
    digests = {}
    for r in rows:
        if r["ok"]:
            digests.setdefault(srt_to_text(r["out"]), []).append(r["alias"])
    collisions = [v for v in digests.values() if len(v) > 1]
    if collisions:
        print("\nHARNESS BUG: identical output from different models: "
              f"{collisions}", file=sys.stderr)
        return 2

    width = max(len(r["alias"]) for r in rows)
    print(f"\n{'alias':<{width}}  {'backend':<12} {'works':<6} {'x-rt':>6} "
          f"{'CER':>7}  cues")
    for r in rows:
        print(f"{r['alias']:<{width}}  {r['backend']:<12} "
              f"{'yes' if r['ok'] else 'NO':<6} "
              f"{r.get('x_realtime', ''):>6} {r.get('cer', ''):>7}  "
              f"{r.get('cues', r.get('error', ''))}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"audio": str(a.audio), "duration_s": duration,
                       "rows": rows}, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {a.json}")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
