"""Sweep Qwen3-ASR's decode-window length on the 7-file subset.

The first experiment this engine needs, because the library default
(`chunk_duration=1200.0`) is a value nobody has measured and it moves three things at
once: how many segments the output has, whether the batched code path can engage at
all (it needs more than one chunk), and peak memory (the audio encoder stays
unquantized at every precision level and its attention mask is materialised densely,
so a longer window costs more than the weight size suggests).

Runs the arms **sequentially**, one model load per arm, and re-reads machine state
between them. Never in parallel: this project has already had to void a session whose
throughput figures were taken while the host was busy with something else, and two
concurrent decodes contend for the same GPU by construction.

    uv run python scripts/benchmarks/sweep_qwen3_chunk.py --corpus DIR --outdir DIR \\
        [--windows 30 60 120 300] [--model qwen3-asr-small]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

from benchmarks.machine_state import machine_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model", default="qwen3-asr")
    p.add_argument("--windows", type=float, nargs="+",
                   default=[30.0, 60.0, 120.0, 300.0])
    p.add_argument("--limit", type=int)
    p.add_argument("--require-idle", action="store_true")
    p.add_argument("--wait-for-idle", type=float, default=20.0)
    a = p.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in a.windows:
        tag = f"{a.model}_c{w:.0f}"
        out = outdir / f"{tag}.json"
        print(f"\n=== window {w:g}s -> {out} ===", flush=True)
        cmd = [sys.executable, str(HERE / "run_qwen3.py"),
               "--corpus", a.corpus, "--model", a.model,
               "--chunk-seconds", str(w), "--json", str(out),
               "--keep-hyp", str(outdir / "hyp"), "--label", tag]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if a.require_idle:
            cmd += ["--require-idle", "--wait-for-idle", str(a.wait_for_idle)]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"arm {w:g}s failed with {r.returncode}; stopping so a partial "
                  f"sweep is not read as a complete one", file=sys.stderr)
            return r.returncode
        d = json.loads(out.read_text())
        rows.append((w, d))
        # Between arms, not just at the start: a host that went busy midway through
        # would otherwise leave one arm's throughput quietly incomparable.
        s = machine_state()
        print(f"[between arms] load {s['load_1min']}, {s['gpu_in_use_gb']}GB GPU, "
              f"busy={s['busy']}")
        time.sleep(5)

    print(f"\n{'window':>8} {'covCER':>8} {'kanaCER':>8} {'lenCER':>8} {'covWER':>8} "
          f"{'x rt':>7} {'segs':>6} {'peakGB':>7}")
    for w, d in rows:
        agg = d["aggregate"]
        segs = sum(r.get("segments", 0) for r in d["results"] if "segments" in r)
        print(f"{w:>7.0f}s {100*agg.get('char', float('nan')):>7.2f}% "
              f"{100*agg.get('char_kana', float('nan')):>7.2f}% "
              f"{100*agg.get('char_lenient', float('nan')):>7.2f}% "
              f"{100*agg.get('word', float('nan')):>7.2f}% "
              f"{d['x_realtime']:>6.1f}x {segs:>6} "
              f"{d.get('peak_memory_gb') or float('nan'):>7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
