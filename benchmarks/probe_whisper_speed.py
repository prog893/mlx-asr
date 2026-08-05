"""Measure decode speed per Whisper model on a fixed 120s slice.

Run before the corpus sweep: the full corpus is 5.2h of audio, so a model at
2x realtime costs 2.5h and one at 15x costs 20 minutes. This prints the
projected corpus wall clock so the sweep can be ordered cheapest-first.

Downloads and weight load are excluded (a 1s silent warm-up call precedes the
timed one), matching how run_whisper.py reports x-realtime.

**Each model is timed in its own subprocess.** `mlx_whisper.load_models` is
`lru_cache`d, so timing several models in one process keeps every previous set of
weights resident; the later ones then contend for memory and read as far slower
than they are. Measured cost of getting this wrong: large-v3 timed 1.85x as the
7th model in-process versus 15.3x alone, an 8x error that would have mis-ordered
the whole sweep. One process per model is the fix.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MODELS = [
    "mlx-community/whisper-tiny-mlx",
    "mlx-community/whisper-base-mlx",
    "mlx-community/whisper-small-mlx",
    "mlx-community/whisper-medium-mlx",
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-large-v2-mlx",
    "mlx-community/whisper-large-v3-mlx",
]


CHILD = """
import json, sys, time
from mlx_whisper.transcribe import transcribe
audio, repo, warm = sys.argv[1], sys.argv[2], sys.argv[3]
opts = json.loads(sys.argv[4])
t0 = time.perf_counter()
transcribe(warm, path_or_hf_repo=repo, language="ja", verbose=False, **opts)
load_s = time.perf_counter() - t0
t0 = time.perf_counter()
r = transcribe(audio, path_or_hf_repo=repo, language="ja", verbose=False, **opts)
wall = time.perf_counter() - t0
print("RESULT" + json.dumps({"load_s": load_s, "wall_s": wall,
      "chars": len(r["text"]), "segments": len(r["segments"])}))
"""


def time_one(audio, repo, warm, opts):
    """Time one model in a fresh interpreter (see the lru_cache note above)."""
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, audio, repo, warm, json.dumps(opts)],
        capture_output=True, text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[6:]), None
    err = (proc.stderr.strip().splitlines() or ["no output"])[-1]
    return None, err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default="/tmp/probe120.wav")
    p.add_argument("--corpus-hours", type=float, default=5.2)
    p.add_argument("--models", nargs="*", default=MODELS)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--no-condition", action="store_true")
    p.add_argument("--json")
    a = p.parse_args()

    dur = Path(a.audio).stat().st_size / (16000 * 2)
    warm = Path("/tmp/_warm1s.wav")
    if not warm.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "anullsrc=r=16000:cl=mono", "-t", "1", "-c:a",
                        "pcm_s16le", str(warm), "-y"], check=True)

    opts = {}
    if a.greedy:
        opts["temperature"] = 0.0
    if a.no_condition:
        opts["condition_on_previous_text"] = False

    print(f"probe: {dur:.0f}s of audio, projecting to {a.corpus_hours}h corpus")
    print(f"opts: {opts or 'library defaults'} (one subprocess per model)\n")
    print(f"{'model':<40} {'load_s':>7} {'wall_s':>7} {'xRT':>6} {'corpus':>9} "
          f"{'seg':>5} {'chars':>6}")
    rows = []
    for repo in a.models:
        res, err = time_one(a.audio, repo, str(warm), opts)
        if err:
            print(f"{repo:<40} FAILED {err[:40]}")
            rows.append({"model": repo, "error": err[:200]})
            continue
        xrt = dur / res["wall_s"]
        proj = a.corpus_hours / xrt
        print(f"{repo:<40} {res['load_s']:>7.1f} {res['wall_s']:>7.1f} "
              f"{xrt:>6.2f} {proj:>8.2f}h {res['segments']:>5} "
              f"{res['chars']:>6}")
        rows.append({"model": repo, "load_s": round(res["load_s"], 1),
                     "wall_s": round(res["wall_s"], 1),
                     "x_realtime": round(xrt, 2),
                     "projected_corpus_h": round(proj, 2),
                     "chars": res["chars"], "segments": res["segments"]})

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"probe_audio": a.audio, "probe_seconds": round(dur, 1),
             "corpus_hours": a.corpus_hours, "opts": opts,
             "isolated_subprocess": True, "results": rows},
            indent=2, ensure_ascii=False))
        print(f"\n[saved] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
