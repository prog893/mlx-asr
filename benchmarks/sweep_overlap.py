"""Sweep prefix-overlap length: does warm-up context recover seam accuracy?

Chunk seams cost real accuracy (60s chunks 7.2% CER vs 30s chunks 9.1% on
identical audio, purely from having twice as many boundaries). Error positions
confirm the mechanism is missing LEFT context: edit operations are 2.2x enriched
in the first 3s of a chunk versus 1.3x in the last 3s
(benchmarks/probes/probe_seam_errors.py).

Since the encoder and decoder are both causal, appending audio after a chunk
cannot change its tokens, but prepending audio can. So each chunk is given
`overlap` seconds of the preceding audio to warm up on, and those tokens are
discarded. Cost is proportional extra decode steps: overlap/chunk_len more work.

    uv run python benchmarks/sweep_overlap.py --reference ref.srt \
        --chunk-seconds 30 --max-batch 32 --overlaps 0,1.6,3,5,8
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_audio.stt.utils import load as load_model

from evals.eval_cer import cer, load_text, normalize
from mlx_asr.audio import SAMPLE_RATE, load_audio_16k, split_with_overlap
from mlx_asr.decode import transcribe_batch
from mlx_asr.output import build_cues, write_srt


DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--chunk-seconds", type=float, default=30.0)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--delay-ms", type=int, default=2400)
    p.add_argument("--kv-bits", type=int, default=8)
    p.add_argument("--overlaps", default="0,1.6,3,5,8")
    p.add_argument("--json")
    a = p.parse_args()

    audio = load_audio_16k(a.audio)
    duration = len(audio) / SAMPLE_RATE
    ref = normalize(load_text(a.reference))
    model = load_model(a.model)
    quiet = lambda *x: None

    print(f"{a.audio}: {duration:.0f}s, chunk {a.chunk_seconds:.0f}s, "
          f"batch {a.max_batch}, kv {a.kv_bits}\n")
    print(f"{'overlap':>8} {'CER':>8} {'delta':>7} {'wall s':>8} {'x rt':>7} "
          f"{'steps':>7} {'extra audio':>12}")

    rows, base = [], None
    for ov in [float(x) for x in a.overlaps.split(",")]:
        chunks, offsets, warmup = split_with_overlap(
            audio, target_s=a.chunk_seconds, overlap_s=ov
        )
        extra = sum(len(c) for c in chunks) / SAMPLE_RATE - duration

        t0 = time.perf_counter()
        timed_all, steps = [], 0
        for i in range(0, len(chunks), a.max_batch):
            sl = slice(i, i + a.max_batch)
            timed, st = transcribe_batch(
                model, chunks[sl], a.delay_ms, None, a.kv_bits, quiet,
                warmup_tokens=warmup[sl],
            )
            timed_all.extend(timed)
            steps += st["steps"]
        wall = time.perf_counter() - t0

        cues = build_cues(model, timed_all, offsets)
        c = cer(ref, normalize("".join(x[2] for x in cues)))
        if base is None:
            base = c
        Path("bench_out").mkdir(exist_ok=True)
        write_srt(cues, f"bench_out/overlap_{ov:g}s.srt")

        rows.append({"overlap_s": ov, "cer": round(c, 4),
                     "delta": round(c - base, 4), "wall_s": round(wall, 1),
                     "x_realtime": round(duration / wall, 1), "steps": steps,
                     "extra_audio_s": round(extra, 1), "cues": len(cues)})
        print(f"{ov:>7.1f}s {c*100:>7.2f}% {(c-base)*100:>+6.2f} {wall:>8.1f} "
              f"{duration/wall:>7.1f} {steps:>7} {extra:>10.0f}s "
              f"(+{100*extra/duration:.0f}%)")

    best = min(rows, key=lambda r: r["cer"])
    print(f"\nbest CER: overlap {best['overlap_s']:g}s at {best['cer']*100:.2f}% "
          f"({best['x_realtime']}x realtime)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"config": vars(a), "duration_s": duration, "results": rows},
                      f, indent=2)
        print(f"[saved] {a.json}")


if __name__ == "__main__":
    main()
