"""Score kotoba-whisper through its OFFICIAL transformers pipeline.

Why this exists as a separate script from run_whisper.py: kotoba-whisper is a
distil-Whisper derivative with a 2-layer decoder, and its model card runs it with
`pipeline(..., chunk_length_s=15)`, i.e. transformers' *chunked* long-form
algorithm. `mlx-whisper` implements Whisper's *sequential* 30s-window algorithm
instead, and that mismatch, not the model, is what wrecks the MLX numbers:
measured on the same 600s slice, MLX-sequential emitted 2053 characters with 44
zero-duration segments while the official chunked pipeline emitted 2527 and
turbo emitted 2630. Scoring kotoba only through MLX would report an engine
artifact as a model result.

So this runs the model the way its authors say to, on the same audio, scored by
the same `eval_coverage.py` functions, and reports it as its own row. The cost is
that it is torch/MPS rather than MLX, so its throughput is not directly
comparable to the MLX rows; that is stated in the output rather than hidden.

Japanese-only model: the English corpus files are skipped, as in run_whisper.py.

    uv run --python .venv-whisper python benchmarks/run_kotoba_hf.py \\
        --corpus DIR --model kotoba-tech/kotoba-whisper-v2.2 --json out.json
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import pipeline

from benchmarks.run_whisper import find_pairs, to_16k_mono
from evals.eval_coverage import is_space_delimited, load_reference, score_pair

SAMPLE_RATE = 16000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus")
    p.add_argument("--pair", nargs=2, action="append", metavar=("AUDIO", "REF"))
    p.add_argument("--model", default="kotoba-tech/kotoba-whisper-v2.2")
    p.add_argument("--chunk-length-s", type=float, default=15.0,
                   help="the model card's recommended value")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--min-cut", type=int, default=30)
    p.add_argument("--min-cut-words", type=int, default=6)
    p.add_argument("--limit", type=int)
    p.add_argument("--label", default="")
    p.add_argument("--keep-hyp")
    p.add_argument("--json")
    a = p.parse_args()

    pairs = []
    if a.corpus:
        pairs += find_pairs(Path(a.corpus).expanduser())
    for audio, ref in a.pair or []:
        pairs.append((Path(audio).expanduser(), Path(ref).expanduser()))
    if not pairs:
        print("no audio/reference pairs given", file=sys.stderr)
        return 2

    tmpdir = Path(tempfile.gettempdir()) / "voxtral_corpus_16k"
    tmpdir.mkdir(exist_ok=True)
    hypdir = Path(a.keep_hyp) if a.keep_hyp else None
    if hypdir:
        hypdir.mkdir(parents=True, exist_ok=True)

    prepared, skipped = [], []
    for audio, ref in pairs:
        wav = to_16k_mono(audio, tmpdir)
        dur = wav.stat().st_size / (SAMPLE_RATE * 2)
        if is_space_delimited(load_reference(str(ref))):
            skipped.append(audio.stem)  # Japanese-only model
            continue
        prepared.append((audio.stem, wav, ref, dur))
    prepared.sort(key=lambda x: x[3])
    if a.limit:
        prepared = prepared[: a.limit]

    label = a.label or (a.model.split("/")[-1] + f"-hf-chunk{a.chunk_length_s:g}")
    total = sum(d for _, _, _, d in prepared)
    print(f"model:  {a.model}  (official transformers pipeline)")
    print(f"corpus: {len(prepared)} files, {total/3600:.2f}h audio"
          f"{'  (skipped non-CJK: %s)' % ', '.join(skipped) if skipped else ''}")
    print(f"opts:   chunk_length_s={a.chunk_length_s:g} "
          f"batch_size={a.batch_size} device={a.device} dtype=float32")
    print("NOTE: torch/MPS, not MLX. Accuracy is comparable to the MLX rows "
          "(same audio, same scoring); throughput is NOT.\n")
    print(f"{'file':<26} {'u':>1} {'dur':>6} {'ref':>6} {'x rt':>6} {'cov':>6} "
          f"{'covERR':>7} {'plain':>7} {'extra':>6}")

    t_load = time.perf_counter()
    pipe = pipeline("automatic-speech-recognition", model=a.model,
                    dtype=torch.float32, device=a.device,
                    model_kwargs={"attn_implementation": "sdpa"})
    load_s = time.perf_counter() - t_load

    rows, tot_wall, tot_dur = [], 0.0, 0.0
    agg_ref = {"char": 0, "word": 0}
    agg_charged = {"char": 0.0, "word": 0.0}

    for stem, wav, refpath, dur in prepared:
        t0 = time.perf_counter()
        try:
            out = pipe(str(wav), chunk_length_s=a.chunk_length_s,
                       batch_size=a.batch_size,
                       generate_kwargs={"language": "ja", "task": "transcribe"})
        except Exception as e:
            print(f"{stem:<26} FAILED {type(e).__name__}: {str(e)[:40]}")
            rows.append({"file": stem, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        wall = time.perf_counter() - t0

        hyp_text = out["text"]
        if hypdir:
            (hypdir / f"{stem}.{label}.txt").write_text(hyp_text + "\n",
                                                        encoding="utf-8")
        unit, s = score_pair(load_reference(str(refpath)), hyp_text,
                             a.min_cut, a.min_cut_words)
        agg_ref[unit] += s["ref_chars"]
        agg_charged[unit] += s["coverage_cer"] * s["ref_chars"]
        tot_wall += wall
        tot_dur += dur
        rows.append({"file": stem, "duration_s": round(dur, 1), "unit": unit,
                     "x_realtime": round(dur / wall, 1),
                     "requested_language": "ja",
                     **{k: v for k, v in s.items() if k != "excused_runs"},
                     "excused_run_count": len(s["excused_runs"])})
        print(f"{stem:<26} {unit[0]} {dur:>6.0f} {s['ref_chars']:>6} "
              f"{dur/wall:>6.1f} {s['coverage']*100:>5.1f}% "
              f"{s['coverage_cer']*100:>6.2f}% {s['cer']*100:>6.1f}% "
              f"{s['extra_ratio']:>6.2f}")
        if a.json:
            _dump(a, label, agg_ref, agg_charged, rows, tot_dur, tot_wall,
                  load_s, skipped)

    print()
    for unit in ("char", "word"):
        if agg_ref[unit]:
            v = agg_charged[unit] / agg_ref[unit]
            m = "coverageCER" if unit == "char" else "coverageWER"
            n = sum(1 for r in rows if r.get("unit") == unit)
            print(f"aggregate {m} over {n} {unit}-unit files "
                  f"({agg_ref[unit]} ref {unit}s, length-weighted): {v*100:.2f}%")
    print(f"throughput: {tot_dur/max(tot_wall,1e-9):.1f}x realtime "
          f"({tot_dur/3600:.2f}h audio in {tot_wall/60:.1f} min; "
          f"model load {load_s:.0f}s excluded) [torch/MPS]")
    if a.json:
        _dump(a, label, agg_ref, agg_charged, rows, tot_dur, tot_wall, load_s,
              skipped)
        print(f"[saved] {a.json}")
    return 0


def _dump(a, label, agg_ref, agg_charged, rows, tot_dur, tot_wall, load_s,
          skipped):
    summary = {u: round(agg_charged[u] / agg_ref[u], 5)
               for u in ("char", "word") if agg_ref[u]}
    Path(a.json).write_text(json.dumps(
        {"engine": "transformers-chunked", "config": vars(a), "label": label,
         "languages": ["ja"], "skipped_files": skipped, "aggregate": summary,
         "ref_units": agg_ref,
         "x_realtime": round(tot_dur / max(tot_wall, 1e-9), 2),
         "model_load_s": round(load_s, 1),
         "throughput_comparable_to_mlx": False,
         "results": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
