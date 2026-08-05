"""Score Whisper through the insanely-fast-whisper recipe.

insanely-fast-whisper is not a model, it is a *runner*: transformers'
`automatic-speech-recognition` pipeline with `chunk_length_s=30`,
`batch_size=24`, fp16 weights and sdpa attention. That distinction is the whole
reason this is worth measuring separately. mlx-whisper implements Whisper's
**sequential** long-form algorithm (30s windows, decoder state carried across
them); IFW uses the **chunked** algorithm (independent windows, batched, stitched
on timestamps). The kotoba result in RESULTS.md showed that swapping exactly
these two algorithms moved one model by 68 points, so the same swap has to be
tried on the general models before any "Whisper scores X" claim is safe.

Same audio, same `eval_coverage.py` scoring, same per-file language selection as
run_whisper.py, so IFW rows compare directly on accuracy. Throughput is torch/MPS
rather than MLX and is reported but flagged not-comparable, as with
run_kotoba_hf.py.

Deliberately mirrors the upstream CLI's defaults rather than tuning them, since
the question is "what does this tool give you", not "what is the best possible
transformers config". Flags exist to vary chunk length and batch size because the
kotoba sweep showed chunk length is worth up to 23 points.

    uv run --python .venv-whisper python benchmarks/run_ifw.py \\
        --corpus DIR --model openai/whisper-large-v3-turbo --json out.json
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
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument("--chunk-length-s", type=float, default=30.0,
                   help="upstream default; the sweep showed this matters")
    p.add_argument("--batch-size", type=int, default=24,
                   help="upstream default")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--device", default="mps")
    p.add_argument("--language", default=None,
                   help="force one code; default is per-file from the reference")
    p.add_argument("--autodetect", action="store_true")
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

    prepared = []
    for audio, ref in pairs:
        wav = to_16k_mono(audio, tmpdir)
        dur = wav.stat().st_size / (SAMPLE_RATE * 2)
        ref_is_en = is_space_delimited(load_reference(str(ref)))
        lang = None if a.autodetect else (a.language or
                                         ("en" if ref_is_en else "ja"))
        prepared.append((audio.stem, wav, ref, dur, lang, ref_is_en))
    prepared.sort(key=lambda x: x[3])
    if a.limit:
        prepared = prepared[: a.limit]

    if a.language and len({r[5] for r in prepared}) > 1:
        print(f"refusing --language {a.language}: this set mixes CJK and "
              f"space-delimited references. Omit it to take each file's "
              f"language from its reference.", file=sys.stderr)
        return 2

    label = a.label or f"ifw-{a.model.split('/')[-1]}-c{a.chunk_length_s:g}"
    total = sum(d for _, _, _, d, _, _ in prepared)
    langs = sorted({r[4] or "auto" for r in prepared})
    print(f"model:  {a.model}  (insanely-fast-whisper recipe)")
    print(f"corpus: {len(prepared)} files, {total/3600:.2f}h audio")
    print(f"opts:   chunk_length_s={a.chunk_length_s:g} "
          f"batch_size={a.batch_size} dtype={a.dtype} device={a.device} "
          f"attn=sdpa language={'/'.join(langs)}")
    print("NOTE: torch/MPS chunked long-form, not MLX sequential. Accuracy is "
          "comparable to the MLX rows; throughput is NOT.\n")
    print(f"{'file':<26} {'u':>1} {'dur':>6} {'ref':>6} {'x rt':>6} {'cov':>6} "
          f"{'covERR':>7} {'plain':>7} {'extra':>6}")

    dtype = torch.float16 if a.dtype == "float16" else torch.float32
    t_load = time.perf_counter()
    pipe = pipeline("automatic-speech-recognition", model=a.model, dtype=dtype,
                    device=a.device,
                    model_kwargs={"attn_implementation": "sdpa"})
    load_s = time.perf_counter() - t_load

    rows, tot_wall, tot_dur = [], 0.0, 0.0
    agg_ref = {"char": 0, "word": 0}
    agg_charged = {"char": 0.0, "word": 0.0}

    for stem, wav, refpath, dur, lang, _ in prepared:
        t0 = time.perf_counter()
        try:
            out = pipe(str(wav), chunk_length_s=a.chunk_length_s,
                       batch_size=a.batch_size, return_timestamps=True,
                       generate_kwargs={"language": lang, "task": "transcribe"})
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
                     "requested_language": lang or "auto",
                     "chunks": len(out.get("chunks") or []),
                     **{k: v for k, v in s.items() if k != "excused_runs"},
                     "excused_run_count": len(s["excused_runs"])})
        print(f"{stem:<26} {unit[0]} {dur:>6.0f} {s['ref_chars']:>6} "
              f"{dur/wall:>6.1f} {s['coverage']*100:>5.1f}% "
              f"{s['coverage_cer']*100:>6.2f}% {s['cer']*100:>6.1f}% "
              f"{s['extra_ratio']:>6.2f}")
        if a.json:
            _dump(a, label, agg_ref, agg_charged, rows, tot_dur, tot_wall,
                  load_s, langs)

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
              langs)
        print(f"[saved] {a.json}")
    return 0


def _dump(a, label, agg_ref, agg_charged, rows, tot_dur, tot_wall, load_s,
          langs):
    summary = {u: round(agg_charged[u] / agg_ref[u], 5)
               for u in ("char", "word") if agg_ref[u]}
    Path(a.json).write_text(json.dumps(
        {"engine": "insanely-fast-whisper", "config": vars(a), "label": label,
         "languages": langs, "skipped_files": [], "aggregate": summary,
         "ref_units": agg_ref,
         "x_realtime": round(tot_dur / max(tot_wall, 1e-9), 2),
         "model_load_s": round(load_s, 1),
         "throughput_comparable_to_mlx": False,
         "results": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
