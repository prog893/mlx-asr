"""Score kotoba-whisper on MLX with the chunked driver.

The point is to remove an asterisk. kotoba previously had to run through
transformers on torch/MPS to be decoded correctly, which made its throughput
incomparable to every MLX row in the benchmark. `mlx_asr/chunked.py` supplies the
chunked driver on MLX, so this measures the same weights, the same algorithm, and
a directly comparable x-realtime.

Needs an MLX conversion of the weights. `mlx-examples/whisper/convert.py` does it;
note the converter writes `model.safetensors` while mlx-whisper's loader expects
`weights.safetensors`, so the file has to be renamed.

    uv run --python .venv-whisper python scripts/benchmarks/run_kotoba_mlx.py \\
        --corpus DIR --model models/kotoba-whisper-v2.2-mlx-fp16 --json out.json
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

# Both the repo root (for mlx_asr) and scripts/ (for metrics) have to be
# importable; this file lives two levels down from the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx

from benchmarks.run_whisper import find_pairs, to_16k_mono
from metrics.eval_coverage import is_space_delimited, load_reference, score_pair
from mlx_asr.chunked import transcribe_chunked

SAMPLE_RATE = 16000


def read_wav16(path: str):
    """Read a 16kHz mono PCM16 wav as float32.

    Deliberately not `mlx_asr.audio.load_audio_16k`: that pulls in mlx-audio,
    which the whisper venv does not have, and everything here is already
    converted to 16k mono by `to_16k_mono`. The stdlib is enough.
    """
    import wave

    import numpy as np

    with wave.open(path, "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, w.getframerate()
        assert w.getnchannels() == 1, w.getnchannels()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus")
    p.add_argument("--pair", nargs=2, action="append", metavar=("AUDIO", "REF"))
    p.add_argument("--model", required=True,
                   help="path to an MLX conversion of kotoba-whisper")
    p.add_argument("--chunk-length-s", type=float, default=10.0)
    p.add_argument("--language", default="ja")
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
            skipped.append(audio.stem)   # Japanese-only model
            continue
        prepared.append((audio.stem, wav, ref, dur))
    prepared.sort(key=lambda x: x[3])
    if a.limit:
        prepared = prepared[: a.limit]

    label = a.label or f"kotoba-mlx-chunk{a.chunk_length_s:g}"
    total = sum(d for _, _, _, d in prepared)
    print(f"model:  {a.model} (MLX, chunked driver)")
    print(f"corpus: {len(prepared)} files, {total/3600:.2f}h audio"
          f"{'  (skipped non-CJK: %s)' % ', '.join(skipped) if skipped else ''}")
    print(f"opts:   chunk_length_s={a.chunk_length_s:g} language={a.language}")
    print("NOTE: MLX, so this throughput IS comparable to the other MLX rows.\n")
    print(f"{'file':<26} {'u':>1} {'dur':>6} {'ref':>6} {'x rt':>6} {'cov':>6} "
          f"{'covERR':>7} {'plain':>7} {'extra':>6}")

    # Warm the weight cache so load time is excluded, as the other runners do.
    t_load = time.perf_counter()
    import numpy as np
    transcribe_chunked(np.zeros(SAMPLE_RATE, dtype=np.float32), a.model,
                       chunk_length_s=a.chunk_length_s, language=a.language,
                       log=lambda *x: None)
    load_s = time.perf_counter() - t_load

    rows, tot_wall, tot_dur = [], 0.0, 0.0
    agg_ref = {"char": 0, "word": 0}
    agg_charged = {"char": 0.0, "word": 0.0}

    for stem, wav, refpath, dur in prepared:
        audio = read_wav16(str(wav))
        t0 = time.perf_counter()
        try:
            cues, text, meta = transcribe_chunked(
                audio, a.model, chunk_length_s=a.chunk_length_s,
                language=a.language, log=lambda *x: None)
        except Exception as e:
            print(f"{stem:<26} FAILED {type(e).__name__}: {str(e)[:40]}")
            rows.append({"file": stem, "error": f"{type(e).__name__}: {e}"[:200]})
            mx.clear_cache()
            continue
        wall = time.perf_counter() - t0

        if hypdir:
            (hypdir / f"{stem}.{label}.txt").write_text(text + "\n",
                                                        encoding="utf-8")
        unit, s = score_pair(load_reference(str(refpath)), text,
                             a.min_cut, a.min_cut_words)
        agg_ref[unit] += s["ref_chars"]
        agg_charged[unit] += s["coverage_cer"] * s["ref_chars"]
        tot_wall += wall
        tot_dur += dur
        rows.append({"file": stem, "duration_s": round(dur, 1), "unit": unit,
                     "x_realtime": round(dur / wall, 1),
                     "requested_language": a.language, **meta,
                     **{k: v for k, v in s.items() if k != "excused_runs"},
                     "excused_run_count": len(s["excused_runs"])})
        print(f"{stem:<26} {unit[0]} {dur:>6.0f} {s['ref_chars']:>6} "
              f"{dur/wall:>6.1f} {s['coverage']*100:>5.1f}% "
              f"{s['coverage_cer']*100:>6.2f}% {s['cer']*100:>6.1f}% "
              f"{s['extra_ratio']:>6.2f}")
        mx.clear_cache()
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
          f"model load {load_s:.0f}s excluded) [MLX]")
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
        {"engine": "mlx-chunked", "config": vars(a), "label": label,
         "languages": [a.language], "skipped_files": skipped,
         "aggregate": summary, "ref_units": agg_ref,
         "x_realtime": round(tot_dur / max(tot_wall, 1e-9), 2),
         "model_load_s": round(load_s, 1),
         "throughput_comparable_to_mlx": True,
         "results": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
