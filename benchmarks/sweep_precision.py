"""Does weight precision matter? Sweep quantizations at a fixed config.

The point of this sweep is to answer whether the quantized models are costing
accuracy at all, by including the unquantized fp16 weights as the ceiling. It
only makes sense on a machine with enough memory to hold fp16 (~8.9GB of
weights plus KV plus the encoder), i.e. not a 16GB M4.

Everything except the weights is held fixed: same audio, same chunk length,
same batch, same delay, same KV setting, same (optional) prompt. Speed is reported too,
because fp16 reads 3.5x more bytes per decode step than 4-bit and decode is
bandwidth-bound.

    uv run python benchmarks/sweep_precision.py \
        --audio AUDIO --reference REFERENCE.srt \
        --json benchmarks/precision_m2ultra.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_audio.stt.utils import load as load_model

from evals.eval_cer import cer, load_text, normalize
from mlx_asr.audio import SAMPLE_RATE, load_audio_16k, split_with_overlap
from mlx_asr.decode import transcribe_batch
from mlx_asr.output import build_cues

MODELS = [
    ("fp16", "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16"),
    ("8bit", "ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx"),
    ("6bit", "mlx-community/Voxtral-Mini-4B-Realtime-6bit"),
    ("4bit", "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"),
]

def kana(text):
    try:
        import pykakasi

        kks = pykakasi.kakasi()
        return "".join(i["hira"] for i in kks.convert(text))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--chunk-seconds", type=float, default=60.0)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--delay-ms", type=int, default=2400)
    p.add_argument("--kv-bits", type=int, default=None)
    p.add_argument("--overlap-seconds", type=float, default=0.0)
    p.add_argument("--prompt", default="",
                   help="optional vocabulary bias, applied identically to every "
                        "variant; left empty by default so the comparison "
                        "isolates the weights")
    p.add_argument("--models", help="comma-separated subset of: " + ",".join(m[0] for m in MODELS))
    p.add_argument("--json")
    a = p.parse_args()

    audio = load_audio_16k(a.audio)
    duration = len(audio) / SAMPLE_RATE
    chunks, offsets, warmup = split_with_overlap(
        audio, target_s=a.chunk_seconds, overlap_s=a.overlap_seconds
    )
    ref = normalize(load_text(a.reference))
    ref_kana = kana(ref)

    wanted = a.models.split(",") if a.models else [m[0] for m in MODELS]
    todo = [(n, r) for n, r in MODELS if n in wanted]

    print(f"audio {duration:.0f}s -> {len(chunks)} chunks of ~{a.chunk_seconds:.0f}s, "
          f"batch {a.max_batch}, delay {a.delay_ms}ms, "
          f"kv {a.kv_bits or 'fp'}, prompt {'on' if a.prompt else 'off'}\n")
    print(f"{'weights':>8} {'CER':>7} {'kanaCER':>8} {'wall s':>7} {'x rt':>6} "
          f"{'enc s':>6} {'dec s':>6} {'steps/s':>8} {'peak GB':>8}")

    rows = []
    for name, repo in todo:
        t0 = time.perf_counter()
        try:
            model = load_model(repo)
        except Exception as e:
            print(f"{name:>8}  load failed: {type(e).__name__}: {str(e)[:40]}")
            rows.append({"weights": name, "repo": repo, "error": str(e)[:200]})
            continue
        t_load = time.perf_counter() - t0

        prompt_ids = None
        if a.prompt:
            from mistral_common.tokens.tokenizers.tekken import Tekkenizer
            from mlx_audio.stt.utils import get_model_path

            tek = Tekkenizer.from_file(str(get_model_path(repo) / "tekken.json"))
            prompt_ids = tek.encode(a.prompt, bos=False, eos=False)

        mx.reset_peak_memory()
        t0 = time.perf_counter()
        timed_all, stats = [], {"encode_s": 0.0, "decode_s": 0.0, "steps": 0}
        failed = None
        for i in range(0, len(chunks), a.max_batch):
            try:
                timed, st = transcribe_batch(
                    model, chunks[i : i + a.max_batch], a.delay_ms,
                    prompt_ids, a.kv_bits, log=lambda *x: None,
                    warmup_tokens=warmup[i : i + a.max_batch],
                )
            except Exception as e:
                failed = f"{type(e).__name__}: {str(e)[:80]}"
                break
            timed_all.extend(timed)
            for k in stats:
                stats[k] += st[k]
        if failed:
            print(f"{name:>8}  decode failed: {failed}")
            rows.append({"weights": name, "repo": repo, "error": failed})
            del model
            mx.clear_cache()
            continue
        wall = time.perf_counter() - t0
        peak = mx.get_peak_memory() / 1e9

        cues = build_cues(model, timed_all, list(offsets))
        hyp = normalize("".join(c[2] for c in cues))
        c = cer(ref, hyp)
        hk = kana(hyp)
        ck = None
        if ref_kana and hk:
            from rapidfuzz.distance import Levenshtein

            ck = Levenshtein.distance(ref_kana, hk) / max(len(ref_kana), 1)
        sps = stats["steps"] / max(stats["decode_s"], 1e-9)
        rows.append({
            "weights": name, "repo": repo, "cer": round(c, 4),
            "kana_cer": round(ck, 4) if ck else None,
            "wall_s": round(wall, 1), "x_realtime": round(duration / wall, 1),
            "encode_s": round(stats["encode_s"], 1),
            "decode_s": round(stats["decode_s"], 1),
            "steps_per_s": round(sps, 2), "peak_gb": round(peak, 2),
            "model_load_s": round(t_load, 1), "cues": len(cues),
        })
        print(f"{name:>8} {c*100:>6.2f}% {(ck*100 if ck else 0):>7.2f}% {wall:>7.1f} "
              f"{duration/wall:>6.1f} {stats['encode_s']:>6.1f} {stats['decode_s']:>6.1f} "
              f"{sps:>8.2f} {peak:>8.2f}")

        # SRT per variant, so texts can be diffed afterwards.
        out = Path(f"bench_out/precision_{name}.srt")
        out.parent.mkdir(exist_ok=True)
        from mlx_asr.output import write_srt

        write_srt(cues, out)
        del model
        mx.clear_cache()

    ok = [r for r in rows if "cer" in r]
    if len(ok) > 1:
        best = min(ok, key=lambda r: r["cer"])
        fp = next((r for r in ok if r["weights"] == "fp16"), None)
        print(f"\nbest CER: {best['weights']} at {best['cer']*100:.2f}%")
        if fp:
            for r in ok:
                if r["weights"] != "fp16":
                    d = (r["cer"] - fp["cer"]) * 100
                    print(f"  {r['weights']:>5} vs fp16: {d:+.2f} CER points, "
                          f"{r['x_realtime'] / fp['x_realtime']:.2f}x the speed")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"config": vars(a), "duration_s": duration,
                       "chunks": len(chunks), "results": rows}, f, indent=2)
        print(f"[saved] {a.json}")


if __name__ == "__main__":
    main()
