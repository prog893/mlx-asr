"""Does input level matter? Sweep gain and normalization modes.

Motivation, and why this is not obviously a no-op: the mel front end clamps at an
ABSOLUTE floor (``log10(power)`` below ``global_log_mel_max - 8.0`` is flattened),
not at a level relative to the utterance. So quiet input has detail destroyed
before the encoder ever sees it, and loud input does not. Measured on one
recording, the share of mel bins sitting exactly at the floor goes 9.9% at unity
gain, 24% at -6dB, 41% at -12dB, 65% at -20dB.

That makes level a real preprocessing variable rather than a cosmetic one, and it
also means the answer is empirical: lifting bins off the floor only helps if what
was clamped was speech detail rather than noise. Amplifying also raises the noise
floor and risks clipping, which destroys information outright.

Modes compared:

  gain <dB>   fixed gain, clipped at full scale
  chunk       per-chunk normalization: each chunk boosted to hit the target on its
              own. Tests whether file-wide gain leaves quiet chunks under the mel
              floor, at the cost of flattening inter-chunk dynamics
  peak        scale so the maximum sample hits a target (default -1 dBFS)
  rms         scale so the speech-frame RMS hits a target, which is closer to what
              a broadcast loudness normalizer does and is robust to isolated
              transients that peak normalization chases

Scored with the coverage-aware metric, since corpus references may omit audio.

    uv run python benchmarks/sweep_gain.py --corpus DIR --modes 0,-12,peak,rms
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np
from mlx_audio.stt.utils import load as load_model

from evals.eval_coverage import load_reference, score_pair
from mlx_asr.audio import (
    SAMPLE_RATE,
    apply_gain,
    frame_db,
    load_audio_16k,
    per_chunk_gain_db,
    split_with_overlap,
)
from mlx_asr.decode import transcribe_batch
from mlx_asr.text import transcript_text

from benchmarks.run_corpus import find_pairs, to_16k_mono

DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def mel_floor_share(audio: np.ndarray, limit_s: float = 120.0) -> float:
    """Fraction of mel bins clamped at the absolute floor, as a diagnostic."""
    from mlx_audio.stt.models.voxtral_realtime.audio import (
        compute_mel_filters,
        compute_mel_spectrogram,
    )

    fb = mx.array(compute_mel_filters(), dtype=mx.float32)
    seg = audio[: int(limit_s * SAMPLE_RATE)]
    mel = np.array(compute_mel_spectrogram(mx.array(seg, dtype=mx.float32), fb))
    floor = (1.5 - 8.0 + 4.0) / 4.0
    return float(np.mean(np.isclose(mel, floor, atol=1e-6)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--modes", default="0,-12,peak,rms",
                   help="comma-separated: a number is fixed dB, or peak / rms")
    p.add_argument("--chunk-seconds", type=float, default=30.0)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--delay-ms", type=int, default=2400)
    p.add_argument("--kv-bits", type=int, default=8)
    p.add_argument("--peak-dbfs", type=float, default=-1.0)
    p.add_argument("--rms-dbfs", type=float, default=-23.0)
    p.add_argument("--limit", type=int)
    p.add_argument("--json")
    a = p.parse_args()

    corpus = Path(a.corpus).expanduser()
    tmp = Path(__import__("tempfile").gettempdir()) / "voxtral_corpus_16k"
    tmp.mkdir(exist_ok=True)
    prepared = []
    for audio_path, ref in find_pairs(corpus):
        wav = to_16k_mono(audio_path, tmp)
        prepared.append((audio_path.stem, wav, ref))
    prepared.sort(key=lambda x: x[1].stat().st_size)
    if a.limit:
        prepared = prepared[: a.limit]

    model = load_model(a.model)
    modes = [m.strip() for m in a.modes.split(",")]
    print(f"corpus: {len(prepared)} files; modes: {modes}\n")
    print(f"{'mode':>8} {'file':<24} {'peak':>7} {'floor%':>7} {'covERR':>8}")

    out = {}
    for mode in modes:
        agg_ref = {"char": 0, "word": 0}
        agg_charged = {"char": 0.0, "word": 0.0}
        per_file = []
        for stem, wav, refpath in prepared:
            audio = load_audio_16k(str(wav))
            if mode != "chunk":
                audio = apply_gain(audio, mode, a.peak_dbfs, a.rms_dbfs)
            pk = 20 * np.log10(max(abs(audio).max(), 1e-12))
            fl = mel_floor_share(audio)
            chunks, offsets, warm = split_with_overlap(
                audio, target_s=a.chunk_seconds, overlap_s=0.0
            )
            if mode == "chunk":
                # Normalize each chunk independently, after splitting.
                chunks = [
                    np.clip(c * (10 ** (per_chunk_gain_db(c) / 20)), -1.0, 1.0)
                    for c in chunks
                ]
            timed_all = []
            for i in range(0, len(chunks), a.max_batch):
                sl = slice(i, i + a.max_batch)
                timed, _ = transcribe_batch(
                    model, chunks[sl], a.delay_ms, None, a.kv_bits,
                    log=lambda *x: None, warmup_tokens=warm[sl],
                )
                timed_all.extend(timed)
            unit, s = score_pair(load_reference(str(refpath)),
                                 transcript_text(model, timed_all))
            agg_ref[unit] += s["ref_chars"]
            agg_charged[unit] += s["coverage_cer"] * s["ref_chars"]
            per_file.append({"file": stem, "unit": unit,
                             "peak_dbfs": round(float(pk), 1),
                             "mel_floor_share": round(float(fl), 4),
                             "coverage_cer": round(s["coverage_cer"], 5)})
            print(f"{mode:>8} {stem[:24]:<24} {pk:>6.1f} {fl*100:>6.1f}% "
                  f"{s['coverage_cer']*100:>7.2f}%")
            mx.clear_cache()
        summary = {u: (agg_charged[u] / agg_ref[u]) for u in agg_ref if agg_ref[u]}
        out[mode] = {"aggregate": {k: round(v, 5) for k, v in summary.items()},
                     "per_file": per_file}
        print(f"{mode:>8} " + "  ".join(
            f"{'coverageCER' if u=='char' else 'coverageWER'} {v*100:.2f}%"
            for u, v in summary.items()) + "\n")

    print(f"{'mode':>8} " + " ".join(f"{u:>14}" for u in ("char", "word")))
    for mode, d in out.items():
        print(f"{mode:>8} " + " ".join(
            f"{d['aggregate'].get(u, float('nan'))*100:>13.2f}%" for u in ("char", "word")))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"config": vars(a), "results": out}, f, indent=2)
        print(f"[saved] {a.json}")


if __name__ == "__main__":
    main()
