"""qwen3-asr with the library's own whole-file token budget, against the per-window budget.

    .venv/bin/python scripts/benchmarks/probes/probe_qwen3_budget.py --corpus DIR \\
        --size 1.7B --quantization 8bit --chunk-seconds 30 --max-tokens 8192 --json out.json

The row docs/benchmarks/engines/qwen3-asr.md explains: upstream `generate` splits the file
into `chunk_duration` windows itself but spends ONE `max_tokens` budget across all of them,
so a long file stops being transcribed when the budget runs out. mlx_asr.backends.qwen3_decode
moved the loop out to give each window its own budget; this probe calls upstream once on the
whole file, with everything else (spec resolution, language name, scoring) as run_qwen3.py
does it, so the two rows differ only in where the budget applies.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402
from mlx_audio.stt.utils import load as load_model  # noqa: E402

from benchmarks.machine_state import machine_state, warn_if_busy  # noqa: E402
from benchmarks.run_qwen3 import find_pairs  # noqa: E402
from metrics.eval_coverage import is_space_delimited, load_reference, score_pair  # noqa: E402
from mlx_asr.audio import SAMPLE_RATE, load_audio_16k  # noqa: E402
from mlx_asr.backends import qwen3_language  # noqa: E402
from mlx_asr.models import resolve as resolve_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--size", default="1.7B")
    p.add_argument("--quantization", default="8bit")
    p.add_argument("--chunk-seconds", type=float, default=30.0)
    p.add_argument("--max-tokens", type=int, required=True, help="whole-file budget")
    p.add_argument("--min-cut", type=int, default=30)
    p.add_argument("--min-cut-words", type=int, default=6)
    p.add_argument("--label", default="")
    p.add_argument("--json", required=True)
    a = p.parse_args()

    spec = resolve_model("qwen3-asr", a.size)
    spec = replace(spec, repo=spec.repo_for(a.quantization))
    state = machine_state()
    warn_if_busy(state)
    model = load_model(spec.repo)
    rows = []
    for audio_path, ref in find_pairs(Path(a.corpus)):
        reference = load_reference(str(ref))
        lang = qwen3_language(model, spec, "en" if is_space_delimited(reference) else "ja",
                              log=lambda *x: None)
        audio = load_audio_16k(str(audio_path))
        dur = len(audio) / SAMPLE_RATE
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        r = model.generate(audio, language=lang, chunk_duration=a.chunk_seconds,
                           max_tokens=a.max_tokens, verbose=False)
        wall = time.perf_counter() - t0
        segments = getattr(r, "segments", None) or []
        covered = max((s.get("end", 0) for s in segments if isinstance(s, dict)), default=0.0)
        unit, s = score_pair(reference, (getattr(r, "text", "") or "").strip(),
                             a.min_cut, a.min_cut_words)
        rows.append({"file": audio_path.stem, "duration_s": round(dur, 1), "unit": unit,
                     "x_realtime": round(dur / wall, 1),
                     "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
                     "segments": len(segments), "audio_covered_s": round(covered, 1),
                     **{k: v for k, v in s.items() if k != "excused_runs"},
                     "excused_run_count": len(s["excused_runs"])})
        print(f"{audio_path.stem}: {len(segments)} segments, covered {covered:.0f}/{dur:.0f}s, "
              f"coverage CER {s['coverage_cer'] * 100:.2f}%")
    out = {"engine": "mlx-qwen3-wholefile", "config": vars(a), "label": a.label,
           "machine": state, "aggregate": {}, "results": rows}
    Path(a.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
