"""Time and score a competing Whisper runner (whisper.cpp, faster-whisper) on one clip.

    .venv/bin/python scripts/benchmarks/run_competitors.py --runner whisper.cpp \\
        --audio CLIP.wav --reference CLIP.srt --model ggml-large-v3-turbo.bin \\
        --threads 8 --label cpp_turbo_fp16_t8 --json out.json

The method of the 2026-08-04 comparison (docs/benchmarks/engines/whisper.md), kept so
the table can be repeated: `--language ja`, each runner's own default decoding (beam 5,
temperature ladder), one 1s warm-up call first, model load excluded from x-realtime,
plain CER via scripts/metrics/eval_cer.py (the clip's reference is complete). One run per
process, so no weights stay resident between configs.

- whisper.cpp: `whisper-cli` (Homebrew whisper-cpp); x-realtime = duration / (total time
  - load time) from its own `whisper_print_timings`. `--model` is a ggml file path.
- faster-whisper: runs in its own venv (`--fw-python`, e.g. .venv-fw/bin/python with
  `uv pip install faster-whisper`), so this project's venv gets no CTranslate2; CPU only,
  since CTranslate2 has no Metal backend. `--model` is a CTranslate2 repo id.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.machine_state import machine_state, warn_if_busy  # noqa: E402
from metrics.eval_cer import cer, load_text, normalize  # noqa: E402

FW_CHILD = r"""
import json, sys, time, wave
import numpy as np
import faster_whisper, ctranslate2
from faster_whisper import WhisperModel, BatchedInferencePipeline
audio, model_id, compute, batch = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
t = time.perf_counter()
model = WhisperModel(model_id, device="cpu", compute_type=compute)
load_s = time.perf_counter() - t
runner = BatchedInferencePipeline(model) if batch > 1 else model
kw = {"language": "ja", "beam_size": 5, **({"batch_size": batch} if batch > 1 else {})}
list(runner.transcribe(np.zeros(16000, dtype=np.float32), **kw)[0])   # warm-up
# Decoded here with `wave`, before timing: faster-whisper 1.2.1 passes PyAV an argument
# (metadata_errors) that current PyAV rejects, so a file path cannot be handed over.
with wave.open(audio) as w:
    if (w.getsampwidth(), w.getnchannels(), w.getframerate()) != (2, 1, 16000):
        sys.exit(f"{audio}: expected 16 kHz mono 16-bit PCM, got {w.getframerate()} Hz, "
                 f"{w.getnchannels()} ch, {8 * w.getsampwidth()}-bit")
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
pcm = pcm.astype(np.float32) / 32768.0
t = time.perf_counter()
segments, _ = runner.transcribe(pcm, **kw)
text = "\n".join(s.text for s in segments)
wall = time.perf_counter() - t
print(json.dumps({"load_s": load_s, "infer_s": wall, "text": text,
                  "version": faster_whisper.__version__,
                  "ctranslate2": ctranslate2.__version__,
                  "audio_input": "16 kHz PCM array decoded with wave before timing"}))
"""


def silence(path: Path):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000)


def run_cpp(a, audio: str, tmp: Path) -> dict:
    def call(src, out):
        cmd = ["whisper-cli", "-m", a.model, "-f", src, "-l", "ja", "-t", str(a.threads),
               "-otxt", "-of", str(out)]
        return subprocess.run(cmd, capture_output=True, text=True, check=True)
    warm = tmp / "warm1s.wav"
    silence(warm)
    call(str(warm), tmp / "warm")
    p = call(audio, tmp / "hyp")
    ms = {k: float(v) for k, v in re.findall(r"(load|total) time\s*=\s*([\d.]+) ms", p.stderr)}
    version = subprocess.run(["whisper-cli", "--version"], capture_output=True,
                             text=True).stdout.strip() or "unknown"
    return {"load_s": ms["load"] / 1000, "infer_s": (ms["total"] - ms["load"]) / 1000,
            "text": (tmp / "hyp.txt").read_text(encoding="utf-8"), "version": version,
            "gpu_backend_line": next((l for l in p.stderr.splitlines()
                                      if "backend_init_gpu" in l), "")}


def run_fw(a, audio: str) -> dict:
    p = subprocess.run([a.fw_python, "-c", FW_CHILD, audio, a.model, a.compute_type,
                        str(a.batch_size)], capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"faster-whisper child failed:\n{p.stderr[-3000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runner", choices=["whisper.cpp", "faster-whisper"], required=True)
    p.add_argument("--audio", required=True, help="16 kHz mono WAV")
    p.add_argument("--reference", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--threads", type=int, default=4, help="whisper.cpp -t")
    p.add_argument("--fw-python", default=".venv-fw/bin/python")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--batch-size", type=int, default=1, help="faster-whisper; >1 batches")
    p.add_argument("--label", default="")
    p.add_argument("--json", required=True)
    a = p.parse_args()

    state = machine_state()
    warn_if_busy(state)
    audio = str(Path(a.audio).resolve())
    with wave.open(audio) as w:
        dur = w.getnframes() / w.getframerate()
    with tempfile.TemporaryDirectory() as tmp:
        r = run_cpp(a, audio, Path(tmp)) if a.runner == "whisper.cpp" else run_fw(a, audio)
    ref, hyp = normalize(load_text(a.reference)), normalize(r.pop("text"))
    row = {"file": Path(a.audio).stem, "duration_s": round(dur, 1), "unit": "char",
           "x_realtime": round(dur / r["infer_s"], 2), "wall_s": round(r["infer_s"], 2),
           "load_s": round(r["load_s"], 2), "ref_chars": len(ref), "hyp_chars": len(hyp),
           "cer": round(cer(ref, hyp), 4)}
    out = {"engine": a.runner, "config": vars(a), "label": a.label, "machine": state,
           "runner_version": {k: v for k, v in r.items() if k not in ("load_s", "infer_s")},
           "aggregate": {"char": row["cer"]}, "x_realtime": row["x_realtime"],
           "results": [row]}
    Path(a.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"{a.label or a.runner}: {row['x_realtime']}x, plain CER {row['cer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
