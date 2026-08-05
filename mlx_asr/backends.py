"""Non-Voxtral engines, adapted to produce the same cue list.

Voxtral's path stays in cli.py because it is the one that needs the batching,
chunking and hardware-profile machinery. Whisper and kotoba-whisper do their own
long-form segmentation internally, so all they need is an adapter that returns
``[(start_s, end_s, text)]`` in the same shape as ``output.build_cues``, after
which every output format works unchanged.

Everything here is MLX. There was a `transformers` backend that ran kotoba-whisper
through the authors' torch/MPS pipeline, kept as a correctness reference; it is
gone, because the MLX chunked driver matched it to within a point at 1.4x the
throughput (RESULTS.md) and keeping it meant a 2.5GB torch dependency for a path
nobody should choose. Its numbers stay in RESULTS.md as the evidence that the MLX
driver is right.

mlx-whisper is an optional import: a source install ships only the Voxtral path,
so a user who never asks for Whisper never installs it. The Homebrew formula
includes it, so the hint below names the right route for how this copy was
installed rather than assuming pip.
"""

import shutil
import sys
from pathlib import Path


def _die(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def _install_hint(extra: str) -> str:
    """How to get an optional dependency, phrased for how this copy was installed.

    A Homebrew install has no pyproject to sync and its venv is not the user's to
    pip into, so telling a brew user to run `uv sync --extra whisper` sends them
    somewhere that does not exist. Detected by location rather than asked, since
    the answer is unambiguous from where the package sits.
    """
    if "/Cellar/mlx-asr/" in str(Path(__file__).resolve()):
        return ("this looks like a Homebrew install, which should already include "
                "it; try `brew reinstall prog893/tap/mlx-asr`")
    if shutil.which("uv") and (Path(__file__).resolve().parents[1]
                               / "pyproject.toml").exists():
        return f"uv sync --extra {extra}"
    return f"pip install 'mlx-asr[{extra}]'"


def transcribe_mlx_whisper(audio_path: str, model, language=None, log=print,
                           **overrides):
    """Run mlx-whisper. Returns (cues, full_text, meta).

    ``model.opts`` carries `condition_on_previous_text=False` for the large
    models, which is not a stylistic default: leaving conditioning on lets a
    repetition loop feed itself across 30s windows and costs up to 22 points on
    long audio (RESULTS.md). It is applied here rather than left to the user.
    """
    try:
        from mlx_whisper.transcribe import transcribe
    except ImportError:
        _die(f"whisper models need mlx-whisper: {_install_hint('whisper')}")

    opts = dict(model.opts)
    opts.update({k: v for k, v in overrides.items() if v is not None})
    if language:
        opts["language"] = language
    if not opts.get("condition_on_previous_text", True):
        log("[whisper] condition_on_previous_text=False (prevents cross-window "
            "repetition loops on long audio)")
    r = transcribe(audio_path, path_or_hf_repo=model.repo, verbose=False, **opts)
    cues = [(s["start"], s["end"], s["text"].strip())
            for s in r.get("segments", []) if s.get("text", "").strip()]
    meta = {"detected_language": r.get("language"),
            "segments": len(r.get("segments", []))}
    return cues, r.get("text", ""), meta


def transcribe_mlx_chunked(audio_path: str, model, language=None, log=print,
                           **overrides):
    """Chunked long-form on MLX, for distil models that need it.

    Exists because the chunked algorithm is a property of the *driver*, not of the
    framework, so a distil model does not have to leave MLX to be decoded
    correctly. Measured on our corpus at chunk 10s: 27.01% coverage CER at 36.2x
    realtime, against 26.16% at 25.4x through the transformers pipeline. Same
    accuracy to within noise, 1.4x the throughput, and directly comparable to the
    other MLX rows instead of carrying a "not comparable" asterisk.

    Requires an MLX conversion of the weights; see `mlx_asr/chunked.py`.
    """
    try:
        from mlx_whisper.transcribe import transcribe  # noqa: F401
    except ImportError:
        _die(f"the mlx-chunked backend needs mlx-whisper: "
             f"{_install_hint('whisper')}")

    import numpy as np

    from .audio import load_audio_16k
    from .chunked import transcribe_chunked

    opts = dict(model.opts)
    opts.update({k: v for k, v in overrides.items() if v is not None})
    chunk_len = opts.pop("chunk_length_s", 10.0)
    opts.pop("batch_size", None)      # transformers-only knob
    lang = language or (model.languages if model.languages != "multilingual"
                        else None)
    audio = np.asarray(load_audio_16k(audio_path), dtype=np.float32)
    cues, text, meta = transcribe_chunked(
        audio, model.repo, chunk_length_s=chunk_len, language=lang, log=log,
        **opts)
    return cues, text, meta


DISPATCH = {
    "mlx-whisper": transcribe_mlx_whisper,
    "mlx-chunked": transcribe_mlx_chunked,
}


def run(model, audio_path: str, language=None, log=print, **overrides):
    """Dispatch to the right backend. Voxtral is handled in cli.py."""
    fn = DISPATCH.get(model.backend)
    if fn is None:
        _die(f"backend {model.backend!r} is not a non-Voxtral backend")
    return fn(audio_path, model, language=language, log=log, **overrides)
