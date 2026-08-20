"""Non-Voxtral engines, adapted to produce the same cue list.

Voxtral's path stays in cli.py because it is the one that needs the batching,
chunking and hardware-profile machinery. Whisper, kotoba-whisper and Qwen3-ASR do
their own long-form segmentation internally, so all they need is an adapter that
returns ``[(start_s, end_s, text)]`` in the same shape as ``output.build_cues``,
after which every output format works unchanged.

One exception to "every output format works": Qwen3-ASR's segment times are the
chunk boundaries its own splitter chose, so an SRT built from them would carry
cues that do not correspond to speech. `-f srt` and `-f vtt` are refused on that
engine in cli.py rather than approximated here, and the adapter labels its meta
`cue_source: "chunk_boundaries"` so nothing downstream mistakes those times for
the per-token ones the Voxtral path produces.

Everything here is MLX. There was a `transformers` backend that ran kotoba-whisper
through the authors' torch/MPS pipeline, kept as a correctness reference; it is
gone, because the MLX chunked driver matched it to within a point at 1.4x the
throughput (docs/benchmarks/engines.md) and keeping it meant a 2.5GB torch dependency for a path
nobody should choose. Its numbers stay in docs/benchmarks/engines.md as the evidence that the MLX
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
    long audio (docs/benchmarks/engines.md). It is applied here rather than left to the user.
    """
    try:
        from mlx_whisper.transcribe import transcribe
    except ImportError:
        _die(f"whisper models need mlx-whisper: {_install_hint('whisper')}")

    opts = dict(model.opts)
    opts.update({k: v for k, v in overrides.items() if v is not None})
    if language:
        # Normalised rather than passed through. mlx-whisper wants an ISO 639-1 code, and
        # given anything else it treats the string as a decoder prompt hint instead of a
        # language selection, which reads as a mysterious accuracy loss rather than as a
        # rejected argument. "ja_JP", "JA" and "jpn" are all things a user reasonably
        # types; each becomes "ja" here. See mlx_asr/languages.py.
        from .languages import to_iso
        opts["language"] = to_iso(language, model.alias)
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
    # Same normalisation as the whisper path; the chunked driver hands the code to
    # mlx-whisper's decoder in the end. A single-language model's own tag is already a
    # bare code, so it needs no round trip.
    if language:
        from .languages import to_iso
        lang = to_iso(language, model.alias)
    else:
        lang = (model.languages if model.languages != "multilingual" else None)
    audio = np.asarray(load_audio_16k(audio_path), dtype=np.float32)
    cues, text, meta = transcribe_chunked(
        audio, model.repo, chunk_length_s=chunk_len, language=lang, log=log,
        **opts)
    return cues, text, meta


def qwen3_language(loaded, spec, language=None, log=print) -> str:
    """The English language name to prompt Qwen3-ASR with. Never None.

    Split out because the *benchmark* has to resolve this the same way the CLI does.
    Two upstream behaviours make it load-bearing, and neither raises:

    `_build_prompt` looks the string up in the checkpoint's own `support_languages`
    case-insensitively and, on a miss, interpolates it verbatim, so `--language ja`
    yields the prefix `language ja<asr_text>` and a token sequence the model never
    saw in training.

    Passing nothing is worse than passing the wrong thing. On the autodetect path
    (`language=None`) `generate` reassigns its own `language` variable inside the
    chunk loop from the first chunk's detection, so from chunk two onward
    `extract_language` no longer strips the prefix and `language X<asr_text>` stays
    embedded in the transcript text. An explicit name skips that branch entirely,
    which is why this returns a name rather than allowing None.
    """
    # The checkpoint's own vocabulary decides what is accepted, so a build that
    # drops a language stops accepting it with no change here.
    supported = getattr(getattr(loaded, "config", None),
                        "support_languages", None) or None
    if language:
        from .languages import to_english_name
        return to_english_name(language, supported, spec.alias)
    # English is the library's own fallback in `extract_language`, and this is
    # announced rather than assumed silently.
    log(f"[{spec.backend}] no --language given; forcing English. Autodetect is "
        f"not used here: upstream reassigns the language inside the chunk loop, "
        f"which leaves a 'language X<asr_text>' prefix in later chunks' text")
    return "English"


# Token budget per second of audio in a chunk.
#
# The reason this file drives the chunk loop itself rather than handing the whole
# recording to `generate`. Upstream `max_tokens` (default 8192) is a budget for the
# WHOLE FILE, and exhausting it does not raise: the chunk loop simply `break`s, so the
# remaining audio produces no text and the output is a well-formed transcript of the
# beginning of the file.
#
# Measured here on a real recording, which is why this is not a hypothetical: a 1553s
# Japanese file at a 30s window returned ONE segment and 110.77% coverage CER. Its
# first chunk entered a repetition loop (one 7-character phrase 2048 times),
# spent the entire
# 8192-token budget, and left 1523 of 1553 seconds untranscribed. Raising the budget
# does not fix that, it only lets the loop run longer: at a duration-scaled 19950
# tokens the same file still came back 8% covered. The budget has to be per chunk, so
# that one degenerate window costs one window.
#
# 25 tokens/s is 3-4x real speech (Japanese runs 6-9 chars/s at roughly 1 token per
# 1-1.5 chars; English is lower), so it cannot truncate an honest transcript.
TOKENS_PER_SECOND = 25
# Floor per chunk, so a 2-second tail chunk still has room for a sentence.
MIN_CHUNK_MAX_TOKENS = 256

# Characters per second above which a segment's text cannot be speech and is treated
# as a degenerate repetition loop. Japanese narration runs 6-9 chars/s; 25 is far
# enough above that a dense speaker is never flagged.
RUNAWAY_CHARS_PER_S = 25


def _split_for_qwen3(audio, chunk_len: float):
    """Cut the audio into windows, preferring mlx-audio's own splitter.

    Uses upstream's `split_audio_into_chunks` so the cut points are the ones the
    library would have chosen (it searches for a low-energy point within 5s of each
    boundary), keeping this adapter's output comparable to running `generate` directly.
    Falls back to fixed-length slicing if that private helper moves.
    """
    try:
        from mlx_audio.stt.models.qwen3_asr.qwen3_asr import split_audio_into_chunks

        return split_audio_into_chunks(audio, sr=16000, chunk_duration=chunk_len)
    except Exception:      # pragma: no cover - upstream layout change
        step = int(chunk_len * 16000)
        return [(audio[i:i + step], i / 16000)
                for i in range(0, max(len(audio), 1), step)]


def qwen3_decode(loaded, audio, language: str, chunk_len: float, log=print, **opts):
    """Decode one already-loaded array. Returns (cues, full_text, meta).

    Separate from ``transcribe_mlx_qwen3`` so a benchmark can load the weights once
    outside its timing loop and still measure exactly the code the CLI runs. Every
    harness bug this project has published came from a benchmark that reimplemented
    a decode path slightly differently from the shipped one.

    ``audio`` is a decoded 16kHz mono array, never a path: mlx-audio's own
    `load_audio` shells out to ffmpeg for some formats and resamples differently, and
    a second audio front end that nothing tests is how a conversion difference would
    leak into CER.

    **The chunk loop is here, not upstream**, so that the token budget is per window.
    See ``TOKENS_PER_SECOND`` for the measured failure that forces this. Nothing else
    is reimplemented: the cut points come from upstream's splitter and each window
    goes through the same `generate`, one chunk at a time, which is the path a
    `batch_size=1` call takes anyway (`--max-batch` is refused at the CLI).

    **Repetition loops are counted, not fixed.** A window can still emit one phrase
    until its own budget runs out; that is a property of these weights. It is reported
    as `runaway_segments` rather than passed off as a transcript, and it now costs
    that window alone.

    The cue times are **chunk boundaries**: `start = offset`,
    `end = offset + len(chunk)/sr`, i.e. the window the text came from rather than
    when the speech happened. No variant of these weights has finer times, which is
    recorded as `cue_source: "chunk_boundaries"` so a break-F1 or timing-drift figure
    taken from this engine can never be put beside Voxtral's or Whisper's.
    """
    duration = len(audio) / 16000
    windows = _split_for_qwen3(audio, chunk_len)
    # Per chunk, and only if the caller did not ask for something specific: a
    # benchmark arm deliberately testing the library default has to be able to.
    per_chunk = opts.pop("max_tokens", None)

    segments, texts = [], []
    for chunk, offset in windows:
        chunk_dur = len(chunk) / 16000
        budget = per_chunk or max(MIN_CHUNK_MAX_TOKENS,
                                  int(chunk_dur * TOKENS_PER_SECOND))
        # chunk_duration above the chunk's own length, so upstream does not split it
        # again; this call decodes exactly one window.
        r = loaded.generate(chunk, language=language,
                            chunk_duration=max(chunk_dur * 2, 1.0),
                            max_tokens=budget, verbose=False, **opts)
        text = (getattr(r, "text", "") or "").strip()
        texts.append(text)
        segments.append({"start": offset, "end": offset + chunk_dur,
                         "text": text, "language": language})

    cues = [(s["start"], s["end"], s["text"]) for s in segments if s["text"]]
    covered = max((s["end"] for s in segments), default=0.0)
    audio_coverage = covered / duration if duration else 1.0
    runaway = [s for s in segments
               if (s["end"] - s["start"]) > 0
               and len(s["text"]) / (s["end"] - s["start"]) > RUNAWAY_CHARS_PER_S]

    # NOT `detected_language`, which the whisper backend sets and which would be a
    # false claim here: nothing detected anything, the language was forced (see
    # qwen3_language for why autodetect is unusable). A JSON field saying
    # "detected_language": "Japanese" on a run that never ran detection is the same
    # class of quiet untruth as a flag that looks honoured and does nothing.
    meta = {"language_source": "forced",
            "requested_language": language,
            "segments": len(segments),
            # The load-bearing caveat: these cues are windows, not speech.
            "cue_source": "chunk_boundaries",
            "chunk_seconds": chunk_len,
            "max_tokens_per_chunk": per_chunk or "scaled per window",
            "audio_coverage": round(audio_coverage, 4),
            "runaway_segments": len(runaway),
            "empty_segments": sum(1 for s in segments if not s["text"])}

    # Loud, because the output file looks complete either way. A short transcript is
    # not distinguishable from quiet audio without this line.
    if audio_coverage < 0.99:
        log(f"[mlx-qwen3] WARNING: segments span only {covered:.0f}s of "
            f"{duration:.0f}s ({audio_coverage:.0%}); the rest of the file is "
            f"MISSING from this output.")
    if runaway:
        worst = max(runaway,
                    key=lambda s: len(s["text"]) / max(s["end"] - s["start"], 1e-9))
        log(f"[mlx-qwen3] WARNING: {len(runaway)} of {len(segments)} windows look "
            f"like repetition loops (worst: "
            f"{len(worst['text']) / (worst['end'] - worst['start']):.0f} chars/s at "
            f"{worst['start']:.0f}s, against 6-9 for real speech). That text is not "
            f"a transcript of the audio. Each is capped to its own window now, so the "
            f"rest of the file is unaffected.")
    # Space-joined, matching upstream's own `" ".join(all_texts)`, so the text a user
    # gets is the same shape as before this loop moved here.
    return cues, " ".join(t for t in texts if t), meta


def transcribe_mlx_qwen3(audio_path: str, model, language=None, log=print,
                         **overrides):
    """Qwen3-ASR through mlx-audio's own loader. Returns (cues, full_text, meta).

    The loader has been in mlx-audio since 0.3.1 and is registered in its dispatch
    table, so this needs no new dependency; it is the same `load` the Voxtral path
    calls, on a different repo id. See ``qwen3_language`` for why the language is
    always explicit and ``qwen3_decode`` for why the timestamps cannot be trusted.
    """
    try:
        from mlx_audio.stt.utils import load as load_model
    except ImportError:      # pragma: no cover - mlx-audio is a hard dependency
        _die("qwen3-asr models need mlx-audio, which should already be installed; "
             f"try {_install_hint('whisper')}")

    from .audio import load_audio_16k

    opts = dict(model.opts)
    opts.update({k: v for k, v in overrides.items() if v is not None})
    # Our flag name, the library's parameter name.
    chunk_len = opts.pop("chunk_length_s", 30.0)
    opts.pop("batch_size", None)     # refused at the CLI; see UnsupportedFlags

    m = load_model(model.repo)
    lang = qwen3_language(m, model, language, log)
    log(f"[{model.backend}] language={lang}, window {chunk_len:g}s")
    return qwen3_decode(m, load_audio_16k(audio_path), lang, chunk_len, log=log,
                        **opts)


DISPATCH = {
    "mlx-whisper": transcribe_mlx_whisper,
    "mlx-chunked": transcribe_mlx_chunked,
    "mlx-qwen3": transcribe_mlx_qwen3,
}


def run(model, audio_path: str, language=None, log=print, **overrides):
    """Dispatch to the right backend. Voxtral is handled in cli.py."""
    fn = DISPATCH.get(model.backend)
    if fn is None:
        _die(f"backend {model.backend!r} is not a non-Voxtral backend")
    return fn(audio_path, model, language=language, log=log, **overrides)
