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

Everything here runs on MLX, with one measured exception: `reazon-k2` runs the
authors' ONNX files through sherpa-onnx on CPU, because that build is the most
accurate open Japanese ASR on its authors' cited benchmarks and ONNX is what
lets it run here without torch. Its throughput is therefore not comparable to
the GPU rows.

mlx-whisper is an optional import: a source install ships only the Voxtral path,
so a user who never asks for Whisper never installs it. The Homebrew formula
includes it, so the hint below names the right route for how this copy was
installed rather than assuming pip.
"""

import shutil
import sys
from pathlib import Path

SAMPLE_RATE_16K = 16000


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


def _refuse_non_japanese(language, alias):
    """Hard error on a language request these weights cannot honour.

    Both Japanese-only engines take NO language token at all: there is nothing to
    forward, so accepting `--language en` and silently producing Japanese output
    is exactly the honoured-looking-but-ignored failure this CLI refuses
    everywhere else."""
    if not language:
        return
    from .languages import to_iso

    code = to_iso(language, alias)
    if code != "ja":
        _die(f"{alias} is a Japanese-only model and takes no language token; "
             f"--language {language!r} (-> {code}) is refused rather than "
             f"ignored. The weights transcribe Japanese only.")


def parakeet_decode(loaded, audio, chunk_len: float, log=print,
                    overlap_s: float = 2.0):
    """Decode one already-loaded array through mlx-audio's Parakeet driver.

    Returns (cues, full_text, meta). Separate from ``transcribe_mlx_parakeet``
    for the same reason as ``qwen3_decode``: the benchmark has to measure exactly
    the code the CLI runs, so the weights load once outside the timing loop and
    every file goes through this function.

    ``audio`` is a 16kHz mono float32 array (the same `load_audio_16k` contract
    every other engine reads), wrapped in `mx.array` here. One audio front end
    for every engine is what keeps a conversion difference out of CER.

    Chunking is delegated to upstream `generate(chunk_duration=...)`, which cuts
    fixed windows, merges them over the overlap, and returns token times already
    offset. These ARE speech times (TDT predicts a duration per token), so unlike
    Qwen3-ASR this engine is subtitle-capable and says so via
    `cue_source: "token_times"`.

    Nothing detects or forces a language: these weights have one output language.
    `language_source: "single_language_model"` states that rather than borrowing
    the vocabulary of engines that actually ran something.
    """
    import mlx.core as mx

    duration = len(audio) / SAMPLE_RATE_16K
    r = loaded.generate(mx.array(audio), chunk_duration=chunk_len,
                        overlap_duration=overlap_s)
    sentences = list(getattr(r, "sentences", []) or [])
    cues = [(s.start, s.end, s.text.strip()) for s in sentences
            if s.text.strip()]
    text = "".join(c[2] for c in cues)

    token_count = sum(len(s.tokens) for s in sentences)
    span_end = max((c[1] for c in cues), default=0.0)
    meta = {"segments": len(sentences),
            "cue_source": "token_times",
            "chunk_seconds": chunk_len,
            "overlap_seconds": overlap_s,
            "language_source": "single_language_model",
            "token_count": token_count,
            # Speech stops before the file ends whenever there is trailing
            # silence, so this is provenance rather than truncation. Recorded so
            # a near-zero value on long audio is visible without reopening JSON.
            "last_token_end_s": round(span_end, 2)}
    if not token_count and duration > 10:
        log(f"[mlx-parakeet] WARNING: {duration:.0f}s of audio produced no "
            f"tokens at all. That is a degenerate decode, not silence.")
    return cues, text, meta


def transcribe_mlx_parakeet(audio_path: str, model, language=None, log=print,
                            **overrides):
    """NVIDIA Parakeet (Japanese) through mlx-audio's own loader.

    No new dependency: mlx-audio ships the FastConformer/TDT implementation and
    dispatches on the repo name, so this is the same `load` call the Voxtral and
    Qwen3 paths make. See ``parakeet_decode`` for the decode itself.
    """
    try:
        from mlx_audio.stt.utils import load as load_model
    except ImportError:      # pragma: no cover - mlx-audio is a hard dependency
        _die("parakeet needs mlx-audio, which should already be installed")

    _refuse_non_japanese(language, model.alias)

    opts = dict(model.opts)
    opts.update({k: v for k, v in overrides.items() if v is not None})
    chunk_len = opts.pop("chunk_length_s", 120.0)
    overlap_s = opts.pop("overlap_duration_s", 2.0)

    m = load_model(model.repo)
    from .audio import load_audio_16k

    log(f"[{model.backend}] window {chunk_len:g}s, overlap {overlap_s:g}s")
    return parakeet_decode(m, load_audio_16k(audio_path), chunk_len, log=log,
                           overlap_s=overlap_s)


# --- ReazonSpeech k2-v2 (sherpa-onnx) ----------------------------------------
#
# Zipformer transducer published as ONNX by the model's authors, run through
# sherpa-onnx. This is the one engine here that is not MLX, and that is a
# measured trade rather than an oversight: the k2 build is the most accurate
# open Japanese ASR on the public benchmarks its authors cite (TEDxJP-10K 9.09
# CER against 10.42 for their NeMo build), and ONNX is what makes it runnable
# without torch or CUDA. It decodes greedily, so it is deterministic like
# Voxtral; it runs on CPU, so its throughput is not comparable to the GPU rows.

# Sentence-break pieces for grouping token timestamps into cues. These weights
# emit NO punctuation (checked: a 558s file came back without a single 。), so
# the length cap does most of the closing and MAX_CUE_CHARS is set to the same
# 42 the Voxtral cue builder uses.
_SENTENCE_ENDERS = "。！？!?"
MAX_CUE_CHARS = 42
# A gap this long between consecutive token times closes the cue: it is a pause
# in the speech, and a subtitle holding across it is wrong about both sides.
PAUSE_GAP_S = 1.5
# Padding after the final token of a cue, which has no next-token start to end on.
LAST_TOKEN_PAD_S = 0.5


def reazon_k2_tokens_to_cues(times, tokens):
    """Group sherpa-onnx per-token timestamps into subtitle-shaped cues.

    sherpa-onnx returns parallel lists: a start time and a piece per token. A cue
    closes on a sentence ender, on a pause longer than PAUSE_GAP_S, or when it
    reaches MAX_CUE_CHARS, whichever comes first; with no punctuation emitted,
    the cap is what normally fires. A cue ends on the NEXT token's start, except
    across a pause, where it ends shortly after its own last token so neither
    side claims the silence. The file-final cue gets LAST_TOKEN_PAD_S.

    The ▁ word-boundary pieces are joined away: Japanese subtitles do not carry
    spaces, and keeping them would inflate every length check.
    """
    cues = []
    cur_toks: list[str] = []
    cur_start = None

    def close(end_time):
        nonlocal cur_toks, cur_start
        text = "".join(cur_toks).replace("▁", "").strip()
        if text and cur_start is not None:
            cues.append((cur_start, max(end_time, cur_start + 0.2), text))
        cur_toks, cur_start = [], None

    prev_t = None         # start time of the previous token
    for i, (tok, t) in enumerate(zip(tokens, times)):
        nxt = times[i + 1] if i + 1 < len(times) else None
        # Consecutive token STARTS are what reveal a pause: these pieces are
        # sub-syllabic, so a multi-second jump between them is silence.
        if prev_t is not None and cur_toks and t - prev_t > PAUSE_GAP_S:
            close(min(prev_t + LAST_TOKEN_PAD_S, t))
        if cur_start is None:
            cur_start = t
        cur_toks.append(tok)
        if tok in _SENTENCE_ENDERS or len("".join(cur_toks)) >= MAX_CUE_CHARS:
            close(nxt if nxt is not None else t + LAST_TOKEN_PAD_S)
        prev_t = t
    if cur_toks:
        close((prev_t if prev_t is not None else 0.0) + LAST_TOKEN_PAD_S)
    return cues


def reazon_k2_load(repo: str, precision: str = "int8") -> str:
    """Download (or reuse) the authors' ONNX files; return the directory.

    The repo ships fp32 and int8 builds side by side in one repo, so unlike every
    other alias here precision selects FILES rather than repos, and there is
    nothing to expose as `--quantization` over a single-repo lookup. fp32 is
    the default and that is measured here, not inherited: on this corpus's
    material int8 drops whole phrases mid-file (296 against 376 characters on
    one 112s file), which contradicts the near-parity the authors' table shows
    on read-speech benchmarks. Pass precision="int8" explicitly to trade that
    accuracy away for a 4x-smaller download and ~1.5x the speed.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:      # pragma: no cover - hub is a transitive dependency
        _die("reazon-k2 needs huggingface_hub to fetch the ONNX files")
    suffix = ".int8" if precision == "int8" else ""
    allow = [f"*{suffix}.onnx", "tokens.txt"]
    path = snapshot_download(repo, allow_patterns=allow)
    missing = [n for n in ("encoder", "decoder", "joiner")
               if not list(Path(path).glob(f"{n}*{suffix}.onnx"))]
    if missing:
        _die(f"{repo} is missing {', '.join(missing)}{suffix}.onnx; the "
             f"publishers may have renamed the files.")
    return str(path)


def reazon_k2_decode(recognizer, audio, chunk_len: float, log=print):
    """Decode one loaded array window-by-window. Returns (cues, full_text, meta).

    Separate from ``transcribe_reazon_k2`` so a benchmark can load once outside
    its timing loop, same as every other engine here.

    Windows come from `split_with_overlap`, the same energy-minima splitter the
    Voxtral path uses, with no overlap: a transducer cannot merge overlapping
    hypotheses honestly without alignment machinery this does not have, so seams
    are cut at quiet points instead. Token timestamps arrive per window and are
    offset back into file time before grouping.
    """
    from .audio import split_with_overlap

    duration = len(audio) / SAMPLE_RATE_16K
    windows = split_with_overlap(audio, target_s=chunk_len, overlap_s=0.0)
    chunks = windows[0]
    offsets = windows[1]

    import numpy as np

    all_times: list[float] = []
    all_tokens: list[str] = []
    empty_windows = 0
    for chunk, offset in zip(chunks, offsets):
        samples = np.asarray(chunk, dtype=np.float32)
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE_16K, samples)
        recognizer.decode_stream(stream)
        r = stream.result
        toks = list(getattr(r, "tokens", []) or [])
        if not toks:
            empty_windows += 1
            continue
        all_tokens.extend(toks)
        all_times.extend(float(t) + offset for t in
                         getattr(r, "timestamps", []) or [])
    if len(all_tokens) != len(all_times):
        _die("sherpa-onnx returned mismatched token/time lists "
             f"({len(all_tokens)} vs {len(all_times)}); refusing to guess")

    cues = reazon_k2_tokens_to_cues(all_times, all_tokens)
    text = "".join(c[2] for c in cues)
    meta = {"segments": len(cues),
            "cue_source": "token_times",
            "chunk_seconds": chunk_len,
            "windows": len(chunks),
            "empty_windows": empty_windows,
            "token_count": len(all_tokens),
            "language_source": "single_language_model"}
    if empty_windows:
        log(f"[sherpa-onnx] {empty_windows} of {len(chunks)} windows produced "
            f"no tokens")
    return cues, text, meta


def transcribe_reazon_k2(audio_path: str, model, language=None, log=print,
                         **overrides):
    """ReazonSpeech k2-v2 through sherpa-onnx. Returns (cues, full_text, meta)."""
    try:
        import sherpa_onnx
    except ImportError:
        _die(f"reazon-k2 needs sherpa-onnx: {_install_hint('reazon')}")

    _refuse_non_japanese(language, model.alias)

    opts = dict(model.opts)
    opts.update({k: v for k, v in overrides.items() if v is not None})
    chunk_len = opts.pop("chunk_length_s", 30.0)
    precision = opts.pop("precision", "fp32")

    weights_dir = reazon_k2_load(model.repo, precision)
    suffix = ".int8" if precision == "int8" else ""
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(Path(weights_dir) / f"encoder-epoch-99-avg-1{suffix}.onnx"),
        decoder=str(Path(weights_dir) / f"decoder-epoch-99-avg-1{suffix}.onnx"),
        joiner=str(Path(weights_dir) / f"joiner-epoch-99-avg-1{suffix}.onnx"),
        tokens=str(Path(weights_dir) / "tokens.txt"),
        num_threads=4,
    )
    log(f"[{model.backend}] {precision}, window {chunk_len:g}s, CPU decode")
    from .audio import load_audio_16k

    return reazon_k2_decode(recognizer, load_audio_16k(audio_path), chunk_len,
                            log=log)


DISPATCH = {
    "mlx-whisper": transcribe_mlx_whisper,
    "mlx-chunked": transcribe_mlx_chunked,
    "mlx-qwen3": transcribe_mlx_qwen3,
    "mlx-parakeet": transcribe_mlx_parakeet,
    "sherpa-onnx": transcribe_reazon_k2,
}


def run(model, audio_path: str, language=None, log=print, **overrides):
    """Dispatch to the right backend. Voxtral is handled in cli.py."""
    fn = DISPATCH.get(model.backend)
    if fn is None:
        _die(f"backend {model.backend!r} is not a non-Voxtral backend")
    return fn(audio_path, model, language=language, log=log, **overrides)
