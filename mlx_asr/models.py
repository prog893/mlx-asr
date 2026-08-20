"""The model registry: what `--model` accepts and how each one is run.

This exists because the four engines here are not interchangeable in the way a
single `--model` flag makes them look, and pretending otherwise produces bad
numbers. The differences that matter, all measured except where noted (see
docs/benchmarks/engines.md):

  voxtral       MLX. Batched, chunked, greedy: byte-identical across reruns *on one
                machine*, so one run is its score, though GPU reduction order
                differs between chips so a hypothesis does not reproduce on other
                hardware. Takes no language token. Fastest measured here. Per-token
                timestamps, so timing drift is the best measured here: worst slope
                25 ms/min across 7 files, against Whisper's 123.
  whisper       MLX (mlx-whisper), Whisper's *sequential* long-form algorithm.
                Its temperature fallback samples, so it is NOT deterministic;
                repeat runs of one config spread ~0.5 points of CER. Needs
                `condition_on_previous_text=False` on long audio or a repetition
                loop can feed itself across windows (worth up to 22 points).
                Most accurate on our corpus.
  kotoba        MLX, *chunked* long-form via our own driver (chunked.py).
                Chunked rather than sequential because a 2-layer distil decoder
                cannot carry state across windows: the sequential algorithm costs
                these weights 68 points. Japanese only. The weights are a
                third-party MLX conversion, since the authors publish torch only.
  qwen3-asr     MLX, chunked long-form inside mlx-audio's own loader. Greedy
                (`temperature=0.0` becomes `mx.argmax`, no fallback ladder), so
                deterministic like Voxtral. It is the only engine here that does
                real language identification and reports what it found. Its
                timestamps are CHUNK boundaries, not speech boundaries, so
                subtitle formats are refused rather than approximated (see
                `no_speech_timestamps`).

Everything here runs on MLX. A `transformers` backend used to exist, running
kotoba through the authors' torch/MPS pipeline as a correctness reference; the MLX
driver matched it to within a point at 1.4x the throughput, so it was removed
rather than carry a 2.5GB torch dependency for a path nobody should pick.

So each entry carries not just a repo id but the flags that make that engine
behave, and a `deterministic` flag the CLI surfaces rather than hiding.

`--model` accepts a registry alias (`voxtral`, `whisper-turbo`, ...) or any raw
HF repo id, in which case the backend is inferred from the id.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Model:
    """One selectable engine+weights combination."""

    alias: str
    repo: str
    backend: str                  # voxtral | mlx-whisper | mlx-chunked | mlx-qwen3
    label: str
    languages: str = "multilingual"
    deterministic: bool = False
    # Backend-specific defaults, applied unless the user overrides them.
    opts: dict = field(default_factory=dict)
    notes: str = ""
    # Rough VRAM at the default batch, for the memory-derived fallback.
    weights_gb: float = 2.0

    # True when this engine emits no timestamp finer than a chunk boundary, so a
    # subtitle file made from it would be a lie. Data rather than derived from the
    # backend string, because it is a property of the weights (no Qwen3-ASR variant
    # has speech-level times) and the CLI has to refuse `-f srt` on it.
    no_speech_timestamps: bool = False

    @property
    def needs_language(self) -> bool:
        """True if this backend takes a language token, i.e. guessing costs
        accuracy. Whisper autodetects from the first 30s, which on real material
        misfires (it returned Russian for Japanese files here).

        Qwen3-ASR is included even though its own language ID is the best here,
        because its multi-chunk autodetect path is buggy upstream: `language` is
        reassigned inside the loop from the first chunk's detection, so later
        chunks keep the `language X<asr_text>` prefix embedded in their text. An
        explicitly passed language avoids that branch entirely."""
        return self.backend in ("mlx-whisper", "mlx-chunked", "mlx-qwen3")

    @property
    def chunked_long_form(self) -> bool:
        """True if this backend cuts the audio into independently-decoded windows
        whose length the caller chooses.

        This is what makes `--chunk-seconds` meaningful outside Voxtral. It is
        the single largest lever on a chunked engine (kotoba spans 23 points
        across 10-30s windows) and it is material-dependent, so it has to be
        reachable. The sequential `mlx-whisper` driver is excluded because its
        30s window is fixed by the model's positional encoding, not a choice."""
        return self.backend in ("mlx-chunked", "mlx-qwen3")


REGISTRY: dict[str, Model] = {
    m.alias: m
    for m in [
        Model(
            alias="voxtral",
            repo="mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
            backend="voxtral",
            label="Voxtral Realtime 4B (4-bit)",
            deterministic=True,
            weights_gb=2.5,
            notes="fastest here; greedy so reproducible; no language token; "
                  "best timestamp stability",
        ),
        Model(
            alias="whisper-turbo",
            repo="mlx-community/whisper-large-v3-turbo",
            backend="mlx-whisper",
            label="Whisper large-v3-turbo",
            weights_gb=1.6,
            opts={"condition_on_previous_text": False},
            notes="most accurate measured here. cross-window conditioning is "
                  "disabled by default; leaving it enabled costs 10 points on "
                  "long audio",
        ),
        Model(
            alias="whisper-large-v3",
            repo="mlx-community/whisper-large-v3-mlx",
            backend="mlx-whisper",
            label="Whisper large-v3",
            weights_gb=3.1,
            opts={"condition_on_previous_text": False},
            notes="slower than turbo and no more accurate on our corpus; "
                  "disabling cross-window conditioning is worth 22 points here",
        ),
        Model(
            alias="whisper-large-v2",
            repo="mlx-community/whisper-large-v2-mlx",
            backend="mlx-whisper",
            label="Whisper large-v2",
            weights_gb=3.1,
            opts={"condition_on_previous_text": False},
        ),
        Model(
            alias="whisper-medium",
            repo="mlx-community/whisper-medium-mlx",
            backend="mlx-whisper",
            label="Whisper medium",
            weights_gb=1.5,
            opts={"condition_on_previous_text": False},
            notes="strong on English for its size",
        ),
        Model(
            alias="whisper-small",
            repo="mlx-community/whisper-small-mlx",
            backend="mlx-whisper",
            label="Whisper small",
            weights_gb=0.5,
            opts={"condition_on_previous_text": False},
        ),
        Model(
            alias="whisper-base",
            repo="mlx-community/whisper-base-mlx",
            backend="mlx-whisper",
            label="Whisper base",
            weights_gb=0.15,
        ),
        Model(
            alias="whisper-tiny",
            repo="mlx-community/whisper-tiny-mlx",
            backend="mlx-whisper",
            label="Whisper tiny",
            weights_gb=0.08,
        ),
        Model(
            alias="kotoba",
            # The authors' own repo, converted to MLX on first use (convert.py).
            # v2.0 rather than v2.2 because, measured, they are the same ASR
            # model: all 539 tensors are identical in value (max absolute
            # difference exactly 0.0; the files differ only in container metadata
            # and stored dtype). v2.2's card says so
            # too, describing itself as v2.0 plus speaker diarization and
            # punctuation, both of which are separate post-processing models
            # (`diarizers`, `punctuators`) rather than different weights, and
            # both of which want torch. So there is nothing to gain here from
            # v2.2 and a torch dependency to lose.
            repo="kotoba-tech/kotoba-whisper-v2.0",
            backend="mlx-chunked",
            label="kotoba-whisper v2.0 (Japanese, MLX)",
            languages="ja",
            weights_gb=1.6,
            # 10s beat 20s and 30s on spontaneous Japanese (27.0 / 31.3 / 49.7%),
            # and 10s beat 20s on the narration clip too (22.67% vs 27.22%). On
            # clean narration in the earlier sweep 20s won, so sweep it on your
            # own material.
            opts={"chunk_length_s": 10.0},
            notes="Japanese only. Converted to MLX on first use, since the "
                  "authors publish transformers weights only. Needs the CHUNKED "
                  "driver: Whisper's sequential algorithm costs these weights 68 "
                  "points, because a 2-layer distil decoder cannot carry state "
                  "across windows",
        ),
        Model(
            alias="qwen3-asr",
            repo="mlx-community/Qwen3-ASR-1.7B-8bit",
            backend="mlx-qwen3",
            label="Qwen3-ASR 1.7B (8-bit)",
            deterministic=True,
            weights_gb=2.3,
            # 30s, and measured rather than inherited. The library default is
            # 1200s, at which any file under 20 minutes is a single chunk: one
            # segment, and the batched path can never engage since it needs more
            # than one. Swept 15/30/60/120/300s on the 7-file corpus and, unlike
            # every other engine here, shorter is better on ACCURACY, SPEED and
            # MEMORY at once, monotonically: 19.98% / 21.42% / 23.55% / 62.47%
            # coverage CER at 30/60/120/300s. Longer windows give a repetition
            # loop a larger token budget to burn, so one loop wrecks more
            # transcript and costs more time. 15s ties on accuracy (20.04%) and
            # is faster, so this is a plateau rather than a boundary; 30s is the
            # optimum. See docs/benchmarks/qwen3-asr.md.
            opts={"chunk_length_s": 30.0},
            no_speech_timestamps=True,
            notes="greedy, so reproducible. Does real language ID and reports "
                  "it. Timestamps are CHUNK boundaries only, so -f srt and "
                  "-f vtt are refused; use txt or json",
        ),
        Model(
            alias="qwen3-asr-small",
            repo="mlx-community/Qwen3-ASR-0.6B-8bit",
            backend="mlx-qwen3",
            label="Qwen3-ASR 0.6B (8-bit)",
            deterministic=True,
            weights_gb=0.94,
            # Same 30s as its sibling. The window sweep was run on the 1.7B; the
            # mechanism behind it (a longer window gives a repetition loop a
            # bigger budget) is a property of the decoding loop rather than of the
            # size, so it carries over. Not separately measured.
            opts={"chunk_length_s": 30.0},
            no_speech_timestamps=True,
            notes="the 0.6B sibling of qwen3-asr; same caveats, same refused "
                  "output formats",
        ),
    ]
}

DEFAULT_ALIAS = "voxtral"


def infer_backend(repo: str) -> str:
    """Guess a backend for a raw HF repo id the registry does not list."""
    low = repo.lower()
    if "voxtral" in low:
        return "voxtral"
    # Ahead of both the distil and whisper checks. These two name an *architecture*,
    # so they are stronger evidence than the generic words "distil" and "whisper",
    # which a derivative id can carry alongside them
    # ("Qwen3-ASR-1.7B-whisper-distilled"). Routing such an id to either Whisper
    # driver would fail on a config those loaders cannot read.
    if "qwen3-asr" in low or "qwen3_asr" in low:
        return "mlx-qwen3"
    # distil derivatives keep few decoder layers and need the chunked driver;
    # the sequential one costs kotoba 68 points (docs/benchmarks/engines.md)
    if "kotoba" in low or "distil" in low:
        return "mlx-chunked"
    if "whisper" in low:
        return "mlx-whisper"
    return "voxtral"


def resolve(name: str | None) -> Model:
    """Return the Model for an alias or a raw repo id."""
    if not name:
        return REGISTRY[DEFAULT_ALIAS]
    if name in REGISTRY:
        return REGISTRY[name]
    # allow the full repo id of a registered model
    for m in REGISTRY.values():
        if m.repo == name:
            return m
    backend = infer_backend(name)
    # The two greedy backends. An unlisted repo on either one still gets the
    # `deterministic` flag right, because the CLI prints a "this engine samples"
    # caveat off it and a wrong caveat is worse than none.
    deterministic = backend in ("voxtral", "mlx-qwen3")
    opts = {}
    if backend == "mlx-whisper":
        opts = {"condition_on_previous_text": False}
    elif backend == "mlx-qwen3":
        # Same as the registry entries: measured best on this corpus, against a
        # library default of 1200s at which a sub-20-minute file is one chunk and
        # one cue. See docs/benchmarks/qwen3-asr.md.
        opts = {"chunk_length_s": 30.0}
    return Model(alias=name, repo=name, backend=backend,
                 label=name.split("/")[-1],
                 deterministic=deterministic,
                 opts=opts,
                 no_speech_timestamps=(backend == "mlx-qwen3"),
                 notes="not in the registry; defaults inferred from the repo id")


def describe_registry() -> str:
    """Human-readable `--list-models` output."""
    rows = []
    width = max(len(a) for a in REGISTRY)
    for alias, m in REGISTRY.items():
        flags = []
        if m.deterministic:
            flags.append("deterministic")
        if m.languages != "multilingual":
            flags.append(m.languages)
        # A caveat that changes which command a user can run at all, so it belongs
        # in the one-line summary rather than only in the notes below it.
        if m.no_speech_timestamps:
            flags.append("no srt/vtt")
        tag = ("  [" + ", ".join(flags) + "]") if flags else ""
        rows.append(f"  {alias:<{width}}  {m.label}{tag}")
        rows.append(f"  {'':<{width}}  {m.repo}")
        if m.notes:
            rows.append(f"  {'':<{width}}  {m.notes}")
        rows.append("")
    return "\n".join(rows).rstrip()
