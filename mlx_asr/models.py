"""The model registry: what `--model` accepts and how each one is run.

This exists because the three engines here are not interchangeable in the way a
single `--model` flag makes them look, and pretending otherwise produces bad
numbers. The differences that matter, all measured (see RESULTS.md):

  voxtral       MLX. Batched, chunked, greedy: byte-identical across reruns, so
                one run is its score. Takes no language token. Fastest measured
                here. Per-token timestamps, so timing drift stays under
                ~15 ms/min.
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
    backend: str                  # voxtral | mlx-whisper | mlx-chunked
    label: str
    languages: str = "multilingual"
    deterministic: bool = False
    # Backend-specific defaults, applied unless the user overrides them.
    opts: dict = field(default_factory=dict)
    notes: str = ""
    # Rough VRAM at the default batch, for the memory-derived fallback.
    weights_gb: float = 2.0

    @property
    def needs_language(self) -> bool:
        """True if this backend takes a language token, i.e. guessing costs
        accuracy. Whisper autodetects from the first 30s, which on real material
        misfires (it returned Russian for Japanese files here)."""
        return self.backend in ("mlx-whisper", "mlx-chunked")

    @property
    def chunked_long_form(self) -> bool:
        """True if this backend cuts the audio into independently-decoded windows
        whose length the caller chooses.

        This is what makes `--chunk-seconds` meaningful outside Voxtral. It is
        the single largest lever on a chunked engine (kotoba spans 23 points
        across 10-30s windows) and it is material-dependent, so it has to be
        reachable. The sequential `mlx-whisper` driver is excluded because its
        30s window is fixed by the model's positional encoding, not a choice."""
        return self.backend == "mlx-chunked"


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
            # model: all 538 tensors match to within 1e-4. v2.2's card says so
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
    ]
}

DEFAULT_ALIAS = "voxtral"


def infer_backend(repo: str) -> str:
    """Guess a backend for a raw HF repo id the registry does not list."""
    low = repo.lower()
    if "voxtral" in low:
        return "voxtral"
    # distil derivatives keep few decoder layers and need the chunked driver;
    # the sequential one costs kotoba 68 points (RESULTS.md)
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
    return Model(alias=name, repo=name, backend=backend,
                 label=name.split("/")[-1],
                 deterministic=(backend == "voxtral"),
                 opts=({"condition_on_previous_text": False}
                       if backend == "mlx-whisper" else {}),
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
        tag = ("  [" + ", ".join(flags) + "]") if flags else ""
        rows.append(f"  {alias:<{width}}  {m.label}{tag}")
        rows.append(f"  {'':<{width}}  {m.repo}")
        if m.notes:
            rows.append(f"  {'':<{width}}  {m.notes}")
        rows.append("")
    return "\n".join(rows).rstrip()
