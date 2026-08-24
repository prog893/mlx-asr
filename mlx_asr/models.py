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

## Selection is two layers

`--model` picks a **family** (`voxtral`, `whisper`, `kotoba`, `qwen3-asr`), and
`--size` plus `--quantization` pick the variant within it. So eleven entries are
four choices, and the second layer is where "same model, different tradeoff" lives:

    mlx-asr a.wav --model whisper --size small --language ja
    mlx-asr a.wav --model qwen3-asr --size 0.6B --quantization 4bit -f txt

The two second-layer flags are not equivalent, and the difference is measured.
**Size is a large accuracy lever** (43 CER points from `tiny` to `turbo`), so each
family's default is chosen on evidence: `whisper` defaults to `turbo`, not to the
larger `large-v3`, which is both less accurate here and half the speed.
**Precision is not a lever at all** (0.43 points across five variants of Voxtral), so
its default is the cheapest that loses nothing, and the flag exists mainly to let
someone re-check that on their own audio or to fit a smaller machine.

`--model` also takes any HF repo id, in which case the backend is inferred from the
id and both second-layer flags are refused: a repo id already names its variant.

`Model.alias` (`whisper-turbo`, `qwen3-asr-small`) is now **internal identity only**.
Those were the `--model` values up to v0.2.2; passing one now is an error that names
the two-flag replacement, rather than being forwarded to the hub as a repo id.
"""

from dataclasses import dataclass, field


class UnknownSize(ValueError):
    """The requested size is not one this family publishes.

    An error rather than a nearest match, for the same reason as the precision case:
    silently running `small` when `smol` was asked for would hand back a transcript
    from different weights than the user believes, and on this corpus size is worth up
    to 43 CER points, so the substitution would not be subtle.
    """

    def __init__(self, value, available, family=""):
        self.value, self.available, self.family = value, available, family
        shown = ", ".join(available)
        super().__init__(
            f"--size {value!r} is not available for --model {family}. "
            f"Sizes: {shown}."
        )


class UnknownQuantization(ValueError):
    """The requested precision is not published for this model.

    An error rather than a fallback, and it names the accepted set. Guessing would
    mean a 404 from the hub after the user has waited, or worse, silently running a
    different precision than they asked for, which is the same class of quiet
    substitution this CLI refuses for flags and output formats.
    """

    def __init__(self, value, available, alias="", is_repo_id=False, hint=""):
        self.value, self.available, self.alias = value, available, alias
        if available:
            shown = ", ".join(sorted(available, key=_quant_sort_key))
            # Name the unquantized build this model actually publishes: it is bf16 for
            # Qwen3-ASR and fp16 for Voxtral, so a fixed string would be wrong for one
            # of them.
            full = next((c for c in ("bf16", "fp16") if c in available), None)
            none_hint = f" 'none' means {full}." if full else ""
            msg = (f"--quantization {value!r} is not published for --model {alias} "
                   f"in a form that loads. Available: {shown}.{none_hint}"
                   + (f"\n{hint}" if hint else ""))
        elif is_repo_id:
            # Refused rather than ignored, per this CLI's rule: dropping it would
            # hand back a transcript at a different precision than was asked for.
            msg = ("--quantization is only supported for the built-in models "
                   "(see --list-models), so it cannot be combined with a repo id.")
        else:
            msg = (f"--quantization is not supported by --model {alias}: it ships one "
                   f"precision.")
        super().__init__(msg)


@dataclass(frozen=True)
class Model:
    """One selectable engine+weights combination: a family at one size.

    `--model` picks the family and `--size` picks among its entries, so `alias` is
    the internal identity (`whisper:turbo`) rather than something a user types.
    """

    alias: str
    repo: str
    backend: str                  # voxtral | mlx-whisper | mlx-chunked | mlx-qwen3
    label: str
    # The two halves of what a user selects. `family` is the `--model` value and
    # `size` the `--size` value; a single-size family leaves `size` empty and refuses
    # the flag, exactly as a single-precision model refuses --quantization.
    family: str = ""
    size: str = ""
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
    # Precision -> repo id, for the aliases that publish more than one build.
    #
    # Data rather than a template, for the same reason `languages.py` keeps its
    # vocabulary as data: the accepted set is then auditable and matches what is
    # actually published. A template (`f"{base}-{quant}"`) would happily accept
    # `--quantization 3bit` and fail at download time with a 404 from huggingface_hub
    # rather than a usage error naming the real options. Empty means the alias ships
    # one precision and `--quantization` is refused on it.
    quant_repos: dict = field(default_factory=dict)

    # Precision -> weights on disk, GB, where it differs enough to matter.
    #
    # Separate from `quant_repos` and not always populated, because it only feeds the
    # batch-size fallback on the Voxtral path. Getting it wrong there is not cosmetic:
    # `derive_batch` subtracts the weight footprint from the GPU budget, so running
    # fp16 (8.9GB) while claiming 2.5GB would size a batch for memory the machine does
    # not have. Any precision absent here falls back to `weights_gb`.
    quant_weights_gb: dict = field(default_factory=dict)
    # Extra line appended to an UnknownQuantization message, for a family where a
    # precision was MEASURED but cannot be shipped. Voxtral is the case: three rows in
    # docs/benchmarks/quantization.md (8bit affine, mxfp8, nvfp4) came from local
    # conversions, and the only published 8bit/6bit repos crash on load. Without this
    # the doc shows an 8bit result with no route to it.
    quant_hint: str = ""

    # Spellings of "no quantization". The published unquantized build is named bf16
    # for Qwen3-ASR and fp16 for Voxtral, so both are accepted for either and the
    # lookup below resolves whichever that alias actually publishes.
    _FULL_PRECISION_ALIASES = ("none", "full", "fp16", "f16", "bf16")

    def _quant_key(self, quant: str) -> str:
        """Normalise a requested precision to a key in `quant_repos`."""
        key = quant.strip().lower()
        if key not in self._FULL_PRECISION_ALIASES:
            return key
        # Whichever name this alias publishes its unquantized build under.
        for candidate in ("bf16", "fp16"):
            if candidate in self.quant_repos:
                return candidate
        return key

    def repo_for(self, quant: str | None) -> str:
        """The repo id for a requested precision. Raises on an unpublished one.

        This is the whole of what `--quantization` is: a lookup from (alias,
        precision) to a published repo id. Nothing is quantized at runtime, so an
        entry has to exist for the value to be accepted.
        """
        if not quant:
            return self.repo
        key = self._quant_key(quant)
        if key in self.quant_repos:
            return self.quant_repos[key]
        raise UnknownQuantization(quant, self.quant_repos, self.family or self.alias,
                                  is_repo_id=(self.alias == self.repo),
                                  hint=self.quant_hint)

    def weights_gb_for(self, quant: str | None) -> float:
        """Weight footprint for a precision, for the batch-size fallback."""
        if not quant:
            return self.weights_gb
        return self.quant_weights_gb.get(self._quant_key(quant), self.weights_gb)

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
            family="voxtral",
            repo="mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
            backend="voxtral",
            label="Voxtral Realtime 4B (4-bit)",
            deterministic=True,
            weights_gb=2.5,
            # Only the two builds that LOAD. `mlx-community/...-6bit` and
            # `ellamind/...-8bit-mlx` both ship a config.json with no `model_type`,
            # so mlx-audio routes them to the non-realtime `voxtral` loader and dies
            # in post_load_hook with "TokenizersBackend has no attribute tokenizer".
            # Verified still true on 2026-08-20. Listing them would turn a usage
            # error into a crash after a multi-gigabyte download, which is strictly
            # worse than saying they are unavailable.
            #
            # 4-bit is the default and it is measured: fp16 through 4-bit span 0.43
            # CER points on the narration clip (a tie), while fp16 costs 1.6x the
            # wall clock and 15.3GB of peak memory against 9.4GB. So the ladder here
            # exists to let someone verify that on their own audio, not because a
            # higher precision is expected to help.
            quant_repos={
                "4bit": "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
                "fp16": "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16",
            },
            # fp16 is 8.9GB on disk against 4-bit's 3.15GB, and this number is
            # subtracted from the GPU budget when sizing a batch. A 16GB machine
            # cannot run fp16 at all, and it must not be told it can.
            quant_weights_gb={"4bit": 2.5, "fp16": 8.9},
            # 8bit, mxfp8 and nvfp4 all have measured rows in
            # docs/benchmarks/quantization.md, from local conversions rather than the
            # hub, so the route to them is a convert step rather than a repo id.
            quant_hint=(
                "8bit, mxfp8 and nvfp4 were measured from LOCAL conversions (the "
                "published 6bit and 8bit repos crash on load). To reproduce one:\n"
                "  python -m mlx_audio.convert --hf-path "
                "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16 \\\n"
                "    --mlx-path ./voxtral-8bit --quantize --q-bits 8 --q-group-size 64\n"
                "then pass --model ./voxtral-8bit. WORTH DOING: on the 20-file corpus a "
                "local 8bit build scores 15.27% against 4bit's 16.34%, matching fp16 "
                "(15.04%) while running at 4bit's speed and 7.3GB of peak memory "
                "instead of 13.0GB. See docs/benchmarks/quantization.md."
            ),
            notes="fastest here; greedy so reproducible; no language token; "
                  "best timestamp stability. 4bit is LAST of five precisions on "
                  "accuracy and ships anyway, since fp16 needs 13GB and no loadable "
                  "8bit build is published; convert one locally if you have the RAM",
        ),
        Model(
            alias="whisper-turbo",
            family="whisper",
            size="turbo",
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
            family="whisper",
            size="large-v3",
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
            family="whisper",
            size="large-v2",
            repo="mlx-community/whisper-large-v2-mlx",
            backend="mlx-whisper",
            label="Whisper large-v2",
            weights_gb=3.1,
            opts={"condition_on_previous_text": False},
        ),
        Model(
            alias="whisper-medium",
            family="whisper",
            size="medium",
            repo="mlx-community/whisper-medium-mlx",
            backend="mlx-whisper",
            label="Whisper medium",
            weights_gb=1.5,
            opts={"condition_on_previous_text": False},
            notes="strong on English for its size",
        ),
        Model(
            alias="whisper-small",
            family="whisper",
            size="small",
            repo="mlx-community/whisper-small-mlx",
            backend="mlx-whisper",
            label="Whisper small",
            weights_gb=0.5,
            opts={"condition_on_previous_text": False},
        ),
        Model(
            alias="whisper-base",
            family="whisper",
            size="base",
            repo="mlx-community/whisper-base-mlx",
            backend="mlx-whisper",
            label="Whisper base",
            weights_gb=0.15,
        ),
        Model(
            alias="whisper-tiny",
            family="whisper",
            size="tiny",
            repo="mlx-community/whisper-tiny-mlx",
            backend="mlx-whisper",
            label="Whisper tiny",
            weights_gb=0.08,
        ),
        Model(
            alias="kotoba",
            family="kotoba",
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
            family="qwen3-asr",
            size="1.7B",
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
            # 8-bit is the default because it is measured: bf16 scored 20.16%
            # against 8-bit's 19.98% on the 7-file corpus, a tie against the ~3.2
            # points this corpus resolves, while costing 1.36x the wall clock
            # (14.1x vs 19.2x) and 1.4x the peak memory (5.66 vs 4.05GB). The
            # ladder is exposed rather than hidden because a precision is a
            # size/quality knob like a Whisper size, and 4bit at 1.61GB is a real
            # choice on a small machine. See docs/benchmarks/quantization.md.
            quant_repos={
                "4bit": "mlx-community/Qwen3-ASR-1.7B-4bit",
                "5bit": "mlx-community/Qwen3-ASR-1.7B-5bit",
                "6bit": "mlx-community/Qwen3-ASR-1.7B-6bit",
                "8bit": "mlx-community/Qwen3-ASR-1.7B-8bit",
                "bf16": "mlx-community/Qwen3-ASR-1.7B-bf16",
            },
            notes="greedy, so reproducible. Does real language ID and reports "
                  "it. Timestamps are CHUNK boundaries only, so -f srt and "
                  "-f vtt are refused; use txt or json",
        ),
        Model(
            alias="qwen3-asr-small",
            family="qwen3-asr",
            size="0.6B",
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
            # Same default and same reason, and here the cost of bf16 is worse:
            # 26.24% against 23.27%, and 23.0x against 32.8x, which would remove
            # the only reason this alias ships (it is the fastest engine measured
            # in this project).
            quant_repos={
                "4bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
                "5bit": "mlx-community/Qwen3-ASR-0.6B-5bit",
                "6bit": "mlx-community/Qwen3-ASR-0.6B-6bit",
                "8bit": "mlx-community/Qwen3-ASR-0.6B-8bit",
                "bf16": "mlx-community/Qwen3-ASR-0.6B-bf16",
            },
            notes="the fastest engine measured in this project (32.8x), and "
                  "7.1 points behind voxtral on Japanese. Same caveats as 1.7B",
        ),
    ]
}

DEFAULT_ALIAS = "voxtral"

# Default size per multi-size family, and it is a measured choice rather than the
# biggest or the newest.
#
#   whisper    `turbo` and not `large-v3`, which is the counterintuitive one. On this
#              corpus large-v3 at library defaults scores 39.91% against turbo's
#              24.97%, and with no-condition 17.36% against 15.91%. Turbo is both
#              more accurate here and ~2x faster, so picking by size number would be
#              wrong in both directions.
#   qwen3-asr  `1.7B`, which beats 0.6B by 3.9 points on the 20-file corpus (19.33%
#              against 23.27%). The 0.6B is the speed option, not the default.
DEFAULT_SIZE = {"whisper": "turbo", "qwen3-asr": "1.7B"}

# Display order per family: smallest/weakest first, so a list reads as a ladder.
# Explicit rather than sorted, because neither alphabetical nor parameter count gives
# Whisper's real ordering (`turbo` is a distilled large-v3, so it sorts last by
# capability while sorting first alphabetically).
SIZE_ORDER = {
    "whisper": ["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo"],
    "qwen3-asr": ["0.6B", "1.7B"],
}


def families() -> dict[str, list[Model]]:
    """Family name -> its entries, in display order."""
    out: dict[str, list[Model]] = {}
    for m in REGISTRY.values():
        out.setdefault(m.family, []).append(m)
    for fam, entries in out.items():
        order = SIZE_ORDER.get(fam)
        if order:
            entries.sort(key=lambda m: order.index(m.size)
                         if m.size in order else len(order))
    return out


def sizes_for(family: str) -> list[str]:
    """The sizes a family publishes, in display order. Empty for a single-size one."""
    entries = families().get(family, [])
    return [m.size for m in entries if m.size]


def resolve_family(family: str, size: str | None = None) -> Model:
    """Pick an entry by family and size. Raises UnknownSize on a bad size."""
    entries = families()[family]
    available = [m.size for m in entries if m.size]
    if not size:
        if len(entries) == 1:
            return entries[0]
        want = DEFAULT_SIZE[family]
        return next(m for m in entries if m.size == want)
    if not available:
        # Single-size family: refused rather than ignored, so `--size large` on
        # voxtral cannot look honoured.
        raise UnknownSize(size, ["(none: this model has one size)"], family)
    # Case-insensitive, since "1.7b" and "Large-V3" are reasonable to type.
    by_lower = {m.size.lower(): m for m in entries if m.size}
    if size.strip().lower() in by_lower:
        return by_lower[size.strip().lower()]
    raise UnknownSize(size, available, family)


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


class UnknownModel(ValueError):
    """`--model` named neither a family nor something that looks like a repo id.

    Exists because the families replaced the old per-size names: someone typing
    `whisper-turbo` (which worked up to v0.2.2) would otherwise have it treated as a
    repo id, and the failure would arrive from huggingface_hub as a 404 rather than
    from the CLI as "you meant --model whisper --size turbo".
    """

    def __init__(self, value, available, size_hint=""):
        self.value, self.available = value, available
        msg = (f"--model {value!r} is not a built-in model. "
               f"Models: {', '.join(available)}.")
        if size_hint:
            msg += f" Did you mean {size_hint}?"
        msg += " A Hugging Face repo id also works, but must contain a '/'."
        super().__init__(msg)


def _size_hint(name: str) -> str:
    """Turn an old-style `family-size` name into the new two-flag form.

    Only for the error message. The v0.1.0-v0.2.2 names were `whisper-turbo`,
    `qwen3-asr-small` and so on, and those are exactly what a user or an old script
    will type first.
    """
    for fam, entries in families().items():
        if not name.startswith(f"{fam}-"):
            continue
        tail = name[len(fam) + 1:]
        for m in entries:
            if m.size and (m.size.lower() == tail.lower()
                           # `qwen3-asr-small` was the 0.6B; "small" is not its size.
                           or (tail == "small" and m.size == "0.6B")):
                return f"--model {fam} --size {m.size}"
        available = sizes_for(fam)
        if not available:
            # Single-size family, so the suffix was never valid: `--model kotoba` is
            # the whole answer and suggesting a size flag would be wrong.
            return f"--model {fam}"
        return f"--model {fam} --size <one of: {', '.join(available)}>"
    return ""


def resolve(name: str | None, size: str | None = None) -> Model:
    """Return the Model for a family (with optional size) or a raw repo id."""
    if not name:
        return resolve_family(REGISTRY[DEFAULT_ALIAS].family, size)
    fams = families()
    if name in fams:
        return resolve_family(name, size)
    # allow the full repo id of a built-in model
    for m in REGISTRY.values():
        if m.repo == name:
            return m
    # A bare word that is not a family is a mistake, not a repo id: every repo id has
    # an owner prefix. Catching it here turns a 404 after a download attempt into a
    # usage error that names the replacement.
    if "/" not in name:
        raise UnknownModel(name, list(fams), _size_hint(name))
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
                 notes="not a built-in model; defaults inferred from the repo id")


def size_help() -> str:
    """Per-family sizes, for `--help`. Derived, so it cannot drift from the registry."""
    parts = []
    for fam, entries in families().items():
        available = [m.size for m in entries if m.size]
        if not available:
            continue
        default = DEFAULT_SIZE.get(fam)
        shown = [f"{s} (default)" if s == default else s for s in available]
        parts.append(f"{fam}: {', '.join(shown)}")
    return "; ".join(parts) if parts else "no family currently has multiple sizes"


def quantization_help() -> str:
    """Per-alias precision options, for `--help`.

    Derived from the registry rather than written out, so a precision added or
    dropped in one place cannot leave the help text claiming otherwise. That is the
    same failure mode as a flag the CLI accepts and ignores: the text is read as
    authoritative and nothing checks it.
    """
    lines = []
    for alias, m in REGISTRY.items():
        if m.quant_repos:
            opts = sorted(m.quant_repos, key=_quant_sort_key)
            default = next((q for q, r in m.quant_repos.items() if r == m.repo), None)
            shown = [f"{q} (default)" if q == default else q for q in opts]
            lines.append(f"{alias}: {', '.join(shown)}")
    if not lines:
        return "no alias currently publishes a precision choice"
    return "; ".join(lines)


def _quant_sort_key(q: str):
    """Order precisions smallest-first, unquantized last rather than alphabetically.

    Alphabetical would put bf16 and fp16 first and read as though they were the
    default. Anything that is not an `<n>bit` name sorts last, so a precision naming
    scheme this does not know about degrades to the end of the list instead of raising
    (an earlier version did `int(q.rstrip("bit"))` and crashed `--help` on `fp16`).
    """
    digits = q[:-3] if q.endswith("bit") else ""
    return (0, int(digits)) if digits.isdigit() else (1, q)


def describe_registry() -> str:
    """Human-readable `--list-models` output, grouped by family.

    Grouped rather than flat because the flat list had 11 entries for 4 actual
    choices, which read as 11 unrelated models. The family is what `--model` takes;
    everything indented under it is reachable with `--size` and `--quantization`.
    """
    rows = []
    for fam, entries in families().items():
        default = resolve_family(fam)
        flags = []
        if default.deterministic:
            flags.append("deterministic")
        if default.languages != "multilingual":
            flags.append(default.languages)
        # A caveat that changes which command a user can run at all, so it belongs in
        # the family headline rather than only in the notes below it.
        if default.no_speech_timestamps:
            flags.append("no srt/vtt")
        tag = ("  [" + ", ".join(flags) + "]") if flags else ""
        rows.append(f"  {fam}{tag}")

        available = [m.size for m in entries if m.size]
        if available:
            shown = ", ".join(f"{s} (default)" if s == DEFAULT_SIZE.get(fam) else s
                              for s in available)
            rows.append(f"      --size: {shown}")
        # Per family, since which precisions exist is a property of what the
        # converters published for those weights.
        if default.quant_repos:
            opts = sorted(default.quant_repos, key=_quant_sort_key)
            dq = next((q for q, r in default.quant_repos.items()
                       if r == default.repo), None)
            rows.append("      --quantization: " + ", ".join(
                f"{q} (default)" if q == dq else q for q in opts))
        rows.append(f"      default weights: {default.repo}")
        if default.notes:
            rows.append(f"      {default.notes}")
        # A size that carries its own caveat needs it visible, since picking that size
        # is the only way to encounter it.
        for m in entries:
            if m.size and m.notes and m.notes != default.notes:
                rows.append(f"      {m.size}: {m.notes}")
        rows.append("")
    return "\n".join(rows).rstrip()
