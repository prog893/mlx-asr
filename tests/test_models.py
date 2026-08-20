"""Tests for the model registry.

The registry is where a measured finding becomes a default, so the tests here
mostly guard against a default silently reverting. The specific ones that matter:
`condition_on_previous_text=False` on the large Whisper models is worth up to 22
CER points on long audio, and `chunk_length_s` on `qwen3-asr` overrides a library
default (1200s) at which a sub-20-minute file becomes one chunk and one cue. Both
are easy to drop while editing a dataclass, and neither failure is visible in the
output.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_asr.models import (
    DEFAULT_ALIAS,
    REGISTRY,
    describe_registry,
    infer_backend,
    resolve,
)


def test_default_is_registered_and_deterministic():
    m = REGISTRY[DEFAULT_ALIAS]
    assert m.backend == "voxtral"
    # the default must be the reproducible engine; a sampling default would make
    # every benchmark in this repo unrepeatable
    assert m.deterministic


BACKENDS = ("voxtral", "mlx-whisper", "mlx-chunked", "mlx-qwen3")

# The greedy backends. Voxtral decodes with argmax and no temperature ladder;
# Qwen3-ASR's `temperature=0.0` becomes `mx.argmax` for the same reason. Whisper's
# temperature fallback samples, so it is not on this list and its repeat runs
# spread ~0.5 CER points.
GREEDY_BACKENDS = ("voxtral", "mlx-qwen3")


def test_every_entry_is_self_consistent():
    for alias, m in REGISTRY.items():
        assert m.alias == alias
        assert "/" in m.repo, f"{alias} repo should be an HF id"
        assert m.backend in BACKENDS
        assert m.weights_gb > 0
        # `deterministic` drives a user-facing caveat ("this engine samples, so
        # repeat runs differ") and the benchmark decision to give an engine one run
        # rather than a distribution, so it has to track what the decoder actually
        # does. It was `backend == "voxtral"` until qwen3-asr, which is greedy too.
        assert m.deterministic == (m.backend in GREEDY_BACKENDS), alias


@pytest.mark.parametrize("alias", [
    "whisper-turbo", "whisper-large-v3", "whisper-large-v2", "whisper-medium",
    "whisper-small",
])
def test_large_whisper_models_disable_conditioning(alias):
    """Regression guard: leaving condition_on_previous_text on lets a repetition
    loop feed itself across 30s windows (measured up to 22 points on long
    audio)."""
    assert REGISTRY[alias].opts.get("condition_on_previous_text") is False


def test_kotoba_uses_a_chunked_driver_not_the_sequential_one():
    """The distinction that is worth 68 CER points on identical weights.

    What must never happen is kotoba being routed to `mlx-whisper`, whose
    sequential driver a 2-layer distil decoder cannot satisfy.
    """
    m = REGISTRY["kotoba"]
    assert m.backend == "mlx-chunked", m.backend
    assert m.opts.get("chunk_length_s") and m.opts["chunk_length_s"] <= 30
    assert m.languages == "ja"


def test_every_alias_runs_on_mlx():
    """No torch, no MPS. The `transformers` backend that ran kotoba through the
    authors' torch pipeline is gone: the MLX chunked driver matched it to within a
    point at 1.4x the throughput, so keeping it meant a 2.5GB dependency for a
    path nobody should pick."""
    for alias, m in REGISTRY.items():
        assert m.backend in BACKENDS, (alias, m.backend)


def test_kotoba_points_at_the_upstream_repo_not_a_third_party_mirror():
    """The alias must name weights published by the model's authors.

    History: it pointed at `kotoba-tech/kotoba-whisper-v2.2`, which is
    transformers-format with no MLX build, so `--model kotoba` could not run at
    all. The first fix swapped in a third-party MLX conversion, which worked but
    made a default depend on one person's unmaintained repo. Converting on first
    use (chunked.resolve_mlx_weights) removed the need for that, so this asserts
    the repo is upstream.

    v2.0 rather than v2.2 is deliberate and measured: the two share identical ASR
    weights (all 539 tensors identical in value), and v2.2's extras are separate
    post-processing models that need torch."""
    repo = REGISTRY["kotoba"].repo
    assert repo == "kotoba-tech/kotoba-whisper-v2.0"
    assert repo.startswith("kotoba-tech/")


def test_voxtral_needs_no_language_but_whisper_does():
    assert REGISTRY["voxtral"].needs_language is False
    assert REGISTRY["whisper-turbo"].needs_language is True
    assert REGISTRY["kotoba"].needs_language is True
    # Qwen3-ASR has the best language ID here and still needs to be told, because
    # its autodetect path reassigns `language` inside the chunk loop upstream and
    # leaves a `language X<asr_text>` prefix in every chunk after the first.
    assert REGISTRY["qwen3-asr"].needs_language is True


@pytest.mark.parametrize("alias", ["qwen3-asr", "qwen3-asr-small"])
def test_qwen3_is_greedy_and_says_so(alias):
    """It decodes with `mx.argmax` (temperature 0.0, no fallback ladder), so it gets
    one benchmark run rather than a distribution, like voxtral and unlike whisper."""
    assert REGISTRY[alias].deterministic is True
    assert REGISTRY[alias].backend == "mlx-qwen3"


@pytest.mark.parametrize("alias", ["qwen3-asr", "qwen3-asr-small"])
def test_qwen3_declares_it_has_no_speech_timestamps(alias):
    """The flag that makes `-f srt` a hard error.

    Its segments are `start=offset`, `end=offset+len(chunk)/sr`: the decode window
    the text came from, not when the speech happened. No variant of these weights
    has finer times, so a subtitle file would carry cues that do not correspond to
    speech, and at a long window it would be one cue holding the whole transcript.
    """
    assert REGISTRY[alias].no_speech_timestamps is True
    # The engines that do have real timestamps must not pick this up.
    assert REGISTRY["voxtral"].no_speech_timestamps is False
    assert REGISTRY["whisper-turbo"].no_speech_timestamps is False


@pytest.mark.parametrize("alias", ["qwen3-asr", "qwen3-asr-small"])
def test_qwen3_ships_the_measured_window_not_the_librarys(alias):
    """The library default is `chunk_duration=1200.0`, i.e. 20 minutes.

    At that value nearly every real file is a single chunk: one segment, and the
    batched path can never engage since it needs more than one. 30s is the measured
    optimum on the 7-file corpus, where unusually for this project shorter is better
    on accuracy, speed and memory at once, monotonically up to 300s (19.98% / 21.42% /
    23.55% / 62.47% coverage CER at 30/60/120/300s). Regression-guarded because a
    revert to the library value costs 42 points and would look like a tidy-up.
    """
    assert REGISTRY[alias].opts.get("chunk_length_s") == 30.0


def test_qwen3_repos_are_the_official_mlx_conversions():
    """Both aliases must name `mlx-community` builds of the Apache-2.0 originals.

    mlx-audio's loader has handled `qwen3_asr` since 0.3.1 and is in its dispatch
    table, so these need no conversion step and no new dependency.
    """
    assert REGISTRY["qwen3-asr"].repo == "mlx-community/Qwen3-ASR-1.7B-8bit"
    assert REGISTRY["qwen3-asr-small"].repo == "mlx-community/Qwen3-ASR-0.6B-8bit"


# --- --model family / --size ----------------------------------------------
#
# The selection scheme is family on the first layer, size and precision on the second.
# `whisper-turbo` and friends are no longer names a user types; they are internal
# aliases. This deliberately breaks v0.1.0-v0.2.2 invocations, so the error has to name
# the replacement rather than let the old name reach the hub as a repo id.

def test_every_entry_declares_a_family():
    from mlx_asr.models import families

    for alias, m in REGISTRY.items():
        assert m.family, alias
    # Four families for eleven entries, which is the point of the grouping.
    assert set(families()) == {"voxtral", "whisper", "kotoba", "qwen3-asr"}


def test_sizes_are_declared_only_where_there_is_a_choice():
    from mlx_asr.models import sizes_for

    assert sizes_for("whisper") == ["tiny", "base", "small", "medium",
                                    "large-v2", "large-v3", "turbo"]
    assert sizes_for("qwen3-asr") == ["0.6B", "1.7B"]
    # Single-size families report none, which is what makes --size refusable on them.
    assert sizes_for("voxtral") == []
    assert sizes_for("kotoba") == []


def test_whisper_defaults_to_turbo_not_the_largest():
    """The counterintuitive default, and it is measured.

    Picking by size number would take large-v3, which on this corpus scores 39.91% at
    library defaults against turbo's 24.97%, and 17.36% against 15.91% with
    no-condition. Turbo is more accurate here AND ~2x faster.
    """
    assert resolve("whisper").repo == "mlx-community/whisper-large-v3-turbo"
    assert resolve("whisper").size == "turbo"


def test_qwen3_defaults_to_the_larger_size():
    """The opposite case, also measured: 1.7B beats 0.6B by 3.9 points at n=20."""
    assert resolve("qwen3-asr").size == "1.7B"


def test_size_is_case_insensitive():
    """"1.7b" and "Large-V3" are reasonable to type."""
    assert resolve("qwen3-asr", "1.7b").size == "1.7B"
    assert resolve("whisper", "LARGE-V3").size == "large-v3"
    assert resolve("whisper", " turbo ").size == "turbo"


def test_an_unavailable_size_names_what_exists():
    from mlx_asr.models import UnknownSize

    with pytest.raises(UnknownSize) as e:
        resolve("whisper", "huge")
    msg = str(e.value)
    assert "tiny" in msg and "turbo" in msg


def test_size_is_refused_on_a_single_size_family():
    """Refused rather than ignored, so `--size large` on voxtral cannot look
    honoured. Same rule as --quantization on a single-precision model."""
    from mlx_asr.models import UnknownSize

    for fam in ("voxtral", "kotoba"):
        with pytest.raises(UnknownSize):
            resolve(fam, "large")


@pytest.mark.parametrize("old,expected", [
    ("whisper-turbo", "--model whisper --size turbo"),
    ("whisper-large-v3", "--model whisper --size large-v3"),
    ("whisper-tiny", "--model whisper --size tiny"),
    ("qwen3-asr-small", "--model qwen3-asr --size 0.6B"),
])
def test_the_old_per_size_names_name_their_replacement(old, expected):
    """These worked up to v0.2.2, so they are exactly what an old script sends.

    Caught here rather than passed to huggingface_hub, which would 404 without saying
    what to type instead.
    """
    from mlx_asr.models import UnknownModel

    with pytest.raises(UnknownModel) as e:
        resolve(old)
    assert expected in str(e.value), str(e.value)


def test_a_bare_typo_is_a_usage_error_not_a_repo_id():
    """Every repo id has an owner prefix, so a bare word cannot be one."""
    from mlx_asr.models import UnknownModel

    with pytest.raises(UnknownModel) as e:
        resolve("wisper")
    assert "voxtral, whisper, kotoba, qwen3-asr" in str(e.value)
    # A real repo id still resolves, so the guard is on the shape and not a whitelist.
    assert resolve("some/custom-model").repo == "some/custom-model"


def test_a_single_size_family_typo_does_not_suggest_a_size_flag():
    """`kotoba-v2` should point at `--model kotoba`, not at an empty size list."""
    from mlx_asr.models import UnknownModel

    with pytest.raises(UnknownModel) as e:
        resolve("kotoba-v2")
    msg = str(e.value)
    assert "--model kotoba" in msg
    assert "one of:" not in msg, msg


# --- --quantization -------------------------------------------------------
#
# The flag is nothing but a lookup from (alias, precision) to a published repo id;
# nothing is quantized at runtime. So the table has to match what the converters
# actually published, and an unpublished value has to be a usage error rather than a
# 404 from the hub after the user has waited.

def test_quantization_maps_to_published_repos_only():
    from mlx_asr.models import UnknownQuantization

    m = REGISTRY["qwen3-asr"]
    assert m.repo_for("4bit") == "mlx-community/Qwen3-ASR-1.7B-4bit"
    assert m.repo_for("6bit") == "mlx-community/Qwen3-ASR-1.7B-6bit"
    # Every published precision resolves, and each names its own repo.
    for q, repo in m.quant_repos.items():
        assert m.repo_for(q) == repo
        assert q.replace("bit", "bit") in repo or q == "bf16"
    # 2bit and 3bit are not published for these weights, so they must not be
    # accepted by a template that would 404 at download time.
    for absent in ("2bit", "3bit", "nvfp4", "mxfp8", "q8_0"):
        with pytest.raises(UnknownQuantization):
            m.repo_for(absent)


def test_voxtral_offers_only_the_builds_that_load():
    """Two published Voxtral quants CRASH rather than run.

    `mlx-community/...-6bit` and `ellamind/...-8bit-mlx` ship a config.json with no
    `model_type`, so mlx-audio routes them to the non-realtime loader and dies in
    post_load_hook. Listing them would turn a usage error into a crash after a
    multi-gigabyte download, so the ladder holds only the two that work.
    """
    m = REGISTRY["voxtral"]
    assert set(m.quant_repos) == {"4bit", "fp16"}, m.quant_repos
    for repo in m.quant_repos.values():
        assert "6bit" not in repo and "8bit" not in repo, repo
        assert repo.startswith("mlx-community/"), repo


def test_voxtral_weights_gb_travels_with_the_precision():
    """The coupling that would otherwise mis-size a batch silently.

    `derive_batch` subtracts the weight footprint from the GPU budget, so running fp16
    (8.9GB) while still claiming 4-bit's 2.5GB plans for ~6.4GB of memory that is
    already spent. The failure would surface as an OOM, not as a bad default.
    """
    m = REGISTRY["voxtral"]
    assert m.weights_gb_for(None) == m.weights_gb
    assert m.weights_gb_for("4bit") == 2.5
    assert m.weights_gb_for("fp16") == 8.9
    assert m.weights_gb_for("none") == 8.9        # 'none' resolves to fp16 here
    # And it really changes the derived batch, rather than just being recorded.
    from mlx_asr.hardware import derive_batch

    small = derive_batch(gpu_gb=12.7, weights_gb=8.9, chunk_seconds=60.0,
                         gpu_cores=10)
    big = derive_batch(gpu_gb=12.7, weights_gb=2.5, chunk_seconds=60.0, gpu_cores=10)
    assert small < big or small == 1, (small, big)


def test_none_resolves_per_alias_to_whatever_that_model_publishes():
    """Qwen3-ASR publishes its unquantized build as bf16 and Voxtral as fp16, so
    `--quantization none` cannot map to a single fixed name."""
    assert REGISTRY["voxtral"].repo_for("none").endswith("-fp16")
    assert REGISTRY["qwen3-asr"].repo_for("none").endswith("-bf16")
    # And each accepts the other's spelling, since a user should not have to know.
    assert REGISTRY["voxtral"].repo_for("bf16").endswith("-fp16")
    assert REGISTRY["qwen3-asr"].repo_for("fp16").endswith("-bf16")


def test_precisions_sort_smallest_first_with_unquantized_last():
    """Alphabetical would put bf16/fp16 first and read as the default. A name this
    does not parse must sort last rather than raise: an earlier version did
    `int(q.rstrip("bit"))` and crashed --help on `fp16`."""
    from mlx_asr.models import _quant_sort_key

    assert sorted(["fp16", "8bit", "4bit", "6bit"], key=_quant_sort_key) == [
        "4bit", "6bit", "8bit", "fp16"]
    assert sorted(["bf16", "4bit", "weird"], key=_quant_sort_key)[0] == "4bit"


def test_the_error_names_the_right_unquantized_build_per_model():
    """"'none' means bf16" is wrong on Voxtral, whose unquantized build is fp16.

    A fixed string would mislead on one of the two models, and the whole point of the
    message is to tell the user what to type instead.
    """
    from mlx_asr.models import UnknownQuantization

    with pytest.raises(UnknownQuantization) as e:
        REGISTRY["voxtral"].repo_for("8bit")
    assert "'none' means fp16" in str(e.value), str(e.value)
    # And the list is ordered smallest-first, not alphabetically.
    assert "Available: 4bit, fp16" in str(e.value), str(e.value)

    with pytest.raises(UnknownQuantization) as e:
        REGISTRY["qwen3-asr"].repo_for("3bit")
    assert "'none' means bf16" in str(e.value), str(e.value)


def test_none_means_unquantized():
    """What a user means by "no quantization" is bf16 here, and several spellings of
    that are reasonable to type."""
    m = REGISTRY["qwen3-asr"]
    bf16 = "mlx-community/Qwen3-ASR-1.7B-bf16"
    for spelling in ("none", "None", "full", "fp16", "f16", "bf16", "BF16"):
        assert m.repo_for(spelling) == bf16, spelling


def test_omitting_the_flag_gives_the_registry_default():
    for alias in ("qwen3-asr", "qwen3-asr-small"):
        assert REGISTRY[alias].repo_for(None) == REGISTRY[alias].repo


def test_eight_bit_is_the_default_and_it_is_the_measured_one():
    """Regression guard on a measured default.

    bf16 scored 20.16% against 8-bit's 19.98% on the 7-file corpus, a tie against the
    ~3.2 points this corpus resolves, while costing 1.36x the wall clock (14.1x vs
    19.2x) and 1.4x the peak memory (5.66 vs 4.05GB). On the 0.6B the gap is worse
    (26.24% vs 23.27%, 23.0x vs 32.8x). So 8bit is not a guess and a silent switch to
    bf16 would cost real speed for no accuracy.
    """
    for alias in ("qwen3-asr", "qwen3-asr-small"):
        m = REGISTRY[alias]
        assert m.repo == m.quant_repos["8bit"], alias
        assert m.repo.endswith("-8bit"), alias


def test_an_alias_with_one_precision_refuses_the_flag():
    from mlx_asr.models import UnknownQuantization

    for alias in ("whisper-turbo", "whisper-tiny", "kotoba"):
        assert not REGISTRY[alias].quant_repos, alias
        with pytest.raises(UnknownQuantization) as e:
            REGISTRY[alias].repo_for("4bit")
        assert "ships one precision" in str(e.value)


def test_a_raw_repo_id_refuses_the_flag_with_the_real_reason():
    """The flag is a lookup keyed on alias, so a repo id has already answered it.

    Refused rather than ignored, per this CLI's rule everywhere else: dropping it
    would hand back a transcript at a different precision than the user asked for.
    """
    from mlx_asr.models import UnknownQuantization

    m = resolve("mlx-community/Qwen3-ASR-1.7B-4bit")
    assert m.quant_repos == {}
    with pytest.raises(UnknownQuantization) as e:
        m.repo_for("8bit")
    assert "cannot be combined with a repo id" in str(e.value), str(e.value)


def test_the_help_text_is_derived_from_the_registry():
    """A hand-written list of precisions in the help would drift from what ships,
    which is the same failure as a flag that is accepted and ignored."""
    from mlx_asr.models import quantization_help

    text = quantization_help()
    for alias, m in REGISTRY.items():
        if m.quant_repos:
            assert alias in text, alias
            for q in m.quant_repos:
                assert q in text, (alias, q)
        else:
            # An alias with no choice must not appear as though it had one.
            assert f"{alias}:" not in text, alias
    assert "8bit (default)" in text


def test_list_models_shows_the_precisions_per_alias():
    out = describe_registry()
    assert "--quantization: 4bit, 5bit, 6bit, 8bit (default), bf16" in out
    # Ordered smallest-first with bf16 last, not alphabetically (which would put
    # bf16 first and read as the default).
    assert out.index("4bit") < out.index("bf16")


def test_the_smaller_qwen3_is_recorded_as_smaller():
    """`weights_gb` feeds the memory-derived batch fallback, so a wrong value there
    picks a batch size for the wrong model."""
    assert (REGISTRY["qwen3-asr-small"].weights_gb
            < REGISTRY["qwen3-asr"].weights_gb)


# --- the adapter's guards against silent truncation ------------------------
#
# These use a stub model rather than the weights, so they run in CI and on a laptop.
# The failure they cover is the worst kind this project has: upstream `max_tokens` is
# a budget for the WHOLE FILE, and when it runs out the chunk loop just `break`s, so
# the output is a short but perfectly well-formed transcript of the first part of the
# audio. Measured on a real 1553s file: one segment, 110.77% coverage CER, 1523
# seconds silently absent.

class _StubQwen3:
    """Records every `generate` call, and returns canned per-call text.

    ``texts`` is consumed one entry per call, so a test can make a single window
    misbehave and assert that the others are unaffected. A callable gets the chunk
    length, for text whose rate depends on the window.
    """

    def __init__(self, texts):
        self._texts = list(texts) if isinstance(texts, (list, tuple)) else texts
        self.calls = []

    def generate(self, audio, **kw):
        self.calls.append({"samples": len(audio), **kw})
        if callable(self._texts):
            text = self._texts(len(audio) / 16000)
        elif self._texts:
            text = self._texts.pop(0)
        else:
            text = ""
        return type("R", (), {"segments": [], "text": text,
                              "language": ["Japanese"]})()


def _silence(seconds):
    import numpy as np

    return np.zeros(int(seconds * 16000), dtype="float32")


def test_the_token_budget_is_per_window_not_per_file():
    """The bug this whole code path exists for.

    Upstream `max_tokens` is a budget for the WHOLE FILE and running out only makes the
    chunk loop `break`, so the tail of the audio silently produces no text. Measured on
    a real 1553s recording: one segment and 110.77% coverage CER, because the first
    window looped and ate all 8192 tokens. Raising the budget did not help (the same
    file came back 8% covered at 19950 tokens); it has to be per window.
    """
    from mlx_asr.backends import TOKENS_PER_SECOND, qwen3_decode

    stub = _StubQwen3(lambda secs: "あ" * 10)
    qwen3_decode(stub, _silence(600), "Japanese", 60.0, log=lambda *x: None)
    assert len(stub.calls) >= 10, len(stub.calls)
    # Each call's budget is sized to its own window, never to the file.
    for call in stub.calls:
        secs = call["samples"] / 16000
        assert call["max_tokens"] <= max(256, int(secs * TOKENS_PER_SECOND) + 1), call
        assert call["max_tokens"] < 600 * TOKENS_PER_SECOND


def test_a_short_tail_window_keeps_room_for_a_sentence():
    """Scaling down without a floor would be its own bug: a 2s tail chunk must not be
    capped at 50 tokens."""
    from mlx_asr.backends import MIN_CHUNK_MAX_TOKENS, qwen3_decode

    stub = _StubQwen3(lambda secs: "あ")
    qwen3_decode(stub, _silence(3), "Japanese", 60.0, log=lambda *x: None)
    assert stub.calls[0]["max_tokens"] == MIN_CHUNK_MAX_TOKENS


def test_an_explicit_budget_is_not_overridden():
    """A benchmark arm testing the library default has to be able to ask for it."""
    from mlx_asr.backends import qwen3_decode

    stub = _StubQwen3(lambda secs: "あ")
    qwen3_decode(stub, _silence(120), "Japanese", 60.0, log=lambda *x: None,
                 max_tokens=8192)
    assert all(c["max_tokens"] == 8192 for c in stub.calls), stub.calls


def test_one_runaway_window_no_longer_costs_the_rest_of_the_file():
    """The property the per-window budget buys, stated as a test.

    First window loops; every later window must still be decoded and present. Before
    this, the loop consumed the file's budget and everything after it was dropped.
    """
    from mlx_asr.backends import qwen3_decode

    lines = []
    n_windows = 10
    texts = ["ループした文章です、" * 2048] + ["正常な文字起こしです。"] * (n_windows + 4)
    cues, text, meta = qwen3_decode(_StubQwen3(texts), _silence(600), "Japanese",
                                    60.0, log=lines.append)
    assert meta["runaway_segments"] == 1, meta
    # Full audio span, and the healthy windows are in the output.
    assert meta["audio_coverage"] == 1.0, meta
    assert len(cues) >= n_windows, len(cues)
    assert text.count("正常な文字起こしです。") >= n_windows - 1
    assert any("repetition loops" in ln for ln in lines), lines
    assert not any("MISSING" in ln for ln in lines), lines


def test_full_coverage_produces_no_warning():
    from mlx_asr.backends import qwen3_decode

    lines = []
    _, _, meta = qwen3_decode(_StubQwen3(lambda secs: "あ" * int(secs * 7)),
                              _silence(120), "Japanese", 60.0, log=lines.append)
    assert meta["audio_coverage"] == 1.0
    assert meta["runaway_segments"] == 0
    assert lines == [], lines


def test_dense_real_speech_is_not_flagged_as_a_loop():
    """The threshold has to sit above a fast speaker, or the flag is noise. 9 chars/s
    is the top of the range measured on this corpus."""
    from mlx_asr.backends import qwen3_decode

    _, _, meta = qwen3_decode(_StubQwen3(lambda secs: "あ" * int(secs * 9)),
                              _silence(120), "Japanese", 60.0, log=lambda *x: None)
    assert meta["runaway_segments"] == 0, meta


def test_windows_tile_the_audio_without_gaps():
    """Cue times are derived from these offsets, so a gap or overlap would put text at
    the wrong timestamp. Upstream's splitter is used for the cut points, so this is
    really a check that its offsets are threaded through unchanged."""
    from mlx_asr.backends import qwen3_decode

    cues, _, meta = qwen3_decode(_StubQwen3(lambda secs: "あ" * 10),
                                 _silence(300), "Japanese", 60.0,
                                 log=lambda *x: None)
    assert cues[0][0] == 0.0
    for (s0, e0, _), (s1, _, _) in zip(cues, cues[1:]):
        assert abs(e0 - s1) < 1e-6, (e0, s1)
    assert abs(cues[-1][1] - 300.0) < 1.0, cues[-1]
    assert meta["chunk_seconds"] == 60.0


def test_the_meta_does_not_claim_a_language_was_detected():
    """This engine's language ID is its most distinctive feature and is unusable here,
    so the output must not imply it ran.

    The whisper backend sets `detected_language` from a real detection. Copying that
    key here would put "detected_language": "Japanese" in a JSON produced by a run that
    forced Japanese and detected nothing, which is the same quiet untruth as a flag that
    looks honoured and does nothing.
    """
    from mlx_asr.backends import qwen3_decode

    _, _, meta = qwen3_decode(_StubQwen3(lambda secs: "あ" * 10), _silence(60),
                              "Japanese", 30.0, log=lambda *x: None)
    assert "detected_language" not in meta, meta
    assert meta["language_source"] == "forced"
    assert meta["requested_language"] == "Japanese"


def test_an_empty_window_is_counted_not_hidden():
    """A window that returns nothing is a fact about the decode, and it is the shape a
    partial failure takes now that the budget is per window."""
    from mlx_asr.backends import qwen3_decode

    texts = ["ある程度の文字起こし", "", "続きの文字起こしです"] + [""] * 20
    cues, _, meta = qwen3_decode(_StubQwen3(texts), _silence(180), "Japanese", 60.0,
                                 log=lambda *x: None)
    assert meta["empty_segments"] >= 1, meta
    # An empty window produces no cue, so the cue count is below the window count.
    assert len(cues) < meta["segments"]


def test_only_the_chunked_drivers_take_a_window_length():
    """`--chunk-seconds` means different things per engine, and one of them is
    "nothing at all".

    It reaches the chunked drivers, where window length is the biggest single
    lever (23 points across 10-30s on kotoba). It must NOT reach `mlx-whisper`:
    that driver's 30s window is fixed by the model's positional encoding, so a
    flag that appeared to change it would be a lie.
    """
    assert REGISTRY["kotoba"].chunked_long_form is True
    assert REGISTRY["qwen3-asr"].chunked_long_form is True
    assert REGISTRY["whisper-turbo"].chunked_long_form is False
    assert REGISTRY["voxtral"].chunked_long_form is False


def test_every_chunked_engine_ships_a_window_default():
    """The CLI reports the default alongside an override, so a chunked engine
    without one would print a fabricated number."""
    for m in REGISTRY.values():
        if m.chunked_long_form:
            assert m.opts.get("chunk_length_s"), m.alias


def test_resolve_accepts_alias_repo_and_unknown():
    assert resolve("voxtral").alias == "voxtral"
    assert resolve(None).alias == DEFAULT_ALIAS
    # the full repo id of a registered model maps back to that entry
    assert resolve(REGISTRY["whisper-turbo"].repo).alias == "whisper-turbo"
    # an unlisted repo still resolves, with an inferred backend
    m = resolve("openai/whisper-large-v3")
    assert m.backend == "mlx-whisper"
    assert m.opts.get("condition_on_previous_text") is False


@pytest.mark.parametrize("repo,backend", [
    ("mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit", "voxtral"),
    ("kotoba-tech/kotoba-whisper-v2.2", "mlx-chunked"),
    ("some/distil-whisper-large-v3", "mlx-chunked"),
    ("mlx-community/whisper-small-mlx", "mlx-whisper"),
    ("Qwen/Qwen3-ASR-1.7B", "mlx-qwen3"),
    ("mlx-community/Qwen3-ASR-0.6B-8bit", "mlx-qwen3"),
    ("some/qwen3_asr-finetune", "mlx-qwen3"),
    # The qwen3 test runs BEFORE the whisper one, so an id carrying both words
    # routes to the loader that can actually read the config.
    ("some/Qwen3-ASR-1.7B-whisper-distilled", "mlx-qwen3"),
    ("some/unknown-model", "voxtral"),
])
def test_infer_backend(repo, backend):
    assert infer_backend(repo) == backend


def test_an_unlisted_qwen3_repo_gets_the_qwen3_defaults():
    """An inferred entry must carry the same three corrections a registry entry does,
    or `--model Qwen/Qwen3-ASR-1.7B` would silently behave differently from
    `--model qwen3-asr`: a 1200s window, a sampling caveat printed for a greedy
    engine, and an SRT written from chunk boundaries.
    """
    m = resolve("Qwen/Qwen3-ASR-1.7B")
    assert m.backend == "mlx-qwen3"
    assert m.deterministic is True
    assert m.no_speech_timestamps is True
    assert m.opts.get("chunk_length_s") == 30.0
    assert m.opts.get("condition_on_previous_text") is None


def test_the_docs_combination_table_covers_every_reachable_variant():
    """docs/MODELS.md lists every model/size/quant -> repo mapping, and it must not go
    stale: a generated table that has drifted is worse than no table, because it reads
    as authoritative. Checked against the registry rather than against a snapshot, so
    adding a precision fails here until the table is regenerated
    (`scripts/docs/gen_model_matrix.py`).
    """
    doc = (Path(__file__).resolve().parents[1] / "docs" / "MODELS.md").read_text()
    table = doc.split("## Every combination")[1].split("## ")[0]
    for m in REGISTRY.values():
        repos = set(m.quant_repos.values()) | {m.repo}
        for repo in repos:
            assert repo in table, f"{repo} missing from the MODELS.md table"
            # Each row must carry a working hub link, since that is the point of it.
            assert f"huggingface.co/{repo}" in table, repo


def test_describe_registry_lists_every_family_size_and_caveat():
    """`--list-models` is where a user picks, so everything reachable must appear.

    Families rather than internal aliases: `whisper-turbo` is no longer something a
    user types, so listing it would advertise a name the CLI rejects. What has to be
    present is the family, every size within it, and the caveats.
    """
    from mlx_asr.models import families, sizes_for

    out = describe_registry()
    for fam in families():
        assert fam in out, fam
        for size in sizes_for(fam):
            assert size in out, (fam, size)
    # No internal alias should leak into user-facing output.
    assert "whisper-turbo" not in out
    assert "qwen3-asr-small" not in out
    # the caveats a user needs before trusting a number must be visible
    assert "deterministic" in out
    # the language restriction is a caveat a user must see before trusting output
    assert "ja" in out
    # a refused output format has to appear here rather than only as an error after
    # the user has chosen
    assert "no srt/vtt" in out
    # and both second-layer flags, since they are the only route to most variants
    assert "--size:" in out and "--quantization:" in out
