"""Tests for the model registry.

The registry is where a measured finding becomes a default, so the tests here
mostly guard against a default silently reverting. The specific one that matters:
`condition_on_previous_text=False` on the large Whisper models is worth up to 22
CER points on long audio, and it is easy to drop while editing a dataclass.
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


def test_every_entry_is_self_consistent():
    for alias, m in REGISTRY.items():
        assert m.alias == alias
        assert "/" in m.repo, f"{alias} repo should be an HF id"
        assert m.backend in ("voxtral", "mlx-whisper", "mlx-chunked")
        assert m.weights_gb > 0
        # only voxtral is greedy here
        assert m.deterministic == (m.backend == "voxtral")


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
        assert m.backend in ("voxtral", "mlx-whisper", "mlx-chunked"), (alias,
                                                                       m.backend)


def test_kotoba_points_at_the_upstream_repo_not_a_third_party_mirror():
    """The alias must name weights published by the model's authors.

    History: it pointed at `kotoba-tech/kotoba-whisper-v2.2`, which is
    transformers-format with no MLX build, so `--model kotoba` could not run at
    all. The first fix swapped in a third-party MLX conversion, which worked but
    made a default depend on one person's unmaintained repo. Converting on first
    use (chunked.resolve_mlx_weights) removed the need for that, so this asserts
    the repo is upstream.

    v2.0 rather than v2.2 is deliberate and measured: the two share identical ASR
    weights (all 538 tensors within 1e-4), and v2.2's extras are separate
    post-processing models that need torch."""
    repo = REGISTRY["kotoba"].repo
    assert repo == "kotoba-tech/kotoba-whisper-v2.0"
    assert repo.startswith("kotoba-tech/")


def test_voxtral_needs_no_language_but_whisper_does():
    assert REGISTRY["voxtral"].needs_language is False
    assert REGISTRY["whisper-turbo"].needs_language is True
    assert REGISTRY["kotoba"].needs_language is True


def test_only_the_chunked_drivers_take_a_window_length():
    """`--chunk-seconds` means different things per engine, and one of them is
    "nothing at all".

    It reaches the chunked drivers, where window length is the biggest single
    lever (23 points across 10-30s on kotoba). It must NOT reach `mlx-whisper`:
    that driver's 30s window is fixed by the model's positional encoding, so a
    flag that appeared to change it would be a lie.
    """
    assert REGISTRY["kotoba"].chunked_long_form is True
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
    ("some/unknown-model", "voxtral"),
])
def test_infer_backend(repo, backend):
    assert infer_backend(repo) == backend


def test_describe_registry_mentions_every_alias_and_its_caveats():
    out = describe_registry()
    for alias in REGISTRY:
        assert alias in out
    # the caveats a user needs before trusting a number must be visible
    assert "deterministic" in out
    # the language restriction is a caveat a user must see before trusting output
    assert "ja" in out
