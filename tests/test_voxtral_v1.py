"""Tests for the Voxtral v1 backend (Mini 3B / Small 24B, 2507).

Nothing here loads weights. The two things most worth guarding are the prompt, which is
assembled by hand because the library entry point needs soundfile, and the per-window
token budget, whose absence would truncate a transcript without any error.
"""

import io
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlx_asr import backends, voxtral_v1  # noqa: E402
from mlx_asr.models import REGISTRY, families, infer_backend, resolve  # noqa: E402

CLI = [sys.executable, "-m", "mlx_asr.cli"]
V1 = [m for m in REGISTRY.values() if m.family == "voxtral-v1"]


def _tekken():
    """The 3B tokenizer from the local HF cache, or skip. Never downloads."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache("mistralai/Voxtral-Mini-3B-2507", "tekken.json")
    if not isinstance(path, str):
        pytest.skip("tekken.json for Voxtral-Mini-3B-2507 is not in the HF cache")
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    return MistralTokenizer.from_file(path)


@pytest.mark.parametrize("seconds,language", [(7.3, "ja"), (30.0, "en"),
                                              (61.0, None), (95.5, "fr")])
def test_prompt_matches_mistral_common_token_for_token(seconds, language):
    """`build_prompt` must equal `encode_transcription`, the authors' own encoder.

    Covers a sub-chunk window, an exact chunk, a multi-chunk window, and the
    no-language (model detects) form. The reference path needs soundfile, which is not
    a dependency, so this skips where it is absent.
    """
    pytest.importorskip("soundfile")
    from mistral_common.protocol.transcription.request import TranscriptionRequest

    tok = _tekken()
    n = int(seconds * 16000)
    x = (np.random.RandomState(0).randn(n) * 0.1).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    ref = tok.instruct_tokenizer.encode_transcription(
        TranscriptionRequest(audio=buf.getvalue(), language=language, model="x"))
    padded = math.ceil(n / voxtral_v1.CHUNK_SAMPLES) * voxtral_v1.CHUNK_SAMPLES
    assert ref.audios[0].audio_array.shape[0] == padded
    assert voxtral_v1.build_prompt(tok, padded, language) == ref.tokens


def test_prompt_has_one_audio_token_per_80ms_and_ends_on_transcribe():
    """Structure check that runs without soundfile: 375 audio tokens per 30s chunk."""
    tok = _tekken()
    ids = voxtral_v1.build_prompt(tok, 2 * voxtral_v1.CHUNK_SAMPLES, "ja")
    audio_id = tok.instruct_tokenizer.audio_encoder.audio_token
    assert ids.count(audio_id) == 750
    assert ids[-1] == tok.instruct_tokenizer.TRANSCRIBE


def test_features_are_padded_to_whole_chunks_and_extracted_once():
    """(n_chunks, 3000, 128), from one extraction over the padded window.

    Per-chunk extraction would normalise each chunk against its own maximum, which is
    not what the processor does; the first chunk of a two-chunk window must therefore
    equal the first 3000 frames of the whole-window extraction.
    """
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(feature_size=128)
    x = (np.random.RandomState(1).randn(int(61.0 * 16000)) * 0.1).astype(np.float32)
    feats, padded_len = voxtral_v1.input_features(fe, x)
    assert padded_len == 3 * voxtral_v1.CHUNK_SAMPLES
    assert tuple(feats.shape) == (3, 3000, 128)
    # The processor's own audio kwargs (VoxtralProcessorKwargs), on the unpadded input.
    whole = fe(x, sampling_rate=16000, return_tensors="np", padding=True,
               truncation=False, pad_to_multiple_of=voxtral_v1.CHUNK_SAMPLES,
               )["input_features"][0]
    assert whole.shape == (128, 9000)
    for i in range(3):
        np.testing.assert_allclose(np.array(feats[i]).T,
                                   whole[:, 3000 * i:3000 * (i + 1)], rtol=1e-5)


def test_split_repo_reads_the_conversion_suffix_only_on_hub_ids(tmp_path):
    assert voxtral_v1.split_repo("mistralai/Voxtral-Mini-3B-2507:4bit") == (
        "mistralai/Voxtral-Mini-3B-2507", "4bit")
    assert voxtral_v1.split_repo("mistralai/Voxtral-Mini-3B-2507") == (
        "mistralai/Voxtral-Mini-3B-2507", None)
    # A local path is taken as-is even if it happens to contain a colon.
    odd = tmp_path / "a:b"
    odd.mkdir()
    assert voxtral_v1.split_repo(str(odd)) == (str(odd), None)


def test_family_has_both_sizes_and_defaults_to_the_one_a_16gb_mac_can_run():
    assert [m.size for m in families()["voxtral-v1"]] == ["3B", "24B"]
    assert resolve("voxtral-v1").size == "3B"
    assert resolve("voxtral-v1", "24b").size == "24B"


@pytest.mark.parametrize("m", V1, ids=lambda m: m.alias)
def test_weights_come_from_the_authors_repos_only(m):
    """Every precision is mistralai's own weights, as published or converted locally."""
    for quant, repo in m.quant_repos.items():
        source, precision = voxtral_v1.split_repo(repo)
        assert source.startswith("mistralai/Voxtral-"), (quant, repo)
        assert precision in (None, *voxtral_v1.QUANT_BITS), (quant, repo)
        assert (precision is None) == (quant == "bf16"), (quant, repo)
    assert m.repo in m.quant_repos.values()


@pytest.mark.parametrize("m", V1, ids=lambda m: m.alias)
def test_v1_is_greedy_windowed_and_untimed(m):
    assert m.deterministic
    assert m.no_speech_timestamps
    assert m.chunked_long_form
    assert m.opts.get("chunk_length_s")


@pytest.mark.parametrize("repo,backend", [
    ("mistralai/Voxtral-Mini-3B-2507", "mlx-voxtral-v1"),
    ("mistralai/Voxtral-Small-24B-2507", "mlx-voxtral-v1"),
    ("mlx-community/Voxtral-Mini-3B-2507-bf16", "mlx-voxtral-v1"),
    ("mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit", "voxtral"),
    ("mistralai/Voxtral-Mini-4B-Realtime-2602", "voxtral"),
])
def test_hub_ids_route_to_the_right_generation(repo, backend):
    assert infer_backend(repo) == backend


@pytest.mark.parametrize("model_type,backend", [("voxtral", "mlx-voxtral-v1"),
                                                ("voxtral_realtime", "voxtral")])
def test_local_directories_route_by_their_config(tmp_path, model_type, backend):
    """The name of a local conversion says nothing; its config does."""
    d = tmp_path / "voxtral-8bit"
    d.mkdir()
    (d / "config.json").write_text(f'{{"model_type": "{model_type}"}}')
    assert infer_backend(str(d)) == backend


def test_the_budget_is_per_window_not_per_file(monkeypatch):
    """Each window gets its own budget, scaled to its length.

    Upstream's `generate` has one 128-token budget per call. If this loop ever handed
    the file over whole again, a long recording would come back as its first minute
    with no error, which is the failure that truncated Qwen3-ASR.
    """
    seen = []

    def fake_window(loaded, chunk, language, budget):
        seen.append((len(chunk) / 16000, budget, language))
        return "テキスト"

    monkeypatch.setattr(voxtral_v1, "transcribe_window", fake_window)
    audio = np.zeros(16000 * 95, dtype=np.float32)
    cues, text, meta = backends.voxtral_v1_decode(object(), audio, "ja", 30.0,
                                                  log=lambda *a: None)
    assert len(seen) >= 3
    for dur, budget, lang in seen:
        assert budget >= max(backends.MIN_CHUNK_MAX_TOKENS,
                             int(dur * backends.TOKENS_PER_SECOND))
        assert lang == "ja"
    assert meta["audio_coverage"] == pytest.approx(1.0, abs=0.01)
    assert meta["cue_source"] == "chunk_boundaries"
    assert meta["language_source"] == "forced"


def test_no_language_is_recorded_as_the_models_own_detection(monkeypatch):
    monkeypatch.setattr(voxtral_v1, "transcribe_window",
                        lambda loaded, chunk, language, budget: "text")
    _, _, meta = backends.voxtral_v1_decode(object(), np.zeros(16000 * 5, np.float32),
                                            None, 30.0, log=lambda *a: None)
    assert meta["language_source"] == "model"
    assert meta["requested_language"] is None


@pytest.mark.parametrize("fmt", ["srt", "vtt", "all"])
def test_subtitle_formats_are_refused(fmt, tmp_path):
    r = subprocess.run(
        CLI + [str(tmp_path / "nope.wav"), "--model", "voxtral-v1", "-f", fmt],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert "no speech-level timestamps" in r.stderr, r.stderr


def test_max_batch_is_refused(tmp_path):
    r = subprocess.run(
        CLI + [str(tmp_path / "nope.wav"), "--model", "voxtral-v1", "-f", "txt",
               "--max-batch", "8"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "--max-batch" in r.stderr, r.stderr
