"""Tests for the pure logic: no model, no audio files, no network.

Deliberately scoped to the properties that other code and the docs rely on, and
that have actually broken before:

- chunking must not lose or duplicate samples (the docstring promises this, and
  timestamps are derived from cumulative chunk lengths)
- cue building must never emit U+FFFD (a real bug: cue flushes were splitting
  multi-byte characters, which silently inflated CER)
- resolve_profile must pass every key the CLI reads (a real bug: overlap_seconds
  was defined in profiles.json but dropped by resolve_profile, so the default
  overlap silently became 0)
- auto gain must be a no-op above the trigger level, since it is on by default

    uv run pytest -q
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_asr.audio import (
    AUTO_TRIGGER_DBFS,
    apply_gain,
    auto_gain_db,
    compact_silence,
    dbfs,
    map_time,
    per_chunk_gain_db,
    split_at_silences,
    split_with_overlap,
    SAMPLE_RATE,
)
from mlx_asr.chunked import _is_mlx_format, resolve_mlx_weights
from mlx_asr.hardware import resolve_profile
from mlx_asr.output import _complete, build_cues, fmt_ts, wrap_cjk


def synth(seconds=180.0, seed=0):
    """Speech-ish signal: bursts of noise separated by near-silence."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    x = rng.normal(0, 0.002, n).astype(np.float32)
    t = 0
    while t < n:
        burst = int(rng.uniform(1.0, 4.0) * SAMPLE_RATE)
        x[t : t + burst] += rng.normal(0, 0.2, len(x[t : t + burst])).astype(np.float32)
        t += burst + int(rng.uniform(0.3, 1.5) * SAMPLE_RATE)
    return np.clip(x, -1, 1)


# --- chunking -------------------------------------------------------------

@pytest.mark.parametrize("target", [15.0, 30.0, 60.0])
def test_split_preserves_every_sample(target):
    a = synth()
    chunks = split_at_silences(a, target_s=target)
    assert sum(len(c) for c in chunks) == len(a)
    assert np.array_equal(np.concatenate(chunks), a)


def test_split_handles_audio_shorter_than_one_chunk():
    a = synth(seconds=5.0)
    chunks = split_at_silences(a, target_s=60.0)
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], a)


@pytest.mark.parametrize("overlap", [0.0, 4.0, 8.0])
def test_overlap_kept_region_equals_plain_chunk(overlap):
    """The kept region of each overlapped chunk must be the original chunk, and
    warmup_tokens must point exactly at its start."""
    a = synth()
    base = split_at_silences(a, target_s=30.0)
    chunks, offsets, warm = split_with_overlap(a, target_s=30.0, overlap_s=overlap)
    assert len(chunks) == len(base) == len(offsets) == len(warm)
    for i, (c, w) in enumerate(zip(chunks, warm)):
        assert np.array_equal(c[w * 1280 :], base[i])
    assert warm[0] == 0  # nothing precedes the first chunk
    if overlap == 0:
        assert all(w == 0 for w in warm)


def test_overlap_offsets_are_monotonic_and_start_at_zero():
    a = synth()
    _, offsets, _ = split_with_overlap(a, target_s=30.0, overlap_s=4.0)
    assert offsets[0] == 0.0
    assert all(b > x for x, b in zip(offsets, offsets[1:]))


# --- silence compaction and time mapping ---------------------------------

def test_compact_silence_shortens_and_maps_time_back():
    a = synth(seconds=300.0)
    out, tmap = compact_silence(a)
    assert len(out) <= len(a)
    assert tmap[0] == (0.0, 0.0)
    # Mapping is monotonic and never reports a time before its own breakpoint.
    prev = -1.0
    for new_t, old_t in tmap:
        assert old_t >= new_t - 1e-6
        assert new_t > prev
        prev = new_t
    assert map_time(0.0, tmap) == pytest.approx(0.0)


def test_map_time_is_monotonic():
    a = synth(seconds=300.0)
    _, tmap = compact_silence(a)
    xs = np.linspace(0, len(a) / SAMPLE_RATE, 200)
    mapped = [map_time(float(x), tmap) for x in xs]
    assert all(b >= x - 1e-6 for x, b in zip(mapped, mapped[1:]))


# --- gain -----------------------------------------------------------------

def test_auto_gain_is_noop_above_trigger():
    a = synth()
    a = a / max(np.abs(a).max(), 1e-9) * 0.9      # ~-0.9 dBFS
    assert auto_gain_db(a) == 0.0
    assert np.array_equal(apply_gain(a, "auto"), a)


@pytest.mark.parametrize("att_db", [-8.0, -12.0, -20.0, -40.0])
def test_auto_gain_lifts_quiet_audio_without_clipping(att_db):
    a = synth()
    a = (a / max(np.abs(a).max(), 1e-9) * (10 ** (att_db / 20))).astype(np.float32)
    out = apply_gain(a, "auto")
    assert dbfs(out) == pytest.approx(-1.0, abs=0.2)
    assert np.abs(out).max() < 1.0            # never clips
    assert auto_gain_db(a) > 0                 # gain is never negative


def test_auto_gain_trigger_boundary():
    a = synth()
    just_loud = (a / max(np.abs(a).max(), 1e-9)
                 * 10 ** ((AUTO_TRIGGER_DBFS + 0.5) / 20)).astype(np.float32)
    assert auto_gain_db(just_loud) == 0.0


def test_per_chunk_gain_leaves_silence_alone():
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    assert per_chunk_gain_db(silence) == 0.0


def test_gain_none_is_identity():
    a = synth(seconds=10.0)
    assert np.array_equal(apply_gain(a, "none"), a)


# --- profiles -------------------------------------------------------------

REQUIRED_PROFILE_KEYS = {"batch", "chunk_seconds", "kv_bits", "overlap_seconds",
                         "matched"}


def test_resolve_profile_exposes_every_key_the_cli_reads():
    """Regression: overlap_seconds existed in profiles.json but resolve_profile
    dropped it, so the CLI silently defaulted overlap to 0."""
    prof = resolve_profile()
    assert REQUIRED_PROFILE_KEYS <= set(prof)


@pytest.mark.parametrize("gpu_gb,ram", [(4, 8), (12, 16), (30, 36), (60, 64), (115, 128)])
def test_fallback_profiles_are_complete_and_avoid_the_batch_valley(gpu_gb, ram):
    prof = resolve_profile({"chip": "Apple MX Unknown", "ram_gb": ram,
                            "gpu_working_set_gb": gpu_gb})
    assert REQUIRED_PROFILE_KEYS <= set(prof)
    assert prof["batch"] == 1 or prof["batch"] >= 12, (
        "batch 2-8 is measurably slower per step than batch 1"
    )
    assert prof["chunk_seconds"] > 0


def test_shipped_profiles_json_is_valid_and_sane():
    data = json.loads(
        (Path(__file__).resolve().parent.parent
         / "mlx_asr" / "profiles.json").read_text(encoding="utf-8")
    )
    for prof in data["profiles"]:
        assert {"match", "batch", "chunk_seconds"} <= set(prof)
        assert prof["batch"] == 1 or prof["batch"] >= 12
    # The old bucket-table fallback was replaced by hardware.py:derive_batch;
    # what ships now is the formula's inputs, not a list of rules.
    derived = data["derived"]
    assert derived["chunk_seconds_default"] > 0
    assert derived["chunk_seconds_low_core"] > 0
    assert derived.get("kv_bits") in (None, 4, 8)
    assert "fallback" not in data, (
        "the gpu_gb bucket table is gone; batch is derived instead "
        "(see hardware.py)"
    )


# --- output ---------------------------------------------------------------

def test_complete_detects_utf8_boundaries():
    assert _complete("あ".encode())
    assert not _complete("あ".encode()[:1])
    assert _complete(b"")


class FakeTokenizer:
    """Emits one byte per token so cue flushes land mid-character unless guarded."""

    n_special = 0
    special_ids = set()

    def __init__(self, text):
        self.data = text.encode("utf-8")

    def token_bytes(self, t):
        return self.data[t : t + 1]


class FakeModel:
    def __init__(self, text):
        self._tokenizer = FakeTokenizer(text)


def test_build_cues_never_splits_multibyte_characters():
    """Regression: cue text was assembled from raw token bytes and decoded at
    every flush, so a flush inside a multi-byte character produced U+FFFD."""
    text = "".join(chr(0x3042 + (i % 40)) for i in range(400))  # 3 bytes each
    model = FakeModel(text)
    timed = [[(i, (i + 1) * 0.08) for i in range(len(text.encode()))]]
    cues = build_cues(model, timed, [0.0])
    joined = "".join(c[2] for c in cues)
    assert "�" not in joined
    assert joined == text


def test_build_cues_times_are_ordered_and_nonnegative():
    text = "".join(chr(0x3042 + (i % 40)) for i in range(200))
    model = FakeModel(text)
    timed = [[(i, (i + 1) * 0.08) for i in range(len(text.encode()))]]
    cues = build_cues(model, timed, [0.0])
    assert cues
    for start, end, _ in cues:
        assert 0.0 <= start <= end
    assert all(b[0] >= a[0] for a, b in zip(cues, cues[1:]))


def test_build_cues_defaults_are_not_fitted_to_our_references():
    """Pin the shipped defaults, which are deliberately NOT the sweep optimum.

    Every timed reference in this project comes from one production team, so the
    sweep optimum (0.7, 32) encodes that team's subtitle convention rather than
    good segmentation in general. Shipping it would export their house style to
    every user, so the conservative pair stays and both sweeps are documented in
    RESULTS.md instead. A failure here means someone applied a sweep result; read
    the build_cues docstring before changing it.
    """
    import inspect

    sig = inspect.signature(build_cues)
    assert sig.parameters["gap_s"].default == 1.2
    assert sig.parameters["max_chars"].default == 28


def test_build_cues_gap_controls_boundary_count():
    """The sweep's mechanism: a smaller silence gap must produce more cues.

    This is the axis that carried the whole result (F1 44.5% at gap 0.6 down to
    35.9% at 1.2), so it is worth asserting the knob still does what the tuning
    assumed rather than trusting the aggregate number alone.
    """
    text = "".join(chr(0x3042 + (i % 40)) for i in range(120))
    model = FakeModel(text)
    nbytes = len(text.encode())
    # a token every 0.08s, with a 1.0s silence inserted every 30 bytes
    timed, t = [], 0.0
    row = []
    for i in range(nbytes):
        t += 1.0 if i and i % 30 == 0 else 0.08
        row.append((i, t))
    timed.append(row)
    tight = build_cues(model, timed, [0.0], gap_s=0.7, max_chars=10_000)
    loose = build_cues(model, timed, [0.0], gap_s=1.5, max_chars=10_000)
    assert len(tight) > len(loose), (len(tight), len(loose))
    # and neither may lose or reorder text
    assert "".join(c[2] for c in tight) == "".join(c[2] for c in loose) == text


def test_fmt_ts_formats_srt_and_vtt():
    assert fmt_ts(0) == "00:00:00,000"
    assert fmt_ts(3661.5) == "01:01:01,500"
    assert fmt_ts(3661.5, ".") == "01:01:01.500"


def test_wrap_cjk_splits_on_width_without_losing_text():
    s = "".join(chr(0x3042 + i) for i in range(35))
    wrapped = wrap_cjk(s, 16)
    assert wrapped.replace("\n", "") == s
    assert max(len(line) for line in wrapped.split("\n")) <= 16


# --- format detection on the chunked driver ----------------------------------
#
# `--model kotoba` names a transformers-format repo, which the MLX loader cannot
# read: it splats config.json into a dataclass and dies on `_name_or_path`. Such
# weights are now converted rather than rejected (convert.py), so what these
# tests guard is the format decision that routes to the converter.

def _write_config(tmp_path, cfg, name="model"):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    return d


MLX_CONFIG = {
    "n_mels": 128, "n_audio_ctx": 1500, "n_audio_state": 1280,
    "n_audio_head": 20, "n_audio_layer": 32, "n_vocab": 51866,
    "n_text_ctx": 448, "n_text_state": 1280, "n_text_head": 20,
    "n_text_layer": 2, "model_type": "whisper",
}
TRANSFORMERS_CONFIG = {
    "_name_or_path": "kotoba-tech/kotoba-whisper-v2.0",
    "architectures": ["WhisperForConditionalGeneration"],
    "d_model": 1280, "decoder_layers": 2, "encoder_layers": 32,
    "num_mel_bins": 128, "max_source_positions": 1500,
}


def test_mlx_and_transformers_configs_are_told_apart(tmp_path):
    assert _is_mlx_format(_write_config(tmp_path, MLX_CONFIG, "a") / "config.json")
    assert not _is_mlx_format(
        _write_config(tmp_path, TRANSFORMERS_CONFIG, "b") / "config.json")


def test_unreadable_config_defers_to_the_loader(tmp_path):
    """Returning True means "do not touch it", so a config that cannot be parsed
    falls through to mlx-whisper's own error instead of a wrong conversion."""
    assert _is_mlx_format(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert _is_mlx_format(bad)


def test_resolve_passes_through_an_mlx_directory(tmp_path):
    d = _write_config(tmp_path, MLX_CONFIG)
    assert resolve_mlx_weights(str(d)) == str(d)


def test_resolve_converts_a_transformers_directory(tmp_path, monkeypatch):
    """The conversion itself is exercised against real weights elsewhere; here
    what matters is that this format reaches the converter at all, since the
    previous behaviour was to stop with instructions."""
    d = _write_config(tmp_path, TRANSFORMERS_CONFIG)
    called = {}

    # `resolve_mlx_weights` imports the converter at call time (to keep the import
    # off the fast path), so the patch has to land on the module attribute the
    # `from .convert import cached_mlx_copy` line will read.
    import mlx_asr.convert as conv

    def fake(repo, log=print):
        called["repo"] = repo
        return "/converted"

    monkeypatch.setattr(conv, "cached_mlx_copy", fake, raising=True)
    assert resolve_mlx_weights(str(d), log=lambda *a: None) == "/converted"
    assert called["repo"] == str(d)


def test_load_audio_falls_back_when_a_decoder_is_missing(tmp_path, monkeypatch):
    """A Homebrew install has ffmpeg but no miniaudio, and mlx_audio imports its
    decoders lazily *inside* `read`. Guarding only the import line let that
    ImportError escape, so every transcription failed on a brew install with
    `ModuleNotFoundError: No module named 'miniaudio'`.
    """
    import mlx_asr.audio as audio_mod

    called = {}

    def fake_ffmpeg(path):
        called["path"] = path
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(audio_mod, "_load_via_ffmpeg", fake_ffmpeg)

    # a real file, since load_audio_16k now rejects a missing path before it gets
    # as far as choosing a decoder
    src = tmp_path / "some.wav"
    src.write_bytes(b"RIFF....WAVE")

    # a reader that imports its backend lazily, exactly as mlx_audio.audio_io does
    fake_io = types.ModuleType("mlx_audio.audio_io")

    def read(path, dtype=None):
        raise ImportError("No module named 'miniaudio'")

    fake_io.read = read
    fake_utils = types.ModuleType("mlx_audio.utils")
    fake_utils.resample_audio = lambda a, s, t: a
    pkg = types.ModuleType("mlx_audio")
    monkeypatch.setitem(sys.modules, "mlx_audio", pkg)
    monkeypatch.setitem(sys.modules, "mlx_audio.audio_io", fake_io)
    monkeypatch.setitem(sys.modules, "mlx_audio.utils", fake_utils)

    out = audio_mod.load_audio_16k(str(src))
    assert called["path"] == str(src)
    assert len(out) == 8


def test_install_hint_matches_how_the_package_was_installed(monkeypatch):
    """A brew user told to run `uv sync --extra whisper` is sent to a directory
    that does not exist, so the hint is derived from the install location."""
    import mlx_asr.backends as backends

    real = Path(backends.__file__).resolve()
    cellar = Path("/opt/homebrew/Cellar/mlx-asr/0.2.0/libexec/lib/"
                  "python3.13/site-packages/mlx_asr/backends.py")

    monkeypatch.setattr(backends, "__file__", str(cellar))
    assert "brew" in backends._install_hint("whisper")

    # a plain pip install: no pyproject next to the package
    monkeypatch.setattr(backends, "__file__",
                        "/tmp/venv/lib/python3.13/site-packages/mlx_asr/backends.py")
    assert backends._install_hint("whisper") == "pip install 'mlx-asr[whisper]'"

    # and a source checkout, which is where the tests themselves run
    monkeypatch.setattr(backends, "__file__", str(real))
    hint = backends._install_hint("vad")
    assert "vad" in hint


def test_audio_errors_are_one_liners_not_tracebacks(tmp_path):
    """A missing or non-audio input is user error. Both loaders report it badly on
    their own (miniaudio raises FileNotFoundError, ffmpeg exits nonzero with a
    multi-line complaint), so it is caught and re-raised as AudioError, which the
    console-script wrapper turns into a single line."""
    from mlx_asr.audio import AudioError, load_audio_16k

    with pytest.raises(AudioError, match="no such file"):
        load_audio_16k(str(tmp_path / "nope.wav"))

    junk = tmp_path / "notaudio.wav"
    junk.write_text("hello")
    with pytest.raises(AudioError):
        load_audio_16k(str(junk))


def test_cli_wrapper_converts_audio_errors_to_exit_1(tmp_path, capsys):
    """`cli()` exists so `main()` stays raise-y for tests and the benchmarks."""
    from mlx_asr.cli import cli

    rc = cli_with_argv(cli, [str(tmp_path / "missing.wav")])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error: no such file")
    assert "Traceback" not in err


def cli_with_argv(fn, argv):
    """Call a zero-argument console-script entry point with a fake argv."""
    old = sys.argv
    sys.argv = ["mlx-asr", *argv]
    try:
        return fn()
    finally:
        sys.argv = old


# --- the transformers -> MLX converter ---------------------------------------

def test_convert_maps_keys_and_config_without_torch(tmp_path):
    """The conversion is a key rename plus one axis swap, so it is checked on
    synthetic tensors rather than a 1.5GB download.

    The axis swap is the part that would silently produce garbage instead of
    failing: torch stores Conv1d as (out, in, kernel) and MLX as (out, kernel, in).
    """
    import mlx.core as mx

    from mlx_asr.convert import convert_to_mlx

    src = tmp_path / "hf"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({
        "num_mel_bins": 128, "max_source_positions": 1500, "d_model": 8,
        "encoder_attention_heads": 2, "encoder_layers": 1, "vocab_size": 99,
        "max_target_positions": 448, "decoder_attention_heads": 2,
        "decoder_layers": 2, "_name_or_path": "ignored",
    }))
    mx.save_safetensors(str(src / "model.safetensors"), {
        "model.encoder.conv1.weight": mx.zeros((8, 128, 3)),
        "model.encoder.layers.0.self_attn.q_proj.weight": mx.ones((8, 8)),
        "model.encoder.layers.0.self_attn_layer_norm.weight": mx.ones((8,)),
        "model.decoder.embed_tokens.weight": mx.ones((99, 8)),
        "proj_out.weight": mx.ones((99, 8)),      # tied, must be dropped
    })

    out = convert_to_mlx(str(src), tmp_path / "mlx", log=lambda *a: None)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg == {
        "n_mels": 128, "n_audio_ctx": 1500, "n_audio_state": 8,
        "n_audio_head": 2, "n_audio_layer": 1, "n_vocab": 99,
        "n_text_ctx": 448, "n_text_state": 8, "n_text_head": 2,
        "n_text_layer": 2, "model_type": "whisper",
    }

    w = mx.load(str(out / "weights.safetensors"))
    assert "proj_out.weight" not in w
    # (out, in, kernel) -> (out, kernel, in)
    assert w["encoder.conv1.weight"].shape == (8, 3, 128)
    assert "encoder.blocks.0.attn.query.weight" in w
    assert "encoder.blocks.0.attn_ln.weight" in w
    assert "decoder.token_embedding.weight" in w


def test_convert_rejects_a_config_that_is_not_a_whisper_checkpoint(tmp_path):
    """Better than emitting a config with missing dimensions that fails later
    inside the loader."""
    from mlx_asr.convert import convert_to_mlx

    src = tmp_path / "hf"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError, match="transformers Whisper checkpoint"):
        convert_to_mlx(str(src), tmp_path / "out", log=lambda *a: None)


# --- input format handling ----------------------------------------------------

def _sine_file(path, *, codec, rate, channels, sample_fmt=None, layout=None):
    """Write a short tone in a given container/codec, or skip if unavailable."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg needed to synthesize test inputs")
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi",
           "-i", f"sine=f=440:d=1:r={rate}", "-c:a", codec,
           "-ar", str(rate), "-ac", str(channels)]
    if sample_fmt:
        cmd += ["-sample_fmt", sample_fmt]
    cmd += [str(path), "-y"]
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        pytest.skip(f"this ffmpeg cannot encode {codec}")
    return path


@pytest.mark.parametrize("name,codec,rate,channels,sample_fmt", [
    ("a.wav", "pcm_s16le", 16000, 1, None),      # the easy case
    ("b.wav", "pcm_s24le", 8000, 1, None),       # below target rate, 24-bit
    ("c.wav", "pcm_f32le", 96000, 2, None),      # above target rate, stereo
    ("d.flac", "flac", 48000, 1, "s32"),         # lossless container
    ("e.m4a", "aac", 44100, 2, None),            # lossy, stereo, mp4 container
    ("f.mp3", "libmp3lame", 22050, 1, None),
])
def test_any_container_rate_depth_and_channel_count_decodes(
        tmp_path, name, codec, rate, channels, sample_fmt):
    """Input format is not constrained; 16kHz mono float32 is the output contract.

    The duration has to survive resampling in both directions (8kHz up, 96kHz
    down), which is where an off-by-a-buffer in the resampler flush would show up
    as a truncated tail.
    """
    from mlx_asr.audio import SAMPLE_RATE, load_audio_16k

    f = _sine_file(tmp_path / name, codec=codec, rate=rate, channels=channels,
                   sample_fmt=sample_fmt)
    audio = load_audio_16k(str(f))
    assert audio.dtype == np.float32
    assert audio.ndim == 1                      # mono, whatever went in
    assert abs(len(audio) / SAMPLE_RATE - 1.0) < 0.05, len(audio)
    assert np.isfinite(audio).all()


def test_decoding_needs_no_subprocess(tmp_path, monkeypatch):
    """PyAV decodes in-process. The ffmpeg CLI is a last-resort fallback, so on a
    normal install nothing should be spawned: no argv quoting, no pipe buffering,
    no parsing another program's stderr to find out what went wrong."""
    import subprocess

    from mlx_asr.audio import load_audio_16k

    f = _sine_file(tmp_path / "s.wav", codec="pcm_s16le", rate=44100, channels=2)

    def explode(*a, **k):
        raise AssertionError("load_audio_16k spawned a subprocess")

    monkeypatch.setattr(subprocess, "run", explode)
    assert len(load_audio_16k(str(f))) > 0


def test_video_without_audio_is_reported_clearly(tmp_path):
    import shutil
    import subprocess

    from mlx_asr.audio import AudioError, load_audio_16k

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg needed to synthesize test inputs")
    out = tmp_path / "silent.mp4"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=32x32:d=1",
         "-c:v", "libx264", "-preset", "ultrafast", str(out), "-y"],
        capture_output=True)
    if r.returncode != 0:
        pytest.skip("this ffmpeg cannot encode h264")
    with pytest.raises(AudioError, match="no audio stream"):
        load_audio_16k(str(out))
