"""Audio loading and silence-aware chunking."""

import sys
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class AudioError(Exception):
    """Input audio could not be read.

    Exists so the CLI can report a bad path as one line instead of a traceback.
    A file that does not exist, or that is not audio, is ordinary user error, not
    a bug worth a stack trace.
    """


def load_audio_16k(path: str) -> np.ndarray:
    """Load any FFmpeg-readable input as 16kHz mono float32.

    Input format is not constrained: container, codec, sample rate, bit depth and
    channel count are whatever the file has, and FFmpeg's demuxers and resampler
    handle the conversion. Verified across aac/m4a, mp3, flac, opus, alac, mulaw,
    float and 24-bit WAV, 8kHz to 96kHz, mono and stereo, and an mp4 video: every
    one decodes to the same duration at 16kHz mono. 16kHz mono float32 is the
    *output* contract, because that is what the mel front end wants.

    Three loaders, in order, all in-process except the last:

    1. **PyAV** (`av`), which is FFmpeg's libav* libraries as a Python extension.
       No subprocess, no pipe, no argv quoting, and errors arrive as exceptions
       rather than as a parsed stderr tail. Bit-identical to the ffmpeg CLI on
       every format tested except opus, which differs by 3e-06 peak.
    2. **mlx-audio**, already present for the Voxtral path. Kept ahead of the
       subprocess because it is in-process for the formats miniaudio handles,
       though note it shells out to ffmpeg itself for aac/m4a.
    3. **The ffmpeg CLI**, as the last resort, so a machine with neither binding
       installed still works if ffmpeg is on PATH.

    Nothing is ever transcoded to a temporary file: each path decodes straight
    into an array, so the only lossy step is the one the source format already
    implies.
    """
    # Checked up front because each loader reports a bad path differently
    # (miniaudio raises FileNotFoundError, ffmpeg exits nonzero with a multi-line
    # complaint) and none of those is a message worth showing.
    if not Path(path).exists():
        raise AudioError(f"no such file: {path}")

    try:
        return _load_via_av(path)
    except ImportError:
        pass
    except AudioError:
        raise
    except Exception:
        # A container PyAV cannot open may still be readable by another loader,
        # so this falls through rather than failing the run.
        pass

    try:
        from mlx_audio.audio_io import read as audio_read
        from mlx_audio.utils import resample_audio

        audio_np, sr = audio_read(path, dtype="float32")
    except ImportError:
        return _load_via_ffmpeg(path)
    except Exception:
        # mlx-audio's decoders are narrower than ffmpeg's, so anything it cannot
        # read is retried rather than treated as fatal. ffmpeg then either
        # succeeds or raises AudioError with its own reason.
        return _load_via_ffmpeg(path)

    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=-1)
    audio_np = audio_np.flatten()
    if sr != SAMPLE_RATE:
        audio_np = resample_audio(audio_np, sr, SAMPLE_RATE)
    return audio_np.astype(np.float32)


def _load_via_av(path: str) -> np.ndarray:
    """Decode with libav in-process via PyAV.

    Resampling, downmixing and format conversion are handed to libswresample,
    which is the same code the ffmpeg CLI uses, so this is not a reimplementation
    of anything. The resampler must be flushed with `resample(None)`, or its
    final buffered frames are dropped and the tail of the audio goes missing.
    """
    import av
    from av.audio.resampler import AudioResampler

    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise AudioError(f"{path} has no audio stream (is it video-only?)")
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        resampler = AudioResampler(format="fltp", layout="mono",
                                   rate=SAMPLE_RATE)
        chunks = [out.to_ndarray().reshape(-1)
                  for frame in container.decode(stream)
                  for out in resampler.resample(frame)]
        chunks += [out.to_ndarray().reshape(-1)
                   for out in resampler.resample(None)]

    if not chunks:
        raise AudioError(f"{path} decoded to no audio")
    return np.concatenate(chunks).astype(np.float32)


def _load_via_ffmpeg(path: str) -> np.ndarray:
    """Decode to 16kHz mono float32 with ffmpeg, reading PCM off its stdout.

    Two deliberate choices here, both about not degrading the input:

    * **No temporary file.** ffmpeg writes raw PCM to stdout and it is read
      straight into an array, so nothing is transcoded to disk and re-read. The
      decode is the only lossy step, and it is the one ffmpeg would do anyway.
    * **f32le, not s16le.** Asking for 16-bit samples would re-quantize anything
      with more resolution than that (24-bit field recordings, float WAVs) at
      about -92 dBFS of added noise, measured on a 24-bit/48kHz file. Small, but
      the whole point of `--gain auto` is that quiet detail near the mel floor
      matters, so throwing bits away right before that check is the wrong default
      when float output is free.
    """
    import shutil
    import subprocess

    # Checked before running ffmpeg so a typo'd filename says so, rather than
    # surfacing as ffmpeg's own multi-line complaint.
    if not Path(path).exists():
        raise AudioError(f"no such file: {path}")
    if shutil.which("ffmpeg") is None:
        raise AudioError(
            f"cannot read {path}: ffmpeg is not installed and is needed for "
            f"anything other than a 16kHz mono WAV (brew install ffmpeg)")

    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip().splitlines()
        raise AudioError(
            f"ffmpeg could not read {path}: "
            f"{detail[-1] if detail else f'exit {proc.returncode}'}")
    if not proc.stdout:
        raise AudioError(f"{path} decoded to no audio (is it a video with no "
                         f"audio track?)")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def split_at_silences(
    audio: np.ndarray, target_s: float = 60.0, search_s: float = 5.0
) -> list:
    """Split audio into ~target_s chunks, cutting at the quietest 50ms window
    within +/- search_s of each target boundary.

    Cutting mid-word costs accuracy at both sides of the seam, so boundaries
    are placed at local energy minima. Returns a list of float32 arrays whose
    lengths sum to len(audio) exactly (no samples dropped or duplicated).
    """
    n = len(audio)
    target = int(target_s * SAMPLE_RATE)
    search = int(search_s * SAMPLE_RATE)
    win = int(0.05 * SAMPLE_RATE)

    cuts = []
    pos = target
    while n - pos > target // 2:
        lo = max(pos - search, (cuts[-1] if cuts else 0) + win)
        hi = min(pos + search, n - win)
        if hi <= lo:
            break
        seg = audio[lo:hi] ** 2
        energy = np.convolve(seg, np.ones(win) / win, mode="valid")
        cut = lo + int(np.argmin(energy)) + win // 2
        cuts.append(cut)
        pos = cut + target
    bounds = [0] + cuts + [n]
    return [audio[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


TOKEN_SAMPLES = 1280  # 80ms at 16kHz, one decoder position
TOKEN_S = TOKEN_SAMPLES / SAMPLE_RATE  # 0.08s per decoded token


def split_with_overlap(
    audio: np.ndarray,
    target_s: float = 60.0,
    search_s: float = 5.0,
    overlap_s: float = 3.0,
):
    """Split like ``split_at_silences`` but prepend ``overlap_s`` of the
    preceding audio to every chunk after the first.

    Both the encoder and the decoder are causal, so a chunk's first tokens are
    emitted with no left context and are measurably worse: on the reference clip,
    edit operations are 2.2x enriched in the first 3s of a chunk versus 1.3x in
    the last 3s. Trailing overlap therefore cannot help (future audio cannot
    change an already-emitted token) but leading overlap can, by letting the
    model warm up on audio whose transcript is then thrown away.

    Returns (chunks, offsets_s, warmup_tokens) where offsets_s[i] is the audio
    time of each chunk's *kept* region and warmup_tokens[i] is how many decoder
    positions to discard from the front of chunk i. The overlap is rounded to a
    whole number of 80ms frames so the discard boundary lands exactly on a token.
    """
    base = split_at_silences(audio, target_s=target_s, search_s=search_s)
    if overlap_s <= 0:
        offs = np.cumsum([0] + [len(c) for c in base[:-1]]) / SAMPLE_RATE
        return base, list(offs), [0] * len(base)

    starts = np.cumsum([0] + [len(c) for c in base[:-1]]).astype(int)
    ov = int(round(overlap_s * SAMPLE_RATE / TOKEN_SAMPLES)) * TOKEN_SAMPLES

    chunks, offsets, warmup = [], [], []
    for i, (c, s) in enumerate(zip(base, starts)):
        take = min(ov, s)  # chunk 0 has nothing before it
        chunks.append(audio[s - take : s + len(c)])
        offsets.append(s / SAMPLE_RATE)
        warmup.append(take // TOKEN_SAMPLES)
    return chunks, offsets, warmup


AUTO_TRIGGER_DBFS = -6.0   # above this the file is left byte-identical
AUTO_TARGET_DBFS = -1.0    # boost target, kept below 0 so nothing clips


def dbfs(audio: np.ndarray) -> float:
    """Peak level in dBFS."""
    return 20.0 * float(np.log10(max(float(np.abs(audio).max()), 1e-12)))


def gain_scope_note() -> str:
    """One-line description of where gain is applied, for logs and docs."""
    return ("file-wide: one scalar for the whole recording, applied before "
            "chunking, so relative dynamics are preserved exactly")


def auto_gain_db(audio: np.ndarray, trigger_dbfs: float = AUTO_TRIGGER_DBFS,
                 target_dbfs: float = AUTO_TARGET_DBFS) -> float:
    """dB of gain "auto" mode would apply. 0.0 means leave the audio alone.

    Derived from the measured behaviour rather than from a fixed loudness target:

    - Attenuated input is significantly worse (-12dB costs ~3.8 CER points,
      -20dB ~7.8), because the mel front end clamps at an absolute floor and
      quiet detail lands under it. So quiet audio must be brought up.
    - Amplifying audio that is *already* healthy is at best neutral and split by
      content, and every mode that improved anything did so while clipping. So
      loud audio is left exactly as it is.
    - Clipping is the one irreversible loss, so the target sits below full scale
      and the gain is never negative.

    The result is a no-op for well-recorded material and a rescue for quiet
    material, which is why it is safe as a default.

    Scope: this is a single scalar for the WHOLE file, decided from the file peak
    and applied before chunking. It deliberately does not adapt per chunk. Most
    chunks of a normal recording sit well below the file peak (94% of chunks are
    under -6 dBFS on the reference corpus, with up to 70dB of spread inside one
    file), so a per-chunk normalizer would be applying tens of dB of differential
    gain and flattening the loud/quiet structure the model uses. See
    ``per_chunk_gain_db`` and the measurements in docs/benchmarks/input-level.md.
    """
    peak = dbfs(audio)
    if peak >= trigger_dbfs:
        return 0.0
    return target_dbfs - peak


def per_chunk_gain_db(chunk: np.ndarray, target_dbfs: float = AUTO_TARGET_DBFS,
                      max_boost_db: float = 20.0,
                      floor_dbfs: float = -50.0) -> float:
    """Gain that would bring one chunk's peak to ``target_dbfs``.

    Provided so per-chunk normalization can be evaluated, not because it is
    recommended: it destroys inter-chunk level relationships and amplifies chunks
    that are quiet because they contain no speech. Boost is capped and chunks
    below ``floor_dbfs`` are left alone so pure silence is not amplified into
    noise.
    """
    peak = dbfs(chunk)
    if peak <= floor_dbfs or peak >= target_dbfs:
        return 0.0
    return min(target_dbfs - peak, max_boost_db)


def apply_gain(audio: np.ndarray, mode: str = "0", peak_dbfs: float = -1.0,
               rms_dbfs: float = -23.0) -> np.ndarray:
    """Scale the waveform. ``mode`` is "auto", a number of dB, "peak", or "rms".

    Level is not cosmetic for this model: the mel front end clamps at an
    *absolute* floor (log10 power below ``global_log_mel_max - 8``), not at one
    relative to the utterance, so quiet input has low-level detail destroyed
    before the encoder sees it. Amplifying lifts detail off that floor, at the
    cost of raising the noise floor with it.

    "rms" targets the RMS of speech-active frames rather than the whole file, so
    a recording with long pauses is not pushed up by its silence, and it is not
    steered by isolated transients the way peak normalization is. Gain is applied
    in float32 and hard-clipped at full scale; clipping destroys information, so
    the peak target defaults just under 0 dBFS.
    """
    if mode in ("", "none", None):
        return audio
    if mode == "auto":
        g_db = auto_gain_db(audio)
        if g_db == 0.0:
            return audio
        return np.clip(audio.astype(np.float32) * (10 ** (g_db / 20)), -1.0, 1.0)
    if mode == "peak":
        peak = float(np.abs(audio).max())
        if peak <= 0:
            return audio
        g = (10 ** (peak_dbfs / 20)) / peak
    elif mode == "rms":
        db = frame_db(audio)
        if len(db) == 0:
            return audio
        # Speech-active frames only: everything above the 60th percentile of
        # frame energy, which excludes pauses without needing a VAD.
        thresh = np.percentile(db, 60)
        frames = audio[: (len(db) * TOKEN_SAMPLES)].reshape(len(db), TOKEN_SAMPLES)
        active = frames[db > thresh]
        rms = float(np.sqrt((active.astype(np.float64) ** 2).mean())) if len(active) else 0.0
        if rms <= 0:
            return audio
        g = (10 ** (rms_dbfs / 20)) / rms
    else:
        g = 10 ** (float(mode) / 20)
    return np.clip(audio.astype(np.float32) * g, -1.0, 1.0)


def frame_db(audio: np.ndarray, frame: int = 1280) -> np.ndarray:
    """Per-frame RMS in dB at the model's 80ms token granularity."""
    n = len(audio) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-10))


def compact_silence(
    audio: np.ndarray,
    thresh_below_peak_db: float = -40.0,
    min_run_frames: int = 5,
    keep_frames: int = 3,
    frame: int = 1280,
):
    """Shorten long silent runs, returning (new_audio, time_map).

    Decode cost is one step per 80ms frame of audio, so removing silence
    removes decode steps one-for-one. Some silence must stay: it is what the
    model uses to place sentence boundaries, and the cue timing logic reads
    gaps in the token stream.

    time_map is a list of (new_start_s, old_start_s) breakpoints for mapping
    output timestamps back onto the original timeline.
    """
    db = frame_db(audio, frame)
    n = len(db)
    if n == 0:
        return audio, [(0.0, 0.0)]
    peak = float(np.percentile(db, 95))
    quiet = db < (peak + thresh_below_peak_db)

    keep_mask = np.ones(n, dtype=bool)
    i = 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if j - i >= min_run_frames:
                # Keep the head of the pause; drop the middle.
                keep_mask[i + keep_frames : j] = False
            i = j
        else:
            i += 1

    frame_s = frame / SAMPLE_RATE
    time_map = []
    new_i = 0
    prev_kept = False
    for old_i in range(n):
        if keep_mask[old_i]:
            if not prev_kept:
                time_map.append((new_i * frame_s, old_i * frame_s))
            new_i += 1
            prev_kept = True
        else:
            prev_kept = False
    if not time_map:
        time_map = [(0.0, 0.0)]

    idx = np.repeat(keep_mask, frame)
    kept = audio[: n * frame][idx]
    tail = audio[n * frame :]
    return np.concatenate([kept, tail]), time_map


def map_time(t_s: float, time_map) -> float:
    """Map a timestamp on the compacted timeline back to the original."""
    lo, hi = 0, len(time_map) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if time_map[mid][0] <= t_s:
            lo = mid
        else:
            hi = mid - 1
    new_start, old_start = time_map[lo]
    return old_start + (t_s - new_start)
