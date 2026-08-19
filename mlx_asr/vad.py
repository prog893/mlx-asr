"""Silero VAD chunk boundaries (optional; needs `pip install mlx-asr[vad]`).

Why bother, given chunking already cuts at energy minima: RMS energy cannot tell
quiet speech from loud noise, so on noisy material the "quietest window" can land
mid-word. The VAD literature is consistent that a neural detector beats an energy
baseline (e.g. arXiv 2601.17270), and WhisperX places its cut points in
"minimally active speech regions" found by VAD rather than by energy.

The counter-evidence is worth stating: arXiv 2402.09797 reports VAD front-ends
can *increase* deletions by clipping speech. This module therefore never removes
audio. It only chooses *where* to cut, so the concatenated chunks still cover the
input exactly. That keeps the failure mode to "a boundary in a slightly worse
place" rather than "words dropped".

Uses the ONNX build of Silero VAD to avoid a torch dependency. The model is ~2MB
and downloaded once from the silero-vad repo.
"""

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, split_at_silences

# Pinned to a TAG, not to master, and verified by hash. Three refs of this file
# (master, v5.1.2, v6.0) all serve exactly 2327524 bytes with three different sha256s,
# so the weights change while the size does not. The old cache check was
# `st_size < 100_000`, which cannot tell them apart: a user who cached in July and one
# who caches today would silently run different VAD weights and get different chunk
# boundaries. In a project whose central claim is byte-identical reproducibility on a
# given machine, that is the wrong failure to leave open.
SILERO_TAG = "v5.1.2"
MODEL_URL = (
    f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_TAG}/"
    "src/silero_vad/data/silero_vad.onnx"
)
# sha256 of the file at SILERO_TAG, computed from the downloaded bytes on 2026-08-19.
# For the record, the three refs at that date, all 2327524 bytes:
#   v5.1.2  2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f  (this one)
#   v6.0    597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004
#   master  1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
# v5.1.2 is chosen because it is what the cache already held, so pinning it changes no
# existing user's cut points. Moving to v6.0 is a separate, measurable decision.
MODEL_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
WINDOW = 512   # samples of new audio per call at 16kHz
CONTEXT = 64   # samples of the PREVIOUS window that must be prepended


def _cache_path() -> Path:
    base = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ) / "mlx-asr"
    base.mkdir(parents=True, exist_ok=True)
    return base / "silero_vad.onnx"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ensure_model(path: Path) -> Path:
    """Download the pinned VAD weights, verifying the hash rather than the size.

    A size check cannot detect a different revision of this file, because every
    revision is the same length. Re-downloads once if the cached copy does not match,
    which repairs a cache populated by an older version of this code, and refuses
    rather than silently proceeding if the fresh download is also wrong: at that point
    something is serving unexpected bytes and guessing is worse than stopping.
    """
    if path.exists() and _sha256(path) == MODEL_SHA256:
        return path
    urllib.request.urlretrieve(MODEL_URL, path)
    got = _sha256(path)
    if got != MODEL_SHA256:
        raise RuntimeError(
            f"silero-vad weights from {MODEL_URL} hashed {got}, expected "
            f"{MODEL_SHA256}. Refusing to use them: VAD cut points would not be "
            f"reproducible. Delete {path} and retry, or report this."
        )
    return path


def speech_probs(audio: np.ndarray) -> np.ndarray:
    """Per-512-sample speech probability. Requires onnxruntime."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    sess = ort.InferenceSession(
        str(_ensure_model(_cache_path())), sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    n = len(audio) // WINDOW
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = audio[: n * WINDOW].reshape(n, WINDOW).astype(np.float32)

    # Silero v5 keeps a [2, batch, 128] recurrent state across calls AND expects
    # each 512-sample window to be prefixed with the last 64 samples of the
    # previous one. Omitting that context is not an error, it just makes the
    # model output ~0.001 for everything, which reads as "no speech anywhere".
    state = np.zeros((2, 1, 128), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    out = np.zeros(n, dtype=np.float32)
    prev = np.zeros(CONTEXT, dtype=np.float32)
    for i in range(n):
        w = frames[i]
        x = np.concatenate([prev, w])[None, :]
        p, state = sess.run(None, {"input": x, "state": state, "sr": sr})
        out[i] = float(p[0][0])
        prev = w[-CONTEXT:]
    return out


def split_at_vad(
    audio: np.ndarray,
    target_s: float = 60.0,
    search_s: float = 5.0,
    threshold: float = 0.5,
) -> list:
    """Split into ~target_s chunks, cutting at the middle of the longest
    non-speech run within +/- search_s of each target boundary.

    Falls back to the energy splitter for any boundary where the VAD finds no
    non-speech at all, so continuous speech still gets cut somewhere sensible
    rather than not at all.
    """
    probs = speech_probs(audio)
    if len(probs) == 0:
        return split_at_silences(audio, target_s, search_s)
    is_speech = probs >= threshold

    n = len(audio)
    target = int(target_s * SAMPLE_RATE)
    search = int(search_s * SAMPLE_RATE)

    cuts = []
    pos = target
    while n - pos > target // 2:
        lo = max(pos - search, (cuts[-1] if cuts else 0) + WINDOW)
        hi = min(pos + search, n - WINDOW)
        if hi <= lo:
            break
        f_lo, f_hi = lo // WINDOW, min(hi // WINDOW, len(is_speech))
        seg = is_speech[f_lo:f_hi]
        cut = None
        if len(seg) and not seg.all():
            # Longest run of non-speech in the search region.
            best_len = best_start = 0
            i = 0
            while i < len(seg):
                if not seg[i]:
                    j = i
                    while j < len(seg) and not seg[j]:
                        j += 1
                    if j - i > best_len:
                        best_len, best_start = j - i, i
                    i = j
                else:
                    i += 1
            if best_len:
                mid = best_start + best_len // 2
                cut = (f_lo + mid) * WINDOW
        if cut is None:
            # No silence found: fall back to the quietest window here.
            win = int(0.05 * SAMPLE_RATE)
            hi2 = max(hi - win, lo + 1)
            e = np.convolve(audio[lo:hi2] ** 2, np.ones(win) / win, mode="valid")
            cut = lo + int(np.argmin(e)) + win // 2 if len(e) else (lo + hi) // 2
        cut = int(min(max(cut, lo + 1), hi - 1))
        cuts.append(cut)
        pos = cut + target

    bounds = [0] + cuts + [n]
    return [audio[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def split_at_vad_with_overlap(
    audio: np.ndarray,
    target_s: float = 60.0,
    search_s: float = 5.0,
    overlap_s: float = 0.0,
    threshold: float = 0.5,
):
    """VAD boundaries plus the prefix-overlap warm-up of split_with_overlap."""
    from .audio import TOKEN_SAMPLES

    base = split_at_vad(audio, target_s, search_s, threshold)
    starts = np.cumsum([0] + [len(c) for c in base[:-1]]).astype(int)
    if overlap_s <= 0:
        return base, [s / SAMPLE_RATE for s in starts], [0] * len(base)

    ov = int(round(overlap_s * SAMPLE_RATE / TOKEN_SAMPLES)) * TOKEN_SAMPLES
    chunks, offsets, warmup = [], [], []
    for c, s in zip(base, starts):
        take = min(ov, int(s))
        chunks.append(audio[s - take : s + len(c)])
        offsets.append(s / SAMPLE_RATE)
        warmup.append(take // TOKEN_SAMPLES)
    return chunks, offsets, warmup
