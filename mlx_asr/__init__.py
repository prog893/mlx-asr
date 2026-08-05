"""Voxtral Realtime batch transcription for Apple Silicon.

Voxtral Realtime is a *streaming* ASR model: it emits exactly one token per
80ms audio frame. A single stream is therefore pinned near 1x realtime no
matter how fast the hardware is, because the token count is set by the audio
duration, not by the compute available.

This package turns that constraint into a throughput problem: split the audio
at silence points, encode each piece, then decode all pieces in lockstep as
one batch. Decode reads the whole model per step regardless of batch size, so
B rows cost far less than B times one row.
"""

# Read from package metadata rather than hardcoded, because a second copy of the
# version drifts: this said 0.2.0 while pyproject.toml said 0.4.0.
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("mlx-asr")
except PackageNotFoundError:        # running from a source tree, not installed
    __version__ = "unknown"
