# Audio input

Anything libavformat can demux, which is nearly everything: WAV, FLAC, MP3, AAC/M4A,
ALAC, Opus, and the audio track of a video container. Sample rate, bit depth and channel
count do not matter either.

## What happens to your audio

Every model here wants 16kHz mono float32, so three conversions run before the audio
reaches one:

1. **Downmix to mono**, by averaging channels.
2. **Resample to 16kHz.** Most source material is 44.1 or 48kHz, so this is the step that
   usually does something. 96kHz files get resampled too; nothing in the mel front end
   can use the extra bandwidth.
3. **Convert to float32** in the planar `fltp` layout.

libswresample does all three, which is the same code the `ffmpeg` binary uses. On the
PyAV path it runs in-process.

Two things this does *not* do. It never writes a temporary file, so the source decode is
the only lossy step, and it never resamples upward: a 8kHz recording stays at its own
bandwidth rather than being interpolated to look like 16kHz.

Gain is separate and Voxtral-only. `--gain auto` boosts audio quieter than -6 dBFS peak
and leaves anything louder byte-identical, because the mel front end clamps at an absolute
floor and quiet input silently loses detail. See
[benchmarks/input-level.md](benchmarks/input-level.md).

## Verified formats

Each of these decoded to the correct duration:

| input | rates tested | depth and channels |
|---|---|---|
| WAV (`pcm_s16le`, `s24le`, `f32le`, `mulaw`) | 8k, 16k, 96k | 16/24/32-bit, mono and stereo |
| FLAC | 48k | 32-bit |
| AAC/M4A, ALAC | 16k, 44.1k | mono and stereo |
| MP3 | 22.05k | mono |
| Opus | 48k | mono |
| MP4 video | 16k | mono, audio stream extracted |

## Decoders, in order

1. **PyAV**, which binds FFmpeg's libavformat, libavcodec and libswresample as a Python
   extension. No subprocess, so no argv quoting, no pipe buffering, and errors arrive as
   exceptions rather than a parsed stderr tail. Output is bit-identical to the `ffmpeg`
   CLI on every format tested except Opus, which differs by 3e-06 peak.
2. **mlx-audio**, already present for the Voxtral path. In-process for the formats
   miniaudio handles, though it shells out to ffmpeg itself for AAC/M4A.
3. **The `ffmpeg` binary**, so a machine with neither binding still works.

One PyAV detail worth recording because getting it wrong loses audio silently: the
resampler holds buffered frames, and skipping the final `resample(None)` flush drops the
tail of the file.

A video with no audio track, a missing path, or a file that is not audio each produce one
line and exit 1 rather than a traceback.
