# Models and command schemas

Every alias, every option it accepts, and whether the minimal command actually
works. Each "works" cell was produced by running the command through the
Homebrew-installed binary on an M4 16GB, not by reading the code.

Everything here runs on MLX. Nothing installs or imports torch, and no path uses
torch's MPS backend. Verified on the installed tree: `torch` is absent, and
`transformers` is present only as a pure-Python tokenizer/config library, which
prints `PyTorch was not found` on import and is never asked for a model.

## Input audio

Nothing is constrained. Container, codec, sample rate, bit depth and channel count
are whatever the file has, and FFmpeg's demuxers and `libswresample` do the
conversion to the 16kHz mono float32 the mel front end needs.

Verified end to end, each decoding to the correct duration:

| input | rate | depth / channels |
|---|---|---|
| WAV (pcm_s16le, s24le, f32le, mulaw) | 8k, 16k, 96k | 16/24/32-bit, mono + stereo |
| FLAC | 48k | 32-bit |
| AAC / M4 16GBA, ALAC | 16k, 44.1k | mono + stereo |
| MP3 | 22.05k | mono |
| Opus | 48k | mono |
| MP4 video | 16k | mono (audio stream extracted) |

Audio is decoded straight into memory. Nothing is transcoded to disk and read back,
so the only lossy step is whatever the source format already implies.

Decoding goes through PyAV, which is FFmpeg's libav* libraries as a Python
extension, so no subprocess is involved: no argv quoting, no pipe buffering, and
failures arrive as exceptions instead of a parsed stderr tail. Output is
bit-identical to the `ffmpeg` CLI on every format tested except Opus, which differs
by 3e-06 peak. The CLI stays as a fallback if PyAV is missing.

A video file with no audio track, a path that does not exist, or a file that is
not audio each produce one line and exit 1, not a traceback.

## The models

180s Japanese narration clip, plain CER against a complete verbatim reference
(the one file in this project where plain CER is meaningful). `--language ja` for
every engine that takes it. x-realtime includes model load, so it understates the
larger models; treat it as "what you wait for", not a throughput benchmark.

| alias | weights | runtime | long-form algorithm | works | CER | x-rt |
|---|---|---|---|---|---|---|
| `voxtral` **(default)** | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | MLX | batched chunks, our decoder | yes | **11.76%** | 1.7x |
| `whisper-turbo` | `mlx-community/whisper-large-v3-turbo` | MLX | sequential 30s | yes | 13.67% | 15.1x |
| `whisper-large-v3` | `mlx-community/whisper-large-v3-mlx` | MLX | sequential 30s | yes | 14.41% | 6.1x |
| `whisper-large-v2` | `mlx-community/whisper-large-v2-mlx` | MLX | sequential 30s | yes | 13.45% | 4.0x |
| `whisper-medium` | `mlx-community/whisper-medium-mlx` | MLX | sequential 30s | yes | 16.42% | 10.8x |
| `whisper-small` | `mlx-community/whisper-small-mlx` | MLX | sequential 30s | yes | 24.47% | 21.7x |
| `whisper-base` | `mlx-community/whisper-base-mlx` | MLX | sequential 30s | yes | 35.59% | 20.9x |
| `whisper-tiny` | `mlx-community/whisper-tiny-mlx` | MLX | sequential 30s | yes | 32.10% | 51.4x |
| `kotoba` | `kotoba-tech/kotoba-whisper-v2.0` | MLX (converted on first use) | chunked 10s windows (our driver) | yes | 22.67% | 9.8x |

Reproduce with `scripts/benchmarks/matrix_models.py`. How the metrics work and why the
sample sizes constrain what you can read off them:
[benchmarks/README.md](benchmarks/README.md).

Do not pick a model from that table. It is one clip in one language, and the
20-file corpus in [docs/benchmarks/engines.md](benchmarks/engines.md) puts `whisper-turbo` ahead of
`voxtral` on accuracy, the reverse of what you see here. That is why the README
quotes the corpus number instead. `whisper-tiny` scoring better than
`whisper-base` is the same problem showing through: at n=1 that is noise, not a
fact about model size.

Voxtral's x-realtime looks bad because a 16GB M4 16GB is the floor case for it. Its
encoder is compute-bound, and the same code reaches 22.8x on an M2 Ultra 128GB (see
docs/benchmarks/decode-throughput.md).

The `kotoba` alias is v2.0 and gains nothing from v2.2. Both hold the same ASR
weights, identical in value across all 539 tensors (max absolute difference exactly
0.0; the files differ only in container metadata and stored dtype), and v2.2's own
model card describes it as v2.0 plus speaker diarization and punctuation. Those are
separate post-processing models (`diarizers`, `punctuators`), not retrained weights,
and both want torch.

Its weights are converted to MLX the first time you use them, since the authors
publish transformers format only. That takes a few seconds, needs no torch, and
is cached under `~/.cache/huggingface/hub/mlx-asr-converted/`. See
[Converting your own weights](#converting-your-own-weights).

## Minimal command, per model

Every one of these is the complete command. No other flag is required.

```bash
mlx-asr audio.wav                                    # voxtral, writes audio.srt
mlx-asr audio.wav --model whisper-turbo --language ja
mlx-asr audio.wav --model whisper-large-v3 --language ja
mlx-asr audio.wav --model whisper-large-v2 --language ja
mlx-asr audio.wav --model whisper-medium --language ja
mlx-asr audio.wav --model whisper-small --language ja
mlx-asr audio.wav --model whisper-base --language ja
mlx-asr audio.wav --model whisper-tiny --language ja
mlx-asr audio.wav --model kotoba                     # forces ja on its own
```

`--language` is not syntactically required on the `whisper-*` aliases, and they
run without it. It is listed above because omitting it means Whisper guesses from
the first 30 seconds, and on this project's material that guess returned Russian
for Japanese audio, costing 25 CER points. `kotoba` needs no hint because it is
Japanese-only and forces `ja` itself. `voxtral` takes no language token at all and
reports the flag as ignored if you pass it.

## Global options

| flag | values | default | notes |
|---|---|---|---|
| `--model` | alias or HF repo id | `voxtral` | an unknown repo id gets a backend inferred from its name |
| `--list-models` | flag | off | prints the registry with per-model caveats, then exits |
| `-f, --output-format` | `srt` `vtt` `txt` `json` `all` | `srt` | `all` writes one file per format |
| `-o, --output` | path | input stem + extension | with `-f all` this is a path **stem**, not a directory; parent dirs are created |
| `--quiet` | flag | off | silences stdout entirely. Download progress bars still appear: those are huggingface_hub writing to stderr |
| `--stats-json` | path | none | timing, resolved config, and machine info |
| `-h, --help` | flag | | |

## Model-specific options

The engines do not share a long-form algorithm, so most flags apply to exactly one
of them. **A flag the selected engine cannot honour is an error, not a warning**, and
exits 2 without transcribing:

```console
$ mlx-asr audio.wav --model kotoba --max-batch 8 --vad
error: --max-batch, --vad: not supported by --model kotoba. These are Voxtral-only,
because the engines do not share a long-form algorithm. Drop the flag, or use the
default --model voxtral.
```

Nothing is silently ignored. A flag that looks accepted and then does nothing produces
output a user will read as having been produced with it, which is the same class of
mistake as publishing a measurement from a config that was never applied. This project
did exactly that once, with a subtitle-cue setting.

| flag | `voxtral` | `whisper-*` | `kotoba` |
|---|---|---|---|
| `--language` | **error** (takes no language token) | **used** (guessing costs 25 points) | forced to `ja` |
| `--chunk-seconds` | chunk length | **error** (30s window is fixed by the model) | **window length, its biggest lever** |
| `--max-batch` | yes | error | error |
| `--delay-ms` | yes | error | error |
| `--kv-bits` / `--no-kv-quant` | yes | error | error |
| `--fast` | yes | error | error |
| `--overlap-seconds` | yes | error | error |
| `--prompt` | yes | error | error |
| `--vad` | yes | error | error |
| `--compact-silence` | yes | error | error |
| `--gain` / `--peak-dbfs` / `--rms-dbfs` | yes | error | error |
| `--gap-seconds` / `--max-chars` / `--max-dur-seconds` | yes | error | error |

`--delay-ms` and `--gain` have non-default values, so they only count as passed when set
away from the default; the default is not something you asked for.

`--chunk-seconds` is the only flag that means different things on different engines. On
`whisper-*` it is refused rather than approximated, because that driver's 30s window is
set by the model's positional encoding and a flag that appeared to change it would be a
lie.

### Voxtral options

All verified working through the brew binary.

| flag | values | default | measured effect |
|---|---|---|---|
| `--delay-ms` | int | `2400` | the biggest accuracy lever, and free: 25.62% at 480ms, 20.51% at 960ms, 16.44% at 2400ms, at the same speed |
| `--chunk-seconds` | float | per machine (60s) | 60s vs 30s is not resolvable on the corpus, so pick on speed |
| `--max-batch` | int | per machine | throughput is **not monotonic** in this: B=2..8 is slower per step than B=1 |
| `--kv-bits` | `4` or `8` | `8` | free: slightly faster, 39 of 40 scored regions identical to unquantized |
| `--no-kv-quant` | flag | off | disables the above |
| `--fast` | flag | off | halves the chunk, doubles the batch, adds warm-up overlap. Declines automatically when it would not help |
| `--overlap-seconds` | float | `0` | won 1.5-1.8 points at 30s chunks on one clip, **did not reproduce on the corpus** (sign reversed) |
| `--prompt` | text | none | vocabulary bias. **Not an instruction field**: an imperative costs ~6 CER points, because the decoder reads it as text it already emitted. Last ~31 tokens only. Ignored when overlap is active |
| `--gain` | `auto` `none` `peak` `rms` or dB | `auto` | `auto` boosts only below -6 dBFS peak, so it is a no-op on healthy audio. Quiet input costs ~3.8 points because the mel floor is absolute |
| `--peak-dbfs` / `--rms-dbfs` | float | `-1.0` / `-23.0` | targets for `--gain peak` / `--gain rms` |
| `--vad` | flag | off | Silero VAD cut points. **Measured worse on clean speech** by 0.8-3.0 points; for material where energy minima mislead |
| `--compact-silence` | flag | off | drops long pauses before decode, remaps timestamps back. Helps some quantizations, badly hurts others |

### whisper-* options

Only `--language` beyond the global set. The registry also applies
`condition_on_previous_text=False`, which is not exposed as a flag because
enabling it is never right on long audio: it lets a repetition loop feed itself
across 30s windows, worth up to 22.6 CER points.

These engines **sample**, so repeat runs of one config differ. Confirmed directly
rather than inferred: three `whisper-base` runs on identical audio and identical
flags produced three different files (33.69% and 35.59% CER on two of them), while
two `voxtral` runs produced byte-identical output. If you need a reproducible
transcript, or you are benchmarking anything, use `voxtral`.

One qualification on that, measured rather than assumed: Voxtral is byte-identical
across reruns **on one machine**, but not across different Apple Silicon chips. Same
audio, same flags, same weight file gave different output on an M4 16GB and an M2 Ultra 128GB,
because reduction order differs and that flips argmax ties. So a transcript is
reproducible where it was produced, and a config comparison has to stay on one
machine.

### kotoba options

`--chunk-seconds` is the one that matters, and it is material-dependent:

```bash
for w in 10 15 20 30; do
  mlx-asr jp.wav --model kotoba --chunk-seconds $w -o "kotoba_${w}s.srt"
done
```

10s measured best on this clip (22.67% against 27.22% at 20s) and on the
spontaneous corpus (27.0 / 31.3 / 49.7% at 10 / 20 / 30s), but 20s won on clean
narration in the v2.2 sweep. Sweep it on your own audio.

## Using weights that are not in the registry

`--model` takes any HF repo id, and the backend is inferred from the name:
`voxtral` in the id routes to the Voxtral path, `kotoba` or `distil` to the
chunked driver (a distil checkpoint keeps 2-4 decoder layers and cannot carry
state across sequential windows), `whisper` to the sequential driver, and anything
else to Voxtral.

```bash
mlx-asr audio.wav --model mlx-community/whisper-medium-mlx --language de
mlx-asr audio.wav --model ./models/my-conversion            # a local directory
```

## Converting your own weights

Sequential Whisper checkpoints are widely converted under `mlx-community`, but
distil models mostly are not, and the MLX loader cannot read transformers format:
it splats `config.json` into a dataclass and dies on `_name_or_path`.

Rather than refuse those, the chunked driver converts them on first use, which is
why `--model kotoba` works with no setup step:

```console
$ mlx-asr jp.wav --model kotoba
[convert] no MLX build of kotoba-tech/kotoba-whisper-v2.0 is published; converting once
[convert] 539 tensors, 32-layer encoder / 2-layer decoder
[convert] wrote ~/.cache/huggingface/hub/mlx-asr-converted/kotoba-tech--kotoba-whisper-v2.0
```

The conversion is cheap because the two formats hold the same numbers: it renames
keys (`self_attn` to `attn`, `fc1` to `mlp1`, and so on) and swaps one axis on the
conv weights, since torch stores Conv1d as (out, in, kernel) and MLX as (out,
kernel, in). Nothing is recomputed or re-quantized, `mx.load` reads safetensors
directly, and **no torch is involved**. It runs once per machine and the result is
cached next to the downloaded weights.

Verified byte-for-byte: the self-converted v2.0 produces output identical to an
independently published MLX conversion of the same checkpoint.

The same applies to any transformers-format Whisper repo passed to the chunked
driver. It is not applied to the sequential `whisper-*` path, where every registry
alias already points at MLX weights.
