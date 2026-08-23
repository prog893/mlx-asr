# Models and command reference

What each flag accepts and what it does. **Why** the defaults are what they are, with the
measurement behind each one, is in [DEFAULTS.md](DEFAULTS.md).

`--model` picks a family; `--size` and `--quantization` pick the variant inside it.

Supports everything libavformat can read, which is nearly every audio format and the audio
track of a video: [AUDIO.md](AUDIO.md).

MLX only. No path imports torch or uses its MPS backend, and there is no CUDA or CPU
fallback, so this runs on Apple Silicon or not at all.

## The models

### `--model voxtral`

| `--quantization` | weights | download | peak GPU memory |
|---|---|---|---|
| `4bit` **default** | [mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit) | 3.15GB | 6.77GB |
| `fp16` | [mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16) | 8.89GB | 12.98GB |

### `--model whisper`

| `--size` | `--quantization` | weights | download | peak GPU memory |
|---|---|---|---|---|
| `tiny` | - | [mlx-community/whisper-tiny-mlx](https://huggingface.co/mlx-community/whisper-tiny-mlx) | 0.07GB | 3.99GB |
| `base` | - | [mlx-community/whisper-base-mlx](https://huggingface.co/mlx-community/whisper-base-mlx) | 0.14GB | 4.06GB |
| `small` | - | [mlx-community/whisper-small-mlx](https://huggingface.co/mlx-community/whisper-small-mlx) | 0.48GB | 4.38GB |
| `medium` | - | [mlx-community/whisper-medium-mlx](https://huggingface.co/mlx-community/whisper-medium-mlx) | 1.52GB | 5.44GB |
| `large-v2` | - | [mlx-community/whisper-large-v2-mlx](https://huggingface.co/mlx-community/whisper-large-v2-mlx) | 3.08GB | 6.99GB |
| `large-v3` | - | [mlx-community/whisper-large-v3-mlx](https://huggingface.co/mlx-community/whisper-large-v3-mlx) | 3.08GB | 6.99GB |
| `turbo` **default** | - | [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | 1.61GB | 5.52GB |

### `--model kotoba`

| `--quantization` | weights | download | peak GPU memory |
|---|---|---|---|
| - | [kotoba-tech/kotoba-whisper-v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) | 1.52GB | 2.38GB |

### `--model qwen3-asr`

| `--size` | `--quantization` | weights | download | peak GPU memory |
|---|---|---|---|---|
| `0.6B` | `4bit` | [mlx-community/Qwen3-ASR-0.6B-4bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit) | 0.71GB | 2.06GB |
| `0.6B` | `5bit` | [mlx-community/Qwen3-ASR-0.6B-5bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-5bit) | 0.79GB | 2.14GB |
| `0.6B` | `6bit` | [mlx-community/Qwen3-ASR-0.6B-6bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-6bit) | 0.86GB | 2.21GB |
| `0.6B` | `8bit` **default** | [mlx-community/Qwen3-ASR-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit) | 1.01GB | 2.36GB |
| `0.6B` | `bf16` | [mlx-community/Qwen3-ASR-0.6B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-bf16) | 1.57GB | 2.92GB |
| `1.7B` **default** | `4bit` | [mlx-community/Qwen3-ASR-1.7B-4bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-4bit) | 1.61GB | 3.19GB |
| `1.7B` | `5bit` | [mlx-community/Qwen3-ASR-1.7B-5bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-5bit) | 1.82GB | 3.40GB |
| `1.7B` | `6bit` | [mlx-community/Qwen3-ASR-1.7B-6bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-6bit) | 2.04GB | 3.62GB |
| `1.7B` | `8bit` **default** | [mlx-community/Qwen3-ASR-1.7B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit) | 2.47GB | 4.05GB |
| `1.7B` | `bf16` | [mlx-community/Qwen3-ASR-1.7B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16) | 4.08GB | 5.66GB |

The data for these tables comes from `scripts/docs/gen_model_matrix.py`, which reads the
registry and checks every id against the Hugging Face API. `download` covers the whole
repo, weights plus tokenizer and configs.

`peak GPU memory` is what decides whether a model fits, and it tracks the download only
loosely: whisper `tiny` downloads 0.07GB and peaks at 3.99GB, because the working set is
the 30s mel window and decoder activations rather than the weights. That is why the whole
whisper ladder lands between 4 and 7GB regardless of size.

Every figure is the maximum over the 20-file corpus on an M2 Ultra, one method for all of
them, so the column can be read against itself. Method and caveats:
[benchmarks/peak-memory.md](benchmarks/peak-memory.md).

Every run also prints its own peak, since the figure moves with the machine, the audio and
the flags, and `--stats-json` writes it to a file. Treat the table as a guide to what fits
and your own run as the number.

`voxtral` and `qwen3-asr` decode greedily and rerun byte-identically on one machine.
`whisper` samples, and `kotoba` runs on Whisper's decoder so it samples too
([benchmarks/determinism.md](benchmarks/determinism.md)).

Accuracy and throughput per model: [benchmarks/engines.md](benchmarks/engines.md) and
[benchmarks/qwen3-asr.md](benchmarks/qwen3-asr.md), which describe the corpus and method
each figure came from.

The `kotoba` alias points at v2.0 because v2.2 holds identical ASR weights (all 539
tensors equal in value); its diarization and punctuation are separate torch models. It
ships in transformers format, so it converts to MLX on first use and caches under
`~/.cache/huggingface/hub/mlx-asr-converted/`. That first run peaks at 3.03GB rather than
2.38GB, because the conversion loads the whole checkpoint; every run after is the table
figure.

## Minimal commands

```bash
mlx-asr audio.wav                                            # voxtral, writes audio.srt
mlx-asr audio.wav --model whisper --language ja              # turbo
mlx-asr audio.wav --model whisper --size small --language ja
mlx-asr audio.wav --model kotoba                             # forces ja itself
mlx-asr audio.wav --model qwen3-asr --language ja -f txt
mlx-asr audio.wav --model qwen3-asr --size 0.6B --language ja -f json
```

## Global flags

The only flags every model accepts.

| flag | values | default | notes |
|---|---|---|---|
| `--model` | family name or HF repo id | `voxtral` | a repo id gets its backend inferred from the name |
| `--list-models` | flag | off | prints each model with its sizes, precisions and caveats, then exits |
| `-f, --output-format` | `srt` `vtt` `txt` `json` `all` | `srt` | `all` writes one file per format. `srt`, `vtt` and `all` exit 2 on `qwen3-asr` |
| `-o, --output` | path | input stem + extension | with `-f all` this is a path **stem**, not a directory; parent dirs are created |
| `--quiet` | flag | off | silences stdout entirely. Download progress bars still appear: those are huggingface_hub writing to stderr |
| `--stats-json` | path | none | timing, peak memory, resolved config, and machine info |
| `-h, --help` | flag | | |

## Flag support matrix

Most flags belong to one model, because the models do not share a long-form algorithm.
Passing an unsupported flag exits 2 without transcribing, rather than being ignored, so
the output always matches the arguments:

```console
$ mlx-asr audio.wav --model kotoba --max-batch 8 --vad
error: --max-batch, --vad: not supported by --model kotoba. These are Voxtral-only,
because the engines do not share a long-form algorithm. Drop the flag, or use the
default --model voxtral.
```

`-` means the flag exits 2 on that model.

| flag | `voxtral` | `whisper` | `kotoba` | `qwen3-asr` |
|---|---|---|---|---|
| `--size` | - | `tiny` `base` `small` `medium` `large-v2` `large-v3` `turbo` | - | `0.6B` `1.7B` |
| `--quantization` | `4bit` `fp16` | - | - | `4bit` `5bit` `6bit` `8bit` `bf16` |
| `--language` | - | yes | forced `ja` | yes |
| `--chunk-seconds` | yes | - | yes | yes |
| `-f srt` / `vtt` / `all` | yes | yes | yes | - |
| `--delay-ms` | yes | - | - | - |
| `--max-batch` | yes | - | - | - |
| `--kv-bits` / `--no-kv-quant` | yes | - | - | - |
| `--fast` | yes | - | - | - |
| `--overlap-seconds` | yes | - | - | - |
| `--prompt` | yes | - | - | - |
| `--vad` | yes | - | - | - |
| `--compact-silence` | yes | - | - | - |
| `--gain` / `--peak-dbfs` / `--rms-dbfs` | yes | - | - | - |
| `--gap-seconds` / `--max-chars` / `--max-dur-seconds` | yes | - | - | - |

`--chunk-seconds` means something different on each engine that takes it: Voxtral's chunk
is a batched decode unit, while kotoba and qwen3-asr use it as an independent window
length. Whisper refuses it, since its 30s window is fixed by the model's positional
encoding.

`--delay-ms`, `--gain`, `--peak-dbfs` and `--rms-dbfs` ship with working values rather
than off, so the other models accept them at their default and only exit 2 if you change
them.

## `--language`

Set it whenever the engine takes one. **Whisper's autodetect reads only the first 30
seconds, and on this project's material it returned Russian for Japanese audio and cost 25
CER points.** `qwen3-asr` does not autodetect at all: omitting the flag forces English.

Any spelling works on any model that takes one. These are all Japanese:

```bash
mlx-asr a.wav --model whisper   --language ja
mlx-asr a.wav --model whisper   --language jpn
mlx-asr a.wav --model qwen3-asr --language Japanese -f txt
mlx-asr a.wav --model qwen3-asr --language ja_JP    -f txt
```

Each model wants a different form internally (whisper an ISO 639-1 code, qwen3-asr an
English name) and the CLI converts, so you never have to know which. Region and script
subtags are dropped: `en-US` and `en-GB` both give `en`. An unrecognised value exits 2
before the audio is read, listing what that model accepts.

The conversion matters because every engine takes a wrong value silently and transcribes
worse rather than failing. `--language Japanese` reaches whisper as a decoder prompt hint
rather than a language selection, and `--language ja` reaches qwen3-asr as the literal
prefix `language ja<asr_text>`.

Accepted names come from each checkpoint's own published vocabulary, so a model that drops
a language stops accepting it without a change here.

Two models refuse the flag, for different reasons. `voxtral` has no language token to set:
its tokenizer carries a thousand special tokens and not one of them names a language or a
task, unlike Whisper's `<|ja|>`. `kotoba` has the tokens but is finetuned on Japanese
alone, so it is pinned to `ja` and says so.

## Flag values and defaults

### voxtral

| flag | values | default | effect |
|---|---|---|---|
| `--delay-ms` | int | `2400` | transcription delay, and the largest accuracy lever here. 2400 is also the model's maximum |
| `--chunk-seconds` | float | per machine | audio per batch row. A throughput knob rather than an accuracy one |
| `--max-batch` | int | per machine | chunks decoded together. Not monotonic, so the profile avoids the middle of the range |
| `--quantization` | `4bit` `fp16`/`none` | `4bit` | weight precision. Also sets the weight footprint used to size the batch |
| `--kv-bits` | `4` `8` | `8` | quantize the KV cache |
| `--no-kv-quant` | flag | off | disable KV quantization |
| `--fast` | flag | off | halve the chunk, double the batch, add warm-up overlap. Declines when it would not help |
| `--overlap-seconds` | float | `0` | preceding audio prepended to each chunk as context, then discarded |
| `--prompt` | text | none | text the decoder treats as already emitted, which biases vocabulary. Last ~31 tokens only, and ignored when overlap is active |
| `--gain` | `auto` `none` `peak` `rms`, or dB | `auto` | input level. `auto` boosts quiet audio and leaves louder audio untouched |
| `--peak-dbfs` / `--rms-dbfs` | float | `-1.0` / `-23.0` | targets for `--gain peak` / `--gain rms` |
| `--vad` | flag | off | place chunk boundaries with Silero VAD instead of energy minima |
| `--compact-silence` | flag | off | drop long pauses before decode, remap timestamps back |
| `--gap-seconds` | float | `1.2` | split a cue at a pause this long. The one cue knob worth sweeping |
| `--max-chars` | int | `28` | split a cue at this many characters |
| `--max-dur-seconds` | float | `7.0` | hard cap on cue duration |

`--prompt` takes domain terms or a topic sentence in the audio's own language, not an
instruction: an imperative there makes things worse, and on English audio any prompt wrecks
word spacing ([benchmarks/prompt.md](benchmarks/prompt.md)).

### whisper

| flag | values | default | effect |
|---|---|---|---|
| `--size` | see matrix | `turbo` | which checkpoint |
| `--language` | any spelling | autodetect | set it; see above |

`condition_on_previous_text=False` is applied to `small` and larger and is not a flag,
because enabling it is never right on long audio. `tiny` and `base` keep the library
default.

### kotoba

| flag | values | default | effect |
|---|---|---|---|
| `--chunk-seconds` | float | `10` | independent window length, and this model's largest lever. Material-dependent, so sweep it on your own audio |
| `--language` | | forced `ja` | Japanese-only weights |

### qwen3-asr

| flag | values | default | effect |
|---|---|---|---|
| `--size` | `0.6B` `1.7B` | `1.7B` | which checkpoint |
| `--quantization` | `4bit` `5bit` `6bit` `8bit` `bf16`/`none` | `8bit` | weight precision |
| `--chunk-seconds` | float | `30` | decode window. Shorter is better on accuracy, speed and memory here |
| `--language` | any spelling | `English` | never autodetects, so set it |

```bash
mlx-asr jp.wav --model qwen3-asr --language ja -f txt --size 0.6B --quantization 4bit
```

Three behaviours specific to this model:

**It writes no subtitles.** Segment times are decode-window boundaries rather than speech,
so `-f srt`, `-f vtt` and `-f all` exit 2 instead of writing a file whose cue times are
fiction. `txt` and `json` work, and the JSON carries `cue_source: "chunk_boundaries"`.

**`--max-batch` exits 2.** `batch_size` exists upstream but does nothing at the default
window, and no value has been measured for this decoder
([#1](https://github.com/prog893/mlx-asr/issues/1)).

**Repetition loops occur on some material** and are a property of the weights. Each is
capped to one window and the CLI warns when it sees one. `mlx_asr` drives the chunk loop
rather than the library, because upstream's `max_tokens` is a per-file budget whose
exhaustion stops transcription silently.

## Bring your own model

`--model` takes any Hugging Face repo id or local directory, in **MLX format only**:
safetensors plus a `config.json`. The backend is inferred from the id, so
`--model Qwen/Qwen3-ASR-1.7B` behaves like `--model qwen3-asr`, including its 30s window
and refused output formats. `--size` and `--quantization` are refused, since the id
already names the variant.

```bash
mlx-asr audio.wav --model mlx-community/whisper-medium-mlx --language de
mlx-asr audio.wav --model ./models/my-conversion
```

The backend is matched on the id, first hit wins:

1. `voxtral` -> Voxtral decoder
2. `qwen3-asr` -> Qwen3 chunk loop
3. `kotoba` or `distil` -> chunked driver (a distil checkpoint keeps 2-4 decoder layers
   and cannot carry state across sequential windows)
4. `whisper` -> sequential driver
5. anything else -> Voxtral

`qwen3-asr` is checked before `whisper` and `distil` so that an id carrying both,
`Qwen3-ASR-1.7B-whisper-distilled`, reaches the loader that can read its config.

**Rule 5 is a guess, and a wrong guess is not caught gracefully.** An id that names no
known architecture is handed to the Voxtral loader, which fails deep in the stack rather
than at the CLI boundary: an id that does not exist on the hub raises
`RepositoryNotFoundError`, and a real checkpoint of the wrong architecture raises an
`AttributeError` for a config field the model does not have. Both arrive as a Python
traceback, after the audio has been decoded and the weights downloaded. Everything else
here exits 2 with a sentence; this path does not, so check the id before pointing at
something exotic.

### Formats that will not work

GGUF, ONNX, ExecuTorch, OpenVINO and CoreML. `mx.load` reads GGUF tensors, but the
`mlx-audio` and `mlx-whisper` loaders need a safetensors layout plus `config.json`, so
GGUF is not a drop-in even at the tensor level; loading it means adopting llama.cpp's
runtime and losing the multi-stream batching this project's throughput depends on.

That rules out unsloth, which publishes no Voxtral, GGUF only for Qwen3-ASR, and
unquantized transformers weights for Whisper. Quantization saves memory here without
costing accuracy or gaining speed, so little is lost:
[benchmarks/quantization.md](benchmarks/quantization.md).

### Converting transformers weights

The MLX loader cannot read transformers format: it splats `config.json` into a dataclass
and fails on `_name_or_path`. Distil checkpoints are mostly published that way, so the
chunked driver converts on first use, which is why `--model kotoba` needs no setup step:

```console
$ mlx-asr jp.wav --model kotoba
[convert] no MLX build of kotoba-tech/kotoba-whisper-v2.0 is published; converting once
[convert] 539 tensors, 32-layer encoder / 2-layer decoder
[convert] wrote ~/.cache/huggingface/hub/mlx-asr-converted/kotoba-tech--kotoba-whisper-v2.0
```

The conversion renames keys (`self_attn` to `attn`, `fc1` to `mlp1`) and swaps one axis on
the conv weights, since torch stores Conv1d as (out, in, kernel) and MLX as (out, kernel,
in). Nothing is recomputed or re-quantized and no torch is involved. It runs once per
machine, caches next to the downloaded weights, and is deterministic: converting the same
checkpoint twice gives byte-identical output.

Applies to any transformers-format Whisper repo on the chunked driver. Not to the
sequential `whisper` path, where every built-in size already points at MLX weights.

### Local quantization

`mlx_audio.convert` produces MLX quants the hub does not publish, including the 8bit,
mxfp8 and nvfp4 Voxtral variants in
[benchmarks/quantization.md](benchmarks/quantization.md):

```bash
python -m mlx_audio.convert \
  --hf-path mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16 \
  --mlx-path ./voxtral-8bit --quantize --q-bits 8 --q-group-size 64
mlx-asr audio.wav --model ./voxtral-8bit
```
