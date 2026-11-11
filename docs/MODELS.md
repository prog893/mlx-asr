# Models

The models `mlx-asr` runs, the variants of each, where their weights come from, and the
flags each family takes.

`--model` picks a family; `--size` and `--quantization` pick the variant inside it. In the
tables:

- priority: the order variants are tried in when neither flag is passed. The CLI loads
  the first one that fits the Mac's memory, starting from 1. A pinned `--size` or
  `--quantization` keeps only the matching rows, tried in the same order. How the order was
  set: [DEFAULTS.md](DEFAULTS.md#how-a-variant-is-chosen).
- weights: the Hugging Face repo the weights are downloaded from, which also states the
  model's license.
- quantized on first use: no loadable build of that precision is published, so the
  first run downloads the linked repo and quantizes it locally, once.
- converted to MLX on first use: the repo is published in transformers format, so the
  first run converts it to MLX locally, once.
- A family with no `--size` or `--quantization` column has one size or one precision, and
  passing that flag is an error.

Quantized and converted builds are cached in `mlx-asr-converted/` inside the Hugging Face cache
(`~/.cache/huggingface/hub/` by default), and later runs load the cached copy.

## The models

### Voxtral Realtime

`--model voxtral`

Mistral AI's Voxtral Mini 4B Realtime, the 2602 release (February 2026). A streaming
model: its causal audio encoder transcribes at a configurable delay, which `--delay-ms`
sets. The card lists 13 languages. The default model.

| priority | `--quantization` | weights | download | peak GPU memory |
|---|---|---|---|---|
| 3 | `4bit` | [mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit) | 3.15GB | 6.77GB |
| 2 | `8bit` | [mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16), quantized on first use | 8.89GB | 7.29GB |
| 1 | `fp16` | [mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16) | 8.89GB | 12.98GB |

Source: [m2-ultra-voxtral-precision.json](../benchmarks/results/2026-10/m2-ultra-voxtral-precision.json) runs `vox_w4bit_c60b16_kv8`, `vox_w8bit_c60b16_kv8`, `vox_wfp16_c60b16_kv8` (peak; download is the hub repo size)

Example:

```bash
mlx-asr audio.wav --model voxtral # writes audio.srt
```

Compatible flags for `--model voxtral`:

| flag | values | default | effect |
|---|---|---|---|
| `--delay-ms` | int | `2400` | transcription delay, and the largest accuracy lever here. 2400 is also the model's maximum |
| `--chunk-seconds` | float | per machine ([DEFAULTS.md](DEFAULTS.md#chunking-and-windows)) | audio per batch row. A throughput knob rather than an accuracy one |
| `--max-batch` | int | from GPU memory ([DEFAULTS.md](DEFAULTS.md#chunking-and-windows)) | chunks decoded together. Sized from GPU memory, then snapped to batch sizes that measured well: per-step time is not monotonic, so 2-11 are never chosen |
| `--quantization` | `4bit` `8bit` `fp16`/`none` | by memory | weight precision: `fp16` where it fits a useful batch, else `8bit`, else `4bit` ([DEFAULTS.md](DEFAULTS.md#how-a-variant-is-chosen)). `8bit` is quantized from the fp16 build on first use. Also sets the weight footprint used to size the batch |
| `--kv-bits` | `4` `8` | `8` | quantize the KV cache |
| `--no-kv-quant` | flag | off | disable KV quantization |
| `--overlap-seconds` | float | `0` | preceding audio prepended to each chunk as context, then discarded |
| `--prompt` | text | none | text the decoder treats as already emitted, which biases vocabulary. Last ~31 tokens only<!-- not-measured: n_left_pad_tokens - 1, a model config constant -->, and ignored when overlap is active |
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

### Voxtral (2025)

`--model voxtral-v1`

Mistral AI's first Voxtral release, 2507 (July 2025): Voxtral Mini 3B, built on Ministral
3B, and Voxtral Small 24B. Speech-understanding models with a dedicated transcription mode
and automatic language detection; the card lists eight languages.

| priority | `--size` | `--quantization` | weights | download | peak GPU memory |
|---|---|---|---|---|---|
| 6 | `3B` | `4bit` | [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), quantized on first use | 9.37GB | 5.25GB |
| 5 | `3B` | `8bit` | [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), quantized on first use | 9.37GB | 7.28GB |
| 4 | `3B` | `bf16` | [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | 9.37GB | 10.91GB |
| 3 | `24B` | `4bit` | [mistralai/Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507), quantized on first use | 48.54GB | 16.27GB |
| 2 | `24B` | `8bit` | [mistralai/Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507), quantized on first use | 48.54GB | 27.92GB |
| 1 | `24B` | `bf16` | [mistralai/Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) | 48.54GB | 50.08GB |

Source: [m2-ultra-voxtral-v1.json](../benchmarks/results/2026-10/m2-ultra-voxtral-v1.json) runs `3b_4bit_c30`, `3b_8bit_c30`, `24b_4bit_c30`, `24b_8bit_c30`; [m2-ultra-headline-remeasure.json](../benchmarks/results/2026-11/m2-ultra-headline-remeasure.json) runs `3b_bf16_c30`, `24b_bf16_c30` (peak; download is the hub repo size)

The weights come from Mistral's own repos. They are published as safetensors that MLX loads
as they are, so `bf16` needs no conversion. `8bit` and `4bit` are quantized from them on
first use, which downloads the bf16 weights first (9.37GB for 3B, 48.54GB for 24B). The <!-- not-measured: hub repo sizes -->
audio encoder stays bf16 at every precision.

Example:

```bash
mlx-asr audio.wav --model voxtral-v1 --language en -f txt
```

Compatible flags for `--model voxtral-v1`:

| flag | values | default | effect |
|---|---|---|---|
| `--size` | `3B` `24B` | by memory | which checkpoint |
| `--quantization` | `4bit` `8bit` `bf16` | by memory | weight precision |
| `--chunk-seconds` | float | `30` | independent window length |
| `--language` | any spelling | autodetect | set it; see [CLI.md](CLI.md#--language) |

It writes no subtitles: `-f srt`, `-f vtt` and `-f all` are refused with an error.

### Whisper

`--model whisper`

OpenAI's Whisper: the original sizes from September 2022, large-v2 from December 2022,
large-v3 from November 2023, and turbo, large-v3 with its decoder cut from 32 layers to 4,
from October 2024. 99 languages.

| priority | `--size` | weights | download | peak GPU memory |
|---|---|---|---|---|
| 7 | `tiny` | [mlx-community/whisper-tiny-mlx](https://huggingface.co/mlx-community/whisper-tiny-mlx) | 0.07GB | 3.98GB |
| 6 | `base` | [mlx-community/whisper-base-mlx](https://huggingface.co/mlx-community/whisper-base-mlx) | 0.14GB | 4.07GB |
| 5 | `small` | [mlx-community/whisper-small-mlx](https://huggingface.co/mlx-community/whisper-small-mlx) | 0.48GB | 4.37GB |
| 4 | `medium` | [mlx-community/whisper-medium-mlx](https://huggingface.co/mlx-community/whisper-medium-mlx) | 1.52GB | 5.43GB |
| 3 | `large-v2` | [mlx-community/whisper-large-v2-mlx](https://huggingface.co/mlx-community/whisper-large-v2-mlx) | 3.08GB | 6.97GB |
| 1 | `large-v3` | [mlx-community/whisper-large-v3-mlx](https://huggingface.co/mlx-community/whisper-large-v3-mlx) | 3.08GB | 7.00GB |
| 2 | `turbo` | [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | 1.61GB | 5.53GB |

Source: [m2-ultra-whisper-sizes.json](../benchmarks/results/2026-10/m2-ultra-whisper-sizes.json) runs `whisper_tiny_libdef`, `whisper_base_libdef`, `whisper_small_nocond`, `whisper_medium_nocond`, `whisper_large-v2_nocond`, `whisper_large-v3_nocond`, `whisper_turbo_nocond` (peak; download is the hub repo size)

Example:

```bash
mlx-asr audio.wav --model whisper --language ja
```

Compatible flags for `--model whisper`:

| flag | values | default | effect |
|---|---|---|---|
| `--size` | see matrix | by memory: the first of `large-v3`, `turbo`, `large-v2`, `medium`, `small`, `base`, `tiny` that fits | which checkpoint |
| `--language` | any spelling | autodetect | set it; see [CLI.md](CLI.md#--language) |

`condition_on_previous_text=False` is applied to `small` and larger and is not a flag,
because enabling it is never right on long audio. `tiny` and `base` keep the library
default.

### Kotoba-Whisper

`--model kotoba`

Kotoba-Whisper v2.0, from Kotoba Technologies and Asahi Ushio (September 2024). Whisper
large-v3 distilled to 2 decoder layers and trained on the ReazonSpeech corpus. Japanese
only.

| weights | download | peak GPU memory |
|---|---|---|
| [kotoba-tech/kotoba-whisper-v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0), converted to MLX on first use | 1.52GB | 2.38GB |

Source: [m2-ultra-peak-memory-kotoba.json](../benchmarks/results/2026-10/m2-ultra-peak-memory-kotoba.json) runs `pm_kotoba_7f8150fdef0e`, `pm_kotoba_c18b2f5aa2a2`, `pm_kotoba_9aefd8f2352e`, `pm_kotoba_07fd66556725`, `pm_kotoba_c29657df52e9`, `pm_kotoba_cbeb19543902`, `pm_kotoba_b88caf8c0924`, `pm_kotoba_bbdabfabfb4f`, `pm_kotoba_1672083b2499`, `pm_kotoba_5b44e078553d`, `pm_kotoba_a73e94506f13`, `pm_kotoba_bb5cf5d5b247`, `pm_kotoba_0c193bf7ced2`, `pm_kotoba_57d841fe39b5`, `pm_kotoba_a773a315fb42`, `pm_kotoba_71bb5eba3baa`, `pm_kotoba_a1372c4d2900`, `pm_kotoba_c47ad127bc83`, `pm_kotoba_1da58e1b2b69`, `pm_kotoba_b3432b40c079` (peak, maximum over the files, at the 10s window shipped then; download is the hub repo size)

The alias points at v2.0 because v2.2 holds identical ASR weights (all 539 tensors equal in
value); v2.2 adds diarization and punctuation as separate torch models. The conversion to MLX
loads the whole checkpoint, so a run with no converted copy cached yet peaks at 3.03GB[^n7]
instead of 2.38GB[^n2].

Example:

```bash
mlx-asr audio.wav --model kotoba
```

Compatible flags for `--model kotoba`:

| flag | values | default | effect |
|---|---|---|---|
| `--chunk-seconds` | float | `15` | independent window length, and this model's largest lever. Material-dependent, so sweep it on your own audio |
| `--language` | | forced `ja` | Japanese-only weights |

### Qwen3-ASR

`--model qwen3-asr`

Alibaba's Qwen3-ASR (January 2026), built on Qwen3-Omni. The card lists 30 languages and
22 Chinese dialects, with language identification.

| priority | `--size` | `--quantization` | weights | download | peak GPU memory |
|---|---|---|---|---|---|
| 10 | `0.6B` | `4bit` | [mlx-community/Qwen3-ASR-0.6B-4bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit) | 0.71GB | 1.77GB |
| 9 | `0.6B` | `5bit` | [mlx-community/Qwen3-ASR-0.6B-5bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-5bit) | 0.79GB | 1.85GB |
| 8 | `0.6B` | `6bit` | [mlx-community/Qwen3-ASR-0.6B-6bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-6bit) | 0.86GB | 1.92GB |
| 7 | `0.6B` | `8bit` | [mlx-community/Qwen3-ASR-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit) | 1.01GB | 2.07GB |
| 6 | `0.6B` | `bf16` | [mlx-community/Qwen3-ASR-0.6B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-bf16) | 1.57GB | 2.63GB |
| 5 | `1.7B` | `4bit` | [mlx-community/Qwen3-ASR-1.7B-4bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-4bit) | 1.61GB | 2.88GB |
| 2 | `1.7B` | `5bit` | [mlx-community/Qwen3-ASR-1.7B-5bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-5bit) | 1.82GB | 3.09GB |
| 4 | `1.7B` | `6bit` | [mlx-community/Qwen3-ASR-1.7B-6bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-6bit) | 2.04GB | 3.31GB |
| 1 | `1.7B` | `8bit` | [mlx-community/Qwen3-ASR-1.7B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit) | 2.47GB | 3.74GB |
| 3 | `1.7B` | `bf16` | [mlx-community/Qwen3-ASR-1.7B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16) | 4.08GB | 5.35GB |

Source: [m2-ultra-qwen3-precision-c15.json](../benchmarks/results/2026-10/m2-ultra-qwen3-precision-c15.json) runs `qwen3_0.6B_4bit_c15`, `qwen3_0.6B_5bit_c15`, `qwen3_0.6B_6bit_c15`, `qwen3_0.6B_8bit_c15`, `qwen3_0.6B_bf16_c15`, `qwen3_1.7B_4bit_c15`, `qwen3_1.7B_5bit_c15`, `qwen3_1.7B_6bit_c15`, `qwen3_1.7B_8bit_c15`, `qwen3_1.7B_bf16_c15` (peak at the shipped 15s window; download is the hub repo size)

Example:

```bash
mlx-asr audio.wav --model qwen3-asr --language ja -f txt
```

Compatible flags for `--model qwen3-asr`:

| flag | values | default | effect |
|---|---|---|---|
| `--size` | `0.6B` `1.7B` | `1.7B` | which checkpoint |
| `--quantization` | `4bit` `5bit` `6bit` `8bit` `bf16`/`none` | `8bit` | weight precision |
| `--chunk-seconds` | float | `15` | decode window. Shorter is better on accuracy, speed and memory here |
| `--language` | any spelling | `English` | never autodetects, so set it |

It writes no subtitles. Segment times are decode-window boundaries rather than speech,
so `-f srt`, `-f vtt` and `-f all` are refused with an error instead of writing a file
whose cue times are fiction. `txt` and `json` work, and the JSON carries `cue_source: "chunk_boundaries"`.

`--max-batch` is refused. `batch_size` exists upstream, and it was measured to be slower at
every batch size tried, with no accuracy gain: batching whole chunks pads each group to its
longest member and ends with its slowest, and there is no weight read to amortize, so one
chunk at a time wins ([benchmarks/qwen3-batch.md](benchmarks/qwen3-batch.md)).

Repetition loops occur on some material and are a property of the weights. Each is
capped to one window and the CLI warns when it sees one. `mlx_asr` drives the chunk loop
rather than the library, because upstream's `max_tokens` is a per-file budget whose
exhaustion stops transcription silently.

### Parakeet

`--model parakeet`

NVIDIA NeMo's Parakeet TDT-CTC 0.6B for Japanese (May 2024), a hybrid FastConformer model
that transcribes with punctuation. Japanese only. Runs through mlx-audio's own driver.

| weights | download | peak GPU memory |
|---|---|---|
| [mlx-community/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/mlx-community/parakeet-tdt_ctc-0.6b-ja) | 2.49GB | 4.77GB (17-file corpus) |

Source: [m2-ultra-picker-speed.json](../benchmarks/results/2026-10/m2-ultra-picker-speed.json) runs `pk_parakeet_default_s1`, `pk_parakeet_default_s2`, `pk_parakeet_default_s3` (peak; download is the hub repo size)

Example:

```bash
mlx-asr audio.wav --model parakeet
```

Compatible flags for `--model parakeet`:

| flag | values | default | effect |
|---|---|---|---|
| `--chunk-seconds` | float | `120` | independent window length |
| `--language` | | forced `ja` | Japanese-only weights; any other value is an error |

### ReazonSpeech

`--model reazon`

ReazonSpeech k2-v2 from Reazon Human Interaction Lab (July 2024): a character-based
Zipformer RNN-T trained on the ReazonSpeech v2.0 corpus. Japanese only. Runs the authors'
ONNX build on CPU through sherpa-onnx (`uv sync --extra reazon`), so it is not an MLX
engine and its throughput is not comparable to the GPU rows.

| weights | download | peak GPU memory |
|---|---|---|
| [reazon-research/reazonspeech-k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2) | 0.78GB | n/a, CPU |

<!-- not-measured: hub repo sizes only; reazon decodes on CPU and has no GPU peak -->

Example:

```bash
mlx-asr audio.wav --model reazon
```

Compatible flags for `--model reazon`:

| flag | values | default | effect |
|---|---|---|---|
| `--chunk-seconds` | float | `30` | independent window length |
| `--language` | | forced `ja` | Japanese-only weights; any other value is an error |

## About the tables

The data for these tables comes from `scripts/docs/gen_model_matrix.py`, which reads the
registry and checks every id against the Hugging Face API. `download` covers the whole
repo, weights plus tokenizer and configs.

`peak GPU memory` is what decides whether a model fits, and it tracks the download only
loosely: whisper `tiny` downloads 0.07GB<!-- not-measured: 0.07GB is the hub download size --> and peaks at 3.98GB[^n1], because the working set is
the 30s mel window and decoder activations rather than the weights. That is why the whole
whisper ladder lands between 4 and 7GB[^n1] regardless of size.

Every figure is the maximum over the 20-file corpus on an M2 Ultra, one method for all of
them, so the column can be read against itself. Method and caveats:
[benchmarks/reference/peak-memory.md](benchmarks/reference/peak-memory.md).

Every run also prints its own peak, since the figure moves with the machine, the audio and
the flags, and `--stats-json` writes it to a file. Treat the table as a guide to what fits
and your own run as the number.

Accuracy and throughput per model: [benchmarks/engines.md](benchmarks/engines.md),
[benchmarks/engines/qwen3-asr.md](benchmarks/engines/qwen3-asr.md) and
[benchmarks/engines/voxtral-v1.md](benchmarks/engines/voxtral-v1.md), which describe the corpus and method
each figure came from.

## Flag support matrix

Most flags belong to one model, because the models do not share a long-form algorithm.
Passing an unsupported flag stops with an error before transcribing, rather than being ignored, so
the output always matches the arguments:

```console
$ mlx-asr audio.wav --model kotoba --max-batch 8 --vad
error: --max-batch, --vad: not supported by --model kotoba. These are Voxtral-only,
because the engines do not share a long-form algorithm. Drop the flag, or use the
default --model voxtral.
```

`-` means the flag is an error on that model.

| flag | `voxtral` | `voxtral-v1` | `whisper` | `kotoba` | `qwen3-asr` |
|---|---|---|---|---|---|
| `--size` | - | `3B` `24B` | `tiny` `base` `small` `medium` `large-v2` `large-v3` `turbo` | - | `0.6B` `1.7B` |
| `--quantization` | `4bit` `8bit` `fp16` | `4bit` `8bit` `bf16` | - | - | `4bit` `5bit` `6bit` `8bit` `bf16` |
| `--language` | - | yes | yes | forced `ja` | yes |
| `--chunk-seconds` | yes | yes | - | yes | yes |
| `-f srt` / `vtt` / `all` | yes | - | yes | yes | - |
| `--delay-ms` | yes | - | - | - | - |
| `--max-batch` | yes | - | - | - | - |
| `--kv-bits` / `--no-kv-quant` | yes | - | - | - | - |
| `--overlap-seconds` | yes | - | - | - | - |
| `--prompt` | yes | - | - | - | - |
| `--vad` | yes | - | - | - | - |
| `--compact-silence` | yes | - | - | - | - |
| `--gain` / `--peak-dbfs` / `--rms-dbfs` | yes | - | - | - | - |
| `--gap-seconds` / `--max-chars` / `--max-dur-seconds` | yes | - | - | - | - |

`--chunk-seconds` means something different on each engine that takes it: Voxtral's chunk
is a batched decode unit, while kotoba, qwen3-asr and voxtral-v1 use it as an independent
window length. Whisper refuses it, since its 30s window is fixed by the model's positional
encoding.

`--delay-ms`, `--gain`, `--peak-dbfs` and `--rms-dbfs` ship with working values rather
than off, so the other models accept them at their default and only error out if you change
them.

## Bring your own model

`--model` takes any Hugging Face repo id or local directory, in MLX format only:
safetensors plus a `config.json`. The backend is inferred from the id, so
`--model Qwen/Qwen3-ASR-1.7B` behaves like `--model qwen3-asr`, including its 15s window
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

Rule 5 is a guess, and a wrong guess is not caught gracefully. An id that names no
known architecture is handed to the Voxtral loader, which fails deep in the stack rather
than at the CLI boundary: an id that does not exist on the hub raises
`RepositoryNotFoundError`, and a real checkpoint of the wrong architecture raises an
`AttributeError` for a config field the model does not have. Both arrive as a Python
traceback, after the audio has been decoded and the weights downloaded. Everything else
here stops with a one-line error; this path does not, so check the id before pointing at
something exotic.

### Formats that will not work

GGUF, ONNX, ExecuTorch, OpenVINO and CoreML builds do not load: the loaders read MLX
safetensors plus `config.json` only. The precisions each model was measured at are all
reachable through a published repo or a local conversion
([benchmarks/quantization.md](benchmarks/quantization.md)).

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

## Reproducibility

`voxtral`, `voxtral-v1`, `qwen3-asr`, `parakeet` and `reazon` decode greedily and rerun byte-identically
on one machine. `whisper` samples whenever a segment trips its fallback thresholds, and
`kotoba` runs on Whisper's decoder so it samples too, which makes neither safe to rerun for
a comparison ([benchmarks/reference/determinism.md](benchmarks/reference/determinism.md)).

[^n1]: [m2-ultra-whisper-sizes.json](../benchmarks/results/2026-10/m2-ultra-whisper-sizes.json) runs `whisper_tiny_libdef`, `whisper_base_libdef`, `whisper_small_nocond`, `whisper_medium_nocond`, `whisper_large-v2_nocond`, `whisper_large-v3_nocond`, `whisper_turbo_nocond` (the runner's own peak per size, which is what the table column prints)
[^n2]: [m2-ultra-peak-memory-kotoba.json](../benchmarks/results/2026-10/m2-ultra-peak-memory-kotoba.json) runs `pm_kotoba_7f8150fdef0e`, `pm_kotoba_c18b2f5aa2a2`, `pm_kotoba_9aefd8f2352e`, `pm_kotoba_07fd66556725`, `pm_kotoba_c29657df52e9`, `pm_kotoba_cbeb19543902`, `pm_kotoba_b88caf8c0924`, `pm_kotoba_bbdabfabfb4f`, `pm_kotoba_1672083b2499`, `pm_kotoba_5b44e078553d`, `pm_kotoba_a73e94506f13`, `pm_kotoba_bb5cf5d5b247`, `pm_kotoba_0c193bf7ced2`, `pm_kotoba_57d841fe39b5`, `pm_kotoba_a773a315fb42`, `pm_kotoba_71bb5eba3baa`, `pm_kotoba_a1372c4d2900`, `pm_kotoba_c47ad127bc83`, `pm_kotoba_1da58e1b2b69`, `pm_kotoba_b3432b40c079`
[^n7]: [m2-ultra-kotoba-conversion.json](../benchmarks/results/2026-10/m2-ultra-kotoba-conversion.json) runs `pm_kotoba_cold_r112`, `pm_kotoba_warm_r112` (the 112s file through the CLI, with the first-use conversion and then with it cached)
