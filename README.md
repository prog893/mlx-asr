# mlx-asr

One CLI over seven speech-to-text engines on Apple Silicon, and a comparative study of ASR
models focused on Japanese.

Point it at a file, get a subtitle file. Runs locally, with sensible defaults per machine.
Or explore the comparison results: [RESULTS.md](RESULTS.md).

## Install

```bash
brew install prog893/tap/mlx-asr
```

<details>
<summary>From source, for development or to run the benchmarks</summary>

Python 3.12+, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/prog893/mlx-asr && cd mlx-asr
uv sync                                   # Voxtral only
uv sync --extra whisper --extra eval      # + the whisper and kotoba engines
uv run mlx-asr audio.wav
```

| extra | pulls | needed for |
|---|---|---|
| `whisper` | mlx-whisper, numba | `--model whisper` and `--model kotoba` |
| `reazon` | sherpa-onnx | `--model reazon`. Note: on Python 3.14 there is no prebuilt wheel yet, and the source build does not bundle libonnxruntime; copy it from the `onnxruntime` wheel into `sherpa_onnx/lib/libonnxruntime.dylib` if the import fails |
| `vad` | onnxruntime | `--vad` |
| `eval` | pykakasi, rapidfuzz | the `scripts/metrics/` scorers and benchmark scripts |
| `dev` | pytest | the test suite |

The `whisper` extra caps numpy at `<2.5`. That is load-bearing: mlx-whisper needs
numba, numba caps numpy, and without the cap a resolver takes the newest numpy and
then backtracks numba to a 2021 release that refuses to build on anything above
Python 3.9, failing with an error that names llvmlite rather than numpy.

The Homebrew formula lives in [scripts/homebrew/](scripts/homebrew/).

</details>

## Usage

```bash
mlx-asr interview.m4a                     # writes interview.srt next to it
mlx-asr talk.mp4 -f json                  # timestamps + metadata as JSON
mlx-asr talk.mp4 -f all -o out/talk       # srt, vtt, txt and json together
mlx-asr earnings.wav --prompt "EBITDA, ARR, Grafana"   # Voxtral only: bias toward domain terms
mlx-asr interview.m4a --model whisper --language ja      # whisper uses large-v3 by default
mlx-asr interview.m4a --model whisper --size small --language ja   # override the size
mlx-asr --list-models
```

Input is anything libavformat demuxes: wav, flac, mp3, aac/m4a, alac, opus, or the audio
track of a video, at any sample rate or bit depth. It gets downmixed to mono and
resampled to 16kHz before the model sees it. Details in [docs/AUDIO.md](docs/AUDIO.md).

Every flag, and which engine accepts it: [docs/MODELS.md](docs/MODELS.md).

## The engines

| `--model` | what it is |
|---|---|
| `voxtral` (default) | Mistral's 2026 realtime model, and the fastest multilingual engine measured here that writes subtitles. Takes a vocabulary prompt, decodes greedily so reruns on one machine are byte-identical, and has the steadiest timestamps here |
| `voxtral-v1` | Mistral's first-generation Voxtral (2025). `--size 24B` or `3B`: the 24B loads where it fits, the 3B on 16 GB Macs. At 24B, the lowest English error measured here; weak on Japanese, which its card does not claim. Greedy. **Writes no subtitles**, like `qwen3-asr` |
| `whisper` | OpenAI's Whisper. `--size tiny base small medium large-v2 large-v3 turbo`, defaulting to **large-v3**; turbo ties it on accuracy and runs faster. The lowest Japanese error measured here, with Voxtral close behind; on English both voxtral-v1 sizes score lower. It samples, so reruns differ |
| `kotoba` | kotoba-whisper: Whisper large-v3 distilled down to 2 decoder layers, then finetuned on Japanese. Fast, Japanese only, and samples like Whisper |
| `qwen3-asr` | Alibaba's Qwen3-ASR. `--size 1.7B` (default) or `0.6B`, the fastest multilingual engine measured here. Greedy, so reproducible. **Writes no subtitles**: it emits no timestamp finer than its own decode window, so `-f srt` and `-f vtt` are refused and only `txt` and `json` work |
| `parakeet` | NVIDIA's Japanese FastConformer-TDT, through mlx-audio. Greedy, token-level timestamps so subtitles work. The fastest engine measured here (Japanese only), well behind the multilingual defaults on this corpus's Japanese |
| `reazon` | ReazonSpeech k2-v2 (Japanese Zipformer), the authors' ONNX build via sherpa-onnx. Decodes on CPU; needs `uv sync --extra reazon`. Japanese only |
| any HF repo id | the backend is inferred from the name; `--size` and `--quantization` are refused, since the id already names the variant |

`--model` picks the family; `--size` and `--quantization` pick the variant inside it. Size
matters far more than precision. Each family's variants are ranked by measured accuracy,
the larger first within a tie, and the CLI loads the first one that fits the Mac's memory,
so one `--model` can run a different variant on a 16 GB Mac than on a 128 GB one
([how a variant is chosen](docs/DEFAULTS.md#how-a-variant-is-chosen)). For example, on Voxtral
[higher precision scores better](docs/benchmarks/quantization.md#experiment-the-weight-precision-ladder), so it runs fp16 where it fits and 8-bit on 16 GB Macs.

`whisper`, `qwen3-asr` and `voxtral-v1` do better when you set `--language`, and each gets the form it
wants (a code for Whisper and Voxtral v1, an English name for Qwen) from whatever you type. Voxtral takes
no language flag, and the Japanese-only engines (`kotoba`, `parakeet`, `reazon`) refuse
`--language` values other than Japanese rather than silently ignoring them.

Every `--model`, `--size` and `--quantization` combination, with its Hugging Face repo,
download size and peak GPU memory, is listed in [docs/MODELS.md](docs/MODELS.md#the-models).
`--list-models` prints the sizes and precisions each family accepts.

The built-in families load weights only from the model authors' own repos and from
mlx-community. `kotoba` and `voxtral-v1` are converted to MLX on your machine on first use,
from the authors' weights. `reazon` runs the authors' ONNX files on CPU.

What each model scored: [docs/benchmarks/engines.md](docs/benchmarks/engines.md).

## Defaults

The aim is that `mlx-asr file.wav` is good enough with no other flag. The defaults were
chosen by measuring the alternatives, including model sizes and
precisions, on our own corpus of Japanese and English recordings. New models get the same
treatment as they are added.

Actual values used depend on the hardware. Each family's variants are ranked by measured accuracy, and the
CLI loads the best one that fits in memory. Other parameters come from a measured profile if
your chip and RAM are in [mlx_asr/profiles.json](mlx_asr/profiles.json); otherwise they are
derived from GPU cores and memory, and the CLI prints which of the two it used. To contribute a profile for your own Mac by
running the benchmark yourself, see [CONTRIBUTING.md](CONTRIBUTING.md).

Each default and how strong the evidence behind it is: [docs/DEFAULTS.md](docs/DEFAULTS.md). All results:
[RESULTS.md](RESULTS.md).

## License

[MIT](LICENSE)

Model weights are not redistributed here and follow their own licenses. The Hugging Face
repo of every model, where its license is stated, is listed in
[docs/MODELS.md](docs/MODELS.md#the-models).
