# mlx-asr

One CLI over four speech-to-text engines on Apple Silicon. Point it at a file, get
a subtitle file. Runs locally, with sensible defaults per machine.

The tuning and measurement here are aimed at **Japanese**: the defaults were chosen
against Japanese audio, the accuracy work uses character-level metrics suited to a
script with no word boundaries, and one engine (`kotoba`) is Japanese-only. English is
measured and works; other languages are inherited from the upstream models and
untested here. See [docs/benchmarks/](docs/benchmarks/).

## Install

Apple Silicon, macOS 14+.

```bash
brew install prog893/tap/mlx-asr
```

Weights download on first use into `~/.cache/huggingface`, about 3GB for the
default model.

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
mlx-asr lecture.mp4 --fast -f vtt         # faster, slightly less accurate
mlx-asr earnings.wav --prompt "EBITDA, ARR, Grafana"   # bias toward domain terms
mlx-asr interview.m4a --model whisper --language ja        # turbo by default
mlx-asr interview.m4a --model whisper --size small --language ja
mlx-asr --list-models
```

Input is anything libavformat demuxes: wav, flac, mp3, aac/m4a, alac, opus, or the audio
track of a video, at any sample rate or bit depth. It gets downmixed to mono and
resampled to 16kHz before the model sees it. Details in [docs/AUDIO.md](docs/AUDIO.md).

Default parameters are picked based on machine. If your chip and RAM are in
[mlx_asr/profiles.json](mlx_asr/profiles.json), they come from measurements on that
machine; otherwise they are derived from GPU cores and memory, and the CLI prints
which of the two it used. If it says `derived`, you can contribute a profile:
[CONTRIBUTING.md](CONTRIBUTING.md).

These work on every engine:

| flag | default | what it does |
|---|---|---|
| `--model NAME_OR_REPO` | `voxtral` | which model family, or a Hugging Face repo id |
| `--size SIZE` | per family | which size within the family (`whisper`, `qwen3-asr`); errors on families with one |
| `--quantization PRECISION` | per model | weight precision, where more than one build is published |
| `-f, --output-format` | `srt` | `srt`, `vtt`, `txt`, `json`, or `all` |
| `-o, --output PATH` | input stem | output path; with `-f all` it is a path stem |
| `--stats-json PATH` | off | write timing, resolved config and machine info |
| `--quiet` | off | silence progress output |
| `--list-models` | | print the engines and their caveats, then exit |

The rest belong to one engine. Pass one the current engine cannot use and it is an
error, not a warning: nothing is silently ignored.

```console
$ mlx-asr audio.wav --model whisper --max-batch 32 --fast
error: --max-batch, --fast: not supported by --model whisper. These are
Voxtral-only, because the engines do not share a long-form algorithm. Drop the
flag, or use the default --model voxtral.
```

The same applies to output formats an engine cannot honestly produce, so `-f srt` on
`qwen3-asr` exits 2 rather than writing a file whose cue times are decode-window
boundaries.

`--prompt` is one of those, Voxtral only. It takes domain terms or a topic sentence in the
same language as the audio, not an instruction: the decoder treats it as text it already
emitted, so "transcribe accurately" makes things worse. On English audio any prompt wrecks
word spacing, so leave it empty there.

Reference: [docs/MODELS.md](docs/MODELS.md).

## The engines

| `--model` | what it is |
|---|---|
| `voxtral` (default) | Mistral's 2026 realtime model. Takes a vocabulary prompt, decodes greedily so reruns on one machine are byte-identical, and has the steadiest timestamps here |
| `whisper` | OpenAI's Whisper. `--size tiny base small medium large-v2 large-v3 turbo`, defaulting to **turbo**, which ties large-v3 on accuracy at about 2x the speed. The most accurate engine on the test corpus, but it samples, so reruns differ |
| `kotoba` | kotoba-whisper: Whisper large-v3 distilled down to 2 decoder layers, then finetuned on Japanese. Fast, Japanese only, and samples like Whisper |
| `qwen3-asr` | Alibaba's Qwen3-ASR. `--size 1.7B` (default) or `0.6B`, the fastest engine measured here. Greedy, so reproducible. **Writes no subtitles**: it emits no timestamp finer than its own decode window, so `-f srt` and `-f vtt` are refused and only `txt` and `json` work |
| any HF repo id | the backend is inferred from the name; `--size` and `--quantization` are refused, since the id already names the variant |

`--model` picks the family; `--size` and `--quantization` pick the variant inside it. Size
matters far more than precision, by a wide margin, and both defaults are picked on
measurement rather than on the largest number. Voxtral's 4-bit default is a deliberate
trade: higher precisions do score better, but each needs more time or more memory than a
16GB machine has.

`whisper` and `qwen3-asr` do better when you set `--language`, and each gets the form it
wants (a code for Whisper, an English name for Qwen) from whatever you type. Voxtral takes
no language flag, and `kotoba` forces Japanese on its own.

`--list-models` prints the sizes and precisions each family accepts. The complete
mapping from every `--model`/`--size`/`--quantization` combination to its Hugging Face
repo id, with download size and measured peak GPU memory, is in
[docs/MODELS.md](docs/MODELS.md#the-models); it is generated from the registry, not
hand-maintained.

All weights are `mlx-community` MLX builds, except `kotoba` which converts the authors'
own weights locally on first use. **unsloth and GGUF quants cannot be used here**: unsloth
publishes no Voxtral, only GGUF for Qwen3-ASR, and unquantized transformers weights for
Whisper, while GGUF is llama.cpp's format that MLX cannot load. That is a format
constraint rather than a quality judgement
([why](docs/MODELS.md#formats-that-will-not-work)).

What each model scored: [docs/benchmarks/engines.md](docs/benchmarks/engines.md).

## Why the defaults are what they are

Every default was measured, and several went against the obvious choice:
[docs/DEFAULTS.md](docs/DEFAULTS.md) lists each one with the number behind it. The full
findings are in [docs/benchmarks/](docs/benchmarks/), one document per lever, indexed by
[RESULTS.md](RESULTS.md).

The short version: transcription delay is the biggest lever and it is free; throughput is
not monotonic in batch size, so the middle of the range is worse than one at a time; VAD
cut points and dropping silence both turned out to change nothing measurable, so both stay
off; higher precision does help Voxtral but costs more time or memory than a 16GB machine
has; and whisper defaults to turbo rather than large-v3 because it ties it at twice the
speed.

## License

MIT, see [LICENSE](LICENSE). Model weights are not redistributed here and follow
their own licenses; `--list-models` prints the repo id for every model.
