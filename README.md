# mlx-asr

One CLI over three speech-to-text engines on Apple Silicon. Point it at a file, get
a subtitle file. Runs locally, with sensible defaults per machine.

## Install

Apple Silicon, macOS 14+.

```bash
brew install prog893/tap/mlx-asr
```

Weights download on first use into `~/.cache/huggingface`, about 1.5GB for the
default model.

<details>
<summary>From source, for development or to run the benchmarks</summary>

Python 3.12+, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/prog893/mlx-asr && cd mlx-asr
uv sync                                   # Voxtral only
uv sync --extra whisper --extra eval      # + the whisper-* and kotoba engines
uv run mlx-asr audio.wav
```

| extra | pulls | needed for |
|---|---|---|
| `whisper` | mlx-whisper, numba | the `whisper-*` aliases and `--model kotoba` |
| `vad` | onnxruntime | `--vad` |
| `eval` | pykakasi, rapidfuzz | the `evals/` scorers and benchmark scripts |
| `dev` | pytest | the test suite |

The `whisper` extra caps numpy at `<2.5`. That is load-bearing: mlx-whisper needs
numba, numba caps numpy, and without the cap a resolver takes the newest numpy and
then backtracks numba to a 2021 release that refuses to build on anything above
Python 3.9, failing with an error that names llvmlite rather than numpy.

The Homebrew formula lives in [packaging/](packaging/).

</details>

## Usage

```bash
mlx-asr interview.m4a                     # writes interview.srt next to it
mlx-asr talk.mp4 -f json                  # timestamps + metadata as JSON
mlx-asr talk.mp4 -f all -o out/talk       # srt, vtt, txt and json together
mlx-asr lecture.mp4 --fast -f vtt         # faster, slightly less accurate
mlx-asr earnings.wav --prompt "EBITDA, ARR, Grafana"   # bias toward domain terms
mlx-asr interview.m4a --model whisper-turbo --language ja
mlx-asr --list-models
```

Input can be anything ffmpeg reads: any container, codec, sample rate, bit depth or
channel count, including video files.

Default parameters are picked based on machine. If your chip and RAM are in
[mlx_asr/profiles.json](mlx_asr/profiles.json), they come from measurements on that
machine; otherwise they are derived from GPU cores and memory, and the CLI prints
which of the two it used. If it says `derived`, you can contribute a profile:
[CONTRIBUTING.md](CONTRIBUTING.md).

These work on every engine:

| flag | default | what it does |
|---|---|---|
| `--model ALIAS_OR_REPO` | `voxtral` | which engine, by alias or Hugging Face repo id |
| `-f, --output-format` | `srt` | `srt`, `vtt`, `txt`, `json`, or `all` |
| `-o, --output PATH` | input stem | output path; with `-f all` it is a path stem |
| `--stats-json PATH` | off | write timing, resolved config and machine info |
| `--quiet` | off | silence progress output |
| `--list-models` | | print the engines and their caveats, then exit |

The rest belong to one engine. Pass one the current engine cannot use and it says
so:

```console
$ mlx-asr audio.wav --model whisper-turbo --max-batch 32 --fast
[mlx-whisper] ignoring Voxtral-only flags: --max-batch, --fast
```

`--prompt` is one of those, Voxtral only. It takes domain terms or a topic sentence,
not an instruction: the decoder treats it as text it already emitted, so "transcribe
accurately" makes things worse.

Reference: [docs/MODELS.md](docs/MODELS.md).

## The engines

| `--model` | what it is |
|---|---|
| `voxtral` (default) | Mistral's 2026 realtime model. Takes a vocabulary prompt, decodes greedily so reruns are identical, and has the steadiest timestamps here |
| `whisper-*` | OpenAI's Whisper in eight sizes, `tiny` through `large-v3`. The best-understood option, and the most accurate on our corpus |
| `kotoba` | kotoba-whisper: Whisper large-v3 distilled down to 2 decoder layers, then finetuned on Japanese. Fast, and Japanese only |
| any HF repo id | the backend is inferred from the name |

The `whisper-*` aliases do better when you set `--language`. Voxtral takes no
language flag, and `kotoba` forces Japanese on its own.

What each model scored: [docs/MODELS.md](docs/MODELS.md).

## License

MIT, see [LICENSE](LICENSE). Model weights are not redistributed here and follow
their own licenses; `--list-models` prints the repo id for every alias.
