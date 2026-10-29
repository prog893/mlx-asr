# mlx-asr

One CLI over seven speech-to-text engines on Apple Silicon. Point it at a file, get
a subtitle file. Runs locally, with sensible defaults per machine.

The tuning and measurement here are aimed at **Japanese**: the defaults were chosen
against Japanese audio, the accuracy work uses character-level metrics suited to a
script with no word boundaries, and three engines (`kotoba`, `parakeet`, `reazon`) are
Japanese-only. English is
measured and works; other languages are inherited from the upstream models and
untested here. See [docs/benchmarks/](docs/benchmarks/).

## Install

Apple Silicon, macOS 14+.

```bash
brew install prog893/tap/mlx-asr
```

Weights download on first use into `~/.cache/huggingface`, about 9GB for the default <!-- not-measured: Hugging Face repo size of the default weights, not a run -->
model: the fp16 Voxtral build, which a 16GB Mac converts to 8bit once.

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
mlx-asr earnings.wav --prompt "EBITDA, ARR, Grafana"   # bias toward domain terms
mlx-asr interview.m4a --model whisper --language ja        # large-v3 by default
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
$ mlx-asr audio.wav --model whisper --max-batch 32 --vad
error: --max-batch, --vad: not supported by --model whisper-large-v3. These are
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
| `voxtral` (default) | Mistral's 2026 realtime model, and the fastest multilingual engine measured here that writes subtitles (23.9x realtime on the M2 Ultra, where it loads fp16; 16 GB Macs get 8-bit). Takes a vocabulary prompt, decodes greedily so reruns on one machine are byte-identical, and has the steadiest timestamps here |
| `voxtral-v1` | Mistral's first-generation Voxtral (2025). `--size 24B` or `3B`: the 24B loads where it fits, the 3B on 16 GB Macs. At 24B, the lowest English error measured here (on three files); weak on Japanese, which its card does not claim. Greedy. **Writes no subtitles**, like `qwen3-asr` |
| `whisper` | OpenAI's Whisper. `--size tiny base small medium large-v2 large-v3 turbo`, defaulting to **large-v3**; turbo ties it on accuracy at about 1.8x the speed. The lowest Japanese error measured here (turbo 14.17% CER), though on the M2 Ultra Voxtral's fp16 default is within the interval of it (+1.03 points, 95% CI [-0.11, +2.25]); on English both voxtral-v1 sizes score lower (24B 16.86% WER against turbo's 18.65%). It samples, so reruns differ |
| `kotoba` | kotoba-whisper: Whisper large-v3 distilled down to 2 decoder layers, then finetuned on Japanese. Fast, Japanese only, and samples like Whisper |
| `qwen3-asr` | Alibaba's Qwen3-ASR. `--size 1.7B` (default) or `0.6B`, the speed option at 35.5x, the fastest multilingual engine measured here. Greedy, so reproducible. **Writes no subtitles**: it emits no timestamp finer than its own decode window, so `-f srt` and `-f vtt` are refused and only `txt` and `json` work |
| `parakeet` | NVIDIA's Japanese FastConformer-TDT, through mlx-audio. Greedy, token-level timestamps so subtitles work. The fastest engine measured here (261.8x realtime, Japanese only), well behind the multilingual defaults on this corpus's Japanese |
| `reazon` | ReazonSpeech k2-v2 (Japanese Zipformer), the authors' ONNX build via sherpa-onnx. Decodes on CPU (52.9x realtime); needs `uv sync --extra reazon`. Japanese only |
| any HF repo id | the backend is inferred from the name; `--size` and `--quantization` are refused, since the id already names the variant |

Source: [ultra-whisper-sizes.json](benchmarks/results/2026-10/ultra-whisper-sizes.json) runs `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`, `whisper_turbo_nocond_r6`, `whisper_large-v3_nocond`, `whisper_large-v3_nocond_r2`, `whisper_large-v3_nocond_r3`, `whisper_large-v3_nocond_r4`, `whisper_large-v3_nocond_r5`, `whisper_large-v3_nocond_r6`; [ultra-picker-speed.json](benchmarks/results/2026-10/ultra-picker-speed.json) runs `pk_parakeet_default_s1`, `pk_parakeet_default_s2`, `pk_parakeet_default_s3`, `pk_qwen3_1.7B_8bit_c15_s1`, `pk_qwen3_1.7B_8bit_c15_s2`, `pk_qwen3_1.7B_8bit_c15_s3`, `pk_qwen3_0.6B_8bit_c15_s1`, `pk_qwen3_0.6B_8bit_c15_s2`, `pk_qwen3_0.6B_8bit_c15_s3`, `pk_v1_3B_8bit_c30_s1`, `pk_v1_3B_8bit_c30_s2`, `pk_v1_3B_8bit_c30_s3`, `pk_kotoba_c10_s1`, `pk_kotoba_c10_s2`, `pk_kotoba_c10_s3`, `pk_reazon_fp32_default_s1`, `pk_reazon_fp32_default_s2`, `pk_reazon_fp32_default_s3`; [ultra-picker-speed-24b.json](benchmarks/results/2026-10/ultra-picker-speed-24b.json) runs `pk_v1_24B_8bit_c30_s1`, `pk_v1_24B_8bit_c30_s2`, `pk_v1_24B_8bit_c30_s3`; [ultra-voxtral-precision-default.json](benchmarks/results/2026-10/ultra-voxtral-precision-default.json) runs `vox_wfp16_c30b128_kv8`, `vox_wfp16_c30b128_kv8_r2`, `vox_wfp16_c30b128_kv8_r3`

`--model` picks the family; `--size` and `--quantization` pick the variant inside it. Size
matters far more than precision. Each family's variants are ranked by measured accuracy,
the larger first within a tie, and the CLI loads the first one that fits the Mac's memory,
so one `--model` can run a different variant on a 16 GB Mac than on a 128 GB one
([how a variant is chosen](docs/DEFAULTS.md#how-a-variant-is-chosen)). On Voxtral, higher
precision scores better, so it runs fp16 where it fits and 8-bit on 16 GB Macs.

`whisper`, `qwen3-asr` and `voxtral-v1` do better when you set `--language`, and each gets the form it
wants (a code for Whisper and Voxtral v1, an English name for Qwen) from whatever you type. Voxtral takes
no language flag, and the Japanese-only engines (`kotoba`, `parakeet`, `reazon`) refuse
`--language` values other than Japanese rather than silently ignoring them.

`--list-models` prints the sizes and precisions each family accepts. The complete
mapping from every `--model`/`--size`/`--quantization` combination to its Hugging Face
repo id, with download size and measured peak GPU memory, is in
[docs/MODELS.md](docs/MODELS.md#the-models); it is generated from the registry, not
hand-maintained.

All weights are `mlx-community` MLX builds, except `kotoba` and `voxtral-v1`, which use the
authors' own weights and convert them locally on first use, and `reazon`, which runs the
authors' ONNX files. **unsloth and GGUF quants cannot be used here**: unsloth
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

The short version: transcription delay is the biggest lever and it is free; per-step decode
time is not monotonic in batch size (batches of 2 to 8 take longer per step than one at a
time, though they still finish sooner: 12.5x to 25.9x against 7.2x on the M2 Ultra[^m2]), so
the batch is snapped to sizes that measured well; VAD
cut points and dropping silence both turned out to change nothing measurable, so both stay
off; higher precision does help Voxtral, so it runs fp16 where memory allows and 8-bit on
16 GB Macs; and whisper defaults to large-v3 although turbo ties it at 1.8x[^m1] the
speed, because a tie on 20 files is not evidence the smaller decoder is safe elsewhere.

## License

MIT, see [LICENSE](LICENSE). Model weights are not redistributed here and follow
their own licenses; `--list-models` prints the repo id for every model.

[^m1]: [ultra-whisper-sizes.json](benchmarks/results/2026-10/ultra-whisper-sizes.json) runs `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`, `whisper_turbo_nocond_r6`, `whisper_large-v3_nocond`, `whisper_large-v3_nocond_r2`, `whisper_large-v3_nocond_r3`, `whisper_large-v3_nocond_r4`, `whisper_large-v3_nocond_r5`, `whisper_large-v3_nocond_r6`
[^m2]: [ultra-synthetic-decode.json](benchmarks/results/2026-10/ultra-synthetic-decode.json) runs `synth_ultra_4bit_r1`, `synth_ultra_4bit_r2`, `synth_ultra_4bit_r3` (synthetic decode, mean of 3 runs per batch size)
