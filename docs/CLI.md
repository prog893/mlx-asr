# Command-line flags

The flags every model accepts, and how `--language` is read. The flags each model family
takes, and which ones each refuses, are in [MODELS.md](MODELS.md).

## Global flags

The only flags every model accepts.

| flag | values | default | notes |
|---|---|---|---|
| `--model` | family name or HF repo id | `voxtral` | a repo id gets its backend inferred from the name |
| `--list-models` | flag | off | prints each model with its sizes, precisions and caveats, then exits |
| `-f, --output-format` | `srt` `vtt` `txt` `json` `all` | `srt` | `all` writes one file per format. `srt`, `vtt` and `all` are an error on `qwen3-asr` and `voxtral-v1` |
| `-o, --output` | path | input stem + extension | with `-f all` this is a path stem, not a directory; parent dirs are created |
| `--quiet` | flag | off | silences stdout entirely. Download progress bars still appear: those are huggingface_hub writing to stderr |
| `--stats-json` | path | none | timing, peak memory, resolved config, and machine info |
| `-h, --help` | flag | | |

## `--language`

Recommended whenever the engine takes one. Whisper's autodetect reads only the first 30
seconds, and on this project's material it returned Russian for Japanese audio and cost 25[^n4]
CER points. `qwen3-asr` does not autodetect at all: omitting the flag forces English.
`voxtral-v1` detects the language when the flag is omitted, and on Japanese that sends
whole windows out in English or Turkish instead
([benchmarks/engines/voxtral-v1.md](benchmarks/engines/voxtral-v1.md#experiment-leaving-the-language-to-the-model)),
so pass it.

Any spelling works on any model that takes one. These are all Japanese:

```bash
mlx-asr a.wav --model whisper   --language ja
mlx-asr a.wav --model whisper   --language jpn
mlx-asr a.wav --model qwen3-asr --language Japanese -f txt
mlx-asr a.wav --model qwen3-asr --language ja_JP    -f txt
```

Each model wants a different form internally (whisper an ISO 639-1 code, qwen3-asr an
English name) and the CLI converts, so you never have to know which. Region and script
subtags are dropped: `en-US` and `en-GB` both give `en`. An unrecognised value is an error
before the audio is read, listing what that model accepts.

Accepted names come from each checkpoint's own published vocabulary, so a model that drops
a language stops accepting it without a change here.

Two models refuse the flag, for different reasons. `voxtral` has no language token to set:
its tokenizer carries a thousand special tokens and not one of them names a language or a
task, unlike Whisper's `<|ja|>`. `kotoba` has the tokens but is finetuned on Japanese
alone, so it is pinned to `ja` and says so.

[^n4]: [m2-ultra-whisper-language.json](../benchmarks/results/2026-10/m2-ultra-whisper-language.json) runs `lang7_turbo_autodetect_r1`, `lang7_turbo_autodetect_r2`, `lang7_turbo_autodetect_r3`; [m2-ultra-mincut-inputs.json](../benchmarks/results/2026-10/m2-ultra-mincut-inputs.json) runs `mincut7_whisper_turbo_libdef`; [m2-ultra-whisper-condition.json](../benchmarks/results/2026-10/m2-ultra-whisper-condition.json) runs `cond7_whisper_turbo_libdef_r2`, `cond7_whisper_turbo_libdef_r3` (autodetect against forced Japanese, both at library defaults, mean of 3 runs each)
