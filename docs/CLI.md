# Command-line flags

Flags that work the same on every model, and how a `--language` value is read. Which
flags each model takes: [MODELS.md](MODELS.md).

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
