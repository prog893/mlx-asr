# Command-line flags

What works the same on every model: the shared flags, how a `--language` value is read,
input, output files, exit codes and where weights are cached. Which flags each model takes:
[MODELS.md](MODELS.md).

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

## Input

One file per run, in any format libavformat reads, including the audio track of a video.
It is downmixed to mono and resampled to 16kHz before the model sees it. Tested formats
and decoders: [AUDIO.md](AUDIO.md).

## Output files

Without `-o`, each file is written next to the input, named after it: `talk.mp4` gives
`talk.srt`. `-o` sets the path instead; with `-f all` it sets the name without the
extension, so `-o out/talk -f all` writes `out/talk.srt`, `out/talk.vtt`, `out/talk.txt`
and `out/talk.json`. Missing parent directories are created. An existing file of the same
name is overwritten.

`-f json` writes the transcript, its segments and run metadata:

```json
{
  "text": "the whole transcript",
  "segments": [{"id": 0, "start": 0.0, "end": 2.48, "text": "first segment"}],
  "meta": {"model": "mlx-community/...", "backend": "voxtral", "duration_s": 60.0,
           "wall_s": 4.1, "x_realtime": 14.6}
}
```

Times are in seconds. Every model writes the `meta` keys shown; some add their own, such as
`cue_source` on `qwen3-asr`.

## Exit codes

| code | meaning |
|---|---|
| `0` | transcribed |
| `1` | the input could not be read as audio |
| `2` | the command was refused before transcribing: an unsupported flag, or an unknown model, size, precision, language or output format |
| `130` | interrupted |

## Downloads and cache

The first run of a model downloads its weights into the Hugging Face cache
(`~/.cache/huggingface/hub/` by default). Builds that are quantized or converted on first
use are written beside them, in `mlx-asr-converted/`; delete a build's directory there and
the next run makes it again.
