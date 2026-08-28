# Why the defaults are what they are

Every default in [MODELS.md](MODELS.md) and what decided it. Several went against the
obvious choice, which is why they are written down rather than left in the code. Each row
links to the document holding the numbers, the corpus and the method; nothing here restates
them, so that there is one place per finding to keep correct.

## How much to trust each row

The rows below do not rest on equal evidence, and the difference has bitten: `voxtral 4bit`
was justified for weeks by a single-clip tie that the corpus later contradicted. So each
default falls into one of three tiers.

| tier | what it means | rows |
|---|---|---|
| **decided by accuracy** | swept on the multi-file corpus, and the winner beat the alternatives by more than the corpus can confuse | model choice, `whisper --size`, `qwen3-asr --size`, `voxtral` precision, `qwen3-asr --chunk-seconds`, `--prompt` |
| **decided by cost** | swept on the corpus, but the arms tie on accuracy, so the default was chosen on speed, memory or dependencies | `voxtral --chunk-seconds`, `--max-batch`, `--overlap-seconds`, `--delay-ms`, `--gain`, `--vad`, `--compact-silence`, `--kv-bits` |
| **not yet on the corpus** | one recording only, or not measured at all, so unverified on the material this project targets | cue layout, which needs a second reference set that does not exist |

That last tier is where surprises come from, and it has emptied out considerably. `--vad`,
`--compact-silence` and `--kv-bits` all sat there until the corpus shrank their margins to
nothing, and Voxtral's precision default was outright *reversed* by moving off one clip. Not
one clip-only claim tested so far has survived intact. See
[benchmarks/metrics.md](benchmarks/metrics.md) for why a null result on one clip cannot be
used to argue an effect is small.

## Model and size

| default | why | detail |
|---|---|---|
| `voxtral` as the model | faster than whisper turbo, needs no language flag, and reruns byte-identically on one machine. Turbo is slightly more accurate on Japanese, so this trades a little accuracy for speed and reproducibility | [engines.md](benchmarks/engines.md) |
| `whisper --size turbo` | ties `large-v3` on Japanese at roughly twice the speed and less memory, so picking by size number would pay for nothing | [engines.md](benchmarks/engines.md) |
| `qwen3-asr --size 1.7B` | clearly ahead of `0.6B` on accuracy. The `0.6B` is the speed option, and the fastest engine measured here | [qwen3-asr.md](benchmarks/qwen3-asr.md) |
| `parakeet --chunk-seconds 120` | clearly ahead of 300s on the corpus, and cheaper on memory. A one-file pre-check had shown a tie, which reversed at corpus scale. 60s is excluded because it drops content outright | [japanese-only.md](benchmarks/japanese-only.md) |
| `reazon --chunk-seconds 30` | not a sweep: the weights cannot decode a whole file at all (they are trained on short VAD segments), so windows are mandatory, and 30s matches the Voxtral rows to hold the front end constant | [japanese-only.md](benchmarks/japanese-only.md) |

## Precision

One default per model, chosen as the cheapest precision whose accuracy cost is worth its
price. **Measured per model, never inherited between them:** assuming one model's precision
behaviour carries to another is how the Voxtral default came to be justified wrongly for
weeks.

| default | why | detail |
|---|---|---|
| `voxtral` 4bit | **not the most accurate option, deliberately.** Accuracy improves with bit width and 4bit is last of the five measured, but fp16 will not fit a 16GB machine and no loadable 8bit build is published, so 4bit is what works everywhere. If you have the memory, pass `--quantization fp16`, or convert 8bit locally to get fp16's accuracy at roughly half its memory | [quantization.md](benchmarks/quantization.md) |
| `qwen3-asr` 8bit | the full ladder is now swept on both sizes, and they disagree: the `1.7B` ties across every precision, while the `0.6B` is sharply sensitive and 8bit wins each comparison outright. One default suits both, and on the `1.7B` a lower rung is a free speed option | [qwen3-asr.md](benchmarks/qwen3-asr.md) |
| `reazon` fp32 | the publisher's table says int8 is near-parity, but on conversational material int8 drops whole phrases mid-file. Read-speech benchmarks and real audio disagree, and the corpus wins | [japanese-only.md](benchmarks/japanese-only.md) |
| `--kv-bits 8` | a tie with unquantized on the corpus, and faster, so the speed is free | [quantization.md](benchmarks/quantization.md) |

`qwen3-asr`'s ladder below 8bit is exposed but has no accuracy figure at all, so going lower
there is a memory choice made blind. Its memory cost is measured
([peak-memory.md](benchmarks/peak-memory.md)).

## Chunking and windows

| default | why | detail |
|---|---|---|
| `voxtral --chunk-seconds` per machine | 30s and 60s tie on accuracy, so this is purely a throughput choice and belongs to the hardware profile | [chunking.md](benchmarks/chunking.md) |
| `voxtral --max-batch` per machine | throughput is not monotonic in batch size, so the profile avoids the middle of the range entirely | [decode-throughput.md](benchmarks/decode-throughput.md) |
| `voxtral --overlap-seconds 0` | won on one clip at short chunks, then reversed sign on the corpus. Tied to `--fast` rather than defaulted on | [chunking.md](benchmarks/chunking.md) |
| `kotoba --chunk-seconds 10` | this model's largest lever, and 10s clearly won on spontaneous Japanese. Material-dependent, so sweep it on your own audio | [engines.md](benchmarks/engines.md) |
| `qwen3-asr --chunk-seconds 30` | unusually, shorter is better on accuracy, speed and memory at once, because a longer window gives a repetition loop a bigger token budget to burn. 15s ties 30s, so this is a plateau rather than a boundary | [qwen3-asr.md](benchmarks/qwen3-asr.md) |

The library default for `qwen3-asr` is 1200s, so anything shorter than 20 minutes becomes a
single window: one segment, and the batched path never engages.

## Audio and prompt

| default | why | detail |
|---|---|---|
| `--delay-ms 2400` | the largest accuracy lever here, and it costs no speed. Also the model's maximum | [delay.md](benchmarks/delay.md) |
| `--gain auto` | quiet input silently loses detail, because the mel front end clamps at an absolute floor. `auto` boosts only quiet audio, so it is byte-identical on healthy input | [input-level.md](benchmarks/input-level.md) |
| `--vad` off | a tie with energy cut points on the corpus, and it costs an `onnxruntime` dependency. The single-clip result that looked decisive did not reproduce | [chunking.md](benchmarks/chunking.md) |
| `--prompt` empty | it biases register rather than recalling vocabulary, an instruction there actively hurts, and on English audio any prompt wrecks word spacing | [prompt.md](benchmarks/prompt.md) |
| `--compact-silence` off | a tie on accuracy at every precision tested, and slightly faster. Off because it discards audio, which should be opted into rather than inherited, not because it costs accuracy | [chunking.md](benchmarks/chunking.md) |

## Whisper long-form stability

`condition_on_previous_text=False` on `small` and larger, not exposed as a flag.

Leaving it on lets a repetition loop feed itself across 30s windows, and it is by far the
largest artifact in this project's history: at library defaults `large-v3` scores badly
enough to read as a broken model rather than as a bad setting
([engines.md](benchmarks/engines.md) has both numbers).

`tiny` and `base` keep the library default, since they are not used for long-form work
here.

## Cue layout

`--gap-seconds 1.2`, `--max-chars 28`, `--max-dur-seconds 7.0`.

Deliberately **not** the optimum of either sweep that was run. Both sweeps optimised
agreement with one editor's subtitle conventions, and no second reference set exists to
check that against, so adopting their optimum would fit this corpus's house style rather
than anything general ([cue-layout.md](benchmarks/cue-layout.md)).
