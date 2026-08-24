# Why the defaults are what they are

Every default in [MODELS.md](MODELS.md) with the measurement behind it. Several went
against the obvious choice, which is why they are written down rather than left in the
code. Full method and corpus per finding: [benchmarks/](benchmarks/).

## How much to trust each row

The rows below do not rest on equal evidence, and the difference has bitten once already:
`voxtral 4bit` was justified for weeks by a single-clip tie that the corpus later
contradicted by 1.30 points. So each default falls into one of three tiers.

| tier | what it means | rows |
|---|---|---|
| **corpus, resolved** | 20 files, paired bootstrap, CI excludes zero | model choice, `whisper --size`, `qwen3-asr --size`, `voxtral` precision, `qwen3-asr --chunk-seconds`, `--prompt` |
| **corpus, unresolved** | 20 or 7 files, difference inside the resolution floor, so the default is a cost choice rather than an accuracy one | `voxtral --chunk-seconds`, `--max-batch`, `--overlap-seconds`, `qwen3-asr` precision, `--delay-ms`, `--gain` |
| **clip only** | one recording, paired over regions; never run on the corpus | `--vad` off, `--kv-bits 8`, `--compact-silence` off |

A clip-only row is not wrong, it is unverified on the material this project targets. Treat
those three as the ones most likely to move, and see
[benchmarks/metrics.md](benchmarks/metrics.md) on why a null result on one clip cannot be
used to argue the effect is small.

## Model and size

| default | why |
|---|---|
| `voxtral` as the model | 1.35-1.65x faster than whisper turbo, needs no language flag, and reruns byte-identically on one machine. Turbo is ~1.7 points more accurate on Japanese, so the default trades a little accuracy for speed and reproducibility ([engines.md](benchmarks/engines.md)) |
| `whisper --size turbo` | ties `large-v3` on Japanese (14.68% against 14.55%, inside turbo's own ±0.27 rerun spread) at 2.1x the speed and 1.5GB less memory. Picking by size number would take large-v3 and pay for nothing |
| `qwen3-asr --size 1.7B` | 3.9 points ahead of 0.6B at n=20 (19.33% against 23.27%). The 0.6B is the speed option: 32.8x, the fastest engine measured here |

## Precision

One default per model, chosen as the cheapest precision whose accuracy cost is worth its
price. On Voxtral that cost is real and measured across the whole ladder: accuracy improves
monotonically with bit width, and 4-bit is last of five
([quantization.md](benchmarks/quantization.md)). It still ships, because the precisions that
beat it need either time or memory a 16GB machine does not have.

| default | why |
|---|---|
| `voxtral` 4bit | a deliberate accuracy trade, and 4bit is **last of five** on the corpus: 16.34% against 15.27% for 8bit (CI [+0.18, +2.16]) and 15.04% for fp16 (CI [+0.59, +2.26]). It ships because fp16's 12.98GB peak exceeds a 16GB M4's 12.71GB working set at 1.65x the wall clock, and no loadable 8bit build is published. Pass `--quantization fp16` on a large-memory machine, or convert 8bit locally to get fp16 accuracy at 7.3GB and full speed ([quantization.md](benchmarks/quantization.md)) |
| `qwen3-asr` 8bit | bf16 is a tie on the 1.7B (20.16% against 19.98%) at 1.36x the wall clock and 1.4x the memory, and 3 points *worse* on the 0.6B |
| `--kv-bits 8` | faster, and 39 of 40 scored regions identical to unquantized. **Clip-only**, like the row above |

4bit through 6bit on `qwen3-asr` have measured memory
([peak-memory.md](benchmarks/peak-memory.md)) but no accuracy figure, so going below 8bit
there is a size choice on unmeasured accuracy.

## Chunking and windows

| default | why |
|---|---|
| `voxtral --chunk-seconds` per machine | 30s vs 60s is not resolvable at n=20 (+0.10 points, CI [-1.89, +2.03]), so it is a throughput choice and belongs to the hardware profile ([chunking.md](benchmarks/chunking.md)) |
| `voxtral --max-batch` per machine | throughput is not monotonic in batch size: 2-8 is slower per step than 1 on a 16GB M4, so the profile avoids that range entirely ([decode-throughput.md](benchmarks/decode-throughput.md)) |
| `voxtral --overlap-seconds 0` | won 1.5-1.8 points at 30s chunks on one clip, then reversed sign on the corpus. Tied to `--fast` rather than defaulted on |
| `kotoba --chunk-seconds 10` | 27.0% against 31.3% and 49.7% at 10/20/30s on spontaneous Japanese. Material-dependent: 20s won on clean narration, so sweep it on your own audio |
| `qwen3-asr --chunk-seconds 30` | 19.98% / 21.42% / 23.55% / 62.47% at 30/60/120/300s. Unusually, shorter is better on accuracy, speed and memory at once, because a longer window gives a repetition loop a bigger token budget to burn. 15s ties 30s, so this is a plateau rather than a boundary ([qwen3-asr.md](benchmarks/qwen3-asr.md)) |

The library default for `qwen3-asr` is 1200s, at which any recording under 20 minutes is a
single window: one segment, and the batched path never engages.

## Audio and prompt

| default | why |
|---|---|
| `--delay-ms 2400` | the largest accuracy lever here and it costs no speed: 25.62% at 480ms, 20.51% at 960ms, 16.44% at 2400ms. Also the model's maximum ([delay.md](benchmarks/delay.md)) |
| `--gain auto` | quiet input costs ~3.8 points because the mel front end clamps at an absolute floor. `auto` boosts only below -6 dBFS peak, so it is byte-identical on healthy audio ([input-level.md](benchmarks/input-level.md)) |
| `--vad` off | Silero cut points scored 0.8-3.0 points *worse* than energy minima, which is the opposite of what the VAD literature predicts. **Clip-only:** 40 paired regions of one recording, never run on the corpus |
| `--prompt` empty | it biases register rather than recalling vocabulary (term counts move under 7%). An imperative there costs ~6 CER points because the decoder reads it as text it already emitted, and on English audio any prompt wrecks word spacing ([prompt.md](benchmarks/prompt.md)) |
| `--compact-silence` off | helps some quantizations and badly hurts others; the mechanism is still a hypothesis |

## Whisper long-form stability

`condition_on_previous_text=False` on `small` and larger, not exposed as a flag.

Leaving it on lets a repetition loop feed itself across 30s windows. Measured at up to
22.6 CER points on long audio, and it is the single largest artifact in this project's
history: `large-v3` scored 39.91% at library defaults against 14.55% with the flag, which
reads as a broken model rather than a bad default.

`tiny` and `base` keep the library default, since they are not used for long-form work
here.

## Cue layout

`--gap-seconds 1.2`, `--max-chars 28`, `--max-dur-seconds 7.0`.

Deliberately **not** the optimum of either sweep that was run. Both sweeps optimised
agreement with one editor's subtitle conventions, and no second reference set exists to
check that against, so adopting their optimum would fit this corpus's house style rather
than anything general ([cue-layout.md](benchmarks/cue-layout.md)).
