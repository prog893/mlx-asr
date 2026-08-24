# Lever: the Japanese-specialized engines

**Conclusion first.** Neither Japanese-specialized engine comes close to the
multilingual defaults on this corpus. Parakeet scores 26.19% coverage CER against
turbo-nocond's 14.49%, losing on 16 of 17 files; reazon-k2 fp32 scores 30.45%.
What parakeet does win is speed, decisively: 244.6x realtime is a project record,
7x faster than anything else measured here, at 4.77GB of peak GPU memory. So the
result is not "these models are bad": it is that **training-domain match does not
beat Whisper's breadth on spontaneous, code-switching studio audio**, and that if
throughput ever matters more than accuracy, parakeet is in a class of one.

## Corpus and method

All 20 recordings minus the 3 English-reference files: 17 Japanese files, 6.78h.
Coverage CER at `min_cut` 30 (see [metrics.md](metrics.md)), same cached 16kHz
audio as every other row, unit chosen per file from the reference script.

Measured 2026-08-23 on the idle-except-residents M2 Ultra: 45GB of GPU memory was
parked by a resident agent throughout and load sat near 5 of 24 cores. Both new
engines decode greedily, so accuracy is unaffected by contention; throughput
figures are floors, same treatment as the kept busy-flagged run in
[engines.md](engines.md). Result files: `benchmarks/parakeet_c120.json`,
`benchmarks/reazon_fp32_c30.json`, `benchmarks/reazon_int8_c30.json`.

Runners: `scripts/benchmarks/run_parakeet.py`,
`scripts/benchmarks/run_reazon.py`. Both drive the exact functions the CLI runs
(`backends.parakeet_decode`, `backends.reazon_k2_decode`), load once outside the
timing loop, and report kana/lenient CER beside coverage CER because both models'
text style (punctuation, numerals) differs from an editorial reference without
the transcription being wrong.

## What was run

| arm | config | notes |
|---|---|---|
| `parakeet_c120` | `mlx-community/parakeet-tdt_ctc-0.6b-ja`, 120s windows, 2s overlap | the shipped window, and the measured winner (next section) |
| `parakeet_c300` | same weights, 300s windows | run to test the one-file tie; it reversed sign |
| `reazon fp32 c30` | authors' ONNX, fp32 encoder/decoder/joiner, 30s energy-minima windows | the shipped default |
| `reazon int8 c30` | same, int8 files | measured to test the quantization |

One design fact discovered before the sweep: **reazon-k2 cannot be decoded
whole-file.** Fed a 112s file in one stream it returned 124 characters; it is
trained on short VAD segments and skips badly outside that distribution, which is
why its own recipe segments by VAD. Every figure here uses 30s windows cut at
energy minima, the same front end the Voxtral rows use.

## Results

| engine | JP coverage CER | kana CER | x realtime | peak |
|---|---|---|---|---|
| whisper-turbo no-cond (3-run mean) | **14.49%** | - | 18.0-22.0x | 5.53GB |
| voxtral | 16.22% | - | 29.6x | - |
| kotoba chunk10s | 27.01% | - | 36.2x | - |
| **parakeet c120** | 26.19% | 23.35% | **244.6x** | 4.77GB |
| **reazon-k2 fp32 c30** | 30.45% | 27.73% | 51.6x | 3.31GB RSS |
| reazon-k2 int8 c30 | 36.93% | - | 78.9x | 2.90GB RSS |

Per file against turbo (17 common files): parakeet wins once, by 9.0 points on
the prepared-narration-style `rec-12`, and loses everywhere else by 1.2
to 22.3 points (mean +9.8). Reazon wins the same file and two others but loses
the rest by far more. The losses are not concentrated in one or two files that
could be excluded as pathological: they are the shape of the whole corpus.

## Experiment: parakeet's window length, settled at n=17

On one file, 120s and 300s tied (380 characters each) while 60s lost content,
so 300s looked free. At corpus scale it is not:

| window | JP coverage CER | x realtime | peak GPU |
|---|---|---|---|
| **120s** | **26.19%** | 244.6x | 4.77GB |
| 300s | 32.60% | 204.4x | 8.5GB |

Paired over 17 files, 300s is +5.81 points worse on average and loses on 11,
with the damage concentrated where windows are longest relative to speech
density (one file goes 32.6% to 50.3%). The one-file tie was the corpus-size
trap this project has documented before: several single-clip findings here
reversed sign when a real corpus arrived ([corpus.md](corpus.md)). 60s was not
swept at corpus scale because it demonstrably drops content on even one file.

Determinism of both new engines was verified rather than assumed after this:
three decodes each through the CLI code path give byte-identical text AND cues,
so every single-run figure above is a score rather than a draw.

## Three findings worth keeping

**int8 is NOT near-parity here, contradicting the publisher's own table.**
Reazon's release notes put int8 within ~0.3 CER of fp32 on JSUT, Common Voice and
TEDxJP-10K. On this corpus int8 drops whole phrases mid-file: 296 against 376
characters on one 112s recording, and 36.93% against 30.45% over the corpus. Read
speech benchmarks and conversational material disagree about what int8 costs;
this corpus is the closer match to what the CLI is for, so fp32 ships and a test
guards the default.

**Chunk length interacts with model training the opposite way to Whisper.** For
kotoba and qwen3, shorter windows were neutral-to-better. Parakeet needs LONG
windows (60s lost content against 120s), and reazon needs windows at all. A
model trained on short independent segments degrades when asked for more context
than it saw; a model trained to carry state across windows degrades when denied
it. There is no universal chunking advice, only per-model sweeps.

**Speed and accuracy are diverging into separate products.** Parakeet decodes
6.78h of audio in 100 seconds. If the job is "rough transcript of everything,
now", it has no competitor here. If the job is subtitles worth publishing,
turbo remains the accuracy pick and parakeet is not close enough to trade.

## Relation to the fine-tuning question

These two engines are the strongest open evidence this project has against the
"just fine-tune on domain data" hypothesis from the corpus side: parakeet-ja was
TRAINED on tens of thousands of hours of naturalistic Japanese including
spontaneous speech, which is exactly this corpus's material, and still loses to a
generic multilingual Whisper checkpoint by 11.7 points. Domain match alone does
not explain engine quality; instruction breadth, long-form behaviour and
code-switching handling dominate it on real recordings.
