# Results

Findings live in [docs/benchmarks/](docs/benchmarks/), one document per lever, each with
its corpus, method, experiments and conclusion. This page is the index and the list of
what is still open.

## Headline

20 recordings, 7.95h of Japanese and English, on an idle Apple M2 Ultra 128GB (Mac14,14),
re-measured 2026-08-19 after a reference-loading fix (see below):

| engine | JP coverage CER | EN coverage WER | x realtime |
|---|---|---|---|
| voxtral (default) | 16.22% | 21.50% | 29.6x |
| whisper-turbo, no-condition | **14.49%** ±0.27 | **18.34%** ±0.69 | 18.0-22.0x |
| qwen3-asr (1.7B) | 19.33% | 25.45% | 21.8x |
| qwen3-asr-small (0.6B) | 23.27% | 24.26% | **32.8x** |

Whisper is about 1.7 points more accurate on Japanese, and the result holds under all three
tests this project has: 3/3 repeat runs beat the baseline, the run-distribution interval
[13.82, 15.15] is entirely below it, and the bootstrap over *files* gives +1.85 points with
CI [+0.58, +3.33], 13 of 17 files won, sign test p=0.049. That third test is the one that
speaks to generalization and it had never been run at this sample size.

Voxtral is ~1.4x faster, needs no language hint, has better-behaved timestamps, and
reproduces byte-identically on a given machine, which is why it is the default. Whisper's ±
is a run distribution, not a confidence interval on the audio: it samples, so it gets 3 runs
while Voxtral gets one.

The two Qwen3-ASR rows were added on 2026-08-19 and change no default: the 1.7B is last on
accuracy and slower than Voxtral. The 0.6B is the fastest engine measured here (32.8x, in
2.36GB) and the only reason either ships. Both **write no subtitles**, since their
timestamps are decode-window boundaries. See
[qwen3-asr.md](docs/benchmarks/qwen3-asr.md).

**The English figures changed on 2026-08-19** and every earlier English number in this repo
is superseded. Reference lines were being joined with no separator, fusing a word at every
line break on the word-level path; one subtitle-shaped reference carried 131 such fusions
and scored 20.09% where the correct figure is 3.29%. Japanese was never affected, because it
has no word spaces. See [metrics.md](docs/benchmarks/metrics.md), which also explains why
plain CER is meaningless on this editorial material.

## What was measured

| lever | conclusion | detail |
|---|---|---|
| transcription delay | `2400` is worth 9 points and free. The strongest result here. | [delay.md](docs/benchmarks/delay.md) |
| Qwen3-ASR | Neither alias beats an existing default on accuracy; the 0.6B is the fastest engine here. Writes no subtitles. 30s window, measured. | [qwen3-asr.md](docs/benchmarks/qwen3-asr.md) |
| chunking | 30s vs 60s is not resolvable on the corpus (+0.10, CI [-1.89, +2.03]), so choose on speed. Overlap helps only where seams are dense. Energy cuts beat VAD. | [chunking.md](docs/benchmarks/chunking.md) |
| engine choice | Whisper turbo + no-condition is more accurate; Voxtral is faster and reproducible. | [engines.md](docs/benchmarks/engines.md) |
| batch size | Not monotonic. Never use 2-8. | [decode-throughput.md](docs/benchmarks/decode-throughput.md) |
| input level | Quiet input silently costs ~3.8 points; `--gain auto` fixes it. | [input-level.md](docs/benchmarks/input-level.md) |
| `--prompt` | Weak, except that an instruction there costs 6-14 points. | [prompt.md](docs/benchmarks/prompt.md) |
| quantization | Costs accuracy monotonically in bit width; 4-bit is last of five, 1.07 behind 8-bit and 1.30 behind fp16. Ships anyway (no loadable 8-bit build, and fp16 needs 16GB+). `--kv-bits 8` is close to free. | [quantization.md](docs/benchmarks/quantization.md) |
| timestamps | Voxtral holds timing, Whisper places cues better. | [timestamps.md](docs/benchmarks/timestamps.md) |
| cue grouping | Two sweeps run, neither adopted, deliberately. | [cue-layout.md](docs/benchmarks/cue-layout.md) |

Supporting: [corpus.md](docs/benchmarks/corpus.md) (what the material is, how to build
your own), [metrics.md](docs/benchmarks/metrics.md) (which number to trust),
[determinism.md](docs/benchmarks/determinism.md) (what reproduces and what does not).

[JOURNAL.md](JOURNAL.md) is the chronological log the findings were distilled from,
including the runs that produced nothing and the conclusions that were later withdrawn.
Read it for history, not for current numbers.

## Ideas tested and rejected

Check here before proposing performance work. Each of these was measured, not reasoned
about.

| idea | result |
|---|---|
| Batching the encoder | 0.84-0.91x. It is compute-bound at ~4950 FLOP/byte, so batching cannot help. |
| `mx.compile` on the decode step | Equal or worse at every batch size. |
| Reshaping to dodge the batch valley | `fold` is bit-exact but worth 3-7%; splitting is 1.8-4x worse. |
| Forcing MLX's `qmv_wide` kernel at the batch valley | 14-27% *slower* at B=2-8 on a gen-14 Ultra, reproducibly. MLX's gen-15 gate is right. |
| Prefix overlap at long chunks | No benefit once seams are sparse. |
| Carrying decoder context across seams | Recovers 0.17 of the ~1.5 points seams cost, at 2x wall clock. |
| Silero VAD cut points | 0.8-3.0 points worse than energy minima, significantly. |
| Lower sampling rate (8kHz) | No speed change at all; the step count depends on duration. Costs 0.2 CER. |
| Time-stretching audio 1.25x | Cuts steps proportionally but costs 1.9 CER points. |
| fp16 weights | No accuracy gain, 1.6x the wall clock, and does not fit 16GB. |
| Raising `--delay-ms` past 2400 | 2400 is the maximum the model supports. |
| Chunks longer than 60s | Slower, and no better on accuracy. |
| Extending cue ends to the next cue's start | Makes break F1 worse. |

## Open

**CLOSED: the headline is measured.** Both rows were re-measured on 2026-08-06 on an idle
M2 Ultra, arms run sequentially, 3 Whisper runs for its distribution, and both result files
record the machine and its state. Voxtral's accuracy reproduced the original session
exactly; throughput rose (22.8x to 29.8x for Voxtral, 15.0-20.1x to 21.3-23.3x for
Whisper), which suggests the earlier figures were taken on a loaded host. See
[engines.md](docs/benchmarks/engines.md).

**No x-realtime figure should be taken from a busy machine.** An earlier attempt on
2026-08-06 was discarded for this reason: the host was doing unrelated GPU work throughout.
Before any speed measurement, check GPU and memory use, and never run two benchmarks
concurrently. The two corpus runners now check this themselves: they read load average and
GPU memory in use before loading a model, warn when the host is not idle, and record the
machine and its state in the result JSON, which is what the voided run's output lacked.

**Timing quality is n=7** and needs more authored subtitle tracks. None of the 13
recordings added in the final corpus growth had them.

**Cue defaults have never been tested against a reference this project did not author.**
Until there is a second, independent set, "good cue segmentation" is not something this
corpus can measure, only agreement with one editor's conventions.

**The English side is n=3.** Every English conclusion, including the prompt-instruction
result and Whisper's English win, rests on three recordings. The direction is consistent;
the magnitudes should not be quoted as general.

**Competing-runner numbers are n=1.** That is why they support "no fastest claim survives"
and nothing stronger.

**Two quantization repos could not be evaluated** (`...-6bit`, `ellamind/...-8bit-mlx`):
both ship configs mlx-audio misroutes. Low value, since fp16 and 4-bit differ by 0.07
points.

**`--compact-silence` splits by quantization** and the mechanism is a hypothesis. Off by
default.

**Cross-machine output differs, but by far less than first reported.** Identical audio,
config and weights give different output on different chips, so a config comparison must
stay on one machine. Measured over 18 files rather than one, though, 11 score identically
and 16 agree within 0.16 points; the 5.45-point case that established this is the shortest
file in the corpus, and the effect scales inversely with reference length. The earlier
"~1 point per file floor" was an extrapolation from n=1 and is withdrawn. See
[determinism.md](docs/benchmarks/determinism.md).

**CLOSED: chunk length 30s versus 60s is settled as a non-effect.** Re-run on one machine
over all 20 files: +0.10 points, CI [-1.89, +2.03], against +1.67 at n=7. Both chunk lengths
are indistinguishable on this material, so the choice is purely throughput and belongs to
`profiles.json`. This was the strongest candidate for an effect that more audio might
resolve, and more audio dissolved it instead.

Deliberately not open: repeat runs of Voxtral on one machine (byte-identical, verified).

A corpus-wide quantization sweep *was* skipped on the same reasoning, that the effects were
0.07-0.26 points against a 3.2-point resolution. That reasoning was wrong for fp16 versus
4-bit, which resolved cleanly at n=20 once measured (+1.30 points, CI [+0.59, +2.26]). The
power estimate had extrapolated an effect size from a single narration clip where the pair
happens to tie. The other precision pairs remain clip-only for the same discredited reason,
so they are open rather than settled.
