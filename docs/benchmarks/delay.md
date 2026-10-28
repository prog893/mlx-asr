# Lever: transcription delay

`--delay-ms 2400` is the default, and it is the largest accuracy lever measured in this
project at no throughput cost: coverage error drops from 25.62% to 16.44%[^d1] between 480ms
and 2400ms at the same throughput, and 2400ms won on all 7 files individually[^d1] at both
comparison points. 2400 is the maximum the model supports. Voxtral only: the delay is a
property of that model's streaming design, and the other engines have no equivalent and
reject `--delay-ms`.

| setting | default | why |
|---|---|---|
| `--delay-ms` | `2400` | lowest error in both languages, 7-0 on files, no throughput cost; applied on every machine and not part of the hardware profile, since there is no tradeoff to tune |

**Setup:** [7-file subset](reference/corpus.md#the-7-file-subset), M2 Ultra 128GB, Voxtral at 4-bit, 30s <!-- not-measured: config: chunk length -->
chunks, batch 32, kv8, via `scripts/benchmarks/run_corpus.py`; scored by
[coverage CER/WER](reference/metrics.md#coverage-cer-and-why-it-had-to-exist) at `min_cut` 30
characters / 6 words.

## Experiment: transcription delay

**Basis:** [7-file subset](reference/corpus.md#the-7-file-subset), M2 Ultra 128GB, Voxtral at 4-bit, 30s chunks, batch 32, kv8, delay stepped 480ms / 960ms / 2400ms. <!-- not-measured: config: delay values stepped -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/delay-dark.svg">
  <img alt="Error by transcription delay, 2400ms is the default" src="img/delay-light.svg">
</picture>

**Table:** error and throughput at each transcription delay.

| delay | JP coverage CER | EN coverage WER | x realtime |
|---|---|---|---|
| 480ms | 25.62% | 35.06% | 31.8x |
| 960ms | 20.51% | 30.36% | 31.9x |
| **2400ms** | **16.44%** | **26.55%** | 31.5x |

Source: [ultra-voxtral-delay.json](../../benchmarks/results/2026-10/ultra-voxtral-delay.json) runs `delay7_c30b32_d480`, `delay7_c30b32_d960`; [ultra-mincut-inputs.json](../../benchmarks/results/2026-10/ultra-mincut-inputs.json) runs `mincut7_vox_c30b32_kv8`

Error falls monotonically in both languages, and throughput is flat: the three rows span
0.3x[^d1] in x-realtime, and the delay does not change the step count. A rerun of the 2400ms
config reproduced 16.44% / 26.55% byte-identically at 31.1x,[^d2] which confirms determinism.
The 2400ms row and that rerun were both recorded with the machine not idle, so their speed <!-- not-measured: 2400ms is a delay setting -->
difference is not a measure of run-to-run spread.

Paired across files, bootstrapped over files with 20k resamples:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/delay-paired-dark.svg">
  <img alt="Paired difference of 480ms and 960ms against the 2400ms default, both intervals above zero" src="img/delay-paired-light.svg">
</picture>

**Table:** paired per-file difference of each lower delay against 2400ms, pooled over Japanese CER and English WER, with its 95% CI and file wins. <!-- not-measured: CI confidence level -->

| comparison | diff | 95% CI | files won |
|---|---|---|---|
| 480ms vs 2400ms | +9.07 | [+5.26, +14.27] | 7-0 |
| 960ms vs 2400ms | +4.02 | [+1.59, +8.91] | 7-0 |

Source: [ultra-voxtral-delay.json](../../benchmarks/results/2026-10/ultra-voxtral-delay.json) runs `delay7_c30b32_d480`, `delay7_c30b32_d960`; [ultra-mincut-inputs.json](../../benchmarks/results/2026-10/ultra-mincut-inputs.json) runs `mincut7_vox_c30b32_kv8`

Positive means the lower delay is worse. Winning 7 files to 0 at both comparison points
makes this the most robust finding in the project: at n=7 the corpus resolves Japanese
effects of about 2.04 points[^d3] (the 95% half-width from the per-file spread of Voxtral at 4-bit, the default then,
against Whisper turbo-nocond over 6 runs, SD 2.75 points),[^d3] and this one is 9.[^d1]

Every other lever in this project is small, machine-dependent, or trades speed for
accuracy. This one is large, reproduces on every file, holds on prepared speech as well
(see [Superseded](#superseded)), and costs nothing. A consequence for benchmarking
Voxtral elsewhere: a run at the library default delay does not measure the model's
accuracy. It is roughly 9 points[^d1] worse than the model can do at no cost.

## How it works

Voxtral Realtime consumes 80ms of audio per decoder position and emits one token per <!-- not-measured: model design constant -->
position. The transcription delay is how much audio the model is allowed to see before
it must commit to a token. A larger delay means the leading tokens of each chunk cover
more audio, so the model is guessing less.

The model supports multiples of 80ms in [80, 1200], plus 2400 as a standalone value. <!-- not-measured: model design constants -->
There is nothing above 2400 to try; the Voxtral paper's suggestion to raise it further
is already exhausted.

Cost is nil because the step count is set by audio duration rather than by the delay.
Nothing extra is decoded.

## Superseded

On the [single narration clip](reference/corpus.md#the-single-clip) the same lever reads 14.74%[^d4] at
480ms versus 7.40% at 2400ms (M2 Ultra, 60s chunks, batch 8, KV unquantized).[^d4] The corpus result above replaces it as the measurement of
record; the clip confirms that the direction and rough magnitude hold on prepared speech.

## Related

[chunking.md](chunking.md) is the next-largest lever and does trade against speed.

[^d1]: [ultra-voxtral-delay.json](../../benchmarks/results/2026-10/ultra-voxtral-delay.json) runs `delay7_c30b32_d480`, `delay7_c30b32_d960`; [ultra-mincut-inputs.json](../../benchmarks/results/2026-10/ultra-mincut-inputs.json) runs `mincut7_vox_c30b32_kv8`
[^d2]: [ultra-voxtral-prompt-trap.json](../../benchmarks/results/2026-10/ultra-voxtral-prompt-trap.json) runs `prompt_trap7_none`; [ultra-mincut-inputs.json](../../benchmarks/results/2026-10/ultra-mincut-inputs.json) runs `mincut7_vox_c30b32_kv8`
[^d3]: [ultra-resolution-floor.json](../../benchmarks/results/2026-10/ultra-resolution-floor.json) runs `resolution_floor`
[^d4]: [ultra-voxtral-clip-cli.json](../../benchmarks/results/2026-10/ultra-voxtral-clip-cli.json) runs `hk_c60b8_kv0_d480`, `hk_c60b8_kv0_d2400`, `score_clip`
