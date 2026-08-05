# Lever: transcription delay

**Voxtral only.** The delay is a property of that model's streaming design; the other
engines have no equivalent and reject `--delay-ms`.

**Conclusion first.** `--delay-ms 2400` is the largest accuracy lever in this project
and it is **free**: coverage error drops from 25.62% to 16.44% between 480ms and 2400ms
at the same throughput. It won on all 7 corpus files individually at both comparison
points, which makes it the strongest result here. 2400 is the maximum the model
supports, and it is the default.

## What the knob is

Voxtral Realtime consumes 80ms of audio per decoder position and emits one token per
position. The transcription delay is how much audio the model is allowed to see before
it must commit to a token. A larger delay means the leading tokens of each chunk cover
more audio, so the model is guessing less.

The model supports multiples of 80ms in [80, 1200], plus 2400 as a standalone value.
There is nothing above 2400 to try; the Voxtral paper's suggestion to raise it further
is already exhausted.

Cost is nil because the step count is set by audio duration, not by the delay. Nothing
extra is decoded.

## Corpus

7 spontaneous multi-speaker recordings, 5 Japanese and 2 English, editorial references,
scored with the coverage-aware metric at `min_cut` 30 characters / 6 words. Config held
at 30s chunks, batch 32, kv8. `scripts/benchmarks/run_corpus.py`, M2 Ultra 128GB.

## Experiment

| delay | JP coverage CER | EN coverage WER | x realtime |
|---|---|---|---|
| 480ms | 25.62% | 35.06% | 29.5x |
| 960ms | 20.51% | 30.36% | 30.7x |
| **2400ms** | **16.44%** | **26.55%** | 28.9x |

Monotonic in both languages, and flat in throughput: the 1.8x spread in x-realtime
across those rows is machine noise, not the delay. A rerun of the 2400ms config
reproduced 16.44% / 26.55% byte-identically at 31.2x, which both confirms determinism
and shows how much the speed column wanders between runs on a shared machine.

Paired across files, bootstrapped over files with 20k resamples:

| comparison | diff | 95% CI | files won |
|---|---|---|---|
| 480ms vs 2400ms | +9.07 | [+5.41, +14.25] | 7-0 |
| 960ms vs 2400ms | +4.02 | [+1.59, +8.84] | 7-0 |

Positive means the lower delay is worse. Winning 7 files to 0 at both comparison points
is what makes this the most robust finding in the project: the corpus only resolves
effects of about 3.2 points at n=7, and this one is 9.

On the single narration clip the same lever reads 14.74% at 480ms versus 7.49% at
2400ms, so the direction and rough magnitude hold on prepared speech too.

## Why this is worth stating so plainly

Every other lever in this project is either small, machine-dependent, or trades speed
for accuracy. This one is large, reproduces on every file, transfers between material
types, and costs nothing. If you are quoting one number from this work, quote this.

The corollary matters for anyone benchmarking Voxtral elsewhere: a run at the library
default delay is not a measurement of the model's accuracy. It is roughly 9 points worse
than the model can do, for free.

## What ships

`--delay-ms 2400`, unconditionally, on every machine. It is not part of the hardware
profile because there is no tradeoff to tune.

## Related

[chunking.md](chunking.md) is the next-largest lever and does trade against speed.
