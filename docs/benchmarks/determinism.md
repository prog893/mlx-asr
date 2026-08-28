# Determinism: what reproduces, and what does not

**Conclusion first.** Voxtral is byte-identical across reruns **on one machine** and
**not** across machines: the same audio, config and weight file give different output on
an M4 16GB and an M2 Ultra 128GB. Whisper is not reliably reproducible, because its
temperature fallback samples whenever a segment trips a threshold, and `kotoba` inherits
that ladder. So one Voxtral run is its score, reruns add no information, and a config
comparison must stay on a single machine.

"Not reliably" rather than "not at all": a file that never trips fallback decodes greedily
and does repeat exactly. On this corpus one file scored an identical 11.91% in all six
runs. That is a property of the audio, not a guarantee the engine offers, so it cannot be
relied on in advance.

## Why it matters before any number is quoted

Two things follow from it that shape every other document here.

- **Voxtral needs no error bars from repetition.** One run is the answer. Statistical
  power comes only from more audio, which is why the method bootstraps over files.
- **Whisper needs a distribution.** Quoting one Whisper run as "the" result is
  indefensible: the run-to-run spread on this corpus is 2.5 points, larger than most of
  the effects being measured.

## Voxtral: deterministic on a machine

The pipeline samples with `mx.argmax` alone: no temperature, no top-p, no top-k. Verified
rather than asserted:

- Same file, 3 consecutive runs: byte-identical, same md5.
- Batch 4 versus 6 versus 16 on the same audio: byte-identical, 0 characters differing.
  Non-associative GPU reductions could in principle perturb argmax ties across batch
  shapes, but do not.
- Full 7-file corpus, independent rerun of one config: all 7 files byte-identical,
  aggregate reproducing exactly at 16.44%.
- Two installs of the same commit (pip and Homebrew): byte-identical SRT.

## Voxtral: not deterministic across machines

Measured 2026-08-06, and it corrects a claim this project used to make without
qualification.

Identical audio, identical config (30s chunks, batch 32, kv8, 2400ms), identical weight
file (`model.safetensors` md5 matching on both hosts):

| | M4 16GB | M2 Ultra 128GB |
|---|---|---|
| coverage CER on one 112s file | **12.56%** | **18.01%** |
| output | differs from the first sentence | |
| repeat run on the same host | byte-identical | byte-identical |

These are accuracy figures, so machine load does not affect them: contention costs wall
clock, not output. (The Ultra was busy at the time, which is why no speed number from that
session was kept.)

The two diverge early and stay diverged, which is what one flipped token does: everything
after it is conditioned differently.

**Cause.** Different GPU geometry means different reduction order in the matmuls, which
changes the last bits of the logits, which flips argmax ties. That is the same effect the
batch-shape check above looked for and did not find *within* a machine. Determinism holds
where the reduction order is fixed and fails where it is not.

**Consequences.**

- A hypothesis file cannot be validated by re-decoding it elsewhere.
- A config comparison must be run on one machine. Comparing an M4 16GB row against an Ultra row
  measures hardware plus config, not config.
- The cross-machine agreement check in this project (M4 16GB nvfp4 versus Ultra 4-bit, agreeing
  to ~1 point on 5 of 7 files) cannot separate quantization from hardware. It was set up
  as a quantization comparison and is not one. Both candidate causes are small, so the
  conclusion survives, but the clean quantization evidence is the single-machine precision
  sweep in [quantization.md](quantization.md).
## How large is the cross-machine effect, really? Usually zero

The 112s file above was the only file compared when this was first measured, and the
"roughly 1 point per file" floor stated here was an extrapolation from it. Running the
identical config on all 18 files of the corpus on both machines shows that extrapolation was
wrong, and the truth is more useful:

| agreement between M4 16GB and M2 Ultra 128GB, same config | files |
|---|---|
| coverage CER identical to 2 decimal places | 11 of 18 |
| within 0.16 points | 16 of 18 |
| the 112s file | 5.45 points apart (12.56% against 18.01%) |

So divergence is **not** a per-file floor that applies everywhere. Most files decode to the
same score on both chips, several byte-identically, and the aggregate difference is small.
What the 112s file shows is the *worst case*, not the typical one, and it is the shortest
file in the corpus: 422 reference characters, so a single flipped token moves the percentage
several points, where the same flip in a 9830-character file moves it by hundredths.

That reframes the practical rule. Cross-machine divergence is real, it is caused by
reduction order, and it cannot be predicted or averaged away on any individual file. But its
*magnitude* scales inversely with how much text the file contains, which means:

- **Still true:** a config comparison must stay on one machine, and a hypothesis file cannot
  be validated by re-decoding it elsewhere. One flipped token is unbounded in principle.
- **Corrected:** there is no ~1 point floor. Expect near-exact agreement on files of
  substantial length and volatility on short ones, so a short-clip comparison across machines
  is the dangerous case rather than the representative one.
- **Consequence for method:** this is another reason single-clip results in this project
  reversed on a corpus. A 112s clip cannot distinguish a config effect from a reduction-order
  coin flip.

## Whisper: samples, so it needs a run distribution

Three `--size base` runs on identical audio and identical flags produced three
different transcripts. Use `voxtral` or `qwen3-asr` when a repeatable transcript
matters, or when benchmarking anything.

Whisper's temperature-fallback ladder samples whenever a segment trips the
compression-ratio or logprob threshold, so one run is a draw. Six repeat runs of the
identical `turbo-nocond` config on identical audio, 7-file corpus:

| | JP coverage CER | EN coverage WER |
|---|---|---|
| the 6 runs | 14.77, 14.93, 15.90, 16.22, 16.35, 17.29% | 21.17, 21.71, 21.98, 22.02, 22.61, 23.94% |
| mean, SD | 15.91%, SD 0.94 | 22.24%, SD 0.96 |
| 95% t-interval on the mean | [14.92%, 16.90%] | [21.24%, 23.24%] |
| Voxtral (deterministic) | 16.44% | 26.55% |
| runs beating Voxtral | 5 of 6 | 6 of 6 |

The right test here is one-sample: only one side has sampling error, so the Whisper mean
gets a t-interval and Voxtral enters as a constant. Bootstrapping over files answers a
different question ("would this hold on other audio").

**The variance is structural, not uniform.** Whisper wins all 6 runs on three files and
loses all 6 on three others; only one file flips. On one file it produced the identical
11.91% in all six runs, i.e. that file never triggers fallback. So the aggregate is
decided by which files dominate the length weighting, not by sampling luck.

## kotoba inherits it

`kotoba` runs on mlx-whisper's `transcribe` and does not override the sampling ladder, so
it is not reproducible either. Five runs of one 112s clip on identical audio and flags gave
five distinct transcripts, 401 to 409 characters, agreeing through the body and diverging
at the tail.

Worth stating explicitly because the model is a *distil* checkpoint reached through this
project's own chunked driver, which makes it easy to assume the driver decides the
sampling. It does not: the driver fixes `condition_on_previous_text=False` and leaves
temperature alone.

So of the four engines, `voxtral` and `qwen3-asr` reproduce on one machine by
construction, while `whisper` and `kotoba` reproduce only on audio that happens never to
trip fallback, which is not knowable before the run.

Two further observations:

- **SD collapses as the corpus grows**, from 0.94 at 7 files to 0.17 at 12. More files
  means each file's sampling luck matters less to the aggregate. That is a reason to
  distrust small-corpus repeatability figures, not evidence that Whisper became
  deterministic.
- **Greedy decoding is not a fix.** `--greedy` (temperature 0.0, no ladder) collapses to
  84.92% / 93.00%, because the fallback is what rescues segments that enter a repetition
  loop. The ladder is load-bearing and the nondeterminism comes with it.

A live example of the cost, from a run made while writing these documents: one 26-minute
file hit a repetition loop (`えず` repeated hundreds of times), collapsing its output to
2284 characters and scoring 2.0% coverage / 97.97% coverage CER. That single file moved
the run's Japanese aggregate by **3.6 points**, from 14.69% to 18.32%, off 4.4% of the
reference characters. A one-run Whisper figure can be wrong by that much in either
direction.

## Practical rules

- Report Voxtral from one run, and say which machine.
- Report Whisper from at least 3 runs, with mean and spread, and never from one.
- Never compare across machines and attribute the difference to a config.
- If a hypothesis file needs checking, re-run it on the machine that produced it.

## Related

[metrics.md](metrics.md) for the comparison methods that build on this.
[engines.md](engines.md) for the Voxtral-versus-Whisper result these caveats apply to.
