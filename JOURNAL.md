# Journal: the chronological log

**This is history, not the current numbers.** It is kept because it records what was
tried, in what order, and which conclusions were later withdrawn, which is the part that
would otherwise be re-litigated. For what the project currently concludes, read
[docs/benchmarks/](docs/benchmarks/), which is organised by lever and carries the
methodology alongside each result. [RESULTS.md](RESULTS.md) is the index.

Three consequences of reading a log rather than a findings document:

- **Numbers here can be superseded.** Where a later section overturns an earlier one, the
  earlier section is left in place with a note rather than edited, so the reversal itself
  stays visible. Several are marked *Superseded* or *Corrected*.
- **Everything below the "multi-file corpus" heading is single-clip work.** Still correct
  as measured, but the rankings do not generalize; two of them reversed sign when a real
  corpus arrived.
- **References to `scripts/benchmarks/*.json` are historical.** Those files existed while this log
  was being written and are no longer in the repo: they held per-file rows that nobody
  without the private audio could re-derive, and a file of bare numbers with no statement
  of what was tested is not auditable. Every aggregate they carried is in
  [docs/benchmarks/](docs/benchmarks/). Filenames are left in place so it stays clear which
  run a number came from.

> **2026-07-26: a 7-file corpus replaced the single test clip, and it overturns
> two of this file's conclusions.** Read the corpus section before anything below it.

## Gaps in the benchmark record

Known-incomplete measurements, in the order they would be worth redoing. Recorded
here because "not measured" and "measured and fine" read identically in a results
file otherwise. Most were found by auditing this file against the committed JSON
rather than by anything failing, and two turned out to be harness bugs that had been
reported as model behaviour; those are marked CLOSED with what the fix changed.

**1. CLOSED: the 20-file comparison is measured.** The headline table had been scored in a
session whose outputs never landed in a result file, and a first re-run attempt on
2026-08-06 was **discarded**: the host was doing unrelated GPU work throughout, so its
throughput figures were meaningless (it measured ~2.6x where the same config reaches 31x
idle) and the session was voided rather than partly quoted.

Redone later the same day on an idle M2 Ultra: 4 sequential runs (1 Voxtral, 3 Whisper for
its distribution), each recording the machine and its state. **Voxtral's accuracy
reproduced the original session exactly** (16.22% JP, 25.24% EN), so that session's
accuracy work was sound. **Throughput did not**: Voxtral measured 29.8x against 22.8x
published, and Whisper 21.3-23.3x against 15.0-20.1x. Since the original had no recorded
machine state, the likely explanation is that it too was taken on a loaded host. Whisper's
JP mean moved 14.93% to 14.74%, inside its run spread, which is expected of a sampling
engine. Verdict is unchanged: whisper-turbo-nocond wins accuracy on both units with both
intervals entirely below Voxtral, Voxtral wins speed by ~1.35x. Details in
[docs/benchmarks/engines.md](docs/benchmarks/engines.md).

**2. CLOSED: break F1 now describes the shipped cue config.** It was 42.8% measured
at `gap_s=0.7, max_chars=32`, which is not what the CLI ships. Re-measured at the
shipped `1.2, 28`: **37.4%** (`scripts/benchmarks/timing_engines_shipped_cues.json`). The
timing table below uses it. Root cause was that the cue config was neither settable
from the CLI nor recorded in any output, so a run could not be attributed to one;
both are fixed.

**3. CLOSED: the degenerate timing row was a harness bug, not a model result.**
Whisper's row for rec-20 had 9.8% anchor coverage and was discarded by the metric.
Cause: `run_timing_sweep.py` hardcoded `language='ja'`, and rec-20 is the
English-audio half of a dubbed pair, so Whisper was asked for the wrong language and
looped. With the language taken per file from the reference it reaches **93.1%**
coverage, all 7 rows are trustworthy (lowest 75.6%), and the n=6-versus-n=7 choice is
gone. Still n=7 overall, which remains small: none of the 13 recordings added in the
final corpus growth had authored SRTs.

**4. Cue defaults have never been tested against a reference the project did not
author.** All seven timed references share one editor's conventions, which is why
neither sweep result was applied (see "Cue defaults" below). Until there is a
second, independent set of authored subtitles, "good cue segmentation" is not
something this corpus can measure, only "agreement with one house style".

**5. The English side is n=3.** Every English conclusion, including the
+13.77-point prompt-instruction result and Whisper's English win, rests on three
recordings, two of them in the original seven. The direction is consistent across
runs, but no English effect here should be quoted as a general figure.

**6. Competing-runner numbers are n=1.** The whisper.cpp / faster-whisper /
insanely-fast-whisper table is one 935s clip, which is why it supports "no fastest
claim survives" and nothing stronger. Re-running it across the corpus would need
each runner driven over 20 files; the finding it produced (faster-whisper is
CPU-only on Metal) is structural and does not need it.

**7. Two quantization repos were never evaluated.** `Voxtral-Mini-4B-Realtime-6bit`
and `ellamind/...-8bit-mlx` both ship configs mlx-audio misroutes, so the precision
sweep has no 6-bit row. Given fp16 and 4-bit differ by 0.07 points, this is
low-value.

**8. `--compact-silence` splits by quantization and was never explained.** It helps
4-bit affine and badly hurts nvfp4 on identical audio. It is off by default and the
mechanism is a hypothesis, not a measurement.

**9. Cross-machine reproducibility is a floor on every two-machine comparison.**
Measured 2026-08-06: identical audio, config and weight file give byte-identical
output on reruns of one machine but *different* output on an M4 16GB versus an M2 Ultra 128GB
(12.56% against 18.01% coverage CER on one 112s file), because GPU reduction order
flips argmax ties. So the "cross-machine check" table cannot separate quantization
from hardware, and no hypothesis file can be validated by re-decoding it elsewhere.
See "Decoding is deterministic" for the measurement.

Deliberately *not* gaps: repeat runs of Voxtral on one machine (greedy, so
byte-identical, and verified as such), and the quantization corpus sweep (the effects
are 0.07-0.26 points against a 3.2-point resolution at n=7, so it would produce
noise).

## 2026-08-06: the headline re-measured on an idle machine

Closes gap 1. Four runs on `Apple M2 Ultra 128GB (Mac14,14)`, sequentially, never two at
once: one Voxtral (`c30b32_kv8`, delay 2400) and three `whisper-large-v3-turbo` with
`condition_on_previous_text=False`, on all 20 files (7.95h).

| | Voxtral | turbo-nocond, 3 runs | previously published |
|---|---|---|---|
| JP coverageCER, 17 files | 16.22% | 14.48 / 14.77 / 14.96 -> 14.74% +/- 0.24 | 16.22% / 14.93% |
| EN coverageWER, 3 files | 25.24% | 22.42 / 22.84 / 23.16 -> 22.81% +/- 0.37 | 25.24% / 22.54% |
| x realtime | 29.8x | 21.3 / 21.5 / 23.3x | 22.8x / 15.0-20.1x |

Three things worth recording separately from the verdict, which did not change.

**Voxtral's accuracy reproduced to the digit.** 16.22% and 25.24%, identical to the
original session on both aggregates. That session's accuracy work was therefore sound, and
this is what makes the earlier decision to void it whole rather than quote half of it look
right in hindsight rather than merely cautious: the accuracy half was recoverable by
re-running, so nothing was lost by refusing to quote it.

**Throughput moved much more than accuracy.** Voxtral 22.8x to 29.8x and Whisper
15.0-20.1x to 21.3-23.3x. Nothing in the code changed to explain that; the previous numbers
had no recorded machine state, so the straightforward reading is that they were measured on
a loaded host. This is the second time an unrecorded machine state has cost a figure, which
is why `machine_state.py` now runs before the model loads rather than being noted by hand.

**One Whisper run was flagged busy and kept anyway.** It started with 25.9GB of GPU memory
held by an unrelated resident process. Keeping it was justified by measurement, not
judgement: it ran at 21.3x against 21.5x for an idle run, and the fastest run of the three
was also idle at 23.3x. Parked memory on a 115GB working set costs nothing here, so the
flag is conservative. Left conservative on purpose, since the cost of checking a false
warning is a few minutes and the cost of missing a real one is a published number.

Per file, Whisper wins 12 of 20 in all three runs and loses 7 of 20 in all three, with one
file changing sign. The ordering is structural rather than driven by any single file.

## Reading the per-file labels

Recordings appear as `rec-NN`, assigned once globally and stable across every
committed result file, so a per-file row can be compared between runs. `rec-01`
through `rec-17` are the Japanese recordings, `rec-18` and `rec-19` the English
ones, `rec-20` the remaining recording with an authored subtitle track, and
`narration-jp` the single prepared-narration clip that carries a complete verbatim
reference (the one file where plain CER is meaningful).

The original 7-file corpus is `rec-02, rec-03, rec-04, rec-05, rec-06` (Japanese)
plus `rec-18, rec-19` (English), which is why the earliest tables skip numbers.
Labels were renumbered once, during the pre-release audit: they had been assigned
per result file, so the same label meant different recordings in different files.
Any external copy of these numbers predating that is not comparable.

## Multi-file corpus (7 recordings, 5.2h, mixed language)

The corpus is 5 single-speaker Japanese recordings (13-93 min) and 2
multi-speaker English ones, all 96kHz mono downmixed to 16kHz.
`scripts/benchmarks/run_corpus.py`.

### The references deliberately omit audio, so plain CER is unusable

These transcripts were edited for readability rather than for ASR evaluation:
off-topic passages, side conversation in another language, and non-speech segments
were cut. The audio still contains that material, so a *correct* transcription
legitimately includes text the reference lacks. Plain CER reads 130-152% on the
Japanese files as a result, with thousands of "insertions" that are not errors.

`scripts/metrics/eval_coverage.py` treats the reference as a subsequence to locate rather than the
whole truth: substitutions and deletions are charged, and so are short insertions
(which catch real hallucination and repetition loops), but insertion *runs* longer
than a threshold are excused as omitted-from-reference audio. On one file that
turns a meaningless 140% into 19.8% while showing 82% of reference characters
correctly present, with all four excused runs being exactly the cut chatter.

Two further corpus-specific fixes: speaker-label lines (`Name:`) are diarization
metadata and are stripped before scoring, and the scoring *unit* is chosen per
file from the reference script (WER for space-delimited text, CER for CJK), since
`normalize()` strips whitespace and would otherwise collapse English to 45 words.

### Results, M2 Ultra 128GB, 4bit

| config | Japanese coverageCER | English coverageWER | x realtime |
|---|---|---|---|
| 30s b32 kv8 | **16.44%** | **26.55%** | **31.3x** |
| 30s b32 kv8 + 8s overlap | 17.34% | 30.77% | 25.4x |
| 60s b16 kv8 | 18.21% | 27.75% | 21.4x |

Per-file coverage error, showing how much files disagree:

| file | unit | 30s | 30s+ov8 | 60s |
|---|---|---|---|---|
| 13 min JP | char | 11.68% | 16.09% | 11.05% |
| 22 min EN | word | 27.99% | 34.74% | 29.68% |
| 26 min JP | char | 19.76% | 15.39% | 25.12% |
| 35 min EN | word | 25.92% | 29.06% | 26.91% |
| 52 min JP | char | 17.37% | 19.92% | 24.46% |
| 70 min JP | char | 16.50% | 11.38% | 13.19% |
| 93 min JP | char | 16.26% | 20.99% | 18.22% |

### Paired across files: nothing is significant any more

Bootstrapping over *files* (length-weighted, 20k resamples) is the first real n>1
test in this project:

| question | diff | 95% CI | verdict |
|---|---|---|---|
| overlap helps at 30s chunks | -1.47 | [-4.33, +2.36] | **not resolvable** |
| 60s beats 30s chunks | +1.67 | [-1.22, +4.73] | **not resolvable** |
| 60s vs 30s+overlap | +0.20 | [-2.42, +3.35] | not resolvable |

**Both conclusions that the single clip supported are withdrawn.** On one clip,
overlap-at-30s looked like a solid +1.8-point win (CI [+0.62, +3.20]) and
60s-over-30s looked like +1.85 (CI [+0.71, +3.24]). Across seven recordings, the
overlap effect actually *reverses sign* (30s without overlap is nominally better,
5 files to 2) and neither difference clears the noise. Between-file variation
(11-28% coverage CER for the same config) is far larger than any config effect,
which is exactly the failure mode the n=1 caveat warned about.

What survives: **30s chunks at batch 32 is the best config on this corpus on every
axis measured** (best Japanese CER, best English WER, and 31.3x versus 21.4x).
That is a change from the previous 60s recommendation, and it is supported by
point estimates rather than by significance, so it is a preference, not a proof.

Two mechanisms plausibly explain why overlap stopped helping: these recordings
contain long stretches of non-reference material, so a warm-up window often
carries content that is cut from the reference; and the seam-error enrichment that
motivated overlap was measured on a single dense-narration clip and may not hold
for conversational audio with frequent long pauses.

### Corpus-level verification of the two big claims

Power analysis first: the per-file paired difference between two real configs has
SD 3.50 points, so at n=7 the corpus resolves effects of about **3.2 points or
larger**. That is why only large claims were re-tested, and why the quantization
sweep was *not* run (see below).

| claim | diff | 95% CI | files | verdict |
|---|---|---|---|---|
| delay 480ms vs 2400ms | +9.07 | [+5.41, +14.25] | 7-0 | **confirmed** |
| delay 960ms vs 2400ms | +4.02 | [+1.59, +8.84] | 7-0 | **confirmed** |
| prompt instruction vs none | +3.53 | [+0.41, +9.82] | 5-2 | **confirmed** |
| same config, rerun | +0.00 | [+0.00, +0.00] | 0-0 | bit-identical |

**`--delay-ms 2400` is confirmed as the project's largest lever**, and it is still
free: 28.9x realtime at 2400ms versus 29.5x at 480ms. The full curve is monotonic:

| delay | Japanese coverageCER | English coverageWER | x realtime |
|---|---|---|---|
| 480ms | 25.62% | 35.06% | 29.5x |
| 960ms | 20.51% | 30.36% | 30.7x |
| 2400ms | **16.44%** | **26.55%** | 28.9x |

It wins on all 7 files individually at both comparison points, which is a stronger
result than anything else in this project.

**The prompt-instruction warning is confirmed but is language-dependent**, which
the single clip could not show. Putting "Transcribe the audio accurately." in the
prompt field costs **+13.77 points on the English files** (CI [+10.34, +15.25],
both files) but only +1.41 on the Japanese ones (CI spans zero). That fits the
mechanism: the decoder continues the prompt as already-emitted text, so an English
instruction pulls English output off-register while barely perturbing Japanese.
The warning stands; the magnitude depends on whether the prompt language matches
the audio.

### Decoding is deterministic (checked, not assumed)

Generative models are usually stochastic, but this pipeline samples with
`mx.argmax` alone: no temperature, no top-p, no top-k. Verified rather than
asserted:

- Same file, 3 consecutive runs: byte-identical (same md5).
- Batch 4 vs 6 vs 16 on the same audio: byte-identical, 0 characters differing.
  Non-associative GPU reductions could in principle perturb argmax ties, but do
  not here.
- Full corpus, independent rerun of one config: **all 7 files byte-identical**,
  aggregate reproducing exactly (16.44%).

So repeating a run adds no information, and the only useful way to add statistical
power is more *audio*. This is why the methodology bootstraps over files rather
than over repeated runs.

**The guarantee is per machine, not across machines.** Measured 2026-08-06: the same
file, same config (30s/b32/kv8/2400ms), same weights (`model.safetensors` md5
identical on both hosts) gives byte-identical output on repeat runs *on one machine*
and **different** output between an M4 16GB and an M2 Ultra 128GB. On one 112s file the two
diverge in the first sentence and score 12.56% against 18.01% coverage CER.

That is the non-associative-reduction effect the second bullet above looked for and
did not find within a machine: different GPU geometry means different reduction
order, which flips argmax ties, and one flipped token changes everything after it.
It also explains why the cross-machine check further down agrees only to ~1 point per
file rather than exactly.

Practical consequences. A number is reproducible where it was measured, so
determinism still means one run is its score and reruns add nothing. But a config
comparison must come from a single machine, and a hypothesis file cannot be checked
by re-decoding it elsewhere. It also puts a floor under cross-machine agreement that
is not measurement noise: expect ~1 point per file even with everything else fixed.

**Corrected 2026-08-07, by running the identical config on all 18 files on both hosts
rather than one.** There is no ~1 point floor. 11 of 18 files score *identically* to two
decimals across the M4 and the Ultra, 16 of 18 agree within 0.16 points, and the 112s file
quoted above is the worst case rather than the typical one. The magnitude scales inversely
with reference length, which is the obvious explanation in hindsight: this file has 422
reference characters, so one flipped token is worth several points, while the same flip in a
9830-character file is worth hundredths. The qualitative claims stand (divergence is real,
unpredictable per file, and a comparison must stay on one machine); only the magnitude was
wrong, and it was wrong because it was extrapolated from n=1. See
[docs/benchmarks/determinism.md](docs/benchmarks/determinism.md).

### Why the quantization sweep was skipped

Given the 3.2-point resolution at n=7, detecting the measured quantization
differences would need far more material than exists:

| effect | size | files needed |
|---|---|---|
| 8bit vs 4bit | 0.26 pts | ~1,000 |
| fp16 vs 4bit | 0.07 pts | ~15,000 |
| kv8 vs unquantized KV | 0.02 pts | ~180,000 |

fp16 was already tested twice on the single clip (default and tuned config) and
tied both times, and the cross-machine check below shows nvfp4 and 4bit agreeing
within ~1 point per file. Running fp16/nvfp4/8bit across the corpus would consume
~1.5h of GPU time to produce three numbers inside the noise floor.

### Threshold sensitivity of the coverage metric

Every corpus number depends on `min_cut`, the insertion-run length above which
text is excused as omitted-from-reference audio. `scripts/benchmarks/sweep_min_cut.py`
varies it over the saved hypotheses:

| config | mc10 | mc20 | mc30 | mc50 | mc80 |
|---|---|---|---|---|---|
| 30s b32 kv8 | 14.97% | 15.89% | **16.44%** | 17.81% | 20.23% |
| 30s b32 kv8 ov8 | 15.62% | 16.67% | 17.34% | 18.12% | 20.61% |
| 60s b16 kv8 | 16.80% | 17.95% | 18.21% | 19.32% | 20.69% |

Absolute levels drift about 5 points across that range, so **the threshold must be
quoted with any absolute number** (all figures here use 30 chars / 6 words). But
the **ranking is identical at every threshold**, on both the char and word units,
so config comparisons are not an artifact of the choice. That is the property the
tuning conclusions actually depend on.

### Input level matters, but only below about -6 dBFS

The mel front end clamps at an **absolute** floor (`log10(power)` below
`global_log_mel_max - 8.0` is flattened), not one relative to the utterance. So
level is a real preprocessing variable: quiet input has low-level detail destroyed
before the encoder ever sees it. Measured share of mel bins sitting exactly at the
floor, one recording:

| gain | peak dBFS | % mel bins clamped |
|---|---|---|
| -20dB | -25.5 | 65.5% |
| -12dB | -17.5 | 41.5% |
| -6dB | -11.5 | 24.0% |
| 0dB | -5.5 | 9.9% |
| +6dB | 0.0 | 2.8% |

Full corpus sweep (`scripts/benchmarks/sweep_gain.py`, 30s/b32/kv8, coverage metric):

| mode | Japanese coverageCER | English coverageWER |
|---|---|---|
| -20 dB | 23.76% | 36.65% |
| -12 dB | 19.42% | 34.23% |
| unity (as recorded) | **16.44%** | 26.55% |
| peak to -1 dBFS | 16.74% | 26.38% |
| +6 dB | 17.09% | **23.96%** |
| rms to -23 dBFS | 17.09% | **23.93%** |

Paired across files:

| comparison | diff | 95% CI | verdict |
|---|---|---|---|
| -20dB vs unity | +7.79 | [+5.39, +12.08] | **attenuation hurts badly** |
| -12dB vs unity | +3.78 | [+1.82, +7.83] | **attenuation hurts** |
| +6dB vs unity, all files | +0.09 | [-1.21, +0.94] | not resolvable |
| +6dB vs unity, English only | **-2.59** | [-3.19, -1.18] | **amplifying helps** |
| +6dB vs unity, Japanese only | +0.65 | [+0.00, +1.08] | marginally hurts |
| peak-normalize vs unity | +0.21 | [-0.19, +0.64] | not resolvable |

Conclusions:

- **Quiet input is genuinely damaging**, and the mechanism is the absolute mel
  floor rather than bit depth. This is worth knowing because it is a silent
  failure: a -12dB recording loses ~3.8 points with no other symptom.
- **Amplifying already-healthy audio is close to a wash overall**, and the sign
  depends on content: the two English multi-speaker recordings improved by 2.6
  points, while the five Japanese ones got marginally worse. Plausibly the English
  files have quieter off-mic speakers whose detail was being clamped, whereas the
  Japanese ones are close-mic single-speaker where amplification mostly lifts the
  noise floor. n=2 for the English side, so treat that as a lead.
- **This corpus needed no normalization**: it was recorded at -0.5 to -5.3 dBFS
  peak, so unity gain was already near-optimal.

### `--gain auto` (the default)

The two findings above give an asymmetric rule rather than a loudness target:
quiet audio must be lifted, loud audio must be left alone. `auto` boosts only when
the peak is below -6 dBFS, and then only up to -1 dBFS:

| peak dBFS | gain applied | result |
|---|---|---|
| -0.5 to -5.3 (this corpus) | 0.0 dB | byte-identical, no-op |
| -7.4 | +6.4 dB | -1.0 dBFS |
| -13.3 | +12.3 dB | -1.0 dBFS |
| -21.3 | +20.3 dB | -1.0 dBFS |
| -41.3 | +40.3 dB | -1.0 dBFS |

Gain is never negative and the target sits below full scale, so `auto` cannot
clip. Verified as a no-op on all 7 corpus files and on the original single clip.

Recovery check, attenuating a clip by 14 dB and transcribing it three ways:

| condition | CER |
|---|---|
| original level | 8.63% |
| attenuated -14 dB, `--gain none` | 10.84% |
| attenuated -14 dB, `--gain auto` | **8.61%** |

So `auto` recovers essentially all of the 2.2-point loss (8.61% vs 8.63%), which
is what makes it safe to enable by default: it is inert on well-recorded material
and repairs quiet material almost exactly.

On bit depth specifically: the sources are 32-bit float, so gain is mathematically
lossless and there is no dynamic-range argument for pre-normalizing. What matters
is only where the signal sits relative to the model's fixed floor. Clipping is the
one irreversible risk, which is why `peak` targets -1 dBFS rather than 0, and why
the CLI reports the clipped-sample percentage when any gain is applied.

### Cross-machine check

The same config on both machines, with different quantizations, agrees closely,
which says the corpus numbers are a property of the audio rather than of a
particular setup:

| | Japanese coverageCER | English coverageWER | x realtime |
|---|---|---|---|
| M4 16GB, nvfp4 | 17.59% | 26.65% | 7.0x |
| M2 Ultra 128GB, 4bit | 18.21% | 27.75% | 21.4x |

Per-file agreement is within ~1 point on 5 of 7 files (worst case 5.4 points on
the 26-minute file). So quantization choice remains a non-factor for accuracy, as
the single-clip fp16 study also concluded.

One correction to how this row should be read. It was set up as a quantization
comparison (nvfp4 against 4-bit affine), but the two machines do not agree
byte-for-byte even at *identical* weights and config: GPU reduction order differs, so
argmax ties break differently (see "Decoding is deterministic" above, where this was
measured directly). Some of the residual disagreement here is therefore the machine,
not the quantization, and this table cannot separate the two. The conclusion survives
because both candidate causes are small; the clean quantization evidence is the
single-clip precision sweep, which held everything but the weights fixed on one host.

### Why the absolute numbers are higher than the old clip's 7%

16-18% coverage CER against 7.3% before is not a regression; it is a harder and
more representative test set. The old clip was scripted single-speaker narration
with a complete verbatim transcript. This corpus has spontaneous speech,
multi-speaker overlap, code-switching mid-sentence, and heavily edited references
where the coverage metric must guess which insertions are cut material. Treat
16-18% as the new baseline for realistic material and 7% as the best case for
clean prepared narration.

English scores worse than Japanese (26-31% WER vs 16-18% CER), but the units are
not comparable: one substituted word is one WER error, while the same error is a
fraction of the characters in a CJK sentence. Both are consistent with the
transcripts reading as broadly correct on inspection.

## Whisper baseline on the same corpus (2026-07-29, M2 Ultra 128GB)

The obvious question the corpus makes answerable: how does this compare to
Whisper? Every Whisper size was run through `scripts/benchmarks/run_whisper.py`, which
scores with the *same* `scripts/metrics/eval_coverage.py` functions on the *same* cached 16kHz
files (verified byte-identical to what the Voxtral runs read), so these rows sit
next to the Voxtral rows without an asterisk. Engine: `mlx-whisper` 0.4.3,
library defaults (temperature ladder 0.0-1.0, `condition_on_previous_text=True`)
unless a row says otherwise, because that is what `mlx_whisper.transcribe` gives
a user.

| engine / config | JP coverageCER | EN coverageWER | x realtime |
|---|---|---|---|
| whisper large-v3-turbo, no-condition (mean of 6 runs) | **15.91%** ±0.94 | **22.24%** ±0.96 | 15.4x |
| **voxtral 30s b32 kv8** | 16.44% | 26.55% | **31.3x** |
| whisper large-v3, no-condition | 17.36% | 23.57% | 9.2x |
| voxtral 60s b16 kv8 | 18.21% | 27.75% | 21.4x |
| whisper large-v3-turbo | 24.97% | 30.69% | 17.3x |
| whisper large-v2 | 25.02% | 28.05% | 15.0x |
| whisper small | 28.61% | 25.80% | 26.4x |
| whisper medium | 28.93% | 23.52% | 21.8x |
| whisper base | 29.96% | 32.16% | 25.5x |
| whisper large-v3 | 39.91% | 29.37% | 9.1x |
| whisper large-v3-turbo, autodetect | 50.14% | 22.73% | 24.3x |
| whisper tiny | 59.27% | 40.47% | 46.4x |
| whisper large-v3-turbo, greedy | 84.92% | 93.00% | 82.3x |
| kotoba-whisper v2.2, official pipeline, chunk 10s | 26.16% | (JP only) | *not comparable* |
| kotoba-whisper v2.2, via mlx-whisper | 53.53%-94.23% | (JP only) | *see below* |

Kotoba rows are torch/MPS via its official chunked pipeline, so their accuracy is
comparable but their throughput is not; the MLX rows for it are an engine
mismatch, explained below. All other rows are MLX.

### The accuracy winner is whisper large-v3-turbo + no-condition; Voxtral wins speed

**turbo-nocond is more accurate on both languages, and on a 14-file corpus that is
now statistically resolved.** Voxtral's claim is throughput (1.7-2.0x) plus
determinism.

The Japanese question took three corpus sizes to settle, which is itself the
lesson. On 5 Japanese files the gap was 0.5 points and unresolvable; the run-to-run
noise of Whisper's sampler was as large as the effect. On 12 Japanese files the gap
is 2.0 points and the whole interval clears the baseline:

| corpus | JP files | voxtral | turbo-nocond | verdict |
|---|---|---|---|---|
| original | 5 | 16.44% | 15.91% ± 0.94 (n=6) | not resolvable |
| corpus_v2 | 12 | 16.08% | **14.07% ± 0.17** (n=3) | **whisper better** |

| | Japanese | English | speed |
|---|---|---|---|
| accuracy winner | **whisper turbo-nocond** | **whisper turbo-nocond** | - |
| resolved at n=12/2? | yes, 3/3 runs, CI [13.66, 14.48] vs 16.08 | yes, 3/3 runs, CI [20.38, 24.39] vs 26.55 | - |
| speed winner | voxtral, 1.7x | voxtral, 1.7x | **voxtral** |

Note the SD collapsed from 0.94 to 0.17 when the corpus grew. More files means each
file's sampling luck matters less to the aggregate, so the same engine looks far
more reproducible on a bigger set. That is a reason to distrust small-corpus
repeatability figures, not evidence that Whisper became deterministic.

Paired per-file bootstrap (`scripts/benchmarks/compare_engines.py`, 20k resamples,
length-weighted, same method as the verification table above):

| comparison | JP diff | 95% CI | EN diff | 95% CI |
|---|---|---|---|---|
| turbo-nocond vs voxtral | -1.51 | [-4.62, +0.67] | **-4.56** | [-7.27, +1.74] |
| large-v3-nocond vs voxtral | +0.91 | [-2.46, +4.45] | -2.97 | [-7.34, +7.20] |
| turbo (defaults) vs voxtral | **+8.52** | [+1.12, +13.24] | **+4.14** | [+3.09, +6.58] |
| medium vs voxtral | **+12.49** | [+7.47, +21.91] | **-3.02** | [-4.25, -0.17] |
| large-v2 vs voxtral | **+8.58** | [+3.86, +13.13] | **+1.50** | [+1.12, +1.67] |
| large-v3 vs voxtral | **+23.47** | [+11.16, +38.60] | **+2.82** | [+2.30, +4.05] |
| small vs voxtral | **+12.16** | [+4.90, +17.03] | -0.74 | [-2.54, +3.43] |
| tiny vs voxtral | **+42.83** | [+24.83, +67.27] | **+13.92** | [+8.92, +25.58] |

Positive means Whisper is worse. Note these are single-run comparisons, which is
why the repeat-run analysis below supersedes them for turbo-nocond. What holds
up:

**At library defaults, Voxtral beats every Whisper size on Japanese** by +8.5 to
+42.8 points, all CIs excluding zero, at 1.8-3.4x the speed. That is the honest
default-vs-default result, and it is also a statement about Whisper's defaults
being bad for long-form (see the no-condition section).

**On English, Voxtral is behind**, and this is the most robust accuracy finding in
the comparison: whisper-medium by 3.02 points (CI excluding zero) at 0.69x the
speed, and turbo-nocond by 4.56 points in every one of 6 repeat runs.

### Whisper is nondeterministic, so it needs a run distribution, not a rerun

Voxtral decodes greedily and reproduces byte-identically, so one run *is* its
score. Whisper's temperature-fallback ladder samples when a segment trips the
compression-ratio or logprob threshold, so one run is a draw from a distribution.
Six repeat runs of the identical turbo-nocond config on identical audio
(`scripts/benchmarks/repeat_distribution.py`):

| | JP coverageCER | EN coverageWER |
|---|---|---|
| the 6 runs | 14.77, 14.93, 15.90, 16.22, 16.35, 17.29% | 21.17, 21.71, 21.98, 22.02, 22.61, 23.94% |
| mean, SD | 15.91%, SD 0.94 | 22.24%, SD 0.96 |
| 95% t-interval on the mean | [14.92%, 16.90%] | [21.24%, 23.24%] |
| voxtral (deterministic) | 16.44% | 26.55% |
| runs beating voxtral | **5 of 6** | **6 of 6** |
| verdict | not resolved: interval straddles 16.44% | **whisper better: interval entirely below** |

The right test here is one-sample: only one side has sampling error, so the
Whisper mean gets a t-interval and Voxtral enters as a constant. Bootstrapping
over files answers a different question ("would this hold on other audio"), which
`compare_engines.py` covers.

**English is settled: turbo-nocond is better, in every run, by 4.3 points at the
mean.** Japanese is genuinely unresolved rather than a tie: the point estimate
favours turbo-nocond (15.91% vs 16.44%) and 5 of 6 runs beat Voxtral, but the
distribution crosses Voxtral's value, so a single Whisper run can land on either
side. Quoting one Whisper Japanese number as "the" result is what the 2.52-point
run-to-run spread makes indefensible.

Per file, the disagreement is structural rather than noisy. Whisper wins all 6
runs on three files (rec-05, rec-06, rec-19) and loses all 6 on three others
(rec-02, rec-04, rec-18); only rec-03 flips. So the aggregate is decided by which
files dominate the length weighting, not by sampling luck. On rec-04 Whisper
produced the identical 11.91% in all six runs, i.e. that file never triggers
fallback at all.

Pure greedy decoding is not a fix for the nondeterminism: `--greedy` (temperature
0.0, no ladder) collapses to 84.92% / 93.00%, because the fallback is what
rescues segments that enter a repetition loop. The ladder is load-bearing, and
the nondeterminism comes with it.

### `condition_on_previous_text=False` is worth more than model size

The single largest Whisper lever here, bigger than any size step:

| model | defaults | no-condition | change |
|---|---|---|---|
| large-v3 | 39.91% | 17.36% | **-22.6 points** |
| large-v3-turbo | 24.97% | 14.93% | **-10.0 points** |
| kotoba-whisper v2.0 | 91.47% | 53.20% | -38.3 points |

The mechanism is the known Whisper failure mode: conditioning each 30s window on
the previous window's text lets a repetition loop, once started, feed itself
across windows. Per-file evidence on large-v3: the 26-minute file goes 68.8% ->
14.5% and the 52-minute file 58.9% -> 24.1%, while the short 13-minute file
barely moves (15.9% -> 12.4%). Long files are where the loop has room to
compound. Voxtral has no equivalent knob and no equivalent failure, because it
decodes independent chunks.

This also inverts the size ranking: at defaults large-v3 (39.91%) is *worse* than
small (28.61%) on Japanese, purely from loop instability, and turbo beats the
full large-v3 it was distilled from. Anyone quoting "large-v3 is the best
Whisper" on long-form Japanese should check this flag first.

### Whisper needs to be told the language; that is not free

Voxtral takes no language token. Whisper does, and all three ways of supplying it
on this mixed corpus cost something:

| approach | JP coverageCER | note |
|---|---|---|
| per-file, from the reference | 24.97% | what the table above uses |
| Whisper's own 30s autodetect | 50.14% | +25.2 points |
| force `ja` for every file | (unusable) | English files score ~100% WER |

Autodetect returned **Russian** for two Japanese files (and on tiny, 102-106% CER
with extra_ratio 8.5). That is not a random misfire: these recordings genuinely
contain Russian side conversation, so a 30s window is a bad sample of a 90-minute
file. Since the harness knows each file's language from its reference script, the
main table gives Whisper that information for free; the autodetect row is what a
zero-config user would actually get. `run_whisper.py` refuses `--language` on a
mixed-script set rather than silently producing the 100% rows.

### The result is not an artifact of the coverage metric

Coverage CER excuses insertion runs longer than `min_cut` as omitted-from-
reference audio, and Whisper's `extra_ratio` on Japanese (2.2-2.7) is higher than
Voxtral's (2.1), so the threshold could in principle be flattering Whisper.
`scripts/benchmarks/sweep_min_cut.py` across engines says it is not:

| config | mc10 | mc20 | mc30 | mc50 | mc80 |
|---|---|---|---|---|---|
| turbo-nocond | 13.85% | 14.78% | 14.93% | 16.03% | 16.84% |
| voxtral c30b32_kv8 | 14.97% | 15.89% | 16.44% | 17.81% | 20.23% |
| large-v3-nocond | 13.32% | 15.53% | 17.36% | 20.10% | 23.39% |
| large-v3-turbo | 22.76% | 23.99% | 24.97% | 26.58% | 28.05% |
| medium | 21.81% | 26.23% | 28.93% | 32.18% | 36.49% |

The ordering of the two leaders is stable across the whole range, and Voxtral's
disadvantage against turbo-nocond *shrinks* as the threshold tightens (-1.1 at
mc10 vs -3.4 at mc80), i.e. the excusal rule is if anything working against
Voxtral, not for it. One rank swap does occur at mc10, where large-v3-nocond
overtakes turbo-nocond; both are inside the rerun noise of each other anyway.

### kotoba-whisper: 65 points of the result was the runner, not the model

The Japanese-finetuned model (distil-large-v3 based) first measured 91.47%, which
would have placed it last among the models tested. That number was almost entirely an
artifact. Chasing it down took three steps, and each one is a caution about
benchmarking someone else's model through your preferred runtime:

| what was run | corpus coverageCER | narration | extra_ratio | engine |
|---|---|---|---|---|
| `kaiinui/kotoba-whisper-v2.0-mlx` (third-party npz port) | 91.47% | 88.70% | 0.13 | mlx-whisper |
| same, `--no-condition` | 53.20% | - | 1.03 | mlx-whisper |
| official v2.2 weights, own MLX conversion | 94.23% | 79.88% | 0.10 | mlx-whisper |
| same, `--no-condition` | 53.53% | 44.78% | 1.03 | mlx-whisper |
| official v2.2, official pipeline, `chunk_length_s=30` | 49.57% | 39.52% | 1.29 | transformers |
| official v2.2, official pipeline, `chunk_length_s=20` | 27.82% | **16.55%** | 1.35 | transformers |
| official v2.2, official pipeline, `chunk_length_s=15` | 30.40% | 20.88% | 1.37 | transformers |
| official v2.2, official pipeline, `chunk_length_s=10` | **26.16%** | 23.71% | 1.39 | transformers |

**Step 1: convert the official weights yourself.** `kotoba-tech/kotoba-whisper-v2.2`
carries the latest ASR weights (byte-identical `model.safetensors` to v2.1; v2.2
only adds a diarization pipeline).

*Correction, 2026-08-04:* this section originally added "and both supersede the
v2.0 the third-party port was built from". That is wrong, and it was assumed from
the version numbers rather than checked. v2.0 and v2.2 hold **the same ASR
weights**: loading both `model.safetensors` files and differencing all 539 tensors
gives a max absolute difference of exactly 0.0. The files have different sha256s,
but only from container metadata. So the 91.47% figure in the first row above was
entirely the runner (sequential long-form on a 2-layer decoder), with nothing
attributable to the checkpoint being older. v2.2's additions are separate
post-processing models, not retrained weights. Converting with the official `mlx-examples/whisper/convert.py`
worked, one snag: the current converter writes `model.safetensors` while the
installed `mlx-whisper` loader looks for `weights.safetensors`, so the file needs
renaming. This changed nothing about the result (94.23% vs 91.47%), which is the
useful part: **the third-party port was not the problem.**

**Step 2: find the actual failure mode.** Not early termination, as the first pass
assumed. On a 600s slice the MLX run reached the final second of audio but emitted
2053 characters where turbo emitted 2630, with 44 zero-duration and 12 empty
segments. It was under-transcribing continuously, dropping speech inside windows.

**Step 3: run it the way its authors do.** The model card uses
`pipeline(..., chunk_length_s=15)`, i.e. transformers' *chunked* long-form
algorithm. `mlx-whisper` implements Whisper's *sequential* 30s-window algorithm.
That mismatch is the whole story: on the same 600s slice the official chunked
pipeline emitted 2527 characters against MLX-sequential's 2053. Distil models keep
only 2 decoder layers, and the sequential algorithm leans on the decoder to
maintain state across windows, which a 2-layer decoder cannot do.

Same weights, three runners, **68 points of spread** (94.23% to 26.16%). The
chunk-length sensitivity is the signature of the same cause: 30s -> 20s -> 15s ->
10s gives 49.57% -> 27.82% -> 30.40% -> 26.16%, i.e. a large gain from dropping
below 30s and then a shallow, non-monotonic basin from 20s down. The optimum
depends on the material: 10s is best on the spontaneous corpus while 20s is best
on the clean narration clip (16.55%, versus 23.71% at 10s), so the model-card's
15s is a reasonable default rather than a tuned value. Anyone using kotoba should
sweep this on their own audio; it is worth up to 23 points.

Corrected verdict: kotoba-whisper is **not** unusable, but at 26.16% it is still
well behind both Voxtral (16.44%) and turbo-nocond (15.91% mean) on this
Japanese material, despite being the only Japanese-specialised model tested.
Its numbers come from `scripts/benchmarks/run_kotoba_hf.py` on torch/MPS, so its accuracy
is comparable to the MLX rows (same audio, same scoring) but **its throughput is
not** and is excluded from speed claims.

### Clean narration: the gap nearly closes

The same models on the original single narration clip, where the reference is
complete (`extra_ratio` ~1.0) and plain CER is meaningful:

| engine | coverageCER | plain CER | x realtime |
|---|---|---|---|
| **voxtral (M2 Ultra 128GB, 60s b16 kv8)** | - | **7.28%** | 21.2x |
| whisper large-v3-turbo | 8.28% | 9.08% | 44.3x |
| whisper large-v3, no-condition | 8.87% | 8.87% | 24.2x |
| whisper large-v3-turbo, no-condition | 10.42% | 10.42% | 73.5x |
| whisper large-v3 | 12.91% | 13.67% | 16.6x |
| whisper small | 13.96% | 14.89% | 59.2x |
| whisper medium | 15.17% | 15.17% | 29.9x |
| whisper large-v2 | 15.20% | 16.96% | 23.0x |
| kotoba-whisper v2.2, chunk 20s | 16.55% | 16.55% | *n/c* |
| whisper base | 22.73% | 25.97% | 90.5x |
| whisper tiny | 34.51% | 36.27% | 101.8x |
| kotoba-whisper v2.0 (third-party npz port) | 88.70% | 88.70% | 43.4x |

Voxtral's 7.28% leads turbo's 9.08%, a much narrower margin than the corpus
suggests, and turbo is faster on this clip. Two caveats on reading it: this is
n=1, which is the failure mode documented at the top of this file, and
`--no-condition` *hurts* on clean audio (turbo 8.28% -> 10.42%) exactly where it
helped enormously on long spontaneous audio. The flag is a long-form-instability
fix, not a general improvement.

### What this does and does not settle

Settled:
- **Whisper large-v3-turbo with `condition_on_previous_text=False` is the accuracy
  winner**, on English decisively (6/6 runs, mean 4.3 points ahead) and on
  Japanese by point estimate (5/6 runs).
- **Voxtral is the speed winner at comparable accuracy**, 31.3x vs 15.4x, i.e.
  2.0x, and it is deterministic, needs no language hint, and needs no
  long-form-stability flag.
- At library defaults Voxtral beats every Whisper size on Japanese (+8.5 to
  +42.8 points), which says as much about Whisper's defaults as about Voxtral.
- `condition_on_previous_text=False` is mandatory before quoting any long-form
  Whisper number; it is worth up to 22.6 points.
- Running a model through the wrong long-form algorithm can cost more than the
  model choice: kotoba-whisper spans 68 points across three runners on identical
  weights.

Not settled: whether Voxtral or turbo-nocond is better on **Japanese**. The
Whisper run distribution [14.92%, 16.90%] straddles Voxtral's 16.44%, so a single
run can land either side; and separately the corpus resolves only ~3.2 points at
n=7 for generalisation. Both need more audio, not more reruns.

Untested: Whisper's own chunking/VAD front-ends (faster-whisper, whisperX), where
long-form instability is usually handled in practice and which would likely help
turbo further; Whisper quantization other than fp16 defaults; and whether a
chunked runner would lift kotoba nearer its published Japanese numbers.

Reproduce with `scripts/benchmarks/run_whisper_sweep.sh`, `scripts/benchmarks/run_kotoba_hf.py`,
then `summarize_engines.py`, `compare_engines.py` and `repeat_distribution.py`.

## Final corpus: 20 files, ~7.9h (2026-08-04, M2 Ultra 128GB)

The corpus was grown from 7 files to 20 (17 Japanese, 3 English) to settle the
Japanese ordering, which was unresolvable at 5 Japanese files. Sources: 7 more
recordings of the same kind as the original set, and 5 published videos with
human-authored subtitle tracks.

**Vetting mattered more than collecting.** Every candidate was checked by
transcribing a 3-minute mid-file sample with language autodetect and comparing the
spoken language to its reference script:

| source | candidates | accepted | rejected |
|---|---|---|---|
| existing recordings | 15 | 7 | 8 (audio in a different language than the transcript) |
| published videos | 10 | 5 | 5 (English narration with translated Japanese subtitles) |

More than half failed. One candidate had a Japanese transcript over speech in
another language, and Voxtral scored it 199.51% coverage CER, which is the metric
correctly flagging a broken pair rather than a model failure. A downloader also
serves the *default* audio track, so the track language has to be pinned explicitly
or you silently benchmark a dub.

### Accepted recordings open in another language, which is the clearest evidence yet that the coverage metric works

Worth recording because it looks like catastrophic failure and is not. All 8 accepted
session recordings begin with untranscribed studio chatter in Russian before the
Japanese content starts; the references cut it, since it is not part of the material.
Voxtral transcribes it, correctly and fluently, so its output is 4-42% Cyrillic on
those files while every reference is 0%. Spot-checking mid-file confirms it switches
to Japanese and tracks the reference; spot-checking the openings confirms the Russian
is real speech, not transliterated Japanese.

Plain CER therefore reports near-total failure on exactly the files where the model is
doing the right thing. From the 20-file run
(`scripts/benchmarks/corpus_n20_c30b32_kv8.json`, the run later voided for GPU
contention; accuracy is unaffected by contention, and no timing figure is quoted here),
the seven worst by plain CER:

| file | coverage | coverageCER | plain CER | chars excused | extra |
|---|---|---|---|---|---|
| rec-02 | 86.6% | **17.4%** | 150.7% | 8453 | 2.32 |
| rec-06 | 83.4% | **19.9%** | 148.2% | 2966 | 2.23 |
| rec-03 | 86.8% | **17.1%** | 144.5% | 12522 | 2.27 |
| rec-05 | 89.1% | **14.6%** | 134.1% | 8957 | 2.20 |
| rec-07 | 98.1% | **5.5%** | 99.3% | 1002 | 1.97 |
| rec-04 | 90.3% | **11.7%** | 64.4% | 1337 | 1.50 |
| rec-12 | 82.2% | **18.0%** | 37.4% | 82 | 1.18 |

**Four files exceed 100% plain CER while scoring 15-20% on coverage CER**, a gap of up
to 133 points, because the hypothesis is more than twice the reference length
(`extra` above 2) and almost all of the excess is real speech the reference omits.
`rec-07` is the sharpest single case: 99.3% plain against 5.5% coverage, with 98.1% of
reference characters found correctly.

Any evaluation of this corpus on plain CER would rank configs by how much real speech
they *skip*. That is the failure mode `eval_coverage.py` exists for, and this is an
independent demonstration of it on 20 files, most collected long after the metric was
written.

**Two accepted files are the same recording dubbed twice**, and they are worth
calling out because they passed vetting legitimately and then broke a harness. They
are `rec-14` (Japanese audio, Japanese subtitles) and `rec-20` (English audio,
English subtitles): identical duration to the sample (778.346688s) and identical
byte count, different audio, different references. Each pair is internally
consistent, so language-vetting had no reason to reject either, and keeping both is
defensible: they isolate language while holding content, speakers and production
fixed.

What they broke is anything assuming one language per corpus. `run_timing_sweep.py`
hardcoded Japanese and drove `rec-20` into a repetition loop, costing that row (see
the timing section).

They were also a latent hazard for `compare_engines.py` and `repeat_distribution.py`,
which match files across runs by rounded duration rather than by name, deliberately,
so that a scrubbed run can be compared against a local one. A duration-only key
collapses these two into one row and silently drops the other. It had never bitten
because the two had not appeared together in an accuracy run, and the 20-file corpus
is the first set containing both. Both scripts now key on (duration, unit), which
separates a dub from its original because the two differ in script; verified to
reproduce the previously published paired numbers exactly on the 7-file runs.

### Result

| | Voxtral (deterministic) | turbo-nocond, 3 runs | verdict |
|---|---|---|---|
| JP coverageCER, 17 files | 16.22% | 14.69 / 14.88 / 15.22 -> **14.93% +/- 0.27** | CI [14.27, 15.59] entirely below -> **whisper** |
| EN coverageWER, 3 files | 25.24% | 21.73 / 22.48 / 23.42 -> **22.54% +/- 0.84** | CI [20.45, 24.64] entirely below -> **whisper** |
| x realtime | **22.8x** | 15.0-20.1x | **voxtral, ~1.4x** |

Provenance, since the two halves of this table are not equally auditable. Neither
column has a committed measurement: all four accuracy figures and both x-realtime
figures come from the original session and no result file survives from it. See gap 1
at the top of this file. Nothing here is suspected wrong, but nothing here has been
independently reproduced either.

**Superseded by the re-measurement later on 2026-08-06** (see the section near the top of
this file). The verdict held, and Voxtral's two accuracy figures reproduced exactly. The
x-realtime column here should not be used: it was measured on a host whose state was never
recorded, and the same configs measure 29.8x and 21.3-23.3x on a verified-idle machine.
Whisper's accuracy column here is a valid draw from its distribution, just a different one
(14.93% then, 14.74% on the re-run).

**Corrected:** this paragraph previously said Voxtral's column had been re-measured on
2026-08-06, reproduced to 0.07 points, and was committed as
`scripts/benchmarks/corpus_n20_c30b32_kv8.json`, which it offered as evidence that the
original session was sound. That re-run is the one gap 1 records as **voided**: the host
was generating video on the same GPU throughout. Accuracy is unaffected by contention
under greedy decoding, so the 0.07-point agreement was probably real, but the session
was voided whole rather than quoted in halves, precisely so that no later reader could
treat one column of it as confirmed. Citing it here did that. The per-file table earlier
in this section is from the same voided run and is retained only as the demonstration
that plain CER misreads this material, which is a property of the metric and does not
depend on the run's timings.

The ordering held at every corpus size, and only became resolvable as n grew:

| corpus | JP files | voxtral | turbo-nocond | verdict |
|---|---|---|---|---|
| original | 5 | 16.44% | 15.91% +/- 0.94 (n=6) | not resolvable |
| corpus_v2 | 12 | 16.08% | 14.07% +/- 0.17 (n=3) | whisper better |
| final | 17 | 16.22% | 14.93% +/- 0.27 (n=3) | whisper better |

Note Voxtral's own number barely moved (16.44 -> 16.08 -> 16.22) while the corpus
more than tripled, which says the original 5 files were not unrepresentative. What
changed is the *resolution*, not the estimate.

**These rows have no committed JSON**, unlike every other table in this file. The
20-file runs were scored in a session whose result files were not saved into
`scripts/benchmarks/`, so the headline comparison is the one number here a reader cannot
audit. The 7-file and 17-file rows that *are* committed all reproduce exactly, and
`scripts/benchmarks/kotoba_mlx_c*.json` carries the 17-file Japanese set, so the corpus
itself is not in doubt; the aggregate is simply unverifiable from the repo. Re-running
it is the top item under "Gaps in the benchmark record" below.

## Timing quality: drift versus cue breaks (2026-08-04)

Every metric above scores text and discards timestamps. `scripts/metrics/eval_timing.py` scores
the timestamps, on the 7 files that have authored SRT references. Two independent
failure modes, reported separately because a model can pass one and fail the
other:

  * **drift**: words correct, times wrong. Measured by aligning the two texts and
    comparing timestamps at matched anchors, so cut material cannot masquerade as
    drift. Constant offset is separated from slope, because a whole file shifted
    400ms is trivially fixable while error that *grows* through the file is not.
  * **cue breaks**: times right, text chopped mid-phrase. Boundary F1 against the
    author's cue ends, plus the rate of hypothesis cue ends landing inside a
    reference cue.

Measured at the cue grouping the CLI actually ships (`gap_s=1.2, max_chars=28`) and
with Whisper's language taken per file from its reference.
`scripts/benchmarks/timing_engines_shipped_cues.json`, n=7, every row above the metric's
50% anchor-coverage bar (lowest is 75.6%), so nothing is excluded:

| | Voxtral | whisper turbo-nocond |
|---|---|---|
| median timing error | 278 ms | **250 ms** |
| median p95 error | **786 ms** | 1908 ms |
| worst drift slope | **25.3 ms/min** | 122.7 ms/min (unstable, see below) |
| break F1 | 37.4% | **56.0%** |
| mid-phrase splits | 58.1% | **41.9%** |

Read Whisper's *worst* slope as an order of magnitude, not a value. Two runs of the
identical config gave 122.7 and 180.7 ms/min on the file that produces it, because
Whisper samples: its output on that file differed between runs, one cue splitting
where the other did not. Voxtral is greedy and reproduced byte-identically. Every
other Whisper row matched exactly across the two runs, so this is one file's
instability rather than general noise, and it does not touch the comparison: both
values are 5-7x Voxtral's worst.

An earlier version of this table reported 42.8% / 52.6% / 58.3% and needed a
paragraph explaining which of two columns to read. Both problems were harness bugs,
now fixed, and they pulled in opposite directions:

- **Voxtral's 42.8% was measured at `gap_s=0.7, max_chars=32`**, which the project
  deliberately does not ship (see "Cue defaults" below). At the shipped pair it is
  **37.4%**, so the published figure flattered the default a user gets by 5.4 points.
- **Whisper's language was hardcoded to `ja`.** One corpus file is English audio
  with an English reference, so it was transcribed as Japanese; it looped and
  emitted degenerate timestamps (identical starts and ends, one cue ending before
  it began) for 9.8% anchor coverage. The metric then discarded the row, which read
  as Whisper struggling on hard audio rather than the harness asking for the wrong
  language. With the language detected per file the same row reaches **93.1%**
  coverage and scores 43.5% F1, and the n=6-versus-n=7 choice disappears.

Both directions of correction leave the ordering unchanged, which is why the
conclusion below still stands. The magnitude does move: Whisper's break advantage is
larger than published (18.6 points, not 9.8), and Voxtral's drift advantage is
larger too (its worst slope is 25.3 ms/min, not 37.1, because that file's worst case
came from the discarded cue config).

**They split cleanly: Voxtral holds timing, Whisper cuts cues.** Voxtral's worst
drift slope of 25.3 ms/min implies a few hundred ms across an hour, and it is under
26 ms/min on every file. Whisper's 122.7 ms/min is over a second of accumulated
drift on a ten-minute file, which is the failure that makes a subtitle file unusable
and is invisible to CER. Voxtral's p95 error is lower on all 7 files, 2.4x lower at
the median. In the other direction Whisper's cue boundaries match human editing
better on 6 of the 7 files, the exception being the prepared-narration clip.

That asymmetry is actionable: Voxtral is better where it is hard to fix (timestamp
accuracy comes from the model) and worse where it is easy (cue grouping is our own
heuristic, and a deliberately conservative one).

Median error is the one column Whisper nominally wins (250 vs 278 ms), and it should
not be read as a Whisper advantage: the median says nothing about the tail, and the
p95 and slope columns are where a subtitle file becomes unusable.

### Cue defaults: an n=1 tuning that the corpus overturned

This is the clearest worked example of n=1 tuning going wrong in this project, so
it is worth the space.

The first pass swept 376 combinations against the single timed reference then
available and moved the defaults from `(gap_s=1.0, max_chars=32)` to
`(1.2, 28)`, reporting break F1 up from 35.4% to 43.6%. Its own docstring flagged
the risk: n=1, `gap_s` flat by median across 0.6-1.3, and `max_chars=28`
suspiciously close to that reference's mechanical 15-char line-wrap width.

With 7 timed references (`scripts/benchmarks/sweep_cues.py`, re-cueing a cached token
stream rather than re-decoding, so a 72-point grid costs seconds) the picture
inverts. `gap_s` is not flat at all, it is monotonic, and it points the *opposite
way* from the n=1 result:

| gap_s (max_chars 32) | break F1 | mid-phrase | cues/ref |
|---|---|---|---|
| 0.6 | 44.5% | 55.7% | 1.36 |
| 0.7 (n=7 optimum) | 42.3% | 56.0% | 1.25 |
| 0.8 | 40.2% | 56.5% | 1.11 |
| 1.0 | 36.2% | 59.1% | 0.99 |
| 1.2 | 35.9% | 57.8% | 0.92 |
| 1.2 / chars 28 (**shipped**) | 37.0% | 58.1% | 1.01 |

Per file, `1.2/28` against the n=7 optimum:

| file | 1.2/28 (shipped) | 0.7/32 |
|---|---|---|
| narration-jp (**the file 1.2/28 was tuned on**) | **46.1%** | 35.9% |
| rec-16 | 38.2% | **49.2%** |
| rec-13 | 28.6% | **36.5%** |
| rec-14 | 32.4% | **41.2%** |
| rec-17 | 33.3% | **41.2%** |
| rec-15 | 41.9% | **48.8%** |
| rec-20 | 38.8% | **43.6%** |

**The one file where `1.2/28` wins is the one it was fitted to.** It loses on all
six held-out files. That is overfitting visible in a single table, and it is why the
n=1 caveat at the top of this file exists.

**Neither sweep result was applied, and `1.2/28` still ships.** The n=7 sweep is
better evidence than the n=1 one, but it is not evidence of good segmentation: all
seven timed references come from one editor, sharing line-wrap width and pause
conventions, so its optimum is agreement with one house style. Shipping it would
export that style to every user, which is a worse default than a conservative pair
that no reference chose. `mlx_asr/output.py` documents this and a test pins it.

**The published break figure used to describe the config that does not ship.** The
first end-to-end timing run (`scripts/benchmarks/timing_engines_newcues.json`) reported F1
42.8%, but it was decoded at `0.7/32`. Both configs were then re-run on the same
audio with everything else fixed, which isolates the cue grouping as the only cause:

| Voxtral, 7 timed files | break F1 | mid-phrase | median drift | p95 |
|---|---|---|---|---|
| **`1.2/28` (shipped)** | **37.4%** | 58.1% | 278 ms | 786 ms |
| `0.7/32` (n=7 sweep optimum) | 42.9% | 57.0% | 258 ms | 829 ms |
| the original published run | 42.8% | 57.1% | 258 ms | 829 ms |

The sweep-optimum arm reproduces the published run to 0.1 points, confirming what it
was measured at. `scripts/benchmarks/timing_engines_shipped_cues.json` and
`scripts/benchmarks/timing_engines_sweep_optimum.json` hold the two arms.

The timing table above uses the shipped number, since a published figure has to
describe what a user gets. Drift is barely affected, as expected: regrouping cues
cannot move a token's timestamp, and the 258 -> 278 ms median shift is only how
per-cue interpolation samples the timeline. Note the shipped config is actually
*better* on p95 drift (786 vs 829 ms), so the 5.4 break points are not bought with
timing accuracy.

The 37.4% end-to-end figure and the 37.0% sweep row are the same measurement taken
two ways, and the 0.4-point gap is systematic rather than noise. `sweep_cues.py`
scores `build_cues` output directly; the end-to-end run scores the *written SRT*, and
`write_srt` clamps `end = max(end, start + 0.5)`, which nudges short cue ends later.
Measured at identical `0.7/32`: 42.8% via SRT against 42.3% via raw cues, with 26-27
of ~210 cues clamped. Quote the SRT number when describing files a user gets, and the
raw number when comparing grid points.

Why 0.7 rather than the top-scoring 0.6, had a sweep result been applied: F1 keeps
rising as the gap falls, but so does cue count (1.11 at 0.8, 1.25 at 0.7, 1.36 at
0.6). Part of that late gain is recall bought by emitting more boundaries, which F1
rewards and a reader does not.

`max_chars` is worth under a point anywhere in 28-72 and `max_dur_s` is inert on
this material at any value from 4 to 9 seconds, because the other two rules always
fire first. Both are kept as safety valves rather than tuned knobs. Cue placement
trails Whisper at either pair, so this is the weaker half of the project's timing
story, and the gap is wider than previously published (37.4% against 56.0%).

The three knobs are now reachable from the CLI (`--gap-seconds`, `--max-chars`,
`--max-dur-seconds`) so that a user with their own reference SRTs can sweep the one
parameter that matters, which is what this section recommends and what was
previously impossible without editing the library.

**Rejected**: extending cue ends to the next cue's start, the usual subtitle
convention, to fix a systematic early bias. It makes break F1 *worse* (43.6% ->
40.2 / 27.0 / 34.3% at 0.5 / 1 / 2s holds on the n=1 data) because every extended
end stops matching a human cue end. A 0.2s hold was F1-neutral and improved drift,
so it is the only variant worth revisiting on the larger set.

## kotoba-whisper on MLX: the driver was the problem, not the framework

The kotoba section above concluded that the model needs transformers' *chunked*
long-form algorithm, and left it running on torch/MPS with its throughput marked
"not comparable". That conclusion was half right. Chunked long-form is a property
of the **driver**, not of the framework: slice the audio, decode each window
independently, offset the timestamps, stitch. Nothing about that requires torch.

`mlx_asr/chunked.py` is that driver on MLX. Same weights (our own conversion of
`kotoba-tech/kotoba-whisper-v2.2`), 20-file corpus, 17 Japanese files:

| driver | runtime | coverageCER | x realtime | comparable? |
|---|---|---|---|---|
| chunked, 10s windows | **MLX** | 27.01% | **36.2x** | yes |
| chunked, 20s windows | MLX | 31.33% | 68.8x | yes |
| chunked, 30s windows | MLX | 49.71% | 72.7x | yes |
| chunked, 10s (authors' pipeline) | torch/MPS | 26.16% | 25.4x | no |
| sequential 30s windows (mlx-whisper) | MLX | 94.23% | 64.8x | no |

**MLX-chunked matches the torch reference to within a point (27.01% vs 26.16%) at
1.4x the throughput**, and its x-realtime is finally measured on the same footing
as every other row in this file. So the "not comparable" asterisk kotoba carried is
gone, and the dependency on torch for correctness is gone with it.

The chunk-length curve reproduces on MLX exactly as it did on torch (27.0 / 31.3 /
49.7% at 10 / 20 / 30s), which is the signature of the same mechanism: a 2-layer
distil decoder has less state to carry, so shorter independent windows hurt it
less. Note the throughput moves the opposite way (36 / 69 / 73x), because fewer
windows means less per-window overhead. 10s is the accuracy pick on this corpus and
20s won on clean narration, so this is the one knob to sweep per material.

Generalisation, worth stating because it is the reusable part: any distil-derived
Whisper checkpoint (2-4 decoder layers) is a poor match for a sequential driver
and should be routed to the chunked one. `models.infer_backend` now sends both
`kotoba` and `distil` repo ids to `mlx-chunked` for that reason.

## Every alias, run end to end on MLX (2026-08-04, M4 16GB)

Run because "the engines work" had never been checked per alias: Whisper was only
ever spot-checked on `tiny`, and `--model kotoba` turned out not to work at all.
All 9 aliases through the Homebrew binary, which has no torch installed.
`scripts/benchmarks/matrix_models.py`, raw rows in `scripts/benchmarks/matrix_models_m4.json`.

180s Japanese narration clip, plain CER against a complete reference, x-realtime
including model load.

| alias | works | CER | x-rt | cues |
|---|---|---|---|---|
| voxtral | yes | **11.76%** | 1.7x | 40 |
| whisper-turbo | yes | 13.67% | 15.1x | 56 |
| whisper-large-v2 | yes | 13.45% | 4.0x | 46 |
| whisper-large-v3 | yes | 14.41% | 6.1x | 47 |
| whisper-medium | yes | 16.42% | 10.8x | 38 |
| kotoba (v2.0 MLX) | yes | 22.67% | 9.8x | 51 |
| whisper-small | yes | 24.47% | 21.7x | 35 |
| whisper-tiny | yes | 32.10% | 51.4x | 31 |
| whisper-base | yes | 35.59% | 20.9x | 49 |

Read this as a coverage check, not a model ranking: n=1 clip, and it puts Voxtral
ahead of turbo, which the n=20 corpus reverses. `whisper-tiny` beating
`whisper-base` is the clearest sign of how little a single clip resolves.

### `--model kotoba` was broken, and the fix is to convert on first use

The alias pointed at `kotoba-tech/kotoba-whisper-v2.2`, which is
transformers-format. The MLX chunked driver cannot load it, and no MLX build of
v2.2 exists on the Hub, so the alias could not work for anyone without the local
conversion used to produce the numbers above.

The first fix pointed it at `kaiinui/kotoba-whisper-v2.0-mlx`, a published MLX
conversion, which does decode correctly (22.67% at 10s windows, 27.22% at 20s).
But that is one person's repo from September 2024, never updated, ~194 downloads a
month against the authors' 182k, and a default should not depend on it.

So `mlx_asr/convert.py` does the conversion instead, and `chunked.py` calls it
when it sees a transformers config. This is much less work than it sounds: the two
formats hold the same numbers, so it is a key rename (`self_attn` to `attn`, `fc1`
to `mlp1`, ...) plus one axis swap on the conv weights, because torch stores Conv1d
as (out, in, kernel) and MLX as (out, kernel, in). `mx.load` reads safetensors
directly, so **no torch is needed**, and `mlx-examples`' converter only imports it
to read the checkpoint and build a throwaway model object. The result is cached
under `HF_HUB_CACHE/mlx-asr-converted/`, so it costs a few seconds once per
machine.

Verified: the self-converted v2.0 produces output byte-identical to the
third-party conversion of the same checkpoint, on the same audio.

### v2.0 and v2.2 are the same ASR model, measured

This is what settles the version question, and it contradicts what the kotoba
section above originally assumed (see the correction there). Loading both
checkpoints' `model.safetensors` and differencing every tensor:

| | v2.0 | v2.2 |
|---|---|---|
| tensors | 539 | 539 |
| keys | identical | identical |
| max abs difference | | **0.0** |
| file sha256 | `0a96a84e...` | `e0ef3e7b...` |

The files differ only in container metadata. v2.2's own model card agrees, and
describes it as v2.0 plus (i) speaker diarization via `diarizers` and (ii)
punctuation via `punctuators`, both separate models loaded by its custom
transformers pipeline. `punctuators` depends on `torch>=1.9`.

So the answer to "do we need v2.2's improvements" is: they are not weights, they
are two extra torch models bolted on after ASR, and this driver never invoked
them. Choosing v2.0 gives up nothing that was reachable and avoids a torch
dependency. If diarization or punctuation is wanted later, the honest way to add
it is as an explicit post-processing step, not by changing the checkpoint.

### The torch backend is gone

`--model kotoba-torch` ran kotoba through the authors' torch/MPS pipeline and was
kept as a correctness reference. It has served that purpose: the numbers above and
in the kotoba section established that the MLX chunked driver is right, matching it
to within a point at 1.4x the throughput. Keeping the alias meant a ~2.5GB torch
dependency, plus a CUDA subtree, for a path with no reason to be chosen. Removed,
along with the `speed_comparable` field on `Model`, which existed only to mark that
one alias's throughput as incomparable and could no longer be false.

`transformers` itself stays in the tree, as a pure-Python dependency of mlx-audio
and mlx-lm used for tokenizer and config loading. Verified it never pulls torch:
importing it in the Homebrew tree prints `PyTorch was not found` and `torch` is
absent from `sys.modules` afterwards.

### Sampling versus determinism, confirmed directly

Three `whisper-base` runs on identical audio and identical flags produced three
different output files (two of them 33.69% and 35.59% CER). Two `voxtral` runs
produced byte-identical output. This is the documented behaviour, but it had only
been inferred from CER spread before, not from file hashes.

### A harness bug worth recording

The first version of `matrix_models.py` reported a perfect all-green matrix in
which every engine scored identically, because it never passed `--model` and so ran
the default engine nine times. Two different ASR engines cannot agree
byte-for-byte on 3 minutes of spontaneous speech, so the script now hashes each
output and fails with `HARNESS BUG` if any two match. A green matrix that is green
for the wrong reason is worse than a red one.

## Packaging for Homebrew (2026-08-04)

Recorded because every line here was established by trying the documented thing
and watching it fail, and because two of them were bugs in this project rather
than in Homebrew.

### The install is torch-free, which is what makes a formula possible at all

`mlx-whisper` declares `torch` in its metadata, and nothing this CLI calls
imports it: only its unused `torch_whisper.py` does. Verified rather than assumed,
by transcribing with every `whisper-*` alias in a venv with no torch installed.
Dropping it removes ~2.5GB and the entire nvidia/triton/CUDA subtree, which on a
Mac is dead weight in the literal sense. If a future mlx-whisper needs torch for
real, the formula is the thing that breaks first.

### numba pinned numpy, and the resolver's answer was a 2021 release

`uv sync --extra whisper` failed to build on Python 3.13+ with an llvmlite error
that had nothing to do with the real cause. The chain: mlx-whisper needs numba (a
hard import in its word-timestamp path), numba 0.66.0 caps `numpy<2.5`, the lock
had numpy 2.5.1, so the resolver kept numpy and backtracked *numba* to 0.53.1, a
2021 release whose `setup.py` refuses any Python above 3.9. Capping numpy in the
extra keeps numba on 0.66.0, which ships a cp314 wheel and needs no build.

Worth generalising: a package that caps a common dependency can be silently
backtracked years, and the error surfaces in its transitive dependency.

### `--model kotoba` could not have worked for anyone

The MLX chunked driver needs an MLX conversion of the weights, the registry named
`kotoba-tech/kotoba-whisper-v2.2` (transformers format), and no MLX build of v2.2
is published on the Hub. The benchmark numbers above are real, but they were
produced against a local conversion that no user had. The failure surfaced from
inside mlx-whisper as `TypeError: ModelDimensions.__init__() got an unexpected
keyword argument '_name_or_path'`, which names a config key rather than the
missing conversion step.

`chunked.py` now checks the config format up front and raises with the conversion
command and the `kotoba-torch` alternative. It stays quiet when it cannot tell
(uncached repo, path that does not exist yet) so mlx-whisper's own clearer errors
still surface.

### The ffmpeg fallback did not fire, because the ImportError came from a call

`load_audio_16k` prefers mlx-audio and falls back to ffmpeg, with the fallback
guarding the *import*. But `mlx_audio.audio_io.read` imports its decoder
(miniaudio) lazily inside the function, so on an install without miniaudio the
ImportError was raised from the call and escaped the guard. Every transcription
in the Homebrew-shaped environment failed with `No module named 'miniaudio'`. The
`try` now spans the call.

### Homebrew's Python helpers cannot install wheels

| documented approach | what happens |
|---|---|
| `venv.pip_install` | `std_pip_args` hard-codes `--no-binary=:all:`, so it compiles every wheel from source. `mlx` has no sdist, so this cannot succeed. |
| `virtualenv_create(..., without_pip: false)` | `ArgumentError`: deprecated for Python 3.12+. |
| installing `r.cached_download` | Homebrew caches as `<sha256>--name-1.2.3-cp313-...whl`; pip parses wheel filenames strictly and rejects it with "Invalid wheel filename (wrong number of parts)". |
| `pip install <project>` | needs hatchling, which has no Homebrew formula and cannot be fetched in the build sandbox. |
| `depends_on "numpy"` | Homebrew ships 2.5.1, and numba caps `<2.5`. |
| `depends_on "protobuf"` / `"sentencepiece"` | both install C++ libraries only, with no python3.13 bindings, so `import google.protobuf` fails. |

So the formula creates the venv with Homebrew, drives the brewed interpreter's pip
at it with `--python=`, stages each wheel under its original filename, and copies
the pure-Python package into site-packages with hand-written entry points. This is
also why it is a tap and not a homebrew-core submission: core's from-source policy
is incompatible with a dependency tree containing `mlx`.

Verified end to end by reproducing the formula's exact steps into a
`--without-pip` venv: 66 wheels, all checksums matching, both engines
transcribing, and the Voxtral output byte-identical to the pip install (which is
the determinism claim holding across two installation methods).

## Competing Apple Silicon runners (2026-08-04, M2 Ultra 128GB)

Run because the alternative was an unfalsifiable speed claim in the README. Same
935s Japanese narration clip, complete reference so plain CER is valid,
`--language ja`, model load and warm-up excluded, serialized so nothing contends
for the GPU. `scripts/benchmarks/compete_*.json`.

| runner | model / quant | GPU? | x realtime | plain CER |
|---|---|---|---|---|
| mlx-whisper | large-v3-turbo fp16 | yes (MLX) | **44.3x** | 0.0908 |
| whisper.cpp 1.9.1 | large-v3-turbo fp16 (`-t 8`) | yes (Metal) | **41.7x** | **0.0832** |
| whisper.cpp 1.9.1 | large-v3-turbo fp16 (`-t 4`) | yes (Metal) | 41.0x | 0.0832 |
| whisper.cpp 1.9.1 | large-v3-turbo q5_0 | yes (Metal) | 30.0x | 0.0830 |
| mlx-whisper | large-v3 fp16 | yes (MLX) | 16.6x | 0.1367 |
| whisper.cpp 1.9.1 | large-v3 q5_0 | yes (Metal) | 11.7x | 0.1641 |
| insanely-fast-whisper | large-v3-turbo fp16 | yes (torch/MPS) | 10.2x | 0.2495 |
| faster-whisper 1.2.1 | large-v3-turbo int8, batched 16 | **no, CPU** | 6.6x | 0.0828 |
| faster-whisper 1.2.1 | large-v3-turbo int8 | no, CPU | 3.5x | 0.0804 |
| faster-whisper 1.2.1 | large-v3 int8 | no, CPU | 1.1x | 0.1175 |

### No "fastest on Apple Silicon" claim survives this

whisper.cpp (41.7x, CER 0.0832) and mlx-whisper (44.3x, CER 0.0908) are
effectively tied: 6% apart on throughput, 0.8 CER points apart in the other
direction, and this is n=1 on one clip where the documented noise band is about
1.3 points. Anyone claiming one beats the other on this evidence is reading noise.
whisper.cpp also has far fewer dependencies. What this project contributes is the
Voxtral engine, a single interface over three engines, and the measurement harness,
not a faster Whisper.

### faster-whisper cannot use the GPU here at all

This is the most load-bearing finding for anyone choosing a runner, because
faster-whisper is the most-cited "fast whisper" and its reputation is CUDA-shaped.
CTranslate2 4.8.1 has no Metal backend, verified rather than assumed:
`get_cuda_device_count()` returns 0, `get_supported_compute_types("cuda")` raises
`ValueError: This CTranslate2 package was not compiled with CUDA support`, and the
CPU compute types are `{float32, int8, int8_float32}` with **no float16**. During
the run it sat at 460-520% CPU with the 60-core GPU idle. On this machine it is
the slowest runner measured, 6.3x behind whisper.cpp at matched accuracy.

Its accuracy is fine (0.0804 is the best CER in the table), so the problem is
purely throughput, and it is structural rather than a tuning mistake.

### Quantization costs speed on the GPU, not only accuracy

A new result, and it extends the settled quantization finding in this file:

| whisper.cpp turbo | x realtime | plain CER |
|---|---|---|
| fp16 (unquantized) | **41.0x** | 0.0832 |
| q5_0 | 30.0x | 0.0830 |

q5_0 is **27% slower at identical accuracy**, because dequantization is work an
fp16 matmul does not do. So on Apple Silicon, quantize only when memory-bound.
Thread count is irrelevant: `-t 4` and `-t 8` gave 41.0x and 41.7x with
byte-identical output, so the workload is Metal-bound, not CPU-bound.

Also note large-v3 q5_0 at 0.1641 CER is not a quantization artifact but a
repetition loop, with one segment duplicated five times. turbo, with 4 decoder
layers instead of 32, avoided it. That matches the pattern in the Whisper section:
the deeper decoder is the less stable one on long audio.

Not measured: whisperX (its ASR stage is faster-whisper, so its throughput is
bounded above by the CPU-only rows, and installing it resolves 100 packages
including three torch distributions), and whisper.cpp with a CoreML encoder (the
Homebrew bottle is built `COREML = 0`).

## Single-clip work (everything below this line)

Test file: a 935s (15:35) single-speaker Japanese narration recording.
Reference: a human-authored SRT for the same recording (202 cues, 4205 scored
characters after normalization).
Metric: CER (NFKC-normalized, punctuation/whitespace stripped) vs reference SRT text.
Hardware: M4 16GB unless stated; M2 Ultra 128GB rows marked. "x realtime"
includes model load. Analysis of the mechanisms behind these numbers is in
`docs/benchmarks/decode-throughput.md`; all rows are reproducible via `scripts/benchmarks/`.

## Which differences are actually significant

Every config is a single deterministic run, so the way to get a confidence
statement out of one clip is a **paired** test: both configs transcribe the same
audio, so split the reference into 40 regions, compute per-region CER for each,
and bootstrap the per-region differences. Shared difficulty cancels instead of
inflating both intervals. `scripts/benchmarks/compare_configs.py`.

This answers "is A better than B *on this clip*". It cannot answer "on other
audio"; only more clips can.

| comparison | diff (points) | 95% CI | verdict |
|---|---|---|---|
| **prompt: generic instruction vs none** | 5.90 | [+1.40, +11.64] | **no prompt wins by far** |
| Ultra: energy vs VAD boundaries, 60s | 3.00 | [+0.74, +5.93] | **energy wins** |
| M4 16GB: 30s chunks, overlap 8s vs 0s | 1.88 | [+0.73, +3.23] | **overlap wins** |
| Ultra: 60s vs 30s chunks | 1.85 | [+0.71, +3.24] | **60s wins** |
| Ultra: 30s chunks, overlap 8s vs 0s | 1.50 | [+0.36, +2.96] | **overlap wins** |
| U+FFFD fix (same config, before vs after) | 0.40 | [+0.05, +0.81] | **fix wins** |
| Ultra: 60s chunks, overlap 0s vs 8s | 0.69 | [-1.47, +0.07] | not resolvable |
| Ultra: 90s vs 60s chunks | 0.62 | [-0.36, +1.69] | not resolvable |
| prompt: term list vs none | 0.19 | [-0.26, +0.60] | not resolvable |
| prompt: topic sentence vs none | 0.19 | [-0.17, +0.57] | not resolvable |
| prompt: term order fwd vs reversed | 0.12 | [-0.26, +0.59] | not resolvable |
| Ultra: fp16 vs 4bit (tuned cfg) | 0.07 | [-0.33, +0.48] | not resolvable |
| Ultra: kv8 vs unquantized KV | 0.02 | [+0.00, +0.07] | not resolvable |
| M4 16GB: default (60s) vs --fast (30s+ov8) | 0.02 | [-0.81, +1.00] | not resolvable |

Six claims survive as significant: **a generic instruction in the prompt field is
disastrous**, **energy boundaries beat VAD**, **prefix overlap helps at short
chunks on both machines**, **60s chunks beat 30s**, and **the U+FFFD fix helps**.

Everything in the prompt *style* study is noise: the best variant is 0.19 points
from no prompt with a CI spanning zero, and reversing the term order (so
truncation discards the terms that matter) is indistinguishable from keeping them.
So "which prompt wording is best" is not an answerable question on this clip;
only "do not put an instruction there" is.

Two earlier claims in this file were overstated and are corrected below:

- "Overlap actively hurts at 60s chunks" is **not supported**. The point estimate
  does favour no-overlap (7.37% vs 8.06%) but the CI spans zero. The defensible
  statement is that overlap stops paying once chunks are long, not that it harms.
- "Nothing above 60s is worth it" holds on *speed* (90s is 17.2x vs 21.2x) but the
  *accuracy* gap to 90s is not resolvable. 60s remains the right default because
  it is both faster and never worse, not because 90s was proven worse.

`--kv-bits 8` being free is confirmed in the strongest form available: 39 of 40
regions are identical, and the two runs differ by 0.02 points.

## Read this before trusting any number below

**Everything here is n=1: one 935s clip, one language, one speaker, one domain.**
Decoding is greedy/deterministic, so repeating a run reproduces it byte for byte;
repetition would add no information. The uncertainty that matters is how much a
number would move on *different audio*, and a single clip cannot estimate that.
`scripts/benchmarks/rescore_all.py` reports a moving-block bootstrap CI as a partial
proxy. Practical consequence: **differences under roughly half a CER point are
not resolvable here.** Treat 7.25% vs 7.42% as a tie, and read the overlap sweep
below as "≥4s helps by about 1.5 points", not as "8s is the optimum".

Metric note: `scripts/metrics/eval_cer.py` (plain CER) has never changed, so every row is
mutually comparable. `scripts/metrics/eval_cer_lenient.py` was added later and applied to the
saved hypotheses without re-decoding. It supersedes an earlier claim in this file
that ~1.6 CER points were orthographic: that came from kana CER, which
over-forgives because collapsing to readings also excuses genuine homophone
errors (a different word sharing the same reading is a wrong word, not a
spelling choice).
Lenient CER, which only forgives script changes of the *same* word, puts real
orthographic variance at ~4% of the edits, i.e. ~0.3 CER points.

## All three metrics, every significant config, both machines

Produced by `scripts/benchmarks/rescore_all.py` from the saved hypotheses (no re-decoding
needed, since plain CER never changed). CI is a moving-block bootstrap on the
single-config CER, which is why it is wide; use the paired table above to compare
two configs.

**M2 Ultra 128GB, 4bit:**

| config | CER | lenient | kana | 95% CI on CER |
|---|---|---|---|---|
| 60s, b16 | **7.23%** | **6.97%** | 5.60% | [6.14, 8.38] |
| 60s, b16, kv8 | 7.25% | 6.99% | 5.62% | [6.17, 8.43] |
| 60s, b16, kv8, fp16 weights | 7.32% | 7.02% | **5.51%** | [6.12, 8.67] |
| 30s, b32 | 9.13% | 8.82% | 6.77% | [7.36, 11.29] |
| 30s, b32, kv8 | 9.11% | 8.80% | 6.77% | [7.33, 11.24] |
| 30s, b32, kv8, overlap 8s | 7.25% | 7.04% | 5.93% | [5.95, 8.64] |
| 60s, overlap 4s | 7.59% | 7.37% | 6.43% | [6.33, 9.02] |
| 60s, overlap 8s | 8.06% | 7.85% | 6.64% | [6.67, 9.64] |
| 90s, b16 | 7.99% | 7.73% | 5.93% | [6.48, 9.88] |
| 120s, b8 | 7.59% | 7.28% | 5.85% | [6.16, 9.60] |
| 30s, VAD | 10.75% | 10.39% | 8.59% | [6.79, 17.24] |
| 60s, VAD | 10.25% | 9.85% | 7.82% | [7.19, 14.58] |

**M4 16GB, nvfp4:**

| config | CER | lenient | kana | 95% CI on CER |
|---|---|---|---|---|
| 60s, b16 | 7.49% | 7.28% | 5.99% | [5.97, 9.07] |
| 60s, b16, kv8 | **7.44%** | **7.23%** | 5.93% | [5.97, 9.05] |
| 30s, b32 | 9.06% | 8.75% | 6.96% | [7.33, 11.19] |
| 30s, b32, overlap 0s | 9.23% | 8.92% | 7.10% | [7.50, 11.15] |
| 30s, b32, overlap 4s | 7.97% | 7.66% | 6.06% | [6.26, 9.88] |
| 30s, b32, overlap 8s | 7.42% | 7.11% | **5.72%** | [6.14, 8.74] |

**Lenient CER changes no ranking on either machine.** It is uniformly ~0.2-0.3
points below plain CER, because genuine orthographic variance is small and roughly
constant across configs. That is a useful negative result: the metric refinement
was worth doing to correct the *magnitude* claim (0.3 points, not 1.6), but it does
not alter any decision. Kana CER, by contrast, does occasionally reorder configs
(fp16 leads on kana but not on CER or lenient), which is another reason to prefer
lenient as the fair number.

| # | Approach | CER | Time | x realtime | Notes |
|---|----------|-----|------|-----------|-------|
| 1 | 4bit affine, delay 480ms, batch 8 | 14.74% | 431s | 2.2x | sub=230 ins=258 del=132; insertions = hallucinated repetitions |
| 2 | 4bit affine, delay 2400ms, batch 8, SRT out | 7.49% | 424s | 2.2x | sub=151 ins=56 del=108; same speed, delay is free quality; SRT timestamps within ~0.5s of reference |
| 3 | 8bit affine (local convert), delay 2400ms, batch 8 | 7.75% | 296s | 3.2x | sub=162 ins=65 del=99; faster than 4bit (fewer dequant ops, no swap at 4.4GB) |
| 4 | mxfp8 (local convert), delay 2400ms, batch 8 | 7.59% | 320s | 2.9x | sub=161 ins=64 del=94; marginally better CER than 8bit affine, slightly slower |
| 5 | 8bit affine, delay 2400ms, batch 16 (single batch) | 7.75% | 177s | 5.3x | identical text to #3; 16 rows of 8bit KV fit in memory, no pressure collapse |
| 6 | fp16 (8.3GB), delay 2400ms, batch 8 | n/a | killed at 20min | <0.8x | swap-thrashed on 16GB (7% free mem, encoder never finished); fp16 not viable on this machine |
| 7 | 4bit affine, delay 2400ms, batch 16 (single batch) | 7.49% | 188s | 5.0x | same text as #2; best CER overall, slightly slower than 8bit at same batch |
| 8 | nvfp4 (local convert), delay 2400ms, batch 16 | 7.87% | 149s | 6.3x | fastest decode (9.3 steps/s); CER slightly worse than affine 4bit |
| 9 | 4bit + prompt bias (45 tok keyword list in left-pad), delay 2400ms, batch 16 | 7.23% | 273s | 3.4x | sub 151->142, ins 56->51; one mid-frequency term went 1->4 hits but the rarest proper nouns stayed wrong; slower decode was other system load (the mechanism adds no per-step cost) |
| 10 | 8bit + same prompt bias, delay 2400ms, batch 16 | 7.44% | 172s | 5.4x | prompt helps 8bit too (7.75->7.44); confirms prompt adds no speed cost (7.9 steps/s) |
| 11 | nvfp4 + prompt bias, delay 2400ms, batch 16 | 7.49% | n/r | n/r | prompt helps nvfp4 too (7.87->7.49); timing not captured before session ended |
| 12 | 4bit, audio downsampled to 8kHz then back to 16kHz | 7.68% | n/r | n/r | robustness check, not a speed lever (mel always needs 16kHz in); +0.19 CER for free loss of the 4-8kHz band |
| 13 | 4bit, audio time-stretched 1.25x (ffmpeg atempo) | 9.39% | n/r | n/r | 748s instead of 935s so ~20% fewer decode steps, but +1.9 CER; pitch-preserving stretch distorts phones |
| 14 | nvfp4, encode-at-true-length fix, 60s chunks, batch 16 | 7.49% | 130s | 7.2x | byte-identical text to #11 at 6.3x -> #11's encode padded every chunk to the longest one, wasting 10-19% of encoder work for nothing |
| 15 | nvfp4, 30s chunks, batch 32 | 9.06% | 113s | 8.3x | encode 51.8s vs decode 48.5s: the encoder is now the bottleneck. 32 seams instead of 16 cost +1.6 CER |
| 16 | nvfp4, 45s chunks, batch 21 | 9.08% | 120s | 7.8x | between #14 and #15 on speed, no better on CER than #15 |
| 17 | nvfp4, 60s chunks, batch 16, **--kv-bits 8** | **7.44%** | 125s | 7.5x | best M4 16GB accuracy AND faster than #14: quantized KV halves cache reads (decode 78.9 -> 72.8s). Effectively free; needed a fix (QuantizedKVCache returns tuples, needs mlx-lm's quantized SDPA wrapper) |
| 18 | nvfp4, 60s chunks, b16, --compact-silence | 11.63% | 124s | 7.5x | 12% of audio removed, but del 105 -> 222. Loss concentrated in one region (6-8min lost 45% of its text) |
| 19 | nvfp4, 30s chunks, b32, --compact-silence | 13.39% | 111s | 8.5x | del 115 -> 338 |
| 19b | nvfp4, `--fast` (30s/b32 + kv8) | 9.23% | 105s | **8.9x** | best M4 16GB speed; kv8 recovers most of what the extra seams cost (9.06% at 8.3x without kv8) |

## M2 Ultra 128GB (128GB, 60 GPU cores), 4bit affine, delay 2400ms

| # | Approach | CER | Time | x realtime | Notes |
|---|----------|-----|------|-----------|-------|
| 20 | 60s chunks, batch 16 | **7.23%** | 43.9s | 21.3x | best CER of any run on either machine; encode 10.8s, decode 31.8s |
| 21 | 45s chunks, batch 21 | 9.23% | 38.3s | 24.4x | |
| 22 | 30s chunks, batch 32 | 9.13% | 30.2s | 31.0x | |
| 23 | 20s chunks, batch 48 | 12.46% | 36.5s | 25.6x | del 123 -> 299: 20s rows end early and drop text. Also lands in the B=48 throughput dip, so slower *and* worse than #22 |
| 24 | 15s chunks, batch 64 | 9.82% | 32.2s | 29.0x | |
| 25 | 15s chunks, batch 128 | 9.82% | 31.3s | 29.9x | only 64 chunks exist, so batch 128 just means "all in one batch" |
| 26 | 60s chunks, b16, --kv-bits 8 | 7.25% | 43.6s | 21.4x | same conclusion as M4 16GB #17: quantized KV is free (7.23 -> 7.25) and slightly faster |
| 27 | 30s chunks, b32, --kv-bits 8 | 9.11% | 29.8s | **31.3x** | fastest run on either machine |
| 28 | 60s chunks, b16, --compact-silence | 8.23% | 42.1s | 22.2x | on this model compaction *helps* (9.13 -> 8.23 at 30s), unlike on the M4 16GB's nvfp4 |
| 29 | 30s chunks, b32, --compact-silence | 8.59% | 31.3s | 29.8x | del stays flat (123 -> 109), vs 115 -> 338 on M4 16GB nvfp4 |

The Ultra reaches the best accuracy *and* 21x by using long chunks, because the
whole clip fits in one batch. On the M4 16GB that choice costs speed. Chunk length
drives accuracy; batch size drives throughput.

**Silence compaction is model-dependent**, which is the one result that does not
transfer: identical audio and code, 12% of frames removed, yet nvfp4 on the M4 16GB
lost huge amounts of text (deletions 3x) while 4-bit affine on the Ultra
improved by ~0.9 CER points. Chunk cut quality actually got *better* after
compaction (no cut louder than -50dB, vs 3 cuts above -45dB before), so the
cause is not worse seams: removing the pauses appears to destabilize the
weaker-quantized model's own segmentation. Left off by default.

## Weight precision, including unquantized fp16 (M2 Ultra 128GB)

The 16GB M4 16GB cannot hold fp16, so this is the test the Ultra existed for: is
quantization costing accuracy at all? Config held fixed at 60s chunks, batch 16,
delay 2400ms, same prompt; only the weights change. `scripts/benchmarks/sweep_precision.py`.

| # | weights | bytes | CER | kana CER | wall | x realtime | decode steps/s | peak GB |
|---|---------|-------|-----|----------|------|-----------|----------------|---------|
| 30 | **fp16 (unquantized)** | 8.9GB | 7.61% | 6.08% | 68.7s | 13.6x | 14.5 | 15.3 |
| 31 | 8bit affine (local convert) | 4.7GB | 7.49% | 5.99% | 44.4s | 21.1x | 26.4 | - |
| 32 | mxfp8 (local convert) | 4.6GB | 7.66% | 5.97% | 44.1s | 21.2x | 26.5 | - |
| 33 | nvfp4 (local convert) | 2.5GB | 7.49% | 5.99% | 42.9s | 21.8x | 27.1 | - |
| 34 | **4bit affine (hub)** | 2.9GB | **7.23%** | **5.60%** | 42.0s | 22.2x | 26.3 | 9.4 |

Repeated with `--kv-bits 8` to rule out the KV cache as a confound: fp16 7.61%
at 14.3x, 4bit 7.25% at 21.8x. Identical conclusion.

**Re-run at the current best config** (30s chunks, batch 32, kv8, overlap 8s, no
prompt), since the original comparison predated prefix overlap and it was fair to
ask whether fp16 would pull ahead once everything else was tuned:

| weights | CER | kana CER | wall | x realtime | decode steps/s | peak GB |
|---|---|---|---|---|---|---|
| fp16 | 7.32% | **5.51%** | 52.6s | 17.8x | 14.7 | 13.4 |
| 4bit | **7.25%** | 5.93% | 35.1s | **26.7x** | 26.4 | 7.4 |

0.07 CER points apart, i.e. indistinguishable (the CI half-width is ~1.3 points),
for 1.5x the wall clock and 1.8x the memory. One honest nuance: fp16 does win on
*kana* CER (5.51% vs 5.93%), so its errors may be marginally more
phonetically-faithful even though its character-level score is not better. That
gap is also inside the noise band. The recommendation is unchanged: use 4-bit.

**fp16 is not better. It is 0.38 CER points WORSE than 4-bit while running 1.63x
slower and using 15.3GB instead of 9.4GB.**

Is the difference real or noise? The two hypotheses differ by only 65 characters
out of 4205 (1.5%), and fp16's 15 extra errors are spread across categories
rather than concentrated:

| weights | sub | ins | del | total |
|---|---|---|---|---|
| fp16 | 158 | 59 | 103 | 320 |
| 4bit | 149 | 52 | 104 | 305 |

So the honest reading is that **all five precisions are within noise of each
other** (7.23-7.66%, a 0.43-point spread on 4205 characters), and there is no
evidence that quantization costs anything on this workload. 4-bit being nominally
best is a coincidence of this clip, not a claim that quantization improves
accuracy. What is not noise is the speed and memory difference: fp16 costs 1.6x
the wall clock for nothing.

Decode steps/s tracks weight bytes as the bandwidth model predicts: 14.5 steps/s
for 8.9GB versus ~26-27 for the 2.5-4.7GB variants, i.e. roughly 1.8x fewer
steps for 1.9-3.5x more bytes read per step.

Two hub repos could not be tested: `mlx-community/Voxtral-Mini-4B-Realtime-6bit`
and `ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx` both ship raw Mistral-format
config.json with no `model_type`, so mlx-audio routes them to the non-realtime
`voxtral` loader and dies in `post_load_hook`. A repo packaging problem, not a
precision result; the locally converted 8bit above covers that point anyway.

### On unsloth dynamic quants and GGUF (checked 2026-07-26)

Unsloth publishes **no Voxtral repos at all** (searched the `unsloth` org: their
audio work is Whisper-only), so there is nothing of theirs to evaluate. Their
dynamic-quantization idea, keeping sensitive layers at higher precision, is sound
in general but has little room here: fp16 is only 0.07 CER points from 4-bit at
the best config, so the accuracy headroom a smarter quant could recover is
smaller than this clip can even measure.

GGUF quants of Voxtral Realtime do exist and are popular (e.g.
`handy-computer/Voxtral-Mini-4B-Realtime-2602-gguf`, ~357k downloads;
`cstr/voxtral-mini-4b-realtime-GGUF`). They are not usable here for a structural
reason rather than a quality one: GGUF is llama.cpp's format, MLX cannot load it,
and this repo's whole speed result comes from a custom batched MLX decoder. Going
GGUF would mean adopting llama.cpp's runtime, which does not batch multiple audio
streams through this architecture, i.e. giving up the 3-4x that batching buys to
chase a quantization difference measured at 0.07 points. If a GGUF path is ever
wanted, `mx.quantize` already supports affine/mxfp4/mxfp8/nvfp4 locally, which is
how the 8bit/mxfp8/nvfp4 rows above were produced.

## Prefix overlap: the one idea that worked

Chunk seams cost accuracy because a chunk's first tokens are emitted with no left
context. `scripts/benchmarks/probes/probe_seam_errors.py` confirms the mechanism by
locating every edit operation relative to the nearest boundary (30s chunks):

| region | sub | ins | del | total | % of ops | % of audio | enrichment |
|---|---|---|---|---|---|---|---|
| first 3s of a chunk | 33 | 24 | 28 | 85 | 22.3% | 9.9% | **2.24x** |
| last 3s of a chunk | 11 | 28 | 10 | 49 | 12.9% | 9.9% | 1.29x |
| elsewhere | 128 | 42 | 77 | 247 | 64.8% | 80.1% | 0.81x |

Errors concentrate at chunk *starts*, not ends. Both the encoder and the decoder
are causal, so trailing overlap cannot help (future audio cannot change an
already-emitted token) but leading overlap can. `--overlap-seconds N` prepends N
seconds of the preceding audio to each chunk and discards the tokens it produces,
so the model warms up before reaching the region we keep.

M2 Ultra 128GB, 30s chunks, batch 32, kv8 (`scripts/benchmarks/sweep_overlap.py`):

| overlap | CER | delta | x realtime | extra audio |
|---|---|---|---|---|
| 0s | 8.73% | - | 32.0x | +0% |
| 1.6s | 8.09% | -0.64 | 30.6x | +5% |
| 3s | 8.11% | -0.62 | 30.2x | +10% |
| 4s | 7.30% | **-1.43** | 29.2x | +13% |
| 5s | 8.51% | -0.21 | 28.4x | +16% |
| 6s | 7.61% | -1.12 | 27.7x | +20% |
| 7s | 7.63% | -1.09 | 27.0x | +23% |
| 8s | **7.25%** | **-1.47** | 26.3x | +27% |
| 10s | 7.56% | -1.17 | 25.6x | +33% |
| 12s | 7.80% | -0.93 | 23.1x | +40% |
| 15s | 11.20% | +2.47 | 22.2x | +50% |

M4 16GB, 30s chunks, batch 32: 0s 9.23% -> 4s 7.97% (-1.26) -> 8s **7.42%** (-1.81).

The win is real and reproduces on both machines, but the curve is noisy rather
than smooth: it plateaus from ~4s onward in the 7.25-7.80% band with no clean
optimum, and collapses at 15s (half the chunk length, where the warm-up region
starts dominating the chunk and EOS handling gets unreliable). 4s buys most of the
gain for half the cost of 8s.

**Overlap only pays where seams are dense.** At 60s chunks on the Ultra the point
estimates go the other way (0s 7.37%, 4s 7.59%, 8s 8.06%), though a paired test
puts that at -0.69 points with CI [-1.47, +0.07], so "it stops helping" is
supported while "it hurts" is not. Either way there is nothing to gain at 60s, so
overlap is tied to `--fast` (which halves the chunk length) rather than being a
universal default. The 30s win, by contrast, is significant: +1.80 points, CI
[+0.62, +3.20].

## Chunk length beyond 60s: no gain

Testing whether even longer chunks beat 60s (M2 Ultra 128GB, 4bit, overlap 0):

| chunk | batch | CER | x realtime |
|---|---|---|---|
| 60s | 16 | **7.37%** | 21.2x |
| 90s | 16 | 7.99% | 17.2x |
| 120s | 8 | 7.59% | 16.1x |
| 180s | 8 | 7.56% | 11.6x |

60s is the accuracy optimum and the fastest of the four. Past it, accuracy flattens
in the 7.5-8.0% band while speed falls away, because chunks exceed the encoder's
750-frame sliding window (a 60s chunk is already ~1948 conv frames) and the batch
has to shrink to fit memory.

Caveat on rigour: the *speed* ordering is solid, but paired testing shows the
accuracy gap from 60s to 90s (+0.62, CI [-0.36, +1.69]) is **not** resolvable on
this clip. So 60s is the right default because it is simultaneously the fastest
and never measurably worse, not because longer chunks were proven less accurate.
The 60s-vs-30s gap *is* significant (+1.85, CI [+0.71, +3.24]).

## Ideas from a literature/upstream review, all tested (2026-07-26)

Five candidate improvements drawn from MLX upstream issues, the Voxtral paper,
Whisper's long-form algorithm, and antirez/voxtral.c. **All five failed to beat
the current defaults.** Recorded so they are not retried.

| idea | source | result | verdict |
|---|---|---|---|
| **Prefix overlap at chunk seams** | WhisperX, chunk-stitching practice | **-1.5 to -1.8 CER points at 30s chunks** (see above) | **yes, shipped** |
| Dodge the batch valley by reshaping | mlx#1593, #3553 | `fold` ([B,1,d]->[1,B,d]) is bit-exact (max diff 0.000000) but only 3-7% faster; `split` into sub-batches is 1.8-4x *worse* | no |
| `mx.compile` the decode step | MLX docs, launch-overhead lore | plain 22.6/83.4/98.6 ms/step at B=1/16/32 vs compiled 23.2/89.8/106.7; all equal or worse | no |
| Newer MLX with better small-batch kernels | mlx releases | 0.32.0 is already the latest release; nothing to upgrade to | n/a |
| Raise `--delay-ms` past 2400 | Voxtral paper | 2400 is the maximum the model supports (multiples of 80 in [80,1200] plus 2400 standalone); already there | n/a |
| Chunks longer than 60s | - | 90-180s are all worse and slower (see above) | no |
| Carry context via the prompt window | Whisper `condition_on_previous_text` | -0.17 CER for 2x wall clock; superseded by prefix overlap, which attacks the same problem properly | no |

### Batch-valley reshaping (M4 16GB, nvfp4, ms/step)

| B | plain | fold | split2 | split4 |
|---|---|---|---|---|
| 4 | 50.3 | 50.3 | 50.1 | 80.9 |
| 8 | 101.6 | 102.1 | 102.8 | 106.3 |
| 12 | 84.1 | 78.1 | 143.4 | 152.0 |
| 16 | 80.6 | 80.3 | 226.8 | 216.5 |
| 32 | 103.2 | 99.8 | 176.0 | 436.6 |

Upstream confirms the cause is kernel dispatch, not bandwidth: *"The drop from 4
to 8 is that we switch from batched qmv to the qmm"* (ml-explore/mlx discussion
#1593). The crossover threshold is `vector_limit` from `get_qmv_batch_limit`,
which is compile-time C++ and machine-specific (10 on an M4 Pro for K,N>4096, 14
on an M4 Max at size 4352), so it cannot be tuned from Python. A related open
issue (mlx#3553) reports the same non-linear step at M=3 and notes that lowering
`vector_limit` made things *worse*, with no fix identified. Conclusion: avoid the
valley by choosing batch >= 12, which the profiles already do.

### Chunk-seam context carry (M2 Ultra 128GB, 30s chunks, batch 32)

The Voxtral paper notes the decoder reuses KV state as audio is appended, so a
chunk boundary is where this tool discards context. Two ways to give it back,
using the existing per-chunk prompt mechanism (decoder input is
audio_embed + embed(prev_token), so the left pad accepts prior text):

| variant | CER | delta | wall | x realtime | note |
|---|---|---|---|---|---|
| none (current) | 8.73% | - | 27.6s | 33.9x | |
| carry_pair | 8.56% | -0.17 | 55.8s | 16.8x | 2 batched passes; pass 2 gets pass 1's tails |
| carry | 8.56% | -0.17 | 177.6s | 5.3x | strictly sequential, batch 1 |
| static keywords | 9.11% | +0.38 | 27.9s | 33.5x | domain prompt in every chunk |

Carrying context recovers only 0.17 points, nowhere near the ~1.5 points that
seams cost, and doubles wall clock at best. Notably the sequential version
(true left-to-right context, batch 1) is **no better than the cheap two-pass
batched version**, which says the 31-token prompt window is simply too small to
carry meaningful context. Not worth it: just use longer chunks, which recovers
the full 1.5 points for free.

Also fixed while testing: a per-chunk prompt list whose first entry was `None`
crashed the prompt builder (the nested-list check tested `isinstance(x[0], list)`).

## VAD chunk boundaries: cleaner cuts, worse transcripts

Silero VAD (ONNX, no torch; `--vad`, needs the `[vad]` extra) placing boundaries
in the middle of the longest non-speech run near each target, instead of at the
quietest 50ms window. It never removes audio, only chooses where to cut, so the
chunks still cover the input exactly. VAD costs 2.2s for the 935s clip (426x
realtime), i.e. negligible.

M2 Ultra 128GB, 4bit, kv8:

| config | CER | lenient CER | x realtime |
|---|---|---|---|
| 30s, energy | **8.73%** | 8.42% | 21.9x |
| 30s, VAD | 10.75% | 10.39% | 20.4x |
| 30s, energy, overlap 8s | **7.25%** | 7.04% | 18.7x |
| 30s, VAD, overlap 8s | 8.04% | 7.73% | 18.0x |
| 60s, energy | **7.37%** | 7.11% | 21.0x |
| 60s, VAD | 10.25% | 9.85% | 20.1x |

VAD loses by 0.8-2.9 points in every pairing, and this is one of the few results
here that clears a significance test: paired over 40 regions at 60s chunks, energy
beats VAD by 3.00 points, CI [+0.74, +5.93], winning 21 regions to 7 (sign test
p=0.013). That is the opposite of what the VAD literature predicts. The cuts
really are cleaner by the obvious measure:
speech probability in the 1s *after* each cut is 0.316 for VAD versus 0.485 for
energy, and only 10/30 VAD cuts start inside speech versus 17/31 energy cuts.

So "cut where there is no speech" is not the objective that matters for this
model. The energy splitter picks the quietest *instant*, which tends to sit in
the middle of a long pause and hands the next chunk a run of leading silence to
warm up on. The VAD splitter picks the middle of a non-speech *run*, which is
often a short inter-word gap that satisfies the VAD but leaves almost no silence
before speech resumes. Given that prefix overlap (adding leading context) is the
single thing that helped most, "chunks need warm-up room" is the consistent
explanation. Kept in the CLI as an opt-in flag for noisy material where energy
minima may be unreliable, but off by default and not recommended here.

## Encoder batching: tested, does not work

`scripts/benchmarks/probes/probe_encoder_batch.py` reimplements the 32-layer encoder
stack with a batch axis (mlx-audio's version is batch-1 only), matching the
stock chunked path via a RotatingKVCache.

| batch | s/chunk | speedup vs per-chunk |
|---|---|---|
| per-chunk (stock) | 1.497 | 1.00x |
| 1 | 1.651 | 0.91x |
| 2 | 1.642 | 0.91x |
| 4 | 1.738 | 0.86x |
| 8 | 1.773 | 0.84x |

No gain, in fact slightly slower. The reason is arithmetic intensity: one 30s
chunk is ~3270 GFLOP against only 0.66GB of weight reads, i.e. ~4950 FLOP/byte,
so the encoder is **compute-bound** and already running near the M4 16GB's fp16 peak
(817ms of pure math vs 1.5s measured). A decode step is 2 FLOP/byte, i.e.
bandwidth-bound, which is exactly why batching works there and not here. So the
encoder share of wall clock (35-45% on the M4 16GB) is a hard floor on this hardware,
not an optimization opportunity.

## Decode throughput vs batch size (synthetic, no audio)

`scripts/benchmarks/sweep_batch.py`. ms/step is what sets wall clock; x realtime is
steps/s x batch x 0.08.

| batch | M4 16GB ms/step | M4 16GB x rt | M2 Ultra 128GB ms/step | M2 Ultra 128GB x rt |
|---|---|---|---|---|
| 1 | 22.4 | 3.6 | 10.8 | 7.4 |
| 2 | 26.2 | 6.1 | 12.8 | 12.5 |
| 4 | 51.9 | 6.2 | 16.3 | 19.7 |
| 8 | 110.8 | 5.8 | 24.7 | 26.0 |
| 12 | 88.7 | 10.8 | 31.2 | 30.8 |
| 16 | 90.3 | 14.2 | 33.6 | 38.1 |
| 24 | 97.3 | 19.7 | 34.4 | 55.9 |
| 32 | 103.6 | 24.7 | 33.9 | 75.6 |
| 48 | 191.4 | 20.1 | 53.2 | 72.2 |
| 64 | swaps | n/a | 53.0 | 96.5 |
| 128 | n/a | n/a | 99.0 | 103.5 |

B=1 on the M4 16GB is 22.4ms = 2.5GB / 120GB/s, exactly the bandwidth floor, so
nothing is wrong there. But **batch 8 costs 5x more per step than batch 1, and
batch 12 is cheaper than batch 8** on both machines. Batch 2-8 is a valley to
avoid; the project's old default of 8 sat at the bottom of it.

## Cue layout does not affect CER (once split characters are fixed)

Earlier results scored the concatenated SRT, which conflated two things: what the
model recognized, and how cue boundaries were chosen. Cue segmentation is
genuinely underdetermined (many groupings of the same tokens are equally valid),
so that was worth separating.

It also hid a real bug. Cue text was assembled by appending raw token *bytes* and
decoding at each flush. The tokenizer is a byte-level BPE, so a flush landing
mid-character split it and both halves became U+FFFD. **217 stray replacement
characters existed across the saved SRTs**, each counting as a CER error while
saying nothing about recognition. `output.py` now only breaks on a
complete-character boundary, and a plain-text path (`mlx_asr/text.py`)
decodes each chunk's token stream in one call, bypassing cue layout entirely.

Isolated effect of the fix, identical config: **7.44% -> 7.32%** overall, and
paired over regions 0.40 points with CI [+0.05, +0.81], so it is a real if small
improvement rather than noise.

With that fixed, text-only and SRT-joined CER are **identical to four decimals in
all 12 configs across both machines**:

| config | M2 Ultra 128GB text | M2 Ultra 128GB srt | M4 16GB text | M4 16GB srt |
|---|---|---|---|---|
| 60s, b16 | 7.40% | 7.40% | 7.75% | 7.75% |
| 60s, b16, kv8 | 7.28% | 7.28% | 7.73% | 7.73% |
| 30s, b32, kv8 | 8.63% | 8.63% | 9.13% | 9.13% |
| 30s, b32, kv8, ov4 | 7.16% | 7.16% | 7.87% | 7.87% |
| 30s, b32, kv8, ov8 | 7.13% | 7.13% | 7.25% | 7.25% |
| 60s, b16, kv8, compact | 8.30% | 8.30% | 7.97% | 7.97% |

So cue grouping is *not* distorting the metric: concatenating cues loses nothing,
because the only text-changing step was the per-cue `.strip()`, and cue boundaries
land in gaps rather than inside words. Both views are reported from now on
(`scripts/benchmarks/run_matrix.sh` writes all formats per run), but they agree, and the
earlier CER figures were sound apart from the ~0.1 point corruption penalty.

The three output formats are unchanged and all still carry what they should: SRT
and VTT with cue timings, JSON with per-segment start/end plus a clean full-text
field, TXT as the raw transcript.

## `--prompt` conflicts with `--overlap-seconds`

Found while re-running the matrix: the two features together are catastrophic.
Same machine, 30s chunks, batch 32, kv8:

| config | CER |
|---|---|
| no prompt, no overlap | 9.04% |
| prompt only | 9.04% |
| overlap 4s only | 7.16% |
| **prompt + overlap 4s** | **18.64%** |

The output degenerates into repetition loops (a short phrase repeated a dozen times), and 148 characters of duplicated text appear. Both knobs work by
handing the decoder text it should treat as already-emitted, and stacking them
over-conditions it; the positions do not collide (the prompt occupies decoder
positions 0..62, the warm-up region follows it), so this is a behavioural
interaction rather than an indexing bug.

The CLI now ignores `--prompt` when overlap is active and says so, keeping overlap
because it is by far the stronger effect. Benchmarks no longer pass a prompt, so
the overlap rows are clean.

## Prompt style, ordering and separators

`scripts/benchmarks/ab_prompt.py`, now domain-agnostic: pass `--terms` or a
`--prompt-file` of labelled variants. Run on the Ultra, 60s chunks, batch 16, kv8,
scored on text output. 31-token prompt window.

| variant | tokens | kept | CER | vs none |
|---|---|---|---|---|
| ASCII-comma term list | 38 | 31 | **7.09%** | -0.19 |
| term list, importance-first (truncated) | 33 | 31 | **7.09%** | -0.19 |
| topic sentence only | 17 | 17 | **7.09%** | -0.19 |
| topic sentence + terms | 33 | 31 | 7.11% | -0.17 |
| natural sentence using the terms | 23 | 23 | 7.11% | -0.17 |
| space-separated terms | 32 | 31 | 7.16% | -0.12 |
| CJK-comma term list | 33 | 31 | 7.21% | -0.07 |
| short list (3 terms, fits fully) | 16 | 16 | 7.21% | -0.07 |
| **no prompt** | 0 | 0 | 7.28% | - |
| language hint + terms | 23 | 23 | 7.49% | +0.21 |
| single term only | 4 | 4 | 7.59% | +0.31 |
| **generic instruction** ("Transcribe the audio accurately") | 7 | 7 | **13.17%** | **+5.89** |

Two findings, one minor and one important.

Minor: **prompt style does not matter measurably.** Excluding the outlier, ten
variants span 7.09-7.59%, a 0.50-point spread, and paired testing puts every one
of them inside the noise: best-variant-vs-none is 0.19 points with CI
[-0.26, +0.60], topic-sentence-vs-none 0.19 with [-0.17, +0.57], and forward-vs-
reversed term order 0.12 with [-0.26, +0.59]. A topic sentence at 17 tokens does
as well as a 38-token term list, so prose is a reasonable default, but the honest
statement is that no wording choice here is distinguishable from no prompt at all.
That the *order* makes no difference even when truncation discards the important
terms is the strongest hint that the small deltas are not vocabulary effects.
None of the variants recovered the rare proper noun they targeted (0/1 in every
run), and the term-recall column shows the prompt mostly causing *over*-production
(one term went from 1 hit to 4, versus 2 in the reference).

Important, and the only significant result in the study: **the prompt field is
not an instruction field.** Putting a natural ASR instruction there costs 5.90
points, CI [+1.40, +11.64]. The decoder treats this region as
text it already emitted, so an English imperative makes it continue in the wrong
register entirely. This is an easy and expensive mistake for a user to make, so
the CLI help now says "domain keywords" explicitly.

## Prompt bias: separator and ordering (180s clip, 943 ref chars)

`scripts/benchmarks/ab_prompt.py`. Only the last 31 tokens of the prompt survive
truncation, silently.

| variant | tokens | CER M4 16GB (nvfp4) | CER M2 Ultra 128GB (4bit) |
|---|---|---|---|
| no prompt | 0 | 10.05% | **9.03%** |
| Japanese `、`/`。` (original) | 45 | **9.59%** | 9.71% |
| Japanese, keywords reordered last | 40 | 9.71% | 10.05% |
| natural sentence | 26 | 9.71% | 9.71% |
| minimal keyword list | 27 | 10.05% | 9.71% |
| Latin `, ` | 44 | 10.05% | 9.71% |
| Latin `,` no space | 40 | 10.16% | 10.05% |
| bare spaces | 36 | 10.05% | 9.82% |

The machines disagree in sign: every variant helped on the M4 16GB and every one hurt
on the Ultra, whose no-prompt baseline was the best cell in the table. Models
differ (nvfp4 vs 4bit affine) so this is not a clean isolation of the prompt,
but the within-column spread (~0.5 points) is the same size as the
between-column disagreement. Conclusion: prompt bias is a weak, unreliable
lever. Japanese punctuation slightly beats Latin commas, a natural sentence
matches a keyword list at half the tokens, and no variant recovered the target
proper nouns (0 hits for the rarest term everywhere).

## Phonetic (kana) CER
Same texts converted to hiragana readings (pykakasi) before Levenshtein; factors
out orthography (kanji vs katakana, 7 vs 七) that pronounces identically.

| Variant | text CER | kana CER |
|---|---|---|
| 4bit b16 | 7.49% | 5.74% |
| 8bit b16 | 7.75% | 6.06% |
| mxfp8 b8 | 7.59% | 5.78% |
| nvfp4 b16 | 7.87% | 6.27% |
| 4bit + prompt | 7.23% | 5.66% |
| 8bit + prompt | 7.44% | 6.12% |

~1.6 points of the text CER looked orthographic by this (kana) measure.
**Superseded:** kana CER over-forgives, because collapsing to readings also
excuses genuine homophone errors. `scripts/metrics/eval_cer_lenient.py` puts real orthographic
variance at ~0.3 points; see the metric note near the top of this file.

## 2026-08-07: chunk length 30s vs 60s, re-run at n=20, and it dissolves

The 7-file corpus put 60s ahead of 30s by +1.67 points, CI [-1.22, +4.73], which was
"not resolvable" but close enough to the n=7 floor of ~3.2 points to look like the one
finding here that more audio might convert into a decision. Both arms re-run on one
machine (M4 16GB) over all 20 files, sequentially, `--delay-ms 2400`, kv8, each chunk
length at its own profile batch (60s/b16, 30s/b32):

| | 60s / b16 | 30s / b32 | paired diff | 95% CI |
|---|---|---|---|---|
| JP coverageCER, 17 files | 16.29% | 16.19% | +0.10 | [-1.89, +2.03] |
| EN coverageWER, 3 files | 26.14% | 25.24% | +0.90 | [-0.27, +1.69] |

The point estimate fell from +1.67 to +0.10 and neither unit resolves. 30s won 6 of 17
Japanese files, 60s won 10, sign test p=0.454. Per file the differences are large and
two-sided: one file 18.25 points better at 30s, another 10.02 points better at 60s.

This is a better outcome than a resolved effect would have been, because it converts an
ambiguity into a settled negative: chunk length in 30-60s is a throughput knob, not an
accuracy knob, on spontaneous conversational audio. The seam mechanism is real and
measurable on dense narration, and it does not survive contact with this material, which
is the same pattern as overlap.

Speed is not comparable between those two rows: the second arm ran while the host had
unrelated background load and the harness flagged it. Accuracy is unaffected because
decoding is greedy.

A cross-machine comparison came free with this, since the Ultra had already produced the
identical 30s config for the headline. That is what withdrew the ~1 point per-file floor
(see the determinism section above): 11 of 18 files identical, 16 within 0.16 points.

## 2026-08-19: Qwen3-ASR, designed and not yet built

Investigated as a candidate engine and not implemented, so this records what the
investigation established rather than a measurement. Everything below was verified against
source or the HuggingFace API rather than taken from a summary.

**What the engine is.** `Qwen/Qwen3-ASR-1.7B` and `-0.6B`, Apache-2.0, 4.4M downloads,
Japanese explicit on the model card (30 languages plus 22 Chinese dialects). Official MLX
conversions at `mlx-community/Qwen3-ASR-1.7B-8bit` (2.29GiB) and `-0.6B-8bit` (0.94GiB).
`mlx-audio` has had a first-class `qwen3_asr` loader since 0.3.1, registered in its dispatch
table, so **the code is already in this project's venv and the formula's pinned 0.4.7**. Zero
new dependencies. It decodes greedily (`temperature=0.0` becomes `mx.argmax`, no temperature
ladder), so it is deterministic like Voxtral and gets one run rather than a distribution. It
also does real language identification and reports what it detected, which no other engine
here does reliably.

**The blocker, and it is structural.** Its segment timestamps are
`start = offset_sec` and `end = offset_sec + len(chunk_audio)/sample_rate`. Those are chunk
boundaries, not speech boundaries. No variant of the model has finer times, and the streaming
path is worse rather than better: its per-token times are a fraction of the 8192-token
*budget*, not of elapsed time, so at 300 tokens every timestamp lands in the first 4% of the
chunk. With the library default `chunk_duration=1200.0`, any file under 20 minutes yields one
segment, i.e. an SRT containing a single cue holding the whole transcript.

The user's call: not supporting SRT is acceptable **provided the failure is explicit**. So
`-f srt` and `-f vtt` must be a hard error, exit 2, with a message saying this engine
produces no speech-level timestamps; `txt` and `json` work normally. The only real route to
word times is `Qwen3-ForcedAligner-0.6B` (80ms resolution, single forward pass, MLX
conversion exists, 1.19GiB), but Japanese alignment there needs `nagisa` plus DyNet, so it
belongs behind a research script rather than in the shipped CLI.

**Already shipped, do not redo.** `--language` normalisation landed in `mlx_asr/languages.py`
(commit 2025599). Qwen3-ASR wants an English language *name*, and its prompt builder
interpolates an unrecognised string verbatim, so `--language ja` would have silently produced
the prefix `language ja<asr_text>`. That whole class of failure is now handled centrally.

### What is left to do

**Done on 2026-08-20; see the section at the end of this file for what was actually built and measured.** Two items below turned out to be wrong: the sweep found 30s rather than any of the values it proposed as candidates, and the `--max-batch` reasoning missed that the library's token budget is per FILE, which silently truncates long audio and had to be fixed before any number here was trustworthy. Left in place unedited, as the plan it was.

Not started. The work is: two registry entries on a new `mlx-qwen3` backend, an adapter
in `backends.py` returning the usual cue list, `-f srt`/`-f vtt` as a hard error naming
the missing-timestamps reason, and `--max-batch` refused for now because `batch_size` is
a no-op unless `--chunk-seconds` is lowered enough to produce more than one chunk.

Three existing tests will fail and each should be rewritten rather than worked around:
`tests/test_models.py` asserts the backend is one of three strings, and asserts
`deterministic == (backend == "voxtral")`, which was true until this engine.

The first experiment is a `--chunk-seconds` sweep at 30/60/120/300 on the 7-file subset,
because the library default of 1200s is a value nobody has measured and it moves cue
count, batching and peak memory at once. Only then a full-corpus run per alias, one run
each since it is greedy, plus one repeat purely as a determinism check. Report kana and
lenient CER beside coverage CER: the Japanese finetune advertises inverse text
normalisation, so digits versus spelled-out numbers is a real confound against these
references. Record peak memory too, since the audio encoder stays unquantized at every
precision level and the attention mask is materialised densely, so this engine's memory
profile is unlike the others here.

## 2026-08-20: Qwen3-ASR, built and measured

The plan in the section above, executed. It ships as two aliases on a new `mlx-qwen3`
backend and **changes no default**. Three findings were not in the design, and two of
them are bugs the design would have shipped.

### The library silently truncates long audio

This is the one that matters, and it is worth stating precisely because the failure
produces a **well-formed transcript** rather than an error. Upstream `max_tokens`
(default 8192) is a budget for the whole file, and exhausting it only makes the chunk
loop `break`:

```python
remaining_tokens = max_tokens
for chunk_audio, offset_sec in chunk_iter:
    if remaining_tokens <= 0:
        break                       # no exception, no warning
    remaining_tokens -= gen_toks
```

Measured on one 1553s Japanese recording at a 30s window:

| configuration | segments | audio covered | coverage CER |
|---|---|---|---|
| library default (whole-file budget 8192) | 1 | 2% | 110.77% |
| whole-file budget scaled to duration (19950) | 2 | 8% | 96.69% |
| per-window budget (shipped) | 52 | 100% | **19.15%** |

The first window entered a repetition loop (one 7-character phrase repeated 2048 times, exactly 8192 tokens
at 1.75 chars/token), spent the file's whole budget, and the loop broke. 1523 of 1553
seconds were never decoded and nothing said so. **Raising the budget does not fix it**,
as the middle row shows; it only lets the loop run longer. So `backends.py` drives the
chunk loop itself with a per-window budget, using upstream's own cut points and the same
`generate` per window. `audio_coverage` and `runaway_segments` are now recorded per file
and warned about, because this class of failure is invisible in a score.

Found by running the sweep, not by reading the source: the design had read the same
function and missed it. The 96.69% row is the useful one, because it is the fix that
looked obviously right and was not.

### Shorter windows are better on every axis, which no other engine here does

7-file subset, one run per arm, sequential on the idle Ultra:

| window | JP covCER | kanaCER | EN covWER | x rt | peak GB |
|---|---|---|---|---|---|
| 15s | 20.04% | 21.88% | 30.38% | **22.3x** | **3.73** |
| **30s** | **19.98%** | **21.20%** | 31.38% | 19.2x | 4.05 |
| 60s | 21.42% | 22.12% | **29.86%** | 16.7x | 4.10 |
| 120s | 23.55% | 23.27% | 34.47% | 15.5x | 4.68 |
| 300s | 62.47% | 74.08% | 37.24% | 9.4x | 5.77 |

Accuracy, throughput and memory all improve as the window shrinks, so above 30s there is
no trade to make. The mechanism is the loops: a runaway generation runs until its window's
budget is exhausted, and that budget scales with window length, so a longer window means a
loop destroys more transcript *and* costs more decode time. 5 of 7 files looped at 30s; 7
of 7 at 120s and 300s. 300s is catastrophic rather than merely worse (one file at 149.38%).

15s was added after the fact, because the first four arms were monotonic and a sweep whose
winner sits on its own boundary has not found an optimum. It shows the curve flattening
(20.04% against 19.98%, a tie on a corpus resolving ~3.2 points), so 30s is the optimum
and neither value is a boundary any more.

### The headline: last on accuracy, and the 0.6B is the fastest thing here

20 files, 7.95h, one run each (greedy), idle host, `busy: false` recorded, no truncation.

| engine | JP covCER | EN covWER | x rt | peak GB |
|---|---|---|---|---|
| whisper-turbo, no-condition | **14.49%** ±0.27 | **18.34%** ±0.69 | 18.0-22.0x | |
| voxtral (default) | 16.22% | 21.50% | 29.6x | |
| qwen3-asr (1.7B) | 19.33% | 25.45% | 21.8x | 4.05 |
| qwen3-asr-small (0.6B) | 23.27% | 24.26% | **32.8x** | **2.36** |

Nothing recommends the 1.7B: third of three on accuracy, slower than the default. The
0.6B is the reason either ships, and only on axes it was not being judged on: fastest
engine measured in this project, in 2.36GB. Its English WER beating the 1.7B's is n=3 and
should not be quoted.

Determinism confirmed rather than assumed: 7 files decoded twice in one process, **7 of 7
byte-identical**. Wall clock varied up to 4%, which is scheduling noise.

### Kana CER never forgave digits, which was most of the confound

The design predicted an inverse-text-normalisation confound and asked for kana CER to
bound it. Measuring it exposed that **pykakasi does not read numerals**: it converts
`2018年` to `2018ねん`, so the digits-versus-spelled-out difference, the largest
orthographic class on this material, was charged in full by the metric meant to forgive
it. On two spellings of one sentence, kana CER read **55%** with nothing misheard, against
0% once digits are read.

`eval_cer_lenient.read_number` fills the gap and `eval_coverage_kana` (new, coverage-aware
kana and lenient CER) uses it. Opt-in in the older plain scripts, so previously published
kana figures keep their meaning; they understate leniency wherever numbers appear. The
1.7B's kana CER is 1.4 points below its coverage CER, the largest such gap here, which
bounds the orthographic share of its 3.1-point deficit at well under the deficit.

A 25-digit run in a real hypothesis then crashed the reader mid-sweep (past 京, no group
name), which cost an arm. Long runs and leading-zero strings now read digit by digit, and
the two supplementary metrics are wrapped so a scoring failure cannot discard a decode
that already cost GPU time.

### What shipped

Two aliases (`qwen3-asr`, `qwen3-asr-small`), 30s windows, deterministic, multilingual,
zero new dependencies. `-f srt`/`-f vtt`/`-f all` are exit 2 via a new `UnsupportedFormat`,
since the timestamps are decode-window boundaries; `cue_source: "chunk_boundaries"` is in
every result file so no timing figure can be taken from them by mistake. `--language` is
always passed explicitly, mapped to the English name, because autodetect reassigns the
language inside the chunk loop upstream and leaves `language X<asr_text>` embedded in later
chunks' text; with no flag the engine is forced to English and says so. That makes its
language identification, its most distinctive feature, unusable here until upstream is
fixed. `--max-batch` is refused with its own message: `batch_size` exists, but it is a
no-op at the default window and the "never use 2-8" finding was measured on Voxtral's
decoder.

Tests: 469 pass. The three that had to change were rewritten, not worked around
(`deterministic == (backend == "voxtral")` became a greedy-backend set). New guards cover
the truncation fix, the per-window budget, the loop detector, window tiling, the digit
reader, and one that derives the runner's `meta` lookups from the adapter so a renamed key
cannot break a run again, which it had.

## 2026-08-21: whisper sizes were measured at the wrong config, and a doc rewrite

### The size table described configs the CLI does not run

Found while auditing why MODELS.md published one-clip CER numbers under a "do not pick a
model from these" warning. The user's objection was that if a number cannot be used, it
should not be in a reference table, and following that led somewhere worse: the *corpus*
size-sweep table in engines.md measured `large-v2`, `medium` and `small` at mlx-whisper's
library defaults, while the CLI ships those three with `condition_on_previous_text=False`.

Re-measured all seven sizes at the shipped config, 20 files, idle Ultra, one arm at a
time, with peak GPU memory:

| size | JP covCER | EN covWER | x rt | peak GB | old (library defaults) |
|---|---|---|---|---|---|
| tiny | 51.28% | 32.17% | 52.7x | 3.98 | 59.27% |
| base | 29.93% | 27.14% | 34.5x | 4.07 | 29.96% |
| small | 21.33% | 22.68% | 20.1x | 4.37 | 28.61% |
| medium | 21.63% | 18.23% | 21.4x | 5.43 | 28.93% |
| large-v2 | 17.87% | 17.68% | 14.4x | 6.97 | 25.02% |
| large-v3 | **14.55%** | 18.26% | 11.4x | 7.00 | 39.91% |
| turbo | 14.68% | 18.31% | 23.7x | 5.53 | 24.97% |

`large-v3` was off by 25 points, which reads as a broken model rather than a bad default.

**The default's justification changed.** turbo does not beat large-v3 on accuracy; it
*ties* it (14.68 against 14.55, inside turbo's own ±0.27 rerun spread) while running 2.1x
faster in 1.5GB less memory. The old table appeared to show an outright accuracy win, so
the reason for the default was wrong even though the default itself was right.

The turbo arm ran on a busy host, so 23.7x is a floor. Its 14.68% agrees with the
separately published 3-run mean of 14.49% ±0.27, which makes it a harness consistency
check.

### Peak GPU memory replaced download size everywhere

The user's point: disk is not the scarce resource. Both corpus runners and the CLI now
record `mx.get_peak_memory()`, and every model/size/quant cell in MODELS.md has a measured
figure.

The interesting result is that peak memory barely tracks the download. whisper `tiny`
downloads 0.07GB and peaks at **3.98GB**, because the working set is the 30s mel window
and decoder activations rather than the weights. The whole whisper ladder lands between 4
and 7GB, so picking `tiny` to save memory buys almost nothing and costs 37 CER points.

Voxtral on the corpus: 4bit 6.77GB, fp16 12.98GB. Both lower than the 9.36/15.28GB in
quantization.md, which used the narration clip at a different chunk/batch pair.

### An unresolved contradiction, now issue #2

Measuring voxtral fp16 for its memory figure produced 15.12% JP against 4bit's 16.21% on
the corpus. The narration clip had those tied at 0.07 points, CI [-0.33, +0.48], and that
tie is the basis for both the 4bit default and the "precision costs nothing measurable"
conclusion.

It is not run noise: Voxtral decodes greedily and reruns byte-identically on one machine.
It is also not yet a finding: one run each, different material, different chunk/batch. Left
as issue #2 with the paired test named, default unchanged.

### Selection is now two layers, breaking change

`--model` takes a family (`voxtral`, `whisper`, `kotoba`, `qwen3-asr`) and `--size` picks
within it, so 11 flat names became 4 choices. The per-size names (`whisper-turbo`,
`qwen3-asr-small`, ...) are gone with no aliases, on the user's call that there are no
users yet. Passing one exits 2 naming the replacement rather than reaching the hub as a
repo id and 404ing.

The two second-layer flags are deliberately asymmetric, and the asymmetry is measured:
size spans 43 CER points across whisper's range, precision spans 0.43 across five voxtral
builds. So size defaults are evidence-based per family and precision defaults to the
cheapest that loses nothing.

`--quantization` is a lookup from (model, precision) to a published repo id, nothing more.
It also carries `weights_gb`, which matters on the voxtral path because `derive_batch`
subtracts the weight footprint from the GPU budget: on an unprofiled 16GB-class machine
4bit derives batch 32 and fp16 derives batch 1. Without that coupling fp16 would have
planned for memory already spent and failed as an OOM.

### Docs restructured after a hard review

Shipped v0.2.0 through v0.3.0 during this. The review points worth keeping:

- Do not publish numbers you then tell the reader to ignore. Either the measurement is
  usable or it does not belong in a reference table.
- Reference and rationale are different documents. MODELS.md says what is accepted and
  what the values are; DEFAULTS.md (new) says why each default is what it is, with the
  measurement.
- Caveats belong where the flag is documented, not appended to an examples block.
- Upstream limitations belong in issues, not in prose arguing with the reader (#1, #2).
- Audio input is a separate concern from model selection: AUDIO.md (new), which also
  documents the conversion the code actually does (downmix to mono, resample to 16kHz,
  fltp via libswresample) rather than asserting that nothing is constrained.
- Do not state that working code works. Only caveats are worth documenting.

A markdown bug shipped twice before being caught: the paragraph after each generated table
was absorbed into the table, because the splice inserting the tables stripped the
generator's trailing blank line. There is now a test.

## 2026-08-23: the Japanese-specialized engines, and what they say about fine-tuning

The question that started this session was whether to fine-tune Whisper on our own
corpus, which is limited. The counter-hypothesis: before training anything, measure
the engines that were trained on exactly this kind of material at industrial scale.
Two candidates existed for Japanese conversational speech and neither was in the
registry: NVIDIA's parakeet-tdt_ctc-0.6b-ja (FastConformer-TDT, MLX conversion
published) and ReazonSpeech k2-v2 (Zipformer transducer, ONNX).

### Integration notes worth keeping

- parakeet needed no new dependency: mlx-audio 0.4.5 already ships a full Parakeet
  implementation and dispatches on the repo name. Its `generate(chunk_duration=)`
  does fixed-window chunking with an overlap merge internally, so the adapter is
  thin.
- reazon-k2 is the first non-MLX engine here. The decision chain: the NeMo build
  needs torch; the k2 build is ONNX AND more accurate on its authors' own
  benchmarks (TEDxJP-10K 9.09 vs 10.42 CER), so k2 via sherpa-onnx wins twice.
  sherpa-onnx has no py3.14 wheel yet and its source build does not bundle
  libonnxruntime; copying the dylib out of the `onnxruntime` wheel fixes it.
- reazon-k2 CANNOT be decoded whole-file. A 112s stream returned 124 characters;
  the weights are trained on short VAD segments and skip outside that
  distribution, which is why the authors' recipe segments by VAD. All figures use
  30s windows at energy minima, the same front end as the Voxtral rows.
- parakeet's window direction is OPPOSITE to kotoba/qwen3: 60s windows LOSE
  content against 120s (301 vs 380 chars on one file). Models trained to carry
  state across windows degrade when denied it; models trained on short segments
  degrade when given more than they saw. No universal chunking advice exists.

### The int8 surprise

Reazon's release table puts int8 within ~0.3 CER of fp32 on three benchmarks.
Measured here it drops whole phrases mid-file: 296 against 376 characters on one
112s recording, 36.93% against 30.45% corpus-wide. Read-speech benchmarks and
conversational material disagree about what int8 costs. fp32 ships; a test guards
the default. This is also a small warning about trusting publisher quantization
tables measured on read speech.

### Results

17 JP files, 6.78h, coverage CER: parakeet c120 26.19% at 244.6x realtime
(4.77GB peak GPU); reazon fp32 c30 30.45% at 51.6x on CPU; reazon int8 36.93%
at 78.9x. Parakeet loses to turbo-nocond on 16 of 17 files (mean +9.8); its one
win is the cleanest narration file. Full detail in docs/benchmarks/japanese-only.md.

244.6x is a project record, 7x past qwen3-small's 32.8x. Machine caveat: 45GB of
GPU memory parked by a resident agent throughout, load ~5 of 24 cores; both new
engines are greedy so accuracy stands, throughput is a floor. Same treatment as
the kept busy-flagged run of 2026-08-06.

### What this says about the original question

Parakeet-ja was trained on tens of thousands of hours of naturalistic Japanese,
including spontaneous speech, which is precisely this corpus's material. It still
loses to a generic multilingual Whisper checkpoint by 11.7 points. Domain match
alone does not decide engine quality; long-form behaviour, code-switching
handling and instruction breadth dominate on real recordings. Fine-tuning turbo
on ~8h of our audio would be buying domain adaptation from a model that already
wins BECAUSE of breadth, with the documented risks of losing exactly that
breadth. The case for fine-tuning got weaker, not stronger.
