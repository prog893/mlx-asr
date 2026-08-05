# Lever: timestamp quality

**Conclusion first.** Voxtral and Whisper fail in different directions, and the split is
actionable. **Voxtral holds timing** (worst drift slope 25.3 ms/min against Whisper's
122.7, lower p95 error on all 7 files) and **Whisper places cues better** (break F1 56.0%
against 37.4%, better on 6 of 7 files). Voxtral is better where errors are hard to fix,
since timestamps come from the model, and worse where they are easy, since cue grouping is
our own heuristic and deliberately not fitted to these references.

## Two failure modes, never combined into one score

A subtitle file can fail in two independent ways, and a single number would hide both:

- **Drift**: words correct, times wrong. Fatal, and invisible to CER.
- **Cue breaks**: times right, text chopped mid-phrase. Annoying but cosmetically fixable.

`scripts/metrics/eval_timing.py` reports them separately and never averages them.

## Corpus

The 7 recordings with author-written subtitle tracks. Plain-text references cannot support
this at all. This is the smallest sample in the project, and none of the 13 recordings
added in the final corpus growth had authored SRTs, so it stays at n=7.

## Method

Both engines are asked for SRT output on the same audio, then scored by `eval_timing`
(`scripts/benchmarks/run_timing_sweep.py`).

**Drift** is measured at *anchors*, not by comparing cue lists. The two texts are aligned
character by character, and every run of at least 8 matched characters becomes an anchor,
sampled at its midpoint. That is what stops cut material from masquerading as drift. It
reports median and p95 absolute error at anchors, plus a least-squares regression of
signed error on time. **Slope is separated from constant offset** because a whole file
shifted 400ms is trivially correctable while error that grows through the file is not.

**Cue breaks** are boundary F1 of hypothesis cue ends against the author's, within a 0.5s
tolerance, plus the rate of hypothesis cue ends landing strictly inside a reference cue.

`anchor_coverage` is reported alongside: below about 50% the timing estimate is weak and
the row should be dropped rather than averaged in.

Measured at the cue grouping the CLI actually ships (`gap_s=1.2, max_chars=28`) and with
Whisper's language taken per file from its reference. Both of those were bugs once; see
below.

## Experiment

n=7, every row above the 50% anchor-coverage bar (lowest is 75.6%), so nothing is
excluded:

| | Voxtral | whisper turbo-nocond |
|---|---|---|
| median timing error | 278 ms | **250 ms** |
| median p95 error | **786 ms** | 1908 ms |
| worst drift slope | **25.3 ms/min** | 122.7 ms/min (unstable, see below) |
| break F1 | 37.4% | **56.0%** |
| mid-phrase splits | 58.1% | **41.9%** |

Per-file, Voxtral's p95 error is lower on **all 7 files**, and 2.4x lower at the median.
Whisper's break F1 is higher on **6 of 7**, the exception being the prepared-narration
clip.

Median error is the one column Whisper nominally wins, and it should not be read as a
Whisper advantage: the median says nothing about the tail, and the p95 and slope columns
are where a subtitle file becomes unusable. Whisper's 122.7 ms/min is over a second of
accumulated drift on a ten-minute file.

### Whisper's worst slope is unstable

Read it as an order of magnitude, not a value. Two runs of the identical config gave 122.7
and 180.7 ms/min on the file that produces it, because Whisper samples: its output on that
file differed between runs, one cue splitting where the other did not. Every other Whisper
row matched exactly across the two runs, so this is one file's instability rather than
general noise. Both values are 5-7x Voxtral's worst, so the comparison is unaffected. See
[determinism.md](determinism.md).

## Two harness bugs this table used to contain

Worth recording because both looked like model behaviour and neither was.

**Voxtral's figure described a config that does not ship.** The published 42.8% was
measured at `gap_s=0.7, max_chars=32`; the CLI ships `1.2, 28`, which scores **37.4%**. So
the published number flattered the default a user gets by 5.4 points. Root cause: the cue
config was neither settable from the CLI nor recorded in any output, so a run could not be
attributed to one. See [cue-layout.md](cue-layout.md).

**Whisper's language was hardcoded to `ja`.** One corpus file is English audio with an
English reference (it is the English half of a dubbed pair). Transcribed as Japanese it
looped, emitting cues with identical start and end and one cue ending before it began, for
9.8% anchor coverage. The metric then discarded the row as untrustworthy, which read as
Whisper struggling on hard audio rather than the harness asking for the wrong language.
With language detected per file from the reference, that same row reaches **93.1%**
coverage and scores 43.5% break F1.

Fixing the second one removed a real methodological wart: the table used to need a
paragraph explaining whether to read an "all 7" column that included a broken row or a
"trusted 6" column that dropped it. There is now one n=7 table.

Both corrections pulled in opposite directions, and the ordering survived both. The
magnitudes moved: Whisper's break advantage is larger than published (18.6 points, not
9.8) and Voxtral's drift advantage is larger too (worst slope 25.3, not 37.1, because that
file's worst case came from the discarded cue config).

## Reading the asymmetry

Voxtral's timestamps come from the model: one token per decoder position, each covering
80ms of audio, so a timestamp is a position count rather than an estimate. That is why
drift is small and why nothing in the cue knobs can improve it.

Whisper emits segment-level timestamps from a model that was not trained to be precise
about them, but its segments are chosen with more linguistic context than our gap-based
heuristic uses, so its boundaries match a human editor better.

If you need timing accuracy, use Voxtral. If you need cue boundaries that read naturally
and are willing to accept drift, Whisper's are better, or fit `--gap-seconds` to your own
references.

## Reproducing

```bash
uv run python scripts/benchmarks/run_timing_sweep.py --corpus DIR --json out.json
```

Requires audio paired with same-stem `.srt` or `.vtt`. Add `--gap-seconds` /
`--max-chars` to measure a cue config other than the shipped one, and `--language` to
override per-file detection.

## Related

[cue-layout.md](cue-layout.md) for why the shipped cue defaults are not the sweep optimum.
[determinism.md](determinism.md) for why Whisper needs a distribution.
