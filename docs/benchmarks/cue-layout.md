# Lever: subtitle cue grouping

The default is `gap_s=1.2, max_chars=28`, and it is deliberately not the optimum of
either sweep run here. Every timed reference available was authored by one editor, so
optimising against them measures agreement with one subtitle house style rather than good
segmentation. Measured end to end, the choice costs 5.4 break-F1 points[^c2] against `0.7/32`,
the config the earlier published figure used, and is slightly better on p95 drift. `0.7/32`
is not the optimum of the regrouped grid: `0.6/32` scores 44.4% against 42.6%,[^c1] and the best of
the 168 points is `0.4/56` with `max_dur_s=5.0` at 45.35%,[^c1] emitting about twice as many cues as
the references.[^c1] `gap_s` is the largest knob, and it is exposed as `--gap-seconds` so you can
fit it to your own references; `max_chars` and `max_dur_s` also move break F1, by up to 3.7
and 1.9 points around the default.[^c1]

**Voxtral only.** Cue grouping is our heuristic over Voxtral's per-token timestamps. The
other engines emit their own segments, so `--gap-seconds`, `--max-chars` and
`--max-dur-seconds` are rejected there.

| setting | default | why |
|---|---|---|
| `--gap-seconds` (`gap_s`) | 1.2 | conservative; the n=7 sweep favours lower values, but only against one editor's convention, and partly by emitting more cues |
| `--max-chars` (`max_chars`) | 28 | the best value in 28-72 at the default gap (37.43% at 28 against 33.76% at 72, a 3.7-point spread); up to 5.2 points elsewhere in the grid |
| `--max-dur-seconds` (`max_dur_s`) | 7.0 | the best value at the default pair (37.43% at 7s against 35.52% to 35.63% at 4s and 5s); up to 5.1 points elsewhere in the grid |

Source: [ultra-cue-layout.json](../../benchmarks/results/2026-10/ultra-cue-layout.json) runs `cue_sweep_union` (grid points, mean break F1 over the 7 timed files); [ultra-timestamps.json](../../benchmarks/results/2026-10/ultra-timestamps.json) runs `ts_0732#engine voxtral`, `ts_default_r1#engine voxtral`, `ts_default_r2#engine voxtral`, `ts_default_r3#engine voxtral`

**Setup:** 7 files with author-written timed subtitle references (one prepared-narration
clip and six subtitled corpus files, all by one editor; see [timed references](reference/corpus.md#timed-references)), scored by
break F1 and mid-phrase rate from `eval_timing`. The sweep regroups cached Voxtral tokens,
so its scores do not depend on the machine.

## Experiment: gap_s at max_chars 32

**Basis:** the 7 [timed references](reference/corpus.md#timed-references), cached Voxtral tokens regrouped by `sweep_cues.py` (machine-independent), `gap_s` 0.6 to 1.2 at `max_chars=32` plus the default pair.

With seven timed references `gap_s` is not flat, as the earlier n=1 sweep suggested (see
[Superseded](#the-n1-sweep)). It is monotonic, and it points the **opposite
way** from the n=1 result: break F1 keeps rising as `gap_s` falls, to 44.99%[^c1] at 0.4 (not in
the table).[^c1] The last table row is the default pair, which uses
`max_chars=28` and so is not a point on the charted line.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/cue-layout-gap-dark.svg">
  <img alt="Across gap_s 0.6 to 1.2 seconds at max_chars 32, break F1 falls from 44.4% to 36.0% while cues per reference falls from 1.40 to 0.95." src="img/cue-layout-gap-light.svg">
</picture>

**Table:** break F1, mid-phrase rate and cues per reference cue at each `gap_s` with `max_chars=32`, plus the default pair.

| gap_s (max_chars 32) | break F1 | mid-phrase | cues/ref |
|---|---|---|---|
| 0.6 | 44.4% | 56.2% | 1.40 |
| 0.7 | 42.6% | 56.2% | 1.29 |
| 0.8 | 40.6% | 56.7% | 1.16 |
| 1.0 | 36.4% | 59.4% | 1.02 |
| 1.2 | 36.0% | 58.2% | 0.95 |
| 1.2 / chars 28 (**default**) | 37.4% | 58.1% | 1.04 |

Source: [ultra-cue-layout.json](../../benchmarks/results/2026-10/ultra-cue-layout.json) runs `cue_sweep_union` (grid points at max_dur_s 7.0, mean over the 7 timed files)

Per file, the default pair against `0.7/32`, the earlier published config:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/cue-layout-per-file-dark.svg">
  <img alt="Break F1 per file for the default pair 1.2/28 and the earlier published config 0.7/32: the default is higher only on 632a295d981d, 46.1% against 35.9%, and lower on all six other files." src="img/cue-layout-per-file-light.svg">
</picture>

**Table:** break F1 per timed file for the default pair `1.2/28` and the earlier published config `0.7/32`; the higher value in each row is bold.

| file | 1.2/28 (default) | 0.7/32 |
|---|---|---|
| 632a295d981d (**the file 1.2/28 was fitted to**) | **46.1%** | 35.9% |
| a773a315fb42 | 38.2% | **49.2%** |
| 71bb5eba3baa | 31.3% | **38.3%** |
| 1da58e1b2b69 | 32.4% | **41.2%** |
| a1372c4d2900 | 33.3% | **41.2%** |
| c47ad127bc83 | 41.9% | **48.8%** |
| b3432b40c079 | 38.8% | **43.6%** |

Source: [ultra-cue-layout.json](../../benchmarks/results/2026-10/ultra-cue-layout.json) runs `cue_sweep_union` (grid points 1.2/28 and 0.7/32 at max_dur_s 7.0)

**The one file where the default pair wins is the one it was fitted to.** It loses on all
six held-out files. That is overfitting visible in a single table, and it is the clearest
worked example in this project of why n=1 tuning is dangerous.

The n=7 sweep is better evidence than the n=1 one, but it is **not evidence of good
segmentation**. It is evidence of agreement with one subtitle convention: all seven
references share line-wrap width and pause conventions, so "optimal `gap_s`" there may be
a fact about that convention rather than about Japanese subtitles. Fitting the default to
it would export one editor's style to every user, which is a worse default than a
conservative pair that no reference chose.

There is also a mechanical reason to distrust the low end. Lower `gap_s` buys part of its
F1 by emitting more cues (cues/ref 0.95 at 1.2, 1.29 at 0.7, 1.40 at 0.6).[^c1] F1 rewards
that; a reader does not.

`mlx_asr/output.py` documents this and a test pins the defaults, so a future sweep result
cannot be quietly applied.

## Experiment: the default pair against 0.7/32, end to end

**Basis:** the 7 [timed references](reference/corpus.md#timed-references), Voxtral end to end (4-bit weights, the CLI default then) on the same audio with written SRTs scored by `eval_timing`, `1.2/28` against `0.7/32` (the earlier published config, not the grid optimum).

Measured end to end on the same audio with everything else fixed:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/cue-layout-end-to-end-dark.svg">
  <img alt="End to end on the 7 timed files, the default 1.2/28 scores 37.7% break F1 against 43.1% for 0.7/32, with 865 ms p95 drift against 976 ms." src="img/cue-layout-end-to-end-light.svg">
</picture>

**Table:** break F1, mid-phrase rate and timestamp drift of the written SRTs for the default pair and `0.7/32`, Voxtral end to end on the 7 timed files.

| Voxtral, 7 timed files | break F1 | mid-phrase | median drift | p95 drift |
|---|---|---|---|---|
| **`1.2/28` (default)** | **37.7%** | 58.3% | 278 ms | 865 ms |
| `0.7/32` (earlier published config) | 43.1% | 57.4% | 258 ms | 976 ms |

Source: [ultra-timestamps.json](../../benchmarks/results/2026-10/ultra-timestamps.json) runs `ts_0732#engine voxtral`, `ts_default_r1#engine voxtral`, `ts_default_r2#engine voxtral`, `ts_default_r3#engine voxtral`

So the default costs 5.4 break points,[^c2] and it is *better* on p95 drift, so the points are
not bought with timing accuracy.

This comparison also fixed a reporting bug. The published break figure was measured at
`0.7/32` while the CLI default was `1.2/28`. Re-running both arms confirmed the cause:
rerunning the `0.7/32` arm reproduced the published figure closely. The root
problem was that the cue config was neither settable from the CLI nor recorded in any
output, so a run could not be attributed to a configuration. Both are fixed:
`--gap-seconds`, `--max-chars` and `--max-dur-seconds` exist, and `--stats-json` records
the resolved values next to the cue count.

**One measurement subtlety.** `sweep_cues.py` scores `build_cues` output directly; the
end-to-end run scores the *written SRT*, and `write_srt` clamps
`end = max(end, start + 0.5)`, which nudges short cue ends later. Measured at identical
`0.7/32`: **43.1% via SRT against 42.6% via raw cues**.[^c3] How many cues the clamp touches was
counted afterwards on the narration clip's written SRTs: 54 of 271 cues at `0.7/32` and 20 of
206 at the shipped default are exactly 0.5s long,[^c4] an upper bound on the clamped count,
since a cue that is naturally 0.5s also counts. <!-- not-measured: 0.5s is the write_srt minimum cue length -->
Quote the SRT number when describing files a user gets, and the raw number when comparing
grid points. The 37.7% and 37.4% figures[^c3] for the default pair are the same measurement
taken these two ways.

## Experiment: max_chars and max_dur_s

**Basis:** the 7 [timed references](reference/corpus.md#timed-references), the same cached-token regrouping as the `gap_s` sweep, `max_chars` 28 to 72 and `max_dur_s` 4 to 9 seconds.

Neither is inert on this material. At the default gap (1.2) and `max_dur_s=7.0`, `max_chars`
moves break F1 by 3.7 points across 28-72, from 37.43% at 28 to 33.76% at 72,[^c1] and by up to
5.2 points at other grid points.[^c1] At the default pair `1.2/28`, `max_dur_s` moves it by 1.9
points, 35.52% to 35.63% at 4s and 5s against 37.43% at 7s and 37.40% at 9s,[^c1] and by up to 5.1
points elsewhere.[^c1] The shipped 28 and 7.0 are the best measured values at the default gap,
so the defaults stand; these two knobs are smaller than `gap_s` but are not safety valves only.

## How it works

`build_cues` turns per-token timestamps into subtitle cues. A cue breaks on: a silence
gap longer than `gap_s`, text reaching `max_chars`, duration reaching `max_dur_s`, or
sentence-ending punctuation once the cue has some body.

This is our heuristic over the model's output. Voxtral emits per-token timestamps;
grouping them into cues is a downstream choice, which is why cue placement is the weaker
half of this project's timing story while timestamp accuracy is the stronger half. See
[timestamps.md](timestamps.md).

The timed references are the only material in the project that can score cue placement
at all, since it needs real cue boundaries to compare against. That all seven share one
editor's conventions is the fact that decided the outcome.

`scripts/benchmarks/sweep_cues.py` runs the sweep. Cue grouping does not affect decoding,
so it re-runs only `build_cues` over cached `(token, time)` pairs. A 168-point grid[^c1]
therefore costs seconds rather than hours. Break metrics are boundary F1 against the
author's cue ends within a 0.5s tolerance, <!-- not-measured: scorer tolerance setting --> plus the rate of hypothesis cue ends landing
inside a reference cue ("mid-phrase"). Drift is deliberately not scored by the sweep,
since regrouping cues cannot move a token's timestamp. The end-to-end drift columns above
still differ between cue configs (p95 865 against 976 ms),[^c2] because drift is scored on the
written cues.

If you have your own timed references, sweep `gap_s` first:

```bash
for g in 0.6 0.7 0.8 1.0 1.2; do
  mlx-asr audio.wav -f srt --gap-seconds $g -o "cues_$g.srt"
done
(cd scripts && uv run python -m metrics.eval_timing ../reference.srt ../cues_0.7.srt)
```

Or use `scripts/benchmarks/sweep_cues.py --corpus DIR` to do the whole grid from one decode.

## Superseded

### The n=1 sweep

A sweep against the single timed reference then available moved the defaults from
`(1.0, 32)` to `(1.2, 28)`.

Its own docstring flagged the risk at the time: n=1, `gap_s` apparently flat by median
across 0.6-1.3, and `max_chars=28` suspiciously close to that reference's mechanical
15-character line-wrap width, so part of the gain might be matching its wrap arithmetic
rather than real phrase awareness. The n=7 sweep above replaced its reading of `gap_s`;
the `(1.2, 28)` pair it produced remains the default for the reasons given there.

## Not settled

**Extending cue ends.** Extending each cue's end to the next cue's start, the usual subtitle convention, to fix a
systematic early bias. On the n=1 data it made break F1 *worse* at 0.5, 1 and 2s holds, <!-- not-measured: hold lengths are the settings tried -->
because every extended end stops matching a human cue end.
A 0.2s hold was F1-neutral and improved drift, so that is the only variant worth <!-- not-measured: 0.2s is a hold setting -->
revisiting on the larger set. The rejection rests on n=1 data only.

## Related

[timestamps.md](timestamps.md) for the drift half of timing, which these knobs move only
through the written cues.

[^c1]: [ultra-cue-layout.json](../../benchmarks/results/2026-10/ultra-cue-layout.json) runs `cue_sweep_union`
[^c2]: [ultra-timestamps.json](../../benchmarks/results/2026-10/ultra-timestamps.json) runs `ts_0732#engine voxtral`, `ts_default_r1#engine voxtral`, `ts_default_r2#engine voxtral`, `ts_default_r3#engine voxtral`
[^c3]: [ultra-timestamps.json](../../benchmarks/results/2026-10/ultra-timestamps.json) runs `ts_0732#engine voxtral`, `ts_default_r1#engine voxtral`, `ts_default_r2#engine voxtral`, `ts_default_r3#engine voxtral`; [ultra-cue-layout.json](../../benchmarks/results/2026-10/ultra-cue-layout.json) runs `cue_sweep_union`
[^c4]: [ultra-derived-measures.json](../../benchmarks/results/2026-10/ultra-derived-measures.json) runs `derive_cues`
