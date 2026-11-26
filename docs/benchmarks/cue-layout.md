# Lever: subtitle cue grouping

Voxtral returns a timestamp per token, and `build_cues` groups the tokens into subtitle
cues. A new cue starts at a pause longer than `gap_s`, when the text reaches `max_chars`,
when the duration reaches `max_dur_s`, or at sentence-ending punctuation once the cue has
some body. The other engines return their own segments, so `--gap-seconds`, `--max-chars`
and `--max-dur-seconds` are Voxtral only.

The defaults are the grid point closest to the timed references: the highest break F1
among points whose cue count is within 10% of the references'. <!-- not-measured: the selection rule -->
Of the 168 points swept,[^c1] that is `0.7/56/9.0`: 42.65% break F1, 54.1% mid-phrase ends
and 1.07 cues per reference cue, against 37.43%, 58.1% and 1.04[^c1] for the previous
default `1.2/28/7.0`. The best break F1 overall, `0.4/56/5.0` at 45.35%,[^c1] produces 1.97 cues per
reference cue,[^c1] so it is further from the references in shape. `tests/test_core.py`
recomputes the rule from the saved rows and fails if the code defaults differ.

The sweep regrouped the tokens of one Voxtral decode with the 4bit weights at 30s chunks
and batch 32, the CLI default when it ran. The default precision is now fp16 on the M2
Ultra and 8bit on the M4; the rule has not yet been rerun on those tokens, and the new
defaults have not yet been measured end to end.

| setting | default | why |
|---|---|---|
| `--gap-seconds` (`gap_s`) | 0.7 | the swept point closest to the references; shorter gaps agree slightly more on breaks only by producing more cues |
| `--max-chars` (`max_chars`) | 56 | part of the same point |
| `--max-dur-seconds` (`max_dur_s`) | 9.0 | part of the same point; 9.0 is the largest value swept |

**Setup:** 7 files with author-written timed subtitle references (one prepared-narration
clip and six subtitled corpus files; see [timed references](reference/corpus.md#timed-references)), scored by
break F1 and mid-phrase rate from `eval_timing`. The sweep regroups cached Voxtral tokens,
so its scores do not depend on the machine.

## Experiment: gap_s at max_chars 32

**Basis:** the 7 [timed references](reference/corpus.md#timed-references), cached Voxtral tokens regrouped by `sweep_cues.py` (machine-independent), `gap_s` 0.6 to 1.2 at `max_chars=32` plus the previous default pair.

With seven timed references `gap_s` is not flat, as the earlier n=1 sweep suggested (see
[Superseded](#the-n1-sweep)). It is monotonic, and it points the **opposite
way** from the n=1 result: break F1 keeps rising as `gap_s` falls, to 44.99%[^c1] at 0.4 (not in
the table).[^c1] The last table row is the previous default pair, which uses
`max_chars=28` and so is not a point on the charted line.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/cue-layout-gap-dark.svg">
  <img alt="Across gap_s 0.6 to 1.2 seconds at max_chars 32, break F1 falls from 44.4% to 36.0% while cues per reference falls from 1.40 to 0.95." src="img/cue-layout-gap-light.svg">
</picture>

**Table:** break F1, mid-phrase rate and cues per reference cue at each `gap_s` with `max_chars=32`, plus the previous default pair.

| gap_s (max_chars 32) | break F1 | mid-phrase | cues/ref |
|---|---|---|---|
| 0.6 | 44.4% | 56.2% | 1.40 |
| 0.7 | 42.6% | 56.2% | 1.29 |
| 0.8 | 40.6% | 56.7% | 1.16 |
| 1.0 | 36.4% | 59.4% | 1.02 |
| 1.2 | 36.0% | 58.2% | 0.95 |
| 1.2 / chars 28 (previous default) | 37.4% | 58.1% | 1.04 |

Source: [m2-ultra-cue-layout.json](../../benchmarks/results/2026-10/m2-ultra-cue-layout.json) runs `cue_sweep_union` (grid points at max_dur_s 7.0, mean over the 7 timed files)

Per file, the previous default pair `1.2/28` against `0.7/32`, the earlier published config:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/cue-layout-per-file-dark.svg">
  <img alt="Break F1 per file for the previous default pair 1.2/28 and the earlier published config 0.7/32: 1.2/28 is higher only on 632a295d981d, 46.1% against 35.9%, and lower on all six other files." src="img/cue-layout-per-file-light.svg">
</picture>

**Table:** break F1 per timed file for the previous default pair `1.2/28` and the earlier published config `0.7/32`; the higher value in each row is bold.

| file | 1.2/28 (previous default) | 0.7/32 |
|---|---|---|
| 632a295d981d (**the file 1.2/28 was fitted to**) | **46.1%** | 35.9% |
| a773a315fb42 | 38.2% | **49.2%** |
| 71bb5eba3baa | 31.3% | **38.3%** |
| 1da58e1b2b69 | 32.4% | **41.2%** |
| a1372c4d2900 | 33.3% | **41.2%** |
| c47ad127bc83 | 41.9% | **48.8%** |
| b3432b40c079 | 38.8% | **43.6%** |

Source: [m2-ultra-cue-layout.json](../../benchmarks/results/2026-10/m2-ultra-cue-layout.json) runs `cue_sweep_union` (grid points 1.2/28 and 0.7/32 at max_dur_s 7.0)

**The one file where `1.2/28` wins is the one it was fitted to.** It loses on all
six held-out files. That is overfitting visible in a single table, and it is the clearest
worked example in this project of why n=1 tuning is dangerous.

Lower `gap_s` raises break F1 partly by emitting more cues (cues/ref 0.95 at 1.2, 1.29 at
0.7, 1.40 at 0.6),[^c1] which is why the selection rule bounds the cue count.

## Experiment: 1.2/28 against 0.7/32, end to end

**Basis:** the 7 [timed references](reference/corpus.md#timed-references), Voxtral end to end (4-bit weights, the CLI default then) on the same audio with written SRTs scored by `eval_timing`, `1.2/28` against `0.7/32` (the earlier published config, not the grid optimum).

Measured end to end on the same audio with everything else fixed:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/cue-layout-end-to-end-dark.svg">
  <img alt="End to end on the 7 timed files, the previous default 1.2/28 scores 37.7% break F1 against 43.1% for 0.7/32, with 865 ms p95 drift against 976 ms." src="img/cue-layout-end-to-end-light.svg">
</picture>

**Table:** break F1, mid-phrase rate and timestamp drift of the written SRTs for the previous default pair `1.2/28` and `0.7/32`, Voxtral end to end on the 7 timed files.

| Voxtral, 7 timed files | break F1 | mid-phrase | median drift | p95 drift |
|---|---|---|---|---|
| `1.2/28` (previous default) | 37.7% | 58.3% | 278 ms | 865 ms |
| `0.7/32` (earlier published config) | 43.1% | 57.4% | 258 ms | 976 ms |

Source: [m2-ultra-timestamps.json](../../benchmarks/results/2026-10/m2-ultra-timestamps.json) runs `ts_0732#engine voxtral`, `ts_default_r1#engine voxtral`, `ts_default_r2#engine voxtral`, `ts_default_r3#engine voxtral`

So `1.2/28` scores 5.4 break points lower,[^c2] and is better on p95 drift.

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
206 at `1.2/28` are exactly 0.5s long,[^c4] an upper bound on the clamped count,
since a cue that is naturally 0.5s also counts. <!-- not-measured: 0.5s is the write_srt minimum cue length -->
Quote the SRT number when describing files a user gets, and the raw number when comparing
grid points. The 37.7% and 37.4% figures[^c3] for `1.2/28` are the same measurement
taken these two ways.

## Experiment: max_chars and max_dur_s

**Basis:** the 7 [timed references](reference/corpus.md#timed-references), the same cached-token regrouping as the `gap_s` sweep, `max_chars` 28 to 72 and `max_dur_s` 4 to 9 seconds.

Neither is inert on this material. At gap 1.2 and `max_dur_s=7.0`, `max_chars`
moves break F1 by 3.7 points across 28-72, from 37.43% at 28 to 33.76% at 72,[^c1] and by up to
5.2 points at other grid points.[^c1] At `1.2/28`, `max_dur_s` moves it by 1.9
points, 35.52% to 35.63% at 4s and 5s against 37.43% at 7s and 37.40% at 9s,[^c1] and by up to 5.1
points elsewhere.[^c1] Both knobs are smaller than `gap_s`, but they are not only safety valves.

## How it works

`build_cues` turns per-token timestamps into subtitle cues. A cue breaks on: a silence
gap longer than `gap_s`, text reaching `max_chars`, duration reaching `max_dur_s`, or
sentence-ending punctuation once the cue has some body.

This is our heuristic over the model's output. Voxtral emits per-token timestamps;
grouping them into cues is a downstream choice, which is why cue placement is the weaker
half of this project's timing story while timestamp accuracy is the stronger half. See
[timestamps.md](timestamps.md).

The timed references are the only material in the project that can score cue placement
at all, since it needs real cue boundaries to compare against.

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
the `(1.2, 28)` pair it produced was the default until 2026-11-27.

## Not settled

**Extending cue ends.** Extending each cue's end to the next cue's start, the usual subtitle convention, to fix a
systematic early bias. On the n=1 data it made break F1 *worse* at 0.5, 1 and 2s holds, <!-- not-measured: hold lengths are the settings tried -->
because every extended end stops matching a human cue end.
A 0.2s hold was F1-neutral and improved drift, so that is the only variant worth <!-- not-measured: 0.2s is a hold setting -->
revisiting on the larger set. The rejection rests on n=1 data only.

## Related

[timestamps.md](timestamps.md) for the drift half of timing, which these knobs move only
through the written cues.

[^c1]: [m2-ultra-cue-layout.json](../../benchmarks/results/2026-10/m2-ultra-cue-layout.json) runs `cue_sweep_union`
[^c2]: [m2-ultra-timestamps.json](../../benchmarks/results/2026-10/m2-ultra-timestamps.json) runs `ts_0732#engine voxtral`, `ts_default_r1#engine voxtral`, `ts_default_r2#engine voxtral`, `ts_default_r3#engine voxtral`
[^c3]: [m2-ultra-timestamps.json](../../benchmarks/results/2026-10/m2-ultra-timestamps.json) runs `ts_0732#engine voxtral`, `ts_default_r1#engine voxtral`, `ts_default_r2#engine voxtral`, `ts_default_r3#engine voxtral`; [m2-ultra-cue-layout.json](../../benchmarks/results/2026-10/m2-ultra-cue-layout.json) runs `cue_sweep_union`
[^c4]: [m2-ultra-derived-measures.json](../../benchmarks/results/2026-10/m2-ultra-derived-measures.json) runs `derive_cues`
