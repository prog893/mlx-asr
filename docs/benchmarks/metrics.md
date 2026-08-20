# Metrics: which number to trust, and why the usual one is wrong here

**Conclusion first.** Plain CER is the wrong metric for most of this project's material,
because the references deliberately omit audio and plain CER counts correct
transcription of that audio as error. Four corpus files read **over 100% plain CER**
while scoring 15-20% on the coverage-aware metric. Use `coverage CER/WER` for editorial
references, plain CER only for verbatim ones, and lenient CER as the fair Japanese
number. Kana CER is reported but should not be quoted as "the" figure: it over-forgives,
and until 2026-08-19 it also under-forgave, because pykakasi never read numerals (see
below).

## The metrics, and when each is wrong

| metric | forgives | correct when |
|---|---|---|
| `eval_cer` | nothing | the reference is verbatim |
| `eval_coverage` | long insertion runs, as omitted-from-reference audio | the reference is editorial |
| `eval_cer_lenient` | kana/kanji respelling of the **same** word | you want the fair Japanese number |
| `eval_cer_kana` | anything sharing a reading | almost never; see below |
| `eval_coverage_kana` | both of the above at once | an editorial reference *and* a model that reformats numbers |
| `eval_timing` | n/a, scores timestamps | you have a timed reference |

All are NFKC-normalized with punctuation and whitespace stripped. Scoring unit is chosen
per file from the reference script: character-level for CJK, word-level for
space-delimited text. The two are aggregated separately and **never averaged**, because
one substituted word is one word-level error but only a fraction of the characters in a
CJK sentence. That is also why English and Japanese always appear in separate columns
here.

### What stripping punctuation and whitespace hides

Two defects are invisible to every metric above, and both were found only by reading the
transcripts rather than the scores:

- **Word spacing on space-delimited output.** Because whitespace is deleted before
  scoring, a hypothesis can lose almost all its spaces and still score normally on
  character coverage. Conversely, word-level coverage over-punishes it: one arm of the
  prompt experiment emitted 2 spaces where the unprompted run emitted 4016, which read as
  97% coverage WER while the character content was within 3 points of baseline. See
  [prompt.md](prompt.md).
- **Punctuation density.** Voxtral emits Japanese `。` and `、` at roughly a third of the
  rate the references use (7.6 to 10.6 marks per 1000 characters against 30.7), in every
  configuration measured. Since punctuation is stripped, no number in these documents
  reflects it, and it does not appear as a regression anywhere.

Neither is a scoring bug: both metrics do what they are specified to do, and stripping is
what makes the kanji/kana leniency rule and the editorial references workable. But a
config that only damages spacing or punctuation can score clean here, so **read a sample of
the actual output before concluding a config is harmless.**

## Coverage CER, and why it had to exist

The references for the spontaneous recordings were written for readability, not for ASR
evaluation: off-topic passages, side conversation in another language, and non-speech
segments were cut. The audio still contains that material, so a *correct* transcription
legitimately includes text the reference lacks.

Plain CER counted all of it as insertions and read 130-152% on the Japanese files. On the
20-file corpus:

| file | coverage | coverage CER | plain CER | chars excused | extra_ratio |
|---|---|---|---|---|---|
| rec-02 | 86.6% | **17.4%** | 150.7% | 8453 | 2.32 |
| rec-06 | 83.4% | **19.9%** | 148.2% | 2966 | 2.23 |
| rec-03 | 86.8% | **17.1%** | 144.5% | 12522 | 2.27 |
| rec-05 | 89.1% | **14.6%** | 134.1% | 8957 | 2.20 |
| rec-07 | 98.1% | **5.5%** | 99.3% | 1002 | 1.97 |
| rec-04 | 90.3% | **11.7%** | 64.4% | 1337 | 1.50 |
| rec-12 | 82.2% | **18.0%** | 37.4% | 82 | 1.18 |

`rec-07` is the sharpest case: 99.3% plain against 5.5% coverage, with 98.1% of reference
characters found correctly. An evaluation of this corpus on plain CER would rank configs
by **how much real speech they skip**, which is the opposite of what anyone wants.

`extra_ratio` (hypothesis length / reference length) is the tell. Near 1.0 means the pair
is comparable and plain CER can be trusted; well above 1 means the reference is
editorial.

### How it works

`scripts/metrics/eval_coverage.py` treats the reference as a *subsequence to locate* rather than
the whole truth. Using Levenshtein opcodes, it charges substitutions and deletions and
*short* insertions, but excuses insertion runs longer than `min_cut` as
omitted-from-reference audio. So hallucination and repetition loops still count (they
produce short, scattered insertions inside matched text) while correct transcription of
cut material does not.

It also strips speaker-label lines (`Name:` alone on a line), which are diarization
metadata the models never emit.

### The threshold has to be quoted with any absolute number

`scripts/benchmarks/sweep_min_cut.py`, varying `min_cut` over saved hypotheses:

| config | mc10 | mc20 | mc30 | mc50 | mc80 |
|---|---|---|---|---|---|
| 30s b32 kv8 | 14.97% | 15.89% | **16.44%** | 17.81% | 20.23% |
| 30s b32 kv8 ov8 | 15.62% | 16.67% | 17.34% | 18.12% | 20.61% |
| 60s b16 kv8 | 16.80% | 17.95% | 18.21% | 19.32% | 20.69% |

Absolute levels drift about 5 points across that range. But the **ranking is identical at
every threshold**, on both units, which is the property the tuning conclusions actually
depend on. Every figure in these documents uses 30 characters / 6 words.

### It is not flattering one engine

The obvious worry is that excusing insertions helps whichever engine inserts more.
Whisper's `extra_ratio` on Japanese (2.2-2.7) is higher than Voxtral's (2.1), so the
threshold could in principle favour Whisper. Across engines:

| config | mc10 | mc20 | mc30 | mc50 | mc80 |
|---|---|---|---|---|---|
| turbo-nocond | 13.85% | 14.78% | 14.93% | 16.03% | 16.84% |
| voxtral c30b32_kv8 | 14.97% | 15.89% | 16.44% | 17.81% | 20.23% |
| large-v3-nocond | 13.32% | 15.53% | 17.36% | 20.10% | 23.39% |
| large-v3-turbo | 22.76% | 23.99% | 24.97% | 26.58% | 28.05% |
| medium | 21.81% | 26.23% | 28.93% | 32.18% | 36.49% |

The two leaders keep their order across the whole range, and Voxtral's disadvantage
against turbo-nocond *shrinks* as the threshold tightens (-1.1 at mc10 versus -3.4 at
mc80). If anything the excusal rule works against Voxtral. One rank swap occurs at mc10,
where large-v3-nocond overtakes turbo-nocond, and those two are inside each other's
rerun noise anyway.

## How much of Japanese CER is just spelling

Japanese has no fixed orthography, so a reference typist's choice of kanji versus kana is
not a model error. Three metrics on the best run of the narration clip:

| metric | value | forgives |
|---|---|---|
| CER | 7.42% | nothing |
| lenient CER | 7.11% | script changes of the *same* word |
| kana CER | 5.72% | anything sharing a reading |

**Use lenient CER as the fair number.** Kana CER over-forgives: collapsing everything to
readings also excuses genuine homophone errors, where a different word sharing the same
reading is substituted. A reader notices that; kana CER scores it free.

This corrects an earlier claim in this project that ~1.6 CER points were orthographic.
That figure came from the kana measure. The honest number is about 4% of edits, i.e.
**~0.3 CER points**, so the vast majority of remaining error is real misrecognition.

Lenient CER changes no ranking on either machine, sitting uniformly 0.2-0.3 points below
plain CER. That is a useful negative result: the refinement was worth doing to correct
the magnitude claim, but it does not alter any decision. Kana CER, by contrast, does
occasionally reorder configs (fp16 leads on kana but not on CER or lenient), which is a
second reason to prefer lenient.

### Kana CER never forgave digits, which is most of the confound

Found while integrating Qwen3-ASR, and it qualifies every kana figure above:
**pykakasi does not read numerals.** It converts `2018年` to `2018ねん`, not to
`にせんじゅうはちねん`, so the digits-versus-spelled-out difference, the largest
orthographic class on this material, was charged in full by the very metric meant to
forgive it. On two spellings of one sentence (`答えは1500でした` against
`こたえはせんごひゃくでした`, nothing misheard) kana CER read **55%** where the corrected
figure is 0%.

`eval_cer_lenient.read_number` fills the gap, and `eval_coverage_kana` uses it by
default. It is **opt-in** in the older plain scripts (`--read-digits`), so every kana
figure already published in this repo keeps the meaning it had; those figures still
understate leniency wherever the material contains numbers. This matters most for a model
doing inverse text normalisation, which is why it surfaced with Qwen3-ASR rather than
earlier.

Stated limits: counter-specific readings are not modelled (`4年` reads as よんねん here,
not よねん), and a digit string a speaker would read one digit at a time is read as a
quantity unless it is longer than 20 digits or carries a leading zero. Each leaves a small
residual charge, so this narrows the confound rather than erasing it.

## Score the text, not the subtitle layout

Cue segmentation is underdetermined: many groupings of the same tokens are equally valid,
and the choice shifts characters between lines. Scoring a concatenated SRT therefore
risks measuring the cue heuristics rather than the transcription.

That concern also uncovered a real bug. Cue text used to be assembled by appending raw
token *bytes* and decoding at each flush. The tokenizer is a byte-level BPE, so a flush
landing mid-character split it and both halves decoded to U+FFFD. **217 stray replacement
characters existed across the saved SRTs**, each an unearned CER error. Fixing it (break
only on complete-character boundaries) improved CER by 0.40 points paired, CI
[+0.05, +0.81].

With that fixed, text-only CER equals SRT-joined CER **to four decimals across all 12
configs on both machines**:

| config | M2 Ultra 128GB text | M2 Ultra 128GB srt | M4 16GB text | M4 16GB srt |
|---|---|---|---|---|
| 60s, b16 | 7.40% | 7.40% | 7.75% | 7.75% |
| 60s, b16, kv8 | 7.28% | 7.28% | 7.73% | 7.73% |
| 30s, b32, kv8 | 8.63% | 8.63% | 9.13% | 9.13% |
| 30s, b32, kv8, ov4 | 7.16% | 7.16% | 7.87% | 7.87% |
| 30s, b32, kv8, ov8 | 7.13% | 7.13% | 7.25% | 7.25% |
| 60s, b16, kv8, compact | 8.30% | 8.30% | 7.97% | 7.97% |

So cue grouping does not distort the metric: the only text-changing step was a per-cue
`.strip()`, and cue boundaries land in gaps rather than inside words. `mlx_asr/text.py`
decodes each chunk's token stream in one call for the plain-text path, and
`scripts/benchmarks/run_matrix.sh` writes every format per run so both views are always
available.

## How to compare two configs

Not by eyeballing two aggregates. Two methods, for two different questions.

**"Is A better than B on this clip?"** Paired test over regions of the same audio
(`scripts/benchmarks/compare_configs.py`): split the reference into 40 regions, compute per-region
error for each config, bootstrap the per-region differences. Shared difficulty cancels
instead of inflating both intervals. Sign convention in the output is `diff = a - b`, so
**positive means A is worse**.

**"Would this hold on other audio?"** Bootstrap over *files*, length-weighted
(`scripts/benchmarks/compare_engines.py`). This is the only method that speaks to
generalization, and it is much stricter: the per-file paired difference between two real
configs has SD 3.50 points, so the corpus resolves effects of about **3.2 points at n=7
and 1.6 at n=20**. Most lever docs here are n=7; check which the experiment used before
reading "not resolvable" as "small".

That gap between the two is why several single-clip results in this project reversed on a
corpus. A significant paired result on one clip means "real on this clip", not "real".

### The English bootstrap is n=3, and that is worse than it sounds

A bootstrap resamples files with replacement, so the number of *distinct* resamples it can
ever draw is the number of multisets of size n from n items, `C(2n-1, n)`:

| files | distinct resamples possible |
|---|---|
| 3 | **10** |
| 7 | 1,716 |
| 17 | 1.17e9 |
| 20 | 6.89e10 |

At n=3 there are ten possible resamples no matter how many are requested, so the resulting
histogram is a handful of spikes and the 2.5th and 97.5th percentiles land on specific
compositions rather than on a smooth tail. `plot_evidence.py bootstrap` makes this visible
immediately: the Japanese panel is a smooth distribution centred near zero, and the English
panel next to it is five bars.

The consequence is not that the English numbers are wrong, but that **a 95% interval is the
wrong summary for them**. Two English comparisons can both be reported as "not resolvable"
while one has every resample on the same side of zero and the other is split evenly. Read
the per-file numbers and the direction for English, not the interval. Anything that rests
on the English side is a direction with n=3 behind it, which is why those conclusions are
labelled as directional throughout these documents.

**Do not use repeat runs for statistical power.** Voxtral decodes greedily, so a rerun is
byte-identical on the same machine and adds no information; only more audio adds power.
Whisper does sample, so it genuinely needs a run distribution: see
[determinism.md](determinism.md).

## Reproducing

```bash
cd scripts        # metrics is a package here, so -m needs it on the path
uv run python -m metrics.eval_coverage REFERENCE HYPOTHESIS [--verbose]
uv run python -m metrics.eval_cer REFERENCE HYPOTHESIS
uv run python -m metrics.eval_cer_lenient REFERENCE HYPOTHESIS [--read-digits]
uv run python -m metrics.eval_coverage_kana REFERENCE HYPOTHESIS
uv run python -m metrics.eval_timing REF.srt HYP.srt
```

`--verbose` on the coverage scorer prints the longest excused runs with their text, which
is how to check that the excusal rule is catching cut material rather than hallucination.

Needs the `eval` extra: `uv sync --extra eval`.
