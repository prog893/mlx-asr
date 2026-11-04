# Reference: the test corpus

Every accuracy and timing figure in these documents was measured on one of three sets of
material: the 20-file corpus, the 7-file subset it grew from, or a single narration clip.
This page is the one description of all three, and the other pages link here rather than
restating it. It ends with how to build a corpus of your own.

## The 20-file corpus

20 recordings, 8.01h: 17 Japanese (6.85h) and 3 English.[^k1] Files run from 1.9 to 93 minutes;
the shortest is 112s.[^k1] Two kinds of material, which behave differently enough that they are
never pooled:

- **Spontaneous multi-speaker recordings.** Studio conversation, several speakers,
  code-switching mid-sentence, long pauses, and stretches in a third language. Their
  references are editorial transcripts rather than verbatim ones (see
  [Editorial references](#editorial-references-and-why-plain-cer-breaks-on-them)). This
  is the realistic case and where the 16-18%[^k2] coverage-error baseline comes from.
- **Narration and dialogue with author-written subtitle tracks.** Real cue timings,
  which is the only material that can score timestamp quality at all.

### Timed references

Seven files have author-written subtitle tracks with real cue timings: the
[single clip](#the-single-clip), which is outside the 20, and six files that joined the
20-file corpus when it grew. They are the only material that can score drift or
cue placement ([timestamps.md](../timestamps.md), [cue-layout.md](../cue-layout.md)). All seven
were authored by one editor, and no other recording here has an authored subtitle track,
so timing work stays at n=7. The timed set is a different set of files from the
[7-file subset](#the-7-file-subset), whose references are plain transcripts.

### Editorial references

The references were written for readability rather than for ASR evaluation: off-topic
passages, side conversation in another language, and non-speech segments were cut, and
several recordings open with minutes of untranscribed studio talk. The audio still
contains that material, so a correct transcription includes text the reference lacks.
Plain CER counts all of it as insertions: four files exceed 100% plain CER[^k2] while scoring
15-20% on the coverage-aware metric.[^k2]

### How files are scored

Every corpus figure is coverage CER (Japanese) or coverage WER (English) at `min_cut` 30
characters / 6 words, defined in [metrics.md](metrics.md). The scoring unit and the
language are both taken per file from the reference script, the two units are aggregated
separately and never averaged together, and speaker-label lines are stripped before
scoring. Comparisons between configs are a length-weighted bootstrap over files
(`scripts/benchmarks/compare_engines.py`), whose 95% half-width is about **1.20 points[^k4] at
n=20** for Voxtral at 4-bit, the default then, against Whisper turbo-nocond[^k4].

### How it grew

The corpus started as the 7-file subset below and grew in two steps. Measured by Japanese
file count, it went from 5 to 12 to 17 ([engines.md](../engines.md)). Pages that quote
results from the intermediate size say so where they do.

The corpus is not distributable, so the audio, references and transcripts are not in this
repo. The per-file scores are: saved runs are kept as per-file rows, keyed by a hash of
the source id, under `benchmarks/results/`, and every figure these documents cite to a
run-group can be recomputed from those rows. Rescoring a hypothesis with a different metric
still needs the corpus.

## The 7-file subset

The original 7 recordings: 5 Japanese and 2 English, 5.18h,[^k1] editorial references, scored
with the same coverage metric. Peak levels run from -0.5 to -5.3 dBFS
([input-level.md](../input-level.md)). It was the whole corpus before the growth to 20, so
every sweep run before then used it, and some sweeps still do:

- transcription delay ([delay.md](../delay.md))
- input gain ([input-level.md](../input-level.md))
- the window sweeps for `qwen3-asr` and `voxtral-v1` ([qwen3-asr.md](../engines/qwen3-asr.md),
  [voxtral-v1.md](../engines/voxtral-v1.md))
- the early prompt sweeps ([prompt.md](../prompt.md))
- parts of the chunking and determinism work ([chunking.md](../chunking.md),
  [determinism.md](determinism.md))

At n=7 the file-level 95% half-width is about **2.04 points**, against 1.20 at n=20[^k4]. A
"not resolvable" verdict on this subset means the effect is smaller than about 2 points[^k4],
which is weaker than saying it is small; see [metrics.md](metrics.md).

## The single clip

One 935s Japanese prepared-narration recording with a complete verbatim reference[^k1] (4205
scored characters[^k5]). It is the one file where plain CER is meaningful, and most early
lever work was done on it, compared with a paired test over 40 regions of the same audio.
[prompt.md](../prompt.md) also uses a 180s excerpt[^k1] of it (886 characters after the scorer's
normalization[^k5]).

A paired result on the clip means the effect is real on that clip. Several clip findings
reversed sign on the corpus, among them overlap ([chunking.md](../chunking.md)) and the
precision tie ([quantization.md](../quantization.md)). **Where a corpus result exists, it
supersedes the clip result.** Clip results that have no corpus counterpart are labelled as
such on their pages.

## Building your own

**Contributing a hardware profile needs none of this.** `mlx-asr-bench` drives the
decoder with random embeddings, so it measures decode throughput with no audio and no
reference at all, and prints a ready-to-paste issue body. Only accuracy work needs a
corpus.

### Shape on disk

One flat directory. Each audio file is paired with a reference by stem:

```
mycorpus/
  interview-01.wav          interview-01.srt              # timed reference
  interview-02.wav          interview-02_transcript.txt   # text-only reference
  lecture.m4a               lecture.vtt
```

Audio can be anything FFmpeg reads. Reference resolution is by stem, first match
wins, in this order: `<stem>_transcript.txt`, `<stem>.srt`, `<stem>.txt`. Audio with
none of those is skipped silently, so check the file count the harness reports
against what you expected.

### Which reference format to use

**Table:** the reference file format each kind of measurement requires.

| you want to measure | reference needed |
|---|---|
| text accuracy only | `_transcript.txt` is enough |
| timestamp drift, or subtitle cue placement | must be `.srt` or `.vtt` |

Plain text cannot support the timing metrics at all, and `run_timing_sweep.py` and
`sweep_cues.py` simply ignore any file without a timed reference. If you care about
timings, author or obtain real cue times; do not synthesise them from a text
transcript, since the metric would then be scoring your synthesis.

### Vet every pair before trusting a score

This is the step that mattered most here, and more than half of the candidates were
rejected by it. Transcribe a three-minute sample from the middle of the file with
language autodetect and compare the spoken language to the reference script. The
failure it catches is a reference in one language over audio in another, which scores
as catastrophic model failure and is nothing of the kind.

Two specific traps:

- **A dub.** If material exists in several languages, a downloader may hand you the
  default audio track rather than the one matching your subtitles. Pin the track
  language explicitly. This corpus contains one recording present in two languages,
  identical in duration to the sample, which is legitimate to keep but breaks any
  tool that assumes one language per corpus or uses duration as a file key.
- **Untranscribed material at the edges.** Recordings here open with several minutes
  of studio talk in another language that the references omit. The model transcribes
  it correctly, which inflates plain CER past 100%[^k2] on files where the transcription is
  in fact good.

### Editorial references, and why plain CER breaks on them

A reference written for readability rather than for ASR evaluation will omit audio:
off-topic passages, side conversation in another language, non-speech segments. The
audio still contains that material, so a *correct* transcription legitimately includes
text the reference lacks.

Plain CER counts all of it as insertions. On this corpus four files read **over 100%[^k2]
plain CER** while scoring 15-20% on the coverage-aware metric, a gap of up to 133[^k2]
points. The tell is `extra_ratio` (hypothesis length over reference length): near 1.0
means the pair is comparable and plain CER can be trusted; well above 1 means the
reference is editorial and plain CER is meaningless.

Use `scripts/metrics/eval_coverage.py` for such material. It treats the reference as a
subsequence to locate, charging substitutions, deletions and *short* insertions, while
excusing insertion runs longer than a threshold as omitted-from-reference audio. That
keeps hallucination and repetition loops chargeable while not punishing correct
transcription of cut material. Quote the threshold with any absolute number.

If your references are verbatim, none of this applies and plain CER is the right
metric.

### Other things the harness assumes

- **Speaker labels** on their own line (`Name:`) are treated as diarization metadata
  and stripped before scoring, since the models emit no speaker labels. They never
  reach any output.
- **Scoring unit is chosen per file** from the reference script: character-level for
  CJK, word-level for space-delimited text. The two are aggregated separately and
  never averaged together, because one substituted word is one word-level error but
  only a fraction of the characters in a CJK sentence.
- **Language is taken per file** from the reference rather than set once for the
  corpus, which is what lets a mixed-language set work. Whisper's own 30-second
  autodetect is not a safe substitute: on this material it returned Russian for
  four of the five Japanese files of the 7-file subset, costing 25 points.[^k3]
- **Audio is converted once** to 16kHz mono and cached in the system temp directory
  keyed on stem. Two corpora with colliding stems will reuse each other's converted
  audio, so clear that cache when switching corpora.

## How much material is enough

More than feels necessary. Between-file variation on this corpus is larger than most
of the config effects being tested: the same unchanged config spans 5.5% to 24.1% coverage CER across the 17 Japanese files[^k2] and 3.3% to 28.0% coverage WER across the 3 English
ones.[^k2] Resolution depends on how many files an experiment uses: a 95% half-width of
roughly 2.0 points at 7 files and 1.2 at all 20[^k4], so anything smaller than that needs either a paired
test or more audio. Several single-clip findings here reversed sign when a real corpus
arrived ([The single clip](#the-single-clip)).

Voxtral, Voxtral v1, Qwen3-ASR, parakeet and reazon decode deterministically per machine, so
repeating their runs adds no information and only more audio adds statistical power.
Whisper, and kotoba through the MLX driver, sample on fallback, so they need several runs
as well ([determinism.md](determinism.md)).

## Related

- [metrics.md](metrics.md): coverage CER, `min_cut`, and how comparisons are tested
- [determinism.md](determinism.md): run-to-run and cross-machine variation on this material
- [README.md](../README.md): index of every lever page

[^k1]: [corpus.json](../../../benchmarks/results/corpus.json)
[^k2]: [ultra-voxtral-headline.json](../../../benchmarks/results/2026-10/ultra-voxtral-headline.json) runs `vox_default_c30b128_kv8`, `vox_default_c30b128_kv8_r2`, `vox_default_c30b128_kv8_r3`
[^k3]: [ultra-mincut-inputs.json](../../../benchmarks/results/2026-10/ultra-mincut-inputs.json) runs `mincut7_whisper_turbo_libdef`; [ultra-whisper-condition.json](../../../benchmarks/results/2026-10/ultra-whisper-condition.json) runs `cond7_whisper_turbo_libdef_r2`, `cond7_whisper_turbo_libdef_r3`; [ultra-whisper-language.json](../../../benchmarks/results/2026-10/ultra-whisper-language.json) runs `lang7_turbo_autodetect_r1`, `lang7_turbo_autodetect_r2`, `lang7_turbo_autodetect_r3`
[^k4]: [ultra-resolution-floor.json](../../../benchmarks/results/2026-10/ultra-resolution-floor.json) runs `resolution_floor` (per-file SD 2.75 of the paired Japanese differences over 17 files; 95% half-width 1.96 SD / sqrt(n)), computed from [ultra-voxtral-headline.json](../../../benchmarks/results/2026-10/ultra-voxtral-headline.json) runs `vox_default_c30b128_kv8`, `vox_default_c30b128_kv8_r2`, `vox_default_c30b128_kv8_r3`; [ultra-whisper-sizes.json](../../../benchmarks/results/2026-10/ultra-whisper-sizes.json) runs `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`, `whisper_turbo_nocond_r6`
[^k5]: [ultra-derived-measures.json](../../../benchmarks/results/2026-10/ultra-derived-measures.json) runs `derive_reflen_c`, `derive_reflen` (reference characters after the scorer's normalization)
