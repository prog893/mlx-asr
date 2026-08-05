# The test corpus, and how to build your own

## What was used here

20 recordings, about 7.9 hours: 17 Japanese and 3 English. Two kinds of material,
which behave differently enough that they are never pooled:

- **Spontaneous multi-speaker recordings.** Studio conversation, several speakers,
  code-switching mid-sentence, long pauses, and stretches in a third language. Their
  references are editorial transcripts, not verbatim ones (see below). This is the
  realistic case and where the 16-18% coverage-error baseline comes from.
- **Published videos with author-written subtitle tracks.** Narration and dialogue
  with real cue timings, which is the only material that can score timestamp quality
  at all. Seven files have such timed references.

Plus one prepared-narration clip with a complete verbatim reference, which is the one
file where plain CER is meaningful and where most single-clip work was done.

The corpus is not distributable, so none of the per-file numbers in these documents
can be re-derived from this repo. That is why the conclusions are written out rather
than left implicit in data files: see the other documents in this directory.

**Contributing a hardware profile needs none of this.** `mlx-asr-bench` drives the
decoder with random embeddings, so it measures decode throughput with no audio and no
reference at all, and prints a ready-to-paste issue body. Only accuracy work needs a
corpus.

## Building your own

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
  it correctly, which inflates plain CER past 100% on files where the transcription is
  in fact good.

### Editorial references, and why plain CER breaks on them

A reference written for readability rather than for ASR evaluation will omit audio:
off-topic passages, side conversation in another language, non-speech segments. The
audio still contains that material, so a *correct* transcription legitimately includes
text the reference lacks.

Plain CER counts all of it as insertions. On this corpus four files read **over 100%
plain CER** while scoring 15-20% on the coverage-aware metric, a gap of up to 133
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
  Japanese files, costing 25 points.
- **Audio is converted once** to 16kHz mono and cached in the system temp directory
  keyed on stem. Two corpora with colliding stems will reuse each other's converted
  audio, so clear that cache when switching corpora.

## How much material is enough

More than feels necessary. Between-file variation on this corpus is larger than most
of the config effects being tested: the same unchanged config spans 11-28% per-file
coverage error. At 7 files the corpus resolves differences of roughly 3.2 points, so
anything smaller than that needs either a paired test or more audio. Several
single-clip findings here reversed sign when a real corpus arrived.

Decoding is deterministic per machine, so repeating a run adds no information. Only
more audio adds statistical power.
