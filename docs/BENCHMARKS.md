# Benchmarking and metrics

How every number in this project was produced, and why the metrics are unusual.
Read this before quoting a figure or adding a result of your own.

## Read this before quoting a number

Between-file variance here is larger than most of the config effects being
measured. Per-file coverage error spans 11-28% for a single unchanged config, so a
1-point difference between two configs is a tie unless a paired test says
otherwise. At n=7 the corpus resolves effects of roughly 3.2 points.

The samples are also small and uneven. Corpus results are n=20 files in two
languages, mostly from one production pipeline plus 5 YouTube uploads. Timing
results are n=7, since only those files have human-authored SRTs. The cue-layout
defaults were tuned on a single file.

If you are assembling your own corpus, expect to throw a lot away. Vetting rejected
more candidates than it accepted here: 8 of 15 local files and 5 of 10 YouTube
uploads turned out to have audio in a different language than their transcript
claimed.

## The headline comparison

Whisper is about 1.3 points more accurate than Voxtral on the corpus and the gap is
statistically resolved. Voxtral is faster, has better-behaved timestamps, and is
byte-reproducible, which is why it is the default. Against whisper.cpp on one clip
the two are effectively tied, so no "fastest on Apple Silicon" claim survives the
measurement.

20 files, 7.95h of spontaneous multi-speaker Japanese and English, on an M2 Ultra:

| engine | JP coverageCER | EN coverageWER | x realtime |
|---|---|---|---|
| voxtral | 16.22% | 25.24% | **22.8x** |
| whisper-turbo, no-condition | **14.93%** | **22.54%** | 15.0-20.1x |


Whisper wins accuracy on the corpus by about 1.3 points and the result is
statistically resolved: its 3-run CIs sit entirely below Voxtral's value, and all 3
runs beat it in both languages. Voxtral wins throughput and is byte-reproducible,
which is why it is the default.

Timestamps go the other way. Voxtral's worst drift is 37.1 ms/min against Whisper's
170.2, and its p95 error is less than half Whisper's on every file. Whisper's cue
boundaries match human editing better (F1 52.6% against 42.8%). Voxtral is better
where errors are hard to fix, since timestamps come from the model, and worse where
they are easy, since cue grouping is our own heuristic.

Against other Apple Silicon runners on one clip, whisper.cpp at 41.7x / CER 0.0832
and this repo's mlx-whisper at 44.3x / CER 0.0908 are effectively tied, both inside
the n=1 noise band. **No "fastest on Apple Silicon" claim survives the
measurement.** What this repo adds is the Voxtral engine, one interface over all
three, and the harness. Two findings worth carrying anyway: faster-whisper's
reputation does not transfer here, because CTranslate2 has no Metal backend and
runs CPU-only; and quantization costs *speed* on the GPU, with whisper.cpp q5_0
running 27% slower than fp16 at identical CER.

## What changed a default

Findings that set a default in this repo, or that would change how you use it. The
full set, including confidence intervals and the experiments that produced nothing,
is in [RESULTS.md](../RESULTS.md).

`--delay-ms 2400` is the biggest lever here and it is free. Coverage CER goes 25.62%
at 480ms, 20.51% at 960ms, 16.44% at 2400ms, at the same speed, winning on all 7
files individually. It is the default.

`condition_on_previous_text=False` is mandatory before quoting any long-form Whisper
number. large-v3 goes from 39.91% to 17.36% with it off. Conditioning each 30s
window on the previous window's text lets a repetition loop feed itself, and long
files give it room to compound. At library defaults it even inverts the size
ranking, making large-v3 worse than small on Japanese. The registry sets it for
every `whisper-*` alias.

Running a model through the wrong long-form algorithm can cost more than the model
choice. kotoba-whisper spans 68 points across three runners on identical weights,
94.23% down to 26.16%, and its window length is worth up to 23 more.

`--prompt` is not an instruction field. The decoder treats it as text it already
emitted, so "Transcribe the audio accurately" makes it continue off-register: +5.90
points on one clip, and +13.77 WER points on the English corpus files against +1.41
on the Japanese, which fits the mechanism. Use domain terms or a topic sentence.
Prompt bias is a weak lever regardless: 11 style variants spread 0.5 points against
a ~1 point noise floor, and none recovered the rare proper nouns they targeted.

Quiet input hurts silently. The mel front end clamps at an absolute floor, so a
recording at -12 dBFS peak loses ~3.8 points with no other symptom. `--gain auto`
fixes it and is a no-op above -6 dBFS, verified byte-identical on all 7 files of the
original corpus.

Quantization costs nothing measurable. fp16 7.61%, 8-bit 7.49%, mxfp8 7.66%, nvfp4
7.49%, 4-bit 7.23%: the whole spread is 0.43 points on 4205 characters, while fp16
costs 1.6x the wall clock and 15.3GB. Use 4-bit.

Energy-based chunk boundaries beat Silero VAD by 0.8-3.0 points, which is the
opposite of what the VAD literature predicts. The VAD cuts really are cleaner by the
obvious measure; they just leave the next chunk no silence to warm up on.

Batching the encoder does not help. It is compute-bound at ~4950 FLOP/byte for a 30s
chunk, and measured 0.84-0.91x. A decode step is 2 FLOP/byte, which is why batching
works there and not here.

## Metrics

### Coverage CER, and why it exists

Our reference transcripts were edited for storytelling: off-topic chatter,
other-language side conversation, and music during breaks were cut. The audio
still contains that material, so a *correct* transcription legitimately includes
text the reference lacks, and plain CER read 130-152% on the Japanese files
purely from those insertions.

`evals/eval_coverage.py` treats the reference as a subsequence to locate rather than as
the whole truth. It charges substitutions, deletions, and *short* insertions, so
hallucination and repetition loops still count, but excuses insertion runs longer
than `min_cut` as omitted-from-reference audio. It also strips speaker-label
lines (diarization metadata) and picks the scoring unit per file: WER for
space-delimited scripts, CER for CJK, never averaged across units.

Two properties to keep in mind. Absolute levels move ~5 points as `min_cut`
varies from 10 to 80, so **the threshold has to be quoted with any absolute
number** (every figure here uses 30 chars / 6 words). But the *ranking* is
identical at every threshold on both units, and Voxtral's disadvantage against
turbo-nocond shrinks as the threshold tightens, so the excusal rule is not
flattering Whisper. `extra_ratio` well above 1 is the signal that a reference is
edited.

On complete references, accuracy is plain CER, the standard metric for Japanese.
`evals/eval_cer_lenient.py` additionally forgives edits where one side is a pure-kana
respelling of the same word, and is the fair number; `evals/eval_cer_kana.py` converts
both sides to readings and is too generous, because it also forgives homophone
substitutions, which are wrong words rather than spelling choices. On the best
run: CER 7.42%, lenient 7.11%, kana 5.72%, so only ~0.3 points of the error is
genuine orthographic variance.

### eval_timing

`evals/eval_timing.py` scores timestamps against a timed reference, and needs an
authored SRT or VTT; plain-text references cannot support it. It aligns the two
texts character-by-character and treats every run of >= 8 matched characters as
an *anchor*, so cut material cannot masquerade as drift. It reports two things
separately and never combines them into one score:

- **drift**: median and p95 absolute error at anchors, plus a regression of
  signed error on time. Slope is separated from constant offset because a whole
  file shifted 400ms is trivially correctable while error that grows through the
  file is not.
- **cue breaks**: F1 of hypothesis cue boundaries against the author's, within a
  0.5s tolerance, plus the rate of cue ends landing inside a reference cue.

`anchor_coverage` is reported alongside; below ~50% the timing estimate is weak
and should be dropped rather than averaged in.

Voxtral's cue grouping defaults were retuned against this metric over 376
parameter combinations, re-cueing a cached token stream rather than re-decoding:
moving from `(gap_s=1.0, max_chars=32)` to `(1.2, 28)` took break F1 from 35.4%
to 43.6% and mid-phrase splits from 62.4% to 55.3%, with drift slightly better
(204ms vs 214ms median). Caveats kept next to the number: **n=1**, only one timed
reference was available locally; `gap_s` is flat by median across 0.6-1.3, so 1.2
is a peak rather than a robustly better value; and `max_chars=28` sits close to
that reference's mechanical 15-char line-wrap width, so part of the gain may be
matching its wrap arithmetic rather than real phrase awareness.


## Benchmarks

Every number here comes from a script in `benchmarks/`. The committed
`benchmarks/*.json` files are the raw results for the 7-file corpus, the narration
clip and the batch sweeps; run them against your own audio to get comparable rows.
Scrub any JSON before committing it (last command below): it picks up hostnames,
absolute paths and private filenames.

```bash
# decode throughput vs batch size on this machine
uv run python benchmarks/sweep_batch.py --json sweep.json

# end-to-end wall clock + CER across chunk/batch/silence/KV configs
sh benchmarks/run_matrix.sh audio.wav reference.srt

# one config across a whole corpus of audio/transcript pairs
uv run python benchmarks/run_corpus.py --corpus DIR --json out.json

# the same corpus with Whisper, scored identically (needs the whisper extra)
uv run python benchmarks/run_whisper.py --corpus DIR \
    --model mlx-community/whisper-large-v3-turbo --no-condition --json out.json
sh benchmarks/run_whisper_sweep.sh CORPUS_DIR NARRATION_WAV NARRATION_SRT

# kotoba-whisper through the authors' own transformers pipeline, as a reference
# for the MLX driver. This one script needs torch; the CLI never does.
uv run python benchmarks/run_kotoba_hf.py --corpus DIR --chunk-length-s 10 --json out.json

# every registry alias end to end, with a works/does-not-work matrix
uv run python benchmarks/matrix_models.py --audio AUDIO --reference REF --json out.json

# Whisper through the insanely-fast-whisper recipe (chunked transformers pipeline)
uv run python benchmarks/run_ifw.py --corpus DIR \
    --model openai/whisper-large-v3-turbo --json out.json

# order a sweep cheapest-first: per-model speed on a fixed slice
uv run python benchmarks/probe_whisper_speed.py

# paired per-file comparison of any two corpus runs (engines or configs)
uv run python benchmarks/compare_engines.py a.json b.json

# a nondeterministic engine's run distribution vs a deterministic baseline
uv run python benchmarks/repeat_distribution.py --runs run*.json --baseline vox.json

# collect every corpus/narration run into the comparison tables
uv run python benchmarks/summarize_engines.py benchmarks/*.json

# timing quality (drift + cue breaks) for both engines, on timed references only
uv run python benchmarks/run_timing_sweep.py --corpus DIR --json out.json

# how sensitive are corpus results to the insertion-run threshold?
uv run python benchmarks/sweep_min_cut.py --corpus DIR --hyp-dir bench_out/corpus

# input level: does gain matter on this material?
uv run python benchmarks/sweep_gain.py --corpus DIR --json out.json

# rescore saved hypotheses under all metrics, with bootstrap CIs
uv run python benchmarks/rescore_all.py --reference ref.srt bench_out/*.srt

# paired significance test between two configs on the same audio
uv run python benchmarks/compare_configs.py a.srt b.srt --reference ref.srt

# sweep prefix-overlap length
uv run python benchmarks/sweep_overlap.py --audio AUDIO --reference ref.srt \
    --chunk-seconds 30 --max-batch 32 --overlaps 0,4,8

# A/B the vocabulary-bias prompt (style, ordering, separators)
uv run python benchmarks/ab_prompt.py --audio AUDIO --reference REF --terms "a,b,c"

# fp16 vs quantized (needs >=32GB)
uv run python benchmarks/sweep_precision.py --audio AUDIO --reference ref.srt

# strip hostnames, paths and private filenames before committing results
uv run python benchmarks/scrub_results.py benchmarks/*.json
```

Lower-level probes (encoder batching, KV length, seam-error location, step
breakdown, batch reshaping, `mx.compile`) are in `benchmarks/probes/`. Most
recorded a negative result, which is why they are kept: see the "ideas tested and
rejected" table in [RESULTS.md](RESULTS.md) before proposing performance work.

