# Lever: which engine

**Conclusion first.** Whisper large-v3-turbo with `condition_on_previous_text=False` is
the most accurate option on this corpus, by about 1.7 points on Japanese, resolved by every
test available here including a bootstrap over files. Voxtral is 1.35-1.65x faster, needs
no language hint, needs no long-form-stability flag, and is reproducible on a given
machine, which is why it is the default. No "fastest on Apple Silicon" claim survives
measurement.

Whisper's *library defaults* are worth 7 to 25 CER points on this material, all of it
cross-window repetition loops, so any Whisper figure taken without
`condition_on_previous_text=False` describes the defaults rather than the model. The CLI
ships that flag on `small` and larger.

## Corpus

7 recordings for the size sweep (5 Japanese, 2 English), grown to 20 for the final
Voxtral-versus-Whisper comparison. Plus the single prepared-narration clip, where a
complete verbatim reference makes plain CER meaningful. Editorial references elsewhere, so
coverage CER/WER at `min_cut` 30/6. See [corpus.md](corpus.md) and
[metrics.md](metrics.md).

## Method

`scripts/benchmarks/run_whisper.py` scores with the *same* functions `run_corpus.py` uses, on the
*same* cached 16kHz files (verified byte-identical to what the Voxtral runs read), so rows
sit next to each other without an asterisk. Engine: `mlx-whisper` 0.4.3.

Whisper is run at library defaults (temperature ladder 0.0-1.0,
`condition_on_previous_text=True`) unless a row says otherwise, because that is what
`mlx_whisper.transcribe` gives a user. Paired comparisons bootstrap over files,
length-weighted, 20k resamples (`scripts/benchmarks/compare_engines.py`).

Whisper samples, so a single run is a draw from a distribution; see
[determinism.md](determinism.md) for why the headline uses a 3-run mean.

## Experiment: every Whisper size at the config that ships

**Superseded the earlier 7-file size sweep, which measured the wrong thing.** That table
ran `large-v2`, `medium` and `small` at mlx-whisper's library defaults, while the CLI
ships those three with `condition_on_previous_text=False`. Its rows therefore described
configs this CLI does not run, and understated them by 7 to 25 points. The old numbers
are kept below the new ones because the gap between them is the finding.

20 files, 7.95h, idle M2 Ultra, `--no-condition` on `small` and larger (which is what
ships) and library defaults on `tiny` and `base` (also what ships). Peak GPU memory from
`mx.get_peak_memory()`, reset per file. An independent re-run reproduced every peak here
within 0.02GB and found that Whisper's figure grows with audio length rather than being a
property of the size ([peak-memory.md](peak-memory.md)).

| size | JP coverage CER | EN coverage WER | x realtime | peak GPU | old table, library defaults |
|---|---|---|---|---|---|
| `tiny` | 51.28% | 32.17% | **52.7x** | **3.98GB** | 59.27% |
| `base` | 29.93% | 27.14% | 34.5x | 4.07GB | 29.96% |
| `small` | 21.33% | 22.68% | 20.1x | 4.37GB | 28.61% |
| `medium` | 21.63% | 18.23% | 21.4x | 5.43GB | 28.93% |
| `large-v2` | 17.87% | 17.68% | 14.4x | 6.97GB | 25.02% |
| `large-v3` | **14.55%** | 18.26% | 11.4x | 7.00GB | 39.91% |
| `turbo` **(ships)** | 14.68% | **18.31%** | 23.7x | 5.53GB | 24.97% |

Three things follow.

**`turbo` remains the right default**, and now for a clearer reason than before: it ties
`large-v3` on Japanese (14.68% against 14.55%, well inside the ±0.27 that repeat runs of
turbo spread) while running **2.1x faster** and using 1.5GB less memory. The earlier
7-file table appeared to show turbo winning outright on accuracy; at n=20 with both at
their shipped config the honest statement is that they are tied and turbo is cheaper.

**`large-v3` was the worst-served by the old measurement.** 39.91% at library defaults
against 14.55% shipped, a 25-point artifact of cross-window repetition loops that
`condition_on_previous_text=False` prevents. Anyone reading the old table would have
concluded the model was broken.

**Peak memory barely tracks download size.** `tiny` downloads 0.07GB and peaks at 3.98GB,
because the working set is the 30s mel window and decoder activations rather than the
weights. So the whole size ladder fits in 4 to 7GB, and picking `tiny` to save memory buys
almost nothing while costing 37 points.

The `turbo` row was measured while the host was not idle, so its 23.7x is a floor rather
than a clean figure; the accuracy figures are unaffected, since contention costs wall clock
only. Its 14.68% is consistent with the separately published 3-run mean of 14.49% ±0.27.

### The superseded 7-file table

Kept for the record. Library defaults except where a row says otherwise, so the
`large-v2`, `medium`, `small`, `base`, `tiny` and plain `large-v3`/`turbo` rows do not
describe anything the CLI runs.

| engine / config | JP coverage CER | EN coverage WER | x realtime |
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

Paired against Voxtral, positive meaning Whisper is worse:

| comparison | JP diff | 95% CI | EN diff | 95% CI |
|---|---|---|---|---|
| turbo-nocond | -1.51 | [-4.62, +0.67] | **-4.56** | [-7.27, +1.74] |
| large-v3-nocond | +0.91 | [-2.46, +4.45] | -2.97 | [-7.34, +7.20] |
| turbo (defaults) | **+8.52** | [+1.12, +13.24] | **+4.14** | [+3.09, +6.58] |
| large-v2 | **+8.58** | [+3.86, +13.13] | **+1.50** | [+1.12, +1.67] |
| large-v3 | **+23.47** | [+11.16, +38.61] | **+2.82** | [+2.30, +4.05] |
| medium | **+12.49** | [+7.47, +21.91] | **-3.02** | [-4.25, -0.17] |
| small | **+12.16** | [+4.90, +17.04] | -0.74 | [-2.54, +3.43] |
| base | **+13.52** | [+11.32, +15.02] | **+5.61** | [+3.14, +11.36] |
| tiny | **+42.83** | [+24.83, +67.27] | **+13.92** | [+8.91, +25.58] |
| turbo, autodetect | **+33.70** | [+19.65, +42.36] | -3.82 | [-5.85, +0.90] |

**At library defaults, Voxtral beats every Whisper size on Japanese.** That is the honest
default-versus-default result, and it is mostly a statement about Whisper's defaults on
long-form audio (see the no-condition section below).

**On English, Voxtral is behind**, and this is the most robust accuracy finding here:
medium by 3.02 points with a CI excluding zero at 0.69x the speed, and turbo-nocond by
4.56 points in every one of 6 repeat runs. Note the English side is n=2 here, n=3 in the
final corpus, so treat it as a direction rather than a magnitude.

## Experiment: the final comparison, 20 files

Measured 2026-08-06 on an idle `Apple M2 Ultra 128GB (Mac14,14)`, mlx 0.32.0, macOS
26.4.1. 20 recordings, 7.95h, 17 Japanese and 3 English. Voxtral at `--chunk-seconds 30
--max-batch 32 --kv-bits 8 --delay-ms 2400`; Whisper at `large-v3-turbo` with
`condition_on_previous_text=False` and the language taken per file from its reference.
Arms were run one at a time, never concurrently.

**Re-measured 2026-08-19** after fixing a reference-loading defect that fused a word at
every line break on the word-level path. The English figures below supersede the earlier
ones; Japanese is unchanged, because Japanese has no word spaces and could not be affected.

| | Voxtral (deterministic) | turbo-nocond, 3 runs | verdict |
|---|---|---|---|
| JP coverage CER, 17 files | 16.22% | 14.29 / 14.37 / 14.79 -> **14.49% ±0.27** | CI [13.82, 15.15] entirely below -> **whisper** |
| EN coverage WER, 3 files | 21.50% | 17.85 / 18.04 / 19.13 -> **18.34% ±0.69** | CI [16.62, 20.07] entirely below -> **whisper** |
| x realtime | **29.6x** | 18.0 / 21.6 / 22.0x | **voxtral, ~1.4x** |

Qwen3-ASR was added as a fourth engine on 2026-08-19 and is measured on the same 20 files
with the same scorers: 19.33% JP / 25.45% EN at 21.8x for the 1.7B, 23.27% / 24.26% at
32.8x for the 0.6B. Both are behind the two rows above on Japanese, so neither changes the
verdict here; the 0.6B is the fastest engine this project has measured. It gets one run
each, being greedy, and it writes no subtitles. See [qwen3-asr.md](qwen3-asr.md).

### The generalization test, finally run

The interval above answers "does this hold on a rerun", which is what
`repeat_distribution.py` is for. It contains no between-file uncertainty, so on its own it
cannot say whether the result would hold on different audio. That second question needs a
bootstrap over *files*, and it had never been run at this sample size. It resolves:

| test | result | verdict |
|---|---|---|
| 3 repeat runs vs the fixed baseline | 3/3 beat it | whisper |
| run-distribution t-interval | [13.82, 15.15], below 16.22 | whisper |
| **bootstrap over 17 files** | **+1.85, CI [+0.58, +3.33]** | **whisper** |
| sign test over files | 13 of 17 won, p=0.049 | whisper |

All four agree, so the engine conclusion is stronger than it was when it rested on the
rerun interval alone. Note the file bootstrap is the strictest of these and its interval is
much wider, which is the honest measure of how much this generalizes: the lower bound is
+0.58 points, not +1.7.

Whisper wins accuracy on both units, in all 3 runs, with the whole interval below
Voxtral's fixed value. Voxtral is 1.35-1.65x faster depending on the Whisper run, since
Whisper's sampling ladder makes its own throughput vary by more than 20%. Per file the split
is structural rather than noisy: Whisper wins 13 of 20 files in all 3 runs and loses 5 of 20
in all 3, with 2 files changing sign between runs. So the aggregate reflects a consistent
per-file ordering rather than one file carrying the result, and those 5 losses are the reason
the win is under 2 points rather than a rout.

The method matters for the Whisper column. Its temperature-fallback ladder samples, so a
single run is a draw rather than a score; the interval above is a one-sample t-interval on
the Whisper mean with Voxtral as a constant, since only one side has sampling error
(`scripts/benchmarks/repeat_distribution.py`). Voxtral gets no error bar because greedy
decoding reproduces byte-identically on one machine, which is also why its column is one
run rather than three.

The ordering held at every corpus size and only became resolvable as n grew:

| corpus | JP files | voxtral | turbo-nocond | verdict |
|---|---|---|---|---|
| original | 5 | 16.44% | 15.91% ±0.94 (n=6) | not resolvable |
| grown | 12 | 16.08% | 14.07% ±0.17 (n=3) | whisper better |
| final | 17 | 16.22% | 14.49% ±0.27 (n=3) | whisper better, and resolved over files |

Voxtral's own number barely moved (16.44 -> 16.08 -> 16.22) while the corpus more than
tripled, which says the original 5 files were not unrepresentative. What changed is the
*resolution*, not the estimate. That is the argument for more audio rather than more
reruns.

**Provenance.** Both columns are re-measured and both result files record the machine and
its state. This closes the gap the previous version of this section described: the numbers
had come from a single session with no surviving result file, and a re-run attempted
earlier on 2026-08-06 was voided because the host was doing unrelated GPU work throughout.

Two things the re-measurement establishes beyond the comparison itself. Voxtral's Japanese
accuracy reproduced the earlier session **exactly** (16.22%), across three separate sessions
and two reference-loader versions, which is evidence that the voided session's accuracy work
was sound even though its timings were not. Whisper's accuracy did not reproduce exactly,
and is not expected to: its Japanese mean has read 14.93%, 14.74% and 14.49% across
sessions, all inside the run-to-run spread, which is the behaviour that requires 3 runs in
the first place.

The English figures moved for a different and non-statistical reason: a reference-loading
defect fixed on 2026-08-19. Reference lines were joined with no separator, so on the
word-level path the last word of each line fused to the first of the next. One
subtitle-shaped reference carried 131 such fusions and scored 20.09% where the correct
figure is 3.29%, moving the Voxtral English aggregate from 25.24% to 21.50% and Whisper's
from 22.81% to 18.34%. Japanese could not be affected, having no word spaces. Every English
number published before that date is superseded; see [metrics.md](metrics.md).

Throughput changed more than accuracy did. Voxtral now measures 29.6-29.8x against 22.8x
before, and Whisper 18.0-23.3x against 15.0-20.1x. The previous figures came from the
session with no recorded machine state, so the most likely explanation is that they were
taken while the host was busy. That is the argument for recording state automatically
rather than noting it by hand, which the runners now do.

One of the three Whisper runs began with 25.9GB of GPU memory already allocated by an
unrelated resident process and was flagged `busy` by the harness. It is kept, because the
flag turned out to be conservative here: that run measured 21.3x against 21.5x for an idle
run, while the fastest run was also idle at 23.3x. Parked memory on a 115GB working set
cost nothing measurable, and the spread is Whisper's own sampling. The flag stays
deliberately cautious, since the cost of investigating a false warning is far below the
cost of publishing a contaminated figure.

## `condition_on_previous_text=False` is worth more than model size

The single largest Whisper lever here:

| model | defaults | no-condition | change |
|---|---|---|---|
| large-v3 | 39.91% | 17.36% | **-22.6 points** |
| large-v3-turbo | 24.97% | 14.93% | **-10.0 points** |
| kotoba-whisper v2.0 | 91.47% | 53.20% | -38.3 points |

The mechanism is the known Whisper failure mode: conditioning each 30s window on the
previous window's text lets a repetition loop, once started, feed itself across windows.
Per-file on large-v3, the 26-minute file goes 68.8% -> 14.5% and the 52-minute file 58.9%
-> 24.1%, while the short 13-minute file barely moves (15.9% -> 12.4%). **Long files are
where the loop has room to compound.**

This also inverts the size ranking: at defaults large-v3 (39.91%) is *worse* than small
(28.61%) on Japanese, purely from loop instability, and turbo beats the full large-v3 it
was distilled from. Anyone quoting "large-v3 is the best Whisper" on long-form Japanese
should check this flag first. The registry sets it for every `whisper-*` alias.

Voxtral has no equivalent knob and no equivalent failure, because it decodes independent
chunks.

Pure greedy decoding is not a fix for the resulting nondeterminism: `--greedy` collapses
to 84.92% / 93.00%, because the fallback ladder is what rescues looping segments.

## Whisper needs to be told the language, and that is not free

Voxtral takes no language token. Whisper does, and all three ways of supplying it on a
mixed corpus cost something:

| approach | JP coverage CER | note |
|---|---|---|
| per-file, from the reference | 24.97% | what the tables above use |
| Whisper's own 30s autodetect | 50.14% | +25.2 points |
| force `ja` for every file | unusable | English files score ~100% WER |

Autodetect returned **Russian** for two Japanese files (and on tiny, 102-106% CER with
`extra_ratio` 8.5). That is not a random misfire: these recordings genuinely contain
Russian speech at the start, so a 30-second window is a bad sample of a 90-minute file.
Since the harness knows each file's language from its reference, the main tables give
Whisper that information for free; the autodetect row is what a zero-config user gets.
`run_whisper.py` refuses `--language` on a mixed-script set rather than silently producing
the 100% rows.

## kotoba-whisper: the long-form driver accounted for 68 points

Our first measurement of this Japanese-finetuned distil model was 91.47% coverage CER.
That figure reflects our harness, not the model: we ran it under `mlx-whisper`, whose
long-form algorithm is a poor match for a distil checkpoint. Running it the way its
authors document recovers most of the difference. Recorded here because the same mistake
is easy to make with any model whose published recipe differs from your preferred
runtime:

| what was run | corpus coverage CER | narration | extra_ratio | engine |
|---|---|---|---|---|
| third-party npz port | 91.47% | 88.70% | 0.13 | mlx-whisper (sequential) |
| same, `--no-condition` | 53.20% | - | 1.03 | mlx-whisper |
| official weights, own MLX conversion | 94.23% | 79.88% | 0.10 | mlx-whisper |
| same, `--no-condition` | 53.53% | 44.78% | 1.03 | mlx-whisper |
| official pipeline, `chunk_length_s=30` | 49.57% | 39.52% | 1.29 | transformers (chunked) |
| official pipeline, `chunk_length_s=20` | 27.82% | **16.55%** | 1.35 | transformers |
| official pipeline, `chunk_length_s=15` | 30.40% | 20.88% | 1.37 | transformers |
| official pipeline, `chunk_length_s=10` | **26.16%** | 23.71% | 1.39 | transformers |

**The cause is the long-form algorithm, not the weights.** `mlx-whisper` implements
Whisper's *sequential* 30s-window algorithm, which leans on the decoder to carry state
across windows. Distil models keep 2 decoder layers instead of 32 and cannot do that. The
model card uses transformers' *chunked* pipeline, which decodes independent windows.

Evidence it is under-transcribing rather than terminating early: on a 600s slice the MLX
sequential run reached the final second of audio but emitted 2053 characters where turbo
emitted 2630, with 44 zero-duration and 12 empty segments. The official chunked pipeline
emitted 2527 on the same slice.

Same weights, three runners, **68 points of spread**, none of it attributable to the
checkpoint. The chunk-length sensitivity is the signature of the same cause, and the
optimum is material-dependent: 10s is best on spontaneous speech, 20s on clean narration,
so the model card's 15s is a sensible general default. Worth up to 23 points, so sweep it
on your own audio.

### Chunked long-form is a property of the driver, not the framework

That conclusion was half-applied at first: kotoba was left running on torch/MPS with its
throughput marked "not comparable". But slicing audio, decoding each window independently
and offsetting timestamps requires nothing from torch. `mlx_asr/chunked.py` is that driver
on MLX. Same weights, 17 Japanese files:

| driver | runtime | coverage CER | x realtime | comparable? |
|---|---|---|---|---|
| chunked, 10s windows | **MLX** | 27.01% | **36.2x** | yes |
| chunked, 20s windows | MLX | 31.33% | 68.8x | yes |
| chunked, 30s windows | MLX | 49.71% | 72.7x | yes |
| chunked, 10s (authors' pipeline) | torch/MPS | 26.16% | 25.4x | no |
| sequential 30s (mlx-whisper) | MLX | 94.23% | 64.8x | no |

**MLX-chunked matches the torch reference to within a point at 1.4x the throughput**, so
the asterisk is gone and so is the torch dependency. The chunk-length curve reproduces
exactly as it did on torch, confirming the mechanism. Note throughput moves the opposite
way to accuracy, since fewer windows means less per-window overhead.

Generalising, because this is the reusable part: **any distil-derived Whisper checkpoint
(2-4 decoder layers) is a poor match for a sequential driver** and should be routed to a
chunked one. `models.infer_backend` sends both `kotoba` and `distil` repo ids to
`mlx-chunked` for that reason.

Run as its authors intend, kotoba scores 26-27% on this material, behind both Voxtral and
turbo-nocond. Worth noting what that comparison is and is not: it is a 2-layer distil model
being measured against full-size decoders on spontaneous multi-speaker audio with editorial
references, which is not the setting its published numbers describe. Its appeal is
throughput, and there it is competitive (36.2x through our MLX chunked driver).

### v2.0 and v2.2 are the same ASR model

Measured, not assumed from version numbers. Loading both checkpoints' `model.safetensors`
and differencing every tensor: 539 tensors each, identical keys, **max absolute difference
exactly 0.0**. The files differ only in container metadata and stored dtype.

v2.2's own model card agrees, describing itself as v2.0 plus speaker diarization
(`diarizers`) and punctuation (`punctuators`), both separate post-processing models loaded
by its custom pipeline, and `punctuators` needs torch. So v2.2's additions are not weights.
The registry uses v2.0 and converts it to MLX on first use, which gives up nothing
reachable and avoids a torch dependency.

An earlier version of this project's notes claimed v2.2 superseded v2.0 on weights. That
was inferred from the version numbers rather than checked, and it was wrong.

## Competing Apple Silicon runners

Run because the alternative was an unfalsifiable speed claim. Same 935s clip, complete
reference so plain CER is valid, `--language ja`, model load and warm-up excluded,
serialized so nothing contends for the GPU.

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

whisper.cpp used the Metal backend, confirmed by its own
`whisper_backend_init_gpu: using MTL0 backend` log line, with flash attention on by
default in 1.9.1 and no CoreML encoder built.

**No "fastest on Apple Silicon" claim survives this.** whisper.cpp (41.7x, CER 0.0832) and
mlx-whisper (44.3x, CER 0.0908) are effectively tied: 6% apart on throughput, 0.8 CER
points apart in the other direction, on one clip where the documented noise band is about
1.3 points. Anyone claiming one beats the other on this evidence is reading noise.
whisper.cpp also has far fewer dependencies. What this project contributes is the Voxtral
engine, one interface over three engines, and the measurement harness, not a faster
Whisper.

**faster-whisper cannot use the GPU here at all**, which is the most load-bearing finding
for anyone choosing a runner, because it is the most-cited "fast whisper" and its
reputation is CUDA-shaped. CTranslate2 4.8.1 has no Metal backend, verified rather than
assumed: `get_cuda_device_count()` returns 0, `get_supported_compute_types("cuda")` raises
`ValueError: This CTranslate2 package was not compiled with CUDA support`, and the CPU
compute types are `{float32, int8, int8_float32}` with **no float16**. During the run it
sat at 460-520% CPU with the 60-core GPU idle. Its accuracy is fine (0.0804 is the best
CER in the table), so the problem is purely throughput and it is structural.

**Quantization costs speed on the GPU**, not just accuracy: q5_0 is 27% slower than fp16
at identical CER, because dequantization is work an fp16 matmul does not do. So quantize
when memory-bound, not for speed. Thread count is irrelevant (`-t 4` and `-t 8` gave 41.0x
and 41.7x with byte-identical output), so the workload is Metal-bound.

Also note large-v3 q5_0 at 0.1641 is not a quantization artifact but a repetition loop,
with one segment duplicated five times. turbo, with 4 decoder layers instead of 32, avoided
it, which matches the pattern above: the deeper decoder is the less stable one on long
audio.

Not measured: whisperX (its ASR stage is faster-whisper, so its throughput is bounded above
by the CPU-only rows), and whisper.cpp with a CoreML encoder (the Homebrew bottle is built
`COREML = 0`).

## Clean narration: the gap nearly closes

The same models on the single narration clip, where the reference is complete:

| engine | coverage CER | plain CER | x realtime |
|---|---|---|---|
| **voxtral (M2 Ultra 128GB, 60s b16 kv8)** | - | **7.28%** | 21.2x |
| whisper large-v3-turbo | 8.28% | 9.08% | 44.3x |
| whisper large-v3, no-condition | 8.87% | 8.87% | 24.2x |
| whisper large-v3-turbo, no-condition | 10.42% | 10.42% | 73.5x |
| whisper large-v3 | 12.91% | 13.67% | 16.6x |
| whisper small | 13.96% | 14.89% | 59.2x |
| whisper medium | 15.17% | 15.17% | 29.9x |
| whisper large-v2 | 15.20% | 16.96% | 23.0x |
| kotoba-whisper v2.2, chunk 20s | 16.55% | 16.55% | n/c |
| whisper base | 22.73% | 25.97% | 90.5x |
| whisper tiny | 34.51% | 36.27% | 101.8x |

Voxtral's 7.28% leads turbo's 9.08%, a much narrower margin than the corpus suggests, and
turbo is faster on this clip. Two caveats: this is n=1, and **`--no-condition` *hurts* on
clean audio** (turbo 8.28% -> 10.42%) exactly where it helped enormously on long
spontaneous audio. The flag is a long-form-instability fix, not a general improvement.

## What ships, and why the default is not the most accurate engine

`voxtral` is the default despite Whisper being ~1.3 points more accurate, because it is
faster (~1.4-2x), needs no language hint, needs no stability flag, has better-behaved
timestamps ([timestamps.md](timestamps.md)), and reproduces on a given machine
([determinism.md](determinism.md)). Whisper is one flag away for anyone who wants the
accuracy.

The two `qwen3-asr` aliases ship without displacing anything. The 1.7B is last of the four
on accuracy; the 0.6B is the fastest here (32.8x in 2.36GB), which is the case for keeping
it. Both refuse `-f srt`.

## What this does and does not settle

Settled: turbo-nocond is the accuracy winner; Voxtral is the speed winner at comparable
accuracy; `condition_on_previous_text=False` is mandatory before quoting any long-form
Whisper number; running a model through the wrong long-form algorithm can cost more than
the model choice.

Not settled: the English side is n=3. The competing-runner table is n=1. Whisper's own
chunking/VAD front-ends (faster-whisper, whisperX) are untested and would likely help
turbo further.
