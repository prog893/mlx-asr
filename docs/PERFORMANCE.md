# Where the time goes, and why batch size is not monotonic

Everything below is measured on this repo's benchmark scripts. Reproduce with
`mlx-asr-bench` (decode only, no audio needed) or
`sh benchmarks/run_matrix.sh` (end-to-end with CER).

**Scope of the accuracy numbers.** All CER figures come from a single 935s
Japanese narration clip, so they are n=1 on the audio axis. Decoding is greedy
and deterministic, so repeating a run is byte-identical and adds no information;
what a single clip cannot tell you is how a number would move on different
material. Speed numbers are far more reliable, being wall-clock measurements of a
deterministic workload.

For accuracy, compare two configs with a **paired** test over regions of the same
clip (`benchmarks/compare_configs.py`) rather than by eyeballing two overall CERs;
shared difficulty then cancels instead of swamping both intervals. Doing that to
this project's own claims, only three clear significance on this clip:

| claim | diff | 95% CI | verdict |
|---|---|---|---|
| overlap 8s beats 0s at 30s chunks (M4) | 1.80 | [+0.62, +3.20] | significant |
| energy boundaries beat VAD at 60s | 3.00 | [+0.74, +5.93] | significant |
| 60s chunks beat 30s | 1.85 | [+0.71, +3.24] | significant |
| overlap hurts at 60s chunks | 0.69 | [-1.47, +0.07] | **not supported** |
| 60s beats 90s on accuracy | 0.62 | [-0.36, +1.69] | **not supported** |
| 4bit vs fp16 | 0.07 | [-0.33, +0.48] | tie (as claimed) |
| kv8 vs unquantized KV | 0.02 | [+0.00, +0.07] | tie (as claimed) |

The two "not supported" rows are corrections to earlier wording in these docs.
60s is still the right default: it is the fastest option and never measurably
worse, which is a weaker but honest justification than "longer chunks are worse".

## The one architectural fact that shapes everything

Voxtral Realtime is a streaming model. It consumes 80ms of audio per decoder
position and emits exactly one token per position. Two consequences:

- **Decode step count is set by audio duration, not by how much speech there
  is.** 935s of audio is ~11,700 steps whether it is dense narration or mostly
  silence. Faster hardware does not reduce the step count.
- **A single stream cannot exceed 1x realtime by much**, no matter the machine,
  because one step costs at least one full pass over the model weights. On an
  M4 that is 21ms against an 80ms budget, so ~3.8x is the ceiling for B=1; the
  only way past it is to decode several audio streams in the same step.

So this is a throughput problem: split the audio, decode all pieces in
lockstep, and the per-step weight read is amortized across rows.

## Batch size is not monotonic, and the bad region is where you would start

Decode reads every weight once per step, so steps/s "should" be flat in batch
size until memory runs out. It is not. Measured decode throughput, 4-bit model:

| batch | M4 (16GB) ms/step | M2 Ultra (128GB) ms/step |
|---|---|---|
| 1 | 22.4 | 10.8 |
| 2 | 26.2 | 12.8 |
| 4 | 51.9 | 16.3 |
| 8 | **110.8** | 24.7 |
| 12 | 88.7 | 31.2 |
| 16 | 90.3 | 33.6 |
| 24 | 97.3 | 34.4 |
| 32 | 103.6 | 33.9 |
| 48 | 191.4 | 53.2 |
| 64 | n/a (swaps) | 53.0 |
| 128 | n/a | 99.0 |

Read the M4 column carefully: **batch 8 costs 5x more per step than batch 1**,
and batch 12 is *cheaper* than batch 8. B=1 at 22.4ms is exactly
2.5GB / 120GB/s, i.e. the bandwidth floor, so the model is being read once and
nothing is wrong at B=1. The penalty from B=2 upward is a kernel-path effect,
not bandwidth: per-step cost grows roughly linearly with B in the 2..8 range,
as though the batched quantized matmul falls back to a per-row path, then
recovers once B is large enough to select a proper tiled kernel.

The M2 Ultra shows the same shape, much shallower, plus a second cliff at
B=48 and again at B=96.

Because throughput per row still rises even where ms/step rises, the useful
metric is x-realtime = steps/s x B x 0.08:

| batch | M4 x realtime | M2 Ultra x realtime |
|---|---|---|
| 1 | 3.6 | 7.4 |
| 8 | 5.8 | 26.0 |
| 16 | 14.2 | 38.1 |
| 32 | **24.7** | 75.6 |
| 48 | 20.1 | 72.2 |
| 128 | n/a | **103.5** |

**Practical rule: never use batch 2-8.** Either stay at 1 or jump to 12+. The
project's earlier default of batch 8 sat at the bottom of the valley.

## What actually limits each machine

Decode is not the whole story. Splitting wall clock on the 935s clip:

| stage | M4, 60s chunks | M2 Ultra, 60s chunks |
|---|---|---|
| encode (16 chunks) | 47.3s | 10.8s |
| decode (816 steps) | 78.9s | 31.8s |
| total | 130.3s (7.2x) | 43.9s (21.3x) |

The encoder is 36% of M4 wall clock and grows as chunks get shorter (more
chunks, each with its own conv stem and 32-layer causal pass). That caps the
short-chunk strategy: on the M4, going 60s -> 30s cut decode from 78.9s to
48.5s but pushed encode from 47.3s to 51.8s, so the net gain was only 7.2x ->
8.3x.

Mel and conv stem are noise (0.01s and 0.02s per chunk).

### Batching the encoder does not help (measured)

mlx-audio's encoder attention is batch-1 only, so the obvious next move is to
give it a batch axis the way the decoder has one.
`benchmarks/probes/probe_encoder_batch.py` does exactly that and it is **0.84x
to 0.91x, i.e. slightly slower**:

| batch | s/chunk |
|---|---|
| per-chunk (stock) | 1.497 |
| 1 | 1.651 |
| 4 | 1.738 |
| 8 | 1.773 |

Batching amortizes *weight reads*, so it only pays when a stage is
bandwidth-bound. Arithmetic intensity says the encoder is not:

| stage | FLOP per chunk/step | bytes read | FLOP/byte | bound by |
|---|---|---|---|---|
| encoder, one 30s chunk | ~3270 GFLOP | 0.66GB | ~4950 | compute |
| decoder, one step | ~5 GFLOP | 2.5GB | ~2 | bandwidth |

At ~3270 GFLOP against an M4 GPU peak near 4 TFLOP/s, one chunk needs ~817ms of
pure math and measures 1.5s, so the encoder is already running at roughly half
of theoretical peak. Its share of wall clock is a hard floor on this hardware,
not an optimization opportunity. Shorter chunks make it *worse*, which is the
real reason the M4 tops out around 8.5x.

## KV cache growth

steps/s decays as the cache fills, ~25% over 800 steps at B=16 on the M4:

| kv_len | ms/step | peak GB |
|---|---|---|
| 138 | 78.4 | 4.28 |
| 438 | 89.0 | 4.94 |
| 838 | 112.6 | 6.41 |

At B=16 and 838 positions the cache is 1.43GB *read per step*, comparable to
the 2.5GB of weights, so it stops being free. Shorter chunks bound kv_len,
which is a second reason 30s chunks beat 60s on the M4 despite more encode
work.

**`--kv-bits 8` is close to a free win and worth using.** It halves the cache
reads, and on both machines it was faster *and* no less accurate:

| config | CER without | CER with kv8 | decode s without | decode s with |
|---|---|---|---|---|
| M4, 60s/B16 | 7.49% | **7.44%** | 78.9 | 72.8 |
| M2 Ultra, 60s/B16 | 7.23% | 7.25% | 31.8 | 29.9 |
| M2 Ultra, 30s/B32 | 9.13% | 9.11% | 17.2 | 15.6 |

(In mlx 0.32 this needs mlx-lm's `quantized_scaled_dot_product_attention`;
`QuantizedKVCache.update_and_fetch` returns triples that the dense
`mx.fast.scaled_dot_product_attention` rejects.)

## Why the batch valley exists (upstream confirmation)

The valley is not a mystery and not fixable from here. MLX dispatches quantized
matmuls to one of several Metal kernels based on the leading dimension, and the
maintainers state the cause directly:

> "The drop from 4 to 8 is that we switch from batched qmv to the qmm."
> ml-explore/mlx discussion #1593

The threshold is `vector_limit`, computed by `get_qmv_batch_limit`, and it is
compile-time C++ and hardware-specific (10 on an M4 Pro for K,N>4096; 14 on an
M4 Max at size 4352; different again on an M2 Ultra). A related open issue
(mlx#3553) documents the same discontinuity at M=3 and reports that manually
lowering `vector_limit` made things substantially worse, with no fix landed.

Attempts to route around it from Python both failed:

| B | plain ms/step | fold ([B,1,d]->[1,B,d]) | split into 2 | split into 4 |
|---|---|---|---|---|
| 8 | 101.6 | 102.1 | 102.8 | 106.3 |
| 16 | 80.6 | 80.3 | 226.8 | 216.5 |
| 32 | 103.2 | 99.8 | 176.0 | 436.6 |

`fold` is numerically bit-exact (max abs diff 0.000000) and worth 3-7% at best;
splitting is far worse because each sub-batch pays full weight reads. The only
effective mitigation is the one the profiles already apply: **choose batch >= 12**.

`mx.compile` also does not help (22.6 -> 23.2 ms/step at B=1, 83.4 -> 89.8 at
B=16), which is the expected result for a bandwidth-bound loop rather than a
launch-bound one.

For calibration, antirez/voxtral.c, a hand-written C + Metal implementation with
custom attention/RoPE/KV kernels, reports 23.5-31.6 ms/step at batch 1 on an
M3 Max. This implementation does 22.4 ms/step at batch 1 on an M4, so the
single-stream path is already at parity with hand-tuned native code; the win here
comes from batching, not from kernel work.

## Things that did not work

- **Lower sampling rate (8kHz, upsampled back to 16kHz).** No speed change at
  all, because the mel front end always wants 16kHz and the model's step count
  depends on duration, not sample rate. Costs 0.2 CER points from the lost
  4-8kHz band. There is no speed lever in sample rate for this architecture.
- **Time-stretching audio 1.25x** (ffmpeg atempo, pitch preserved). Does cut
  steps proportionally (935s -> 748s) but costs 1.9 CER points. A bad trade
  versus shortening chunks, which is free.
- **fp16 weights on 16GB.** 8.3GB of weights plus KV swap-thrashes; the run
  never finished. Not viable below ~32GB.
- **Batching the encoder.** See above: compute-bound, so 0.84-0.91x.

## Silence compaction: works on one model, breaks the other

`--compact-silence` drops the middle of pauses longer than 400ms, keeping the
first 240ms. Since decode cost is one step per 80ms frame, removing silence
removes steps one-for-one; on the reference clip it removed 12% of the audio.
Timestamps are mapped back to the original timeline, and the resulting chunk
cuts were measurably *cleaner* (no cut louder than -50dB, versus 3 cuts above
-45dB without it).

The accuracy result nevertheless splits by model:

| config | CER baseline | CER compacted | deletions |
|---|---|---|---|
| M4, nvfp4, 60s/B16 | 7.49% | 11.63% | 105 -> 222 |
| M4, nvfp4, 30s/B32 | 9.06% | 13.39% | 115 -> 338 |
| M2 Ultra, 4bit affine, 60s/B16 | 7.23% | 8.23% (at 30s: 9.13% -> 8.23%) | 103 -> 118 |
| M2 Ultra, 4bit affine, 30s/B32 | 9.13% | 8.59% | 123 -> 109 |

On nvfp4 the deletions triple and the loss is concentrated rather than spread:
one two-minute stretch lost 45% of its text after only 4.8s of silence was
removed there, so this is not proportional information loss. The model appears
to rely on pauses for its own segmentation, and the more aggressively quantized
weights tolerate their removal much worse. Off by default; if you want the
speed, verify CER on your own audio and model first.

## Quantization: including unquantized fp16 as the ceiling

The obvious worry with a 4-bit model is that quantization is quietly costing
accuracy. It is not. Measured on a 128GB M2 Ultra (the 16GB M4 cannot hold fp16
at all), everything fixed except the weights, 60s chunks / batch 16 / 2400ms:

| weights | bytes | CER | kana CER | x realtime | decode steps/s |
|---|---|---|---|---|---|
| fp16 (unquantized) | 8.9GB | 7.61% | 6.08% | 13.6x | 14.5 |
| 8-bit affine | 4.7GB | 7.49% | 5.99% | 21.1x | 26.4 |
| mxfp8 | 4.6GB | 7.66% | 5.97% | 21.2x | 26.5 |
| nvfp4 | 2.5GB | 7.49% | 5.99% | 21.8x | 27.1 |
| 4-bit affine | 2.9GB | 7.23% | 5.60% | 22.2x | 26.3 |

The whole spread is 0.43 CER points on 4205 reference characters, and the fp16
and 4-bit hypotheses differ by just 65 characters (1.5%), with fp16's 15 extra
errors spread across substitutions, insertions and deletions rather than
concentrated anywhere. **Treat all five as tied on accuracy.** 4-bit landing
nominally best is a quirk of this clip, not evidence that quantization helps.

What is not noise: fp16 costs 1.63x the wall clock and 15.3GB of peak memory
versus 9.4GB, for no measurable accuracy return. Decode steps/s tracks weight
bytes the way the bandwidth model predicts (14.5 for 8.9GB versus ~26-27 for the
2.5-4.7GB variants), which is just the same bandwidth story as everywhere else
in this document.

Re-running the comparison with `--kv-bits 8` changed nothing (fp16 7.61% at
14.3x, 4-bit 7.25% at 21.8x), so the KV cache is not confounding it.

Re-checked at the tuned config (30s chunks, batch 32, kv8, 8s overlap), in case
fp16 only pulls ahead once everything else is optimal: fp16 7.32% at 17.8x versus
4-bit 7.25% at 26.7x. Still a tie on CER, still 1.5x the cost. One nuance worth
recording: fp16 wins on *kana* CER there (5.51% vs 5.93%), so its errors may be
slightly more phonetically faithful even where its character score is not better.
That gap is also inside the noise band.

Practical guidance: use 4-bit. There is no accuracy argument for anything larger,
and on a 16GB machine fp16 does not merely run slower, it swap-thrashes and never
finishes.

## How much of the CER is just spelling?

Japanese has no fixed orthography, so a reference typist's choice of kanji vs
kana is not a model error. Three metrics, increasingly lenient:

| metric | value on the best run | what it forgives |
|---|---|---|
| CER | 7.42% | nothing |
| lenient CER | 7.11% | script changes of the *same* word (a kanji word spelled in kana, or katakana vs hiragana) |
| kana CER | 5.72% | anything sharing a reading |

Use lenient CER (`evals/eval_cer_lenient.py`) rather than kana CER as the "fair" number.
Kana CER over-forgives: collapsing everything to readings also excuses genuine
homophone errors, where a different word sharing the same reading is substituted.
A reader notices that, but kana CER scores it as free. Lenient CER requires that one side be a pure-kana respelling of the other,
so it charges homophone swaps as errors.

This corrects an earlier claim that ~1.6 CER points were orthographic; that came
from the kana figure. The honest number is ~4% of edits, i.e. **~0.3 CER points**,
so the vast majority of the remaining error is real misrecognition.

## Chunk seams, and the prefix-overlap fix

Every chunk boundary costs accuracy, and the cost is asymmetric. Locating each
edit operation relative to the nearest boundary (30s chunks) shows errors
concentrate at chunk *starts*:

| region | share of edits | share of audio | enrichment |
|---|---|---|---|
| first 3s of a chunk | 22.3% | 9.9% | **2.24x** |
| last 3s of a chunk | 12.9% | 9.9% | 1.29x |
| elsewhere | 64.8% | 80.1% | 0.81x |

That is what you would expect from a causal model: at position 0 of a chunk it
has no left context. It also tells you which direction of overlap can possibly
help. Appending audio after a chunk cannot change tokens already emitted;
prepending audio can. `--overlap-seconds N` prepends N seconds of the preceding
audio and discards the tokens produced from it, so the model warms up before the
region we keep.

| overlap | CER (M2 Ultra, 30s/B32) | CER (M4, 30s/B32) | extra decode |
|---|---|---|---|
| 0s | 8.73% | 9.23% | +0% |
| 4s | 7.30% | 7.97% | +13% |
| 8s | 7.25% | 7.42% | +27% |
| 15s | 11.20% | - | +50% |

Worth 1.4-1.8 CER points at 30s chunks, which is most of what the extra seams
cost in the first place, and it survives a paired test (+1.80, CI [+0.62, +3.20]).
Two caveats:

- **The curve is noisy, not smooth.** Between 4s and 12s it wanders in the
  7.25-7.80% band with no clean optimum, a spread smaller than the measurement
  noise. "≥4s helps by about 1.5 points" is defensible; "8s is optimal" is not.
- **It stops paying where seams are sparse.** At 60s chunks the point estimates
  reverse (7.37% -> 7.59% at 4s -> 8.06% at 8s), though paired testing puts that
  at -0.69 with CI [-1.47, +0.07], so the honest claim is "no benefit", not
  "harmful". Either way there is nothing to gain, so the CLI enables overlap with
  `--fast` (which halves the chunk length) and leaves it off otherwise.

At 15s overlap on 30s chunks it collapses entirely (11.20%): the warm-up region
is then half the chunk, and rows start hitting EOS inside it.

## Chunk length: 60s is the sweet spot

| chunk | batch | CER | x realtime |
|---|---|---|---|
| 20s | 48 | 12.46% | 25.6x |
| 30s | 32 | 9.13% | 31.0x |
| 60s | 16 | **7.37%** | 21.2x |
| 90s | 16 | 7.99% | 17.2x |
| 120s | 8 | 7.59% | 16.1x |
| 180s | 8 | 7.56% | 11.6x |

(M2 Ultra, no overlap.) Accuracy improves up to 60s, then flattens in the
7.5-8.0% band while speed falls off, because chunks beyond ~60s exceed the
encoder's 750-frame sliding window and the batch has to shrink to fit memory.
The 60s-over-30s gain is significant (+1.85, CI [+0.71, +3.24]); the 60s-over-90s
gain is not (+0.62, CI [-0.36, +1.69]). So nothing above 60s is worth the wall
clock, but that is a speed argument, not a proven accuracy one.

## How the default batch size is chosen

Two tiers, in `mlx_asr/hardware.py`.

A machine listed in `mlx_asr/profiles.json`, matched on chip and RAM, uses its
measured numbers outright. That is the point of the file: what to pick is not
predictable from specs, since decode throughput is not monotonic in batch size (see
above), so a formula fitted to one machine mispredicts the next one.

For anything unbenchmarked the batch is derived as the smaller of two caps, then
snapped onto the sizes that measured well (1, 12, 16, 24, 32, 64, 128):

- a **memory cap**: half the GPU working set, minus model weights and ~0.6GB fixed
  overhead, divided by ~0.002 GB per row-second of chunk audio, which is the
  asymptote of both batch sweeps.
- a **compute cap**: 3 rows per GPU core, since both sweeps reach 90% of peak
  throughput at 1.1-3.2 rows per core.

Anything landing in the 2-8 valley falls back to 1, because the real choice there is
"12 or more" versus "1", not a point on a smooth curve.

## VAD boundaries: cleaner cuts, worse output

`--vad` uses Silero VAD (ONNX, no torch) to cut in the middle of the longest
non-speech run near each target, rather than at the quietest 50ms window. It
never removes audio. VAD inference is negligible (2.2s for 935s, 426x realtime).

It loses in every pairing: 30s chunks 8.73% -> 10.75%, 60s chunks 7.37% ->
10.25%, and with 8s overlap 7.25% -> 8.04%. Unlike most differences here this one
is significant (paired at 60s: 3.00 points, CI [+0.74, +5.93], 21 of 40 regions
to 7). It is also the opposite of what the VAD literature predicts, and the cuts
genuinely are cleaner by the obvious metric (speech probability in the 1s after a
cut: 0.316 for VAD vs 0.485 for energy).

The likely reason ties back to overlap: the energy splitter picks the quietest
*instant*, which lands mid-pause and leaves the next chunk a run of leading
silence to warm up on. VAD picks the middle of a non-speech *run*, often a short
inter-word gap that satisfies the detector but leaves almost no silence before
speech resumes. Warm-up room, not cut cleanliness, is what this model wants.
Kept as an opt-in flag for noisy material where energy minima may mislead.

## Accuracy knobs that cost nothing

- **`--delay-ms 2400`** (vs the 480ms default): CER 14.74% -> 7.49% at
  identical speed. The model is allowed to see more audio before committing to
  a token. This is the single largest quality win available and it is free.
  Higher delays mean the leading tokens of each chunk cover more audio.
- **Chunk length.** Every chunk seam costs accuracy: the model restarts with no
  context. 60s chunks gave 7.23% on the M2 Ultra; 30s gave 9.13%; 20s gave
  12.46% (mostly deletions, as short rows end early). If wall clock allows,
  prefer long chunks.
- **`--prompt`**, see below.

## Input level: the mel floor is absolute

Worth knowing because it fails silently. The mel front end clamps at a fixed
floor, `log10(power) < global_log_mel_max - 8.0`, which is an *absolute* level and
not one relative to the utterance. Quiet input therefore has low-level detail
flattened before the encoder sees it:

| gain | peak dBFS | % mel bins clamped | Japanese coverageCER |
|---|---|---|---|
| -20dB | -25.5 | 65.5% | 23.76% |
| -12dB | -17.5 | 41.5% | 19.42% |
| unity | -5.5 | 9.9% | **16.44%** |
| +6dB | 0.0 | 2.8% | 17.09% |

Attenuation is significantly harmful (-12dB costs +3.78 points, CI [+1.82, +7.83];
-20dB costs +7.79). Amplifying audio that is already healthy is a wash overall
(+0.09, CI [-1.21, +0.94]), though it split by content: the multi-speaker English
recordings gained 2.6 points while close-mic single-speaker Japanese lost ~0.6.

`--gain` therefore defaults to `auto`, which encodes the asymmetry: boost only
below -6 dBFS peak, and only up to -1 dBFS, never attenuate, never clip. That is a
no-op on all 7 corpus recordings and on the original clip, while recovering
essentially all of the loss on quiet input (a clip attenuated 14 dB scores 10.84%
raw, 8.61% with auto, against 8.63% at its original level).

The older advice was to leave levels alone; there is still no reason to normalize
audio already peaking near full scale, which is exactly what `auto` declines to do. Bit depth is not the
issue: 32-bit float sources make gain mathematically lossless, so only the
position relative to the model's floor matters. Clipping is the one irreversible
risk, so `--gain peak` targets -1 dBFS and the CLI reports clipped samples.

## Metric hygiene: score text, not subtitle layout

Cue segmentation is underdetermined (many groupings of the same tokens are equally
valid), so scoring a concatenated SRT risks measuring the cue heuristics rather
than the transcription. Worse, cue text used to be built by appending raw token
bytes and decoding at each flush; the tokenizer is a byte-level BPE, so a flush
landing mid-character split it into two U+FFFD replacement characters. There were
217 of those across the saved outputs, each an unearned CER error. Fixing it
(break only on complete-character boundaries) improved CER by 0.12 points at
identical config.

With that fixed, text-only CER equals SRT-joined CER to four decimals across all
12 configs on both machines, so cue layout is confirmed not to distort the metric.
`mlx_asr/text.py` decodes each chunk's token stream in one call for the
plain-text path; `benchmarks/run_matrix.sh` writes every format per run so both
views are always available.

## The prompt field is not an instruction field

This is the most expensive single mistake available in the CLI. The decoder reads
the prompt region as text it has already emitted, so an ASR-style imperative makes
it continue in the wrong register:

| prompt style | tokens | CER |
|---|---|---|
| best (topic sentence, or a comma-separated term list) | 17-38 | 7.09% |
| none | 0 | 7.28% |
| a generic instruction such as "Transcribe the audio accurately" | 7 | **13.17%** |

Nearly six points worse than saying nothing. Use domain vocabulary or a short
topic sentence describing the recording.

Across ten sane variants the spread is 0.50 points against a ~1-point noise floor,
so style choice is close to irrelevant: a 17-token topic sentence matches a
38-token term list. Ordering does not matter either, which is the informative part.
Reversing the term list so truncation discards the *important* terms scored
identically to keeping them, meaning the small gain is not really about the
specific vocabulary reaching the model.

Also: `--prompt` and `--overlap-seconds` conflict badly (9.04% and 7.16% alone,
18.64% together, with visible repetition loops). Both hand the decoder
already-emitted text and stacking them over-conditions it. The CLI keeps overlap
and ignores the prompt.

## The prompt window is 31 tokens, and it truncates silently

Bias text is injected into the decoder's 32-token left-pad region, so only the
**last 31 tokens** survive; earlier tokens are dropped with no warning (the CLI
now warns). Put the terms you care about *last*.

Separator choice mostly changes how many keywords fit. Measured on the same
180s clip on both machines, one row per variant:

| variant | tokens | CER, M4 (nvfp4) | CER, M2 Ultra (4bit affine) |
|---|---|---|---|
| no prompt | 0 | 10.05% | **9.03%** |
| Japanese `、`/`。` | 45 | **9.59%** | 9.71% |
| Japanese, reordered | 40 | 9.71% | 10.05% |
| natural sentence | 26 | 9.71% | 9.71% |
| minimal keyword list | 27 | 10.05% | 9.71% |
| Latin `, ` | 44 | 10.05% | 9.71% |
| Latin `,` no space | 40 | 10.16% | 10.05% |
| bare spaces | 36 | 10.05% | 9.82% |

The two machines disagree in sign: every prompt variant *helped* on the M4 and
every one *hurt* on the Ultra, where the no-prompt baseline was already the
best result. Since the models differ (nvfp4 vs 4-bit affine) the two columns
are not a clean A/B of the prompt alone, but the spread within each column
(~0.5 points, on 943 reference characters) is the same size as the disagreement
between them.

The defensible conclusions:

- **Prompt bias is a weak and unreliable lever.** It is worth a few tenths of a
  CER point at best and can be negative. It did not recover any of the proper
  nouns it was aimed at (zero hits for the rarest term in every variant on both
  machines).
- Japanese punctuation is a slightly better separator than Latin commas, and a
  natural sentence performs as well as a keyword list at half the token cost,
  so prefer prose if using this at all.
- If you use `--prompt`, verify on your own audio rather than trusting these
  numbers; the effect is small enough to flip with the model or the clip.
