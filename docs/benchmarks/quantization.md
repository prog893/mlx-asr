# Lever: weight and KV-cache precision

**`--kv-bits` is Voxtral only**, since it applies to our own batched decoder.
**`--quantization` works on any model that publishes more than one build**, currently
`voxtral` and the two `qwen3-asr*` models; the `whisper-*` aliases point at fp16 MLX
conversions with no quantized builds in use here, so it errors there. whisper.cpp
quantization is covered in [engines.md](engines.md), where it costs speed rather than
accuracy.

**Conclusion first.** Quantization does cost accuracy on this workload, monotonically in bit
width, and 4-bit is the worst of the five precisions measured: **1.07 points behind 8-bit
and 1.30 behind fp16** over the 20-file corpus, both significant. 4-bit still ships as the
safe default, because 8-bit is not published in a loadable form and fp16 will not fit a 16GB
machine. **If you have the memory, `--quantization fp16` is the accuracy option, and a
locally converted 8-bit build matches it at 7.3GB and full speed.** `--kv-bits 8` is
separately close to free and is on by default.

That replaces an earlier "quantization costs nothing measurable", which came from a single
prepared-narration clip where the five precisions span 0.43 points and 4-bit ranks nominally
best. Both measurements are sound on their own material and they disagree; the corpus one
governs, because it is the material this project is tuned for. The clip's cost figures
(wall clock, memory, decode steps/s) are unaffected and still stand.

## The policy: one measured default per model, and `--quantization` to override it

Each model ships **one default precision, chosen by measurement where a choice exists**,
and `--quantization` exposes the rest where the converters published more than one build.

The default is picked by this rule, in this order:

1. **Take the cheapest precision whose accuracy cost is worth its price.** On Voxtral that
   cost is now measured: 4-bit gives up ~1.1 points against 8-bit and ~1.3 against fp16. It
   still ships, because fp16 needs 1.65x the time and more memory than a 16GB machine has,
   and no loadable 8-bit build is published.
2. **Where only one build exists, ship that.** Nothing to choose.

| model | default | other options | why that default |
|---|---|---|---|
| `voxtral` | 4-bit | `fp16` | fp16 is 8.9GB of weights, 15.3GB peak and 1.6x the wall clock, for 0.07 CER points. Measured below |
| `whisper-*` | fp16 | none | 0.08-3.1GB; no quantized MLX builds in use here |
| `kotoba` | fp16 | none | 1.6GB, converted on first use |
| `qwen3-asr` | **8-bit** | 4bit, 5bit, 6bit, bf16 | bf16 **tied** it on accuracy (20.16% vs 19.98%) while costing **1.36x the wall clock** (14.1x vs 19.2x) and **1.4x the peak memory** (5.66 vs 4.05GB) |
| `qwen3-asr-small` | **8-bit** | 4bit, 5bit, 6bit, bf16 | same, and the gap is wider: bf16 26.24% vs 23.27%, and 23.0x vs 32.8x, which would remove the only reason this model ships |

So the shorthand is a precision name rather than a repo id:

```bash
mlx-asr jp.wav --model qwen3-asr --language ja -f txt                      # 8-bit
mlx-asr jp.wav --model qwen3-asr --language ja -f txt --quantization 4bit  # 1.61GB
mlx-asr jp.wav --model qwen3-asr --language ja -f txt --quantization none  # bf16
```

`--quantization` is a **lookup from (model, precision) to a published repo id**, not a
runtime conversion, so only precisions that exist are accepted and an unpublished one is
an error naming what does exist rather than a 404 mid-download. It therefore does not
combine with a repo id passed to `--model`, which already names its own precision, and it
errors on models that ship a single build. `--list-models` prints the options per model.

`none` resolves to whichever unquantized build a model publishes, which is not the same
name everywhere: bf16 for Qwen3-ASR, fp16 for Voxtral. Both spellings work on either.

On `voxtral` the flag also moves the **weight footprint used to size the batch**. That is
not bookkeeping: `derive_batch` subtracts it from the GPU budget, so on an unprofiled
16GB-class machine 4-bit derives batch 32 and fp16 derives batch 1. Claiming 4-bit's
2.5GB while loading 8.9GB would plan for memory already spent, and the failure would
surface as an OOM rather than as a bad default.

**Two published Voxtral quants are deliberately absent**, because they crash rather than
run: `mlx-community/Voxtral-Mini-4B-Realtime-6bit` and
`ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx` ship a `config.json` with no `model_type`,
so mlx-audio routes them to the non-realtime loader and dies in `post_load_hook`
(re-verified 2026-08-20). Listing them would turn a usage error into a crash after a
multi-gigabyte download.

**A full ladder is not swept per model, and that policy is now on notice.** The clip sweep
below answered the general question once (is quantization silently costing quality?) with
"no", and every later model inherited that answer instead of being measured. The corpus
result above shows the inheritance was unsound for at least one pair: fp16 versus 4-bit is
a tie on the clip and a significant 1.30 points on the corpus.

The mechanism of the mistake is worth naming, because it is not "the sample was small". It
is that a **power calculation was fed an effect size measured on unrepresentative
material**. The estimate said 8-bit versus 4-bit would need ~1,000 files and fp16 versus
4-bit ~15,000, both derived from clip differences of 0.07-0.26 points. Twenty files
resolved fp16 cleanly, so the input to that calculation was wrong by two orders of
magnitude, and any other number it produced is equally untrustworthy. A power calculation
inherits the validity of its effect size, and an effect size from one recording of prepared
narration does not describe spontaneous multi-speaker audio.

What that does *not* undermine: the cost figures (wall clock, memory, decode steps/s), which
are properties of the weights and the hardware rather than of the audio, and which the
corpus run reproduces. **4bit through 6bit on `qwen3-asr*` remain exposed but unmeasured**,
which the `--help` text says.

The one thing worth checking afresh on a new engine is not accuracy but **decoder
degeneracy**, where the effect size is large rather than fractional. On Qwen3-ASR it made
no difference: bf16 and 8-bit produced near-identical repetition-loop counts (2/0/18/0/31
against 3/0/19/0/31 per file), so its loops are not a quantization artifact. See
[qwen3-asr.md](qwen3-asr.md).

## The full ladder on the corpus, which overturns the clip result

**Precision does cost accuracy on this material, and 4-bit is the worst of the five.** Read
this section first; the clip sweep further down is superseded on accuracy and still valid on
cost.

Five precisions, 20-file corpus, one config (60s chunks, batch 16, kv8, delay 2400ms), idle
M2 Ultra. Every arm through `run_corpus.py`, paired with `compare_engines.py`:

| weights | on disk | JP coverage CER | vs 4-bit | x realtime | peak GPU |
|---|---|---|---|---|---|
| fp16 | 8.9GB | **15.04%** | +1.30, CI [+0.59, +2.26] | 11.2x | 12.98GB |
| **8-bit** | 4.7GB | **15.27%** | +1.07, CI [+0.18, +2.16] | **19.8x** | 7.29GB |
| mxfp8 | 4.6GB | 15.86% | +0.48, CI [-0.43, +1.45] | 19.4x | 7.14GB |
| nvfp4 | 2.5GB | 16.07% | +0.27, CI [-0.47, +1.23] | 19.6x | **5.09GB** |
| 4-bit (ships) | 2.9GB | 16.34% | | 18.5x | 6.77GB |

The ordering is monotonic in bit width, which is what one would naively expect and what the
clip sweep denied. Two comparisons clear significance against 4-bit (fp16 and 8-bit); mxfp8
and nvfp4 land inside the resolution floor.

**8-bit is the interesting result.** Against fp16 it is a tie (+0.23 points, CI [-0.12,
+0.70]), so it captures the full accuracy of unquantized weights while running at 4-bit's
speed (19.8x against 18.5x) and 7.29GB instead of 12.98GB. It also beats nvfp4 (+0.80, CI
[+0.03, +1.66]).

### Why this is not a metric artifact

For the fp16/4-bit pair, fp16 leads on all three error types counted separately, and
coverage excusal moves the wrong way to manufacture the result:

| | substitutions | deletions | insertions counted |
|---|---|---|---|
| 4-bit | 5,749 | 3,444 | 2,313 |
| fp16 | **5,185** | **3,274** | **2,132** |

The result also survives dropping the largest contributors one at a time and both together
(+1.20, +1.03, +0.93; every CI still clear of zero), so it is not one outlier file. Both
material types agree (spontaneous +1.58, published-video +0.73).

### What ships, and why it is still 4-bit for now

Nothing changed yet, and the reason is deliberate. Two things need settling before moving a
default that every user gets:

- **8-bit is not on the hub in a loadable form.** The two repos that advertise it ship raw
  Mistral `config.json` with no `model_type` and crash the loader (see below). Shipping
  8-bit by default would mean either a local conversion step on first use or a new upload,
  which is a distribution decision rather than a benchmark one.
- **These are per-machine trades, not one ranking.** On a 16GB machine fp16 cannot load at
  all and nvfp4's 5.09GB is the only comfortable option; on a 128GB machine 8-bit is
  clearly right. A single global default cannot express that, which is what
  `--quantization` is for.

So 4-bit remains the default as the safe option, now documented as a ~1.1-point accuracy
cost rather than as free. **If you have the memory, pass `--quantization fp16`, or convert
8-bit locally and point `--model` at it.**

### Still open: why the clip and the corpus disagree

The clip put fp16 versus 4-bit at 0.07 points, CI [-0.33, +0.48], on 40 paired regions, and
ranked 4-bit nominally best of five. The corpus reverses the ranking and resolves it. Both
tests are sound on their own material. Material type alone does not explain it, since both
kinds of corpus audio favour the higher precisions. Untested candidates: reference style
(verbatim against editorial), and that a 935s single-speaker recording may simply not
exercise whatever quantization degrades. Not resolvable without a second corpus.

## Corpus for the rest of this page

The single 935s Japanese prepared-narration clip with a complete verbatim reference, so
plain CER is meaningful. The remaining levers here are deliberately **not** measured on the
multi-file corpus, for a reason worth stating: at n=7 the corpus resolves about 3.2 points,
and these effects are 0.07-0.26 points. Detecting them would need roughly 1,000 files for
8-bit versus 4-bit and ~180,000 for kv8. Running the sweep across the corpus would consume
about 1.5h of GPU time to produce numbers inside the noise floor.

That power calculation is what the fp16 result above breaks: it predicted ~15,000 files
were needed for fp16 versus 4-bit, and 20 sufficed. So the calculation was extrapolating an
effect size measured on the wrong material, which is a caution about the estimates for the
other pairs too rather than a reason to trust them.

The cross-machine check gives an independent read instead: the same config on an M4 16GB
(nvfp4) and an M2 Ultra 128GB (4-bit affine) agreed to within ~1 point on 5 of 7 files. Note
that comparison cannot fully separate quantization from hardware, because the two
machines do not produce byte-identical output even at identical weights (see
[determinism.md](determinism.md)).

That confound is now better bounded, and it favours this reading. Running one identical
4-bit config on both machines over 18 files put 11 of them at *identical* coverage CER and
16 within 0.16 points, so the hardware term in that check is much smaller than the ~1 point
it was assumed to be. Whatever separated the nvfp4 and 4-bit rows was therefore mostly not
hardware, and it was still under a point, which strengthens rather than weakens the
conclusion that precision is a non-factor here.

## Method

`scripts/benchmarks/sweep_precision.py`. Everything fixed except the weights: 60s chunks, batch
16, delay 2400ms, same prompt. M2 Ultra 128GB, since a 16GB M4 cannot hold fp16 at all.
Differences are checked with a paired test over 40 regions of the same clip
(`scripts/benchmarks/compare_configs.py`) rather than by eyeballing two overall CERs.

## Experiment: weight precision

| weights | bytes | CER | kana CER | x realtime | decode steps/s | peak GB |
|---|---|---|---|---|---|---|
| fp16 (unquantized) | 8.9GB | 7.61% | 6.08% | 13.6x | 14.5 | 15.28 |
| 8-bit affine (local convert) | 4.7GB | 7.49% | 5.99% | 21.1x | 26.4 | - |
| mxfp8 (local convert) | 4.6GB | 7.66% | 5.97% | 21.2x | 26.5 | - |
| nvfp4 (local convert) | 2.5GB | 7.49% | 5.99% | 21.8x | 27.1 | - |
| **4-bit affine (hub)** | 2.9GB | **7.23%** | **5.60%** | **22.2x** | 26.3 | 9.36 |

The whole spread is 0.43 CER points. The fp16 and 4-bit hypotheses differ by 65
characters out of 4205 (1.5%), and fp16's 15 extra errors are spread across categories
rather than concentrated:

| weights | sub | ins | del | total |
|---|---|---|---|---|
| fp16 | 158 | 59 | 103 | 320 |
| 4bit | 149 | 52 | 104 | 305 |

Paired, fp16 versus 4-bit is 0.07 points with CI [-0.33, +0.48]. **On this clip, treat all
five as tied.** That does not generalise: fp16 beats 4-bit by 1.30 points on the 20-file
corpus, significantly, which is the section at the top of this page. The other four
comparisons here have not been re-tested on the corpus and so remain clip-only results.

What is *not* noise is the cost. Decode steps/s tracks weight bytes the way the
bandwidth model predicts (14.5 for 8.9GB versus 26-27 for the 2.5-4.7GB variants), which
is the same bandwidth story as [decode-throughput.md](decode-throughput.md).

Two confounds were ruled out explicitly:

- **The KV cache.** Repeating the comparison with `--kv-bits 8`: fp16 7.61% at 14.3x,
  4-bit 7.25% at 21.8x. Identical conclusion.
- **An untuned baseline.** Re-run at the best config found later (30s chunks, batch 32,
  kv8, 8s overlap), in case fp16 only pulls ahead once everything else is optimal: fp16
  7.32% at 17.8x versus 4-bit 7.25% at 26.7x. Still a tie, still 1.5x the cost.

One nuance worth recording rather than hiding: at that tuned config fp16 wins on *kana*
CER (5.51% vs 5.93%), so its errors may be marginally more phonetically faithful even
where its character score is not better. That gap is also inside the noise band.

## Experiment: KV cache quantization

`--kv-bits 8` halves the cache bytes read per step, which matters because at batch 16 and
838 positions the cache is 1.43GB per step against 2.5GB of weights.

| config | CER without | CER with kv8 | decode s without | decode s with |
|---|---|---|---|---|
| M4 16GB, 60s/B16 | 7.49% | **7.44%** | 78.9 | 72.8 |
| M2 Ultra 128GB, 60s/B16 | 7.23% | 7.25% | 31.8 | 29.9 |
| M2 Ultra 128GB, 30s/B32 | 9.13% | 9.11% | 17.2 | 15.6 |

Faster on both machines and no less accurate. Paired, kv8 versus unquantized KV is 0.02
points with CI [+0.00, +0.07], and **39 of 40 scored regions are identical**, which is
the strongest form of "free" available from this method. On by default in both hardware
profiles.

Implementation note: in mlx 0.32 this needs mlx-lm's
`quantized_scaled_dot_product_attention`, because `QuantizedKVCache.update_and_fetch`
returns triples that the dense `mx.fast.scaled_dot_product_attention` rejects.

## Two repos that could not be evaluated

`mlx-community/Voxtral-Mini-4B-Realtime-6bit` and
`ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx` both fail to load with
`TokenizersBackend has no attribute tokenizer`. They ship raw Mistral-format
`config.json` with no `model_type`, so mlx-audio routes them to the non-realtime
`voxtral` loader and dies in `post_load_hook`. That is a repo packaging problem, not a
precision result, and the locally converted 8-bit row above covers the same point.

Still true on 2026-08-20, which is why `--quantization` on `voxtral` offers only `4bit`
and `fp16`: those are the builds that load.

## Adjacent questions that came up and were closed

**Unsloth dynamic quants.** Unsloth publishes no Voxtral repos at all (their audio work
is Whisper-only), so there is nothing of theirs to evaluate. The idea of keeping
sensitive layers at higher precision is sound in general but has little room here: fp16
is 0.07 points from 4-bit at the best config, so the headroom a smarter quantization
could recover is smaller than this clip can measure.

**GGUF.** GGUF quants of Voxtral Realtime exist and are popular. They are unusable here
for a structural reason rather than a quality one: GGUF is llama.cpp's format, MLX cannot
load it, and this project's speed result comes entirely from a custom batched MLX
decoder. Adopting llama.cpp's runtime would mean giving up multi-stream batching, which
is worth 3-4x, to chase a quantization difference measured at 0.07 points. If a GGUF path
is ever wanted, `mx.quantize` already supports affine/mxfp4/mxfp8/nvfp4 locally, which is
how the 8-bit, mxfp8 and nvfp4 rows above were produced.

**Quantization on other runners is a different story.** On whisper.cpp, q5_0 is 27%
*slower* than fp16 at identical CER, because dequantization is work an fp16 matmul does
not do. So on Apple Silicon, quantize when memory-bound, not for speed. See
[engines.md](engines.md).

## What ships

`mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` with `--kv-bits 8`. Note the M4 16GB
profile was measured on nvfp4 while the Ultra profile was measured on 4-bit affine, which
is why those two rows are not a clean A/B of anything but hardware.

## Related

[decode-throughput.md](decode-throughput.md) for why bytes-per-step sets the speed.
[metrics.md](metrics.md) for why kana CER is reported but not trusted as the fair number.
