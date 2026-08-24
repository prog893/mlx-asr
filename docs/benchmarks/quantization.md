# Lever: weight and KV-cache precision

**`--kv-bits` is Voxtral only**, since it applies to our own batched decoder.
**`--quantization` works on any model that publishes more than one build**, currently
`voxtral` and the two `qwen3-asr*` models; the `whisper-*` aliases point at fp16 MLX
conversions with no quantized builds in use here, so it errors there. whisper.cpp
quantization is covered in [engines.md](engines.md), where it costs speed rather than
accuracy.

**Conclusion first.** Quantization costs about **1.3 CER points** on the 20-file corpus,
where fp16 beats 4-bit significantly (95% CI [+0.59, +2.26]). **Use 4-bit anyway**: fp16
costs 1.65x the wall clock and 1.9x the memory, and its 12.98GB peak does not fit a 16GB
machine at all, so the accuracy is unbuyable for most users. `--kv-bits 8` is separately
close to free and is on by default.

That figure replaces an earlier "quantization costs nothing measurable", which came from a
single prepared-narration clip where the same pair spans 0.43 points and is a tie. Both
measurements are sound on their own material and they disagree; the corpus one governs,
because it is the material this project is tuned for. Details and what remains unexplained
are in the next section.

## The policy: one measured default per model, and `--quantization` to override it

Each model ships **one default precision, chosen by measurement where a choice exists**,
and `--quantization` exposes the rest where the converters published more than one build.

The default is picked by this rule, in this order:

1. **Take the cheapest precision whose accuracy cost is worth its price.** On Voxtral that
   cost is now measured at ~1.3 CER points, and 4-bit still wins the trade because fp16
   needs 1.65x the time, 1.9x the memory, and more memory than a 16GB machine has.
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

**A full ladder is not swept per model.** The sweep below answered the general question
once (is quantization silently costing quality?) and the answer was no, by a margin far
under what this corpus can resolve: at n=7 it resolves about 3.2 points, and 8-bit versus
4-bit would need roughly 1,000 files. New models therefore inherit that conclusion, and a
single bf16-versus-default check is run to confirm the default rather than a five-arm
ladder. **4bit through 6bit on `qwen3-asr*` are exposed but unmeasured here**, which the
`--help` text says.

The one thing worth checking afresh on a new engine is not accuracy but **decoder
degeneracy**, where the effect size is large rather than fractional. On Qwen3-ASR it made
no difference: bf16 and 8-bit produced near-identical repetition-loop counts (2/0/18/0/31
against 3/0/19/0/31 per file), so its loops are not a quantization artifact. See
[qwen3-asr.md](qwen3-asr.md).

## fp16 versus 4-bit is resolved on the corpus, and fp16 wins

**This overturns the conclusion below for that one pair.** Everything else on this page
stands; read this section first.

A paired bootstrap over the 20-file corpus, both arms produced in the same sweep on an idle
M2 Ultra at one config:

| | JP coverage CER | EN coverage WER | x realtime | peak GPU |
|---|---|---|---|---|
| 4-bit (ships) | 16.34% | 22.43% | 18.50x | 6.77GB |
| **fp16** | **15.04%** | 21.64% | 11.24x | 12.98GB |

    paired difference, 17 char-unit files, 70,419 reference chars
      +1.30 points, 95% CI [+0.59, +2.26]
      fp16 wins 13 of 17 files, sign test p = 0.049

The CI excludes zero, so this is a real difference on this material rather than the tie the
clip reported. It survives dropping the largest contributors one at a time and both
together (+1.20, +1.03, +0.93; every CI still clear of zero), so it is not one outlier
file.

It is also not a metric artifact. fp16 is ahead on all three error types counted
independently, and coverage excusal moves the wrong way to explain it:

| | substitutions | deletions | insertions counted |
|---|---|---|---|
| 4-bit | 5,749 | 3,444 | 2,313 |
| fp16 | **5,185** | **3,274** | **2,132** |

Both material types in the corpus agree (spontaneous +1.58, published-video +0.73), so the
disagreement with the clip result below is **not** explained by material alone.

**What has not changed: 4-bit still ships.** fp16 costs 1.65x the wall clock and 1.9x the
memory, and its 12.98GB peak does not fit a 16GB machine at all
([peak-memory.md](peak-memory.md)), so the default cannot be fp16 for most users even
though it is more accurate here. What changed is the *reason*: 4-bit is a deliberate
accuracy-for-memory trade of about 1.3 points, not a free lunch.

**Still open: why the clip and the corpus disagree.** The clip put the same pair at 0.07
points, CI [-0.33, +0.48], on 40 paired regions. Both tests are sound on their own
material, so something differs between prepared narration with a verbatim reference and
this corpus that quantization interacts with. Untested candidates: chunk/batch config
(the clip ran 60s/B16 fixed, the corpus at the per-machine profile), and reference style.
Not resolvable without another corpus.

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
