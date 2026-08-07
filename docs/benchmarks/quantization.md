# Lever: weight and KV-cache precision

**Voxtral only.** `--kv-bits` and the weight choice apply to our own batched decoder. The
`whisper-*` aliases point at fp16 MLX conversions and take no precision flag; whisper.cpp
quantization is covered in [engines.md](engines.md), where it costs speed rather than
accuracy.

**Conclusion first.** Quantization costs nothing measurable in accuracy on this
workload. fp16 through 4-bit span 0.43 CER points on 4205 reference characters, which is
inside the noise, while fp16 costs 1.6x the wall clock and 15.3GB of peak memory against
9.4GB. **Use 4-bit.** `--kv-bits 8` is separately close to free and is on by default.
Do not read 4-bit landing nominally best as evidence that quantization helps.

## Corpus

The single 935s Japanese prepared-narration clip with a complete verbatim reference, so
plain CER is meaningful. This lever is deliberately **not** measured on the multi-file
corpus, for a reason worth stating: at n=7 the corpus resolves about 3.2 points, and
these effects are 0.07-0.26 points. Detecting them would need roughly 1,000 files for
8-bit versus 4-bit, ~15,000 for fp16 versus 4-bit, and ~180,000 for kv8. Running the
sweep across the corpus would consume about 1.5h of GPU time to produce three numbers
inside the noise floor.

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

Paired, fp16 versus 4-bit is 0.07 points with CI [-0.33, +0.48]. **Treat all five as
tied.**

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
