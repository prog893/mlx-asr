# Does batching help the Qwen3-ASR decoder? (issue #1)

**No.** Throughput falls monotonically with batch size and accuracy does not move, so
`--max-batch` stays refused on `qwen3-asr`. This closes
[#1](https://github.com/prog893/mlx-asr/issues/1).

## Result

20 files, 7.95h, 15s windows, `Qwen3-ASR-1.7B-8bit`, `--language ja`, M2 Ultra.

| batch | x realtime | JP coverage CER | EN coverage WER | peak GPU |
|---|---|---|---|---|
| **1 (ships)** | **23.18x** | **19.51%** | **27.93%** | 3.73GB |
| 2 | 20.72x | 20.16% | 28.68% | 3.77GB |
| 4 | 14.75x | 19.92% | 28.58% | 3.82GB |
| 8 | 9.91x | 19.79% | 28.49% | 6.16GB |

Batch 1 is fastest, and by a widening margin: 2.3x faster than batch 8. Accuracy across
the four arms spans 0.65 CER points, which is inside this corpus's noise, so there is no
accuracy case for batching either.

Batch 16 was killed by memory pressure partway through (the machine reached 16.1GB of
17.4GB swap). Its one completed file scored 4.1x against 36.2x for batch 1 on the same
file, consistent with the trend, but it is one file and is not in the table.

## Why batching loses here, when it wins on Voxtral

Voxtral batches **decode steps**: rows share one weight read per step, which is the whole
point on a bandwidth-bound decoder. The Qwen3-ASR path batches **whole chunks**, and two
things in that design cancel the benefit.

**Padding.** Chunks in a group are zero-padded to the longest member before the encoder
runs (`pad_to = max(len(c[0]) for c in group)`). A group holding a 15s chunk and a 2s tail
chunk pays 15s of encoder cost on both. The encoder is a real share of wall clock here, so
this is not a rounding error.

**Whole-group lockstep.** The group decodes until every member hits EOS or the budget runs
out. A batch finishes when its *slowest* member finishes, so one long chunk holds the other
rows' slots open while they emit nothing.

Batch 1 has neither cost, and the decode it does is the same decode; there is no weight
read to amortize across chunks because each chunk's decode is already sequential over its
own tokens.

## The shared token budget, which is a separate bug

Upstream's batched path decrements one budget across the whole file
(`remaining_tokens -= group_tokens` in `_transcribe_batched`), so the per-window cap that
`mlx_asr`'s own chunk loop exists to enforce does not survive batching. One runaway chunk
can then spend the budget its batch-mates needed.

Measured on one 14.7-minute file at batch 4, with upstream's accounting against a
per-group budget:

| | chars emitted | chars/s of audio | CER | x realtime | peak |
|---|---|---|---|---|---|
| batch 1 (ships) | 4,409 | 5.0 | 17.56% | 36.2x | 3.73GB |
| batch 4, budget per group | 5,950 | 6.7 | 17.40% | 26.2x | 3.77GB |
| batch 4, upstream's file-wide budget | 33,389 | **37.8** | **49.86%** | 1.8x | 13.09GB |

Real Japanese speech runs 6-9 chars/s and `RUNAWAY_CHARS_PER_S` is 25, so 37.8 chars/s is
a repetition loop running unchecked: 32 CER points worse and 20x slower than batch 1, with
peak memory at 13.09GB against 3.73GB. The per-group arm sits at 6.7 chars/s, inside the
range for real speech.

**The table at the top of this document uses the per-group budget**, which is the fairest
version of batching available. Batching loses even then, which is the finding. Had the
sweep used upstream's accounting alone, the result would have looked like a catastrophe
caused by batching rather than by budget accounting, and the two are worth separating.

## Method

`scripts/benchmarks/sweep_qwen3_batch.py`. The batch-1 arm calls `qwen3_decode`, the
shipped code path, so the baseline is not a reimplementation. Batch > 1 has to call
upstream's `generate(batch_size=)` directly because `mlx_asr`'s loop is per-chunk by
construction; that asymmetry is the thing being measured.

15s windows rather than the shipped 30s, so that a file yields enough chunks to fill
several groups at batch 8 and 16. 15s ties 30s on accuracy
([qwen3-asr.md](qwen3-asr.md)), so the baseline is comparable.

**Caveat on the throughput figures.** This is a shared machine and two arms began with
another process holding GPU memory (48.8GB at the start of batch 2, 8.9GB at batch 4), which
the harness recorded per arm. Wall clock is therefore not clean. Two things make the
conclusion hold anyway:

- Accuracy is unaffected by contention, because decode is greedy. The paired per-file
  comparison shows batch 2 **worse on 18 of 20 files**, which is the robust signal.
- The slowdown is monotonic and large (2.3x by batch 8), while the contention was worst
  during batch 2, the arm that lost *least*. Contention cannot explain a trend that runs
  opposite to it.

A re-run on an idle machine would tighten the x-realtime column. It would not change the
decision, which is to keep refusing the flag.
