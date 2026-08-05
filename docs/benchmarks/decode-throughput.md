# Lever: batch size and decode throughput

**Voxtral only.** Multi-stream batching is what this project added; the other engines
decode one stream at a time and reject `--max-batch`.

**Conclusion first.** Batch size is the single largest speed lever and it is **not
monotonic**: on every machine measured, batch 2-8 is *slower per step* than batch 1.
Either stay at 1 or jump to 12 or more. The cause is upstream kernel dispatch in MLX,
not anything fixable here, so the defaults encode the sizes that measured well rather
than a formula. Batching the encoder, splitting the batch, and `mx.compile` were all
tried and none helped.

## Why this lever exists at all

Voxtral Realtime is a streaming model: it consumes 80ms of audio per decoder position
and emits exactly one token per position. Two consequences set everything else.

- **Decode step count is fixed by audio duration**, not by how much speech there is.
  935s of audio is ~11,700 steps whether it is dense narration or mostly silence.
  Faster hardware does not reduce the step count.
- **A single stream cannot exceed about 1x realtime by much**, on any machine, because
  one step costs at least one full pass over the model weights. On the M4 that is 21ms
  against an 80ms budget, so ~3.8x is the ceiling at batch 1.

So this is a throughput problem, not a latency one: split the audio, decode the pieces
in lockstep, and the per-step weight read amortizes across rows.

## Method

`mlx-asr-bench`. It drives the real decoder with random embeddings, which measures the
quantity that actually sets wall clock (decode steps per second), and therefore needs
**no audio and no reference**. That also makes it the one measurement in this project
anyone can reproduce on their own hardware, which is why it is the basis of the
contributed-profile flow.

400 steps per batch size, reported in four blocks so decay within a measurement is
visible. `x realtime = steps/s x batch x 0.08`, since each row advances 80ms per step.
`peak GB` is the MLX peak during the run.

Machine state matters more than it looks. A host doing other GPU work reports several
times lower throughput, badly enough that one run in this project was thrown away rather
than reported. Check GPU and memory use before starting, run one benchmark at a time, and
treat any speed number without a stated machine state as unreliable.

## Experiment: the batch ladder on two machines

M4 17GB, 10 GPU cores, nvfp4 weights, mlx 0.32.0:

| batch | steps/s | ms/step | x realtime | peak GB |
|---|---|---|---|---|
| 1 | 44.63 | 22.4 | 3.6 | 2.79 |
| 2 | 38.19 | 26.2 | 6.1 | 2.91 |
| 4 | 19.26 | 51.9 | 6.2 | 3.25 |
| 8 | 9.03 | **110.8** | 5.8 | 3.78 |
| 12 | 11.28 | 88.7 | 10.8 | 4.45 |
| 16 | 11.07 | 90.3 | 14.2 | 4.94 |
| 24 | 10.28 | 97.3 | 19.7 | 5.92 |
| 32 | 9.66 | 103.6 | **24.7** | 6.84 |
| 48 | 5.23 | 191.4 | 20.1 | 8.42 |

M2 Ultra 137GB, 60 GPU cores, 4-bit affine, mlx 0.32.0:

| batch | steps/s | ms/step | x realtime | peak GB |
|---|---|---|---|---|
| 1 | 92.60 | 10.8 | 7.4 | 5.15 |
| 2 | 78.35 | 12.8 | 12.5 | 5.32 |
| 4 | 61.45 | 16.3 | 19.7 | 5.65 |
| 8 | 40.55 | 24.7 | 26.0 | 6.28 |
| 12 | 32.09 | 31.2 | 30.8 | 6.75 |
| 16 | 29.74 | 33.6 | 38.1 | 7.43 |
| 24 | 29.10 | 34.4 | 55.9 | 8.38 |
| 32 | 29.54 | 33.9 | 75.6 | 9.06 |
| 48 | 18.81 | 53.2 | 72.2 | 10.73 |
| 64 | 18.85 | 53.0 | 96.5 | 12.41 |
| 96 | 12.08 | 82.8 | 92.8 | 16.16 |
| 128 | 10.11 | 99.0 | **103.5** | 19.61 |

Read the M4 column carefully: **batch 8 costs 5x more per step than batch 1**, and
batch 12 is *cheaper* than batch 8. Batch 1 at 22.4ms is exactly 2.5GB / 120GB/s, the
bandwidth floor, so nothing is wrong at batch 1; the penalty from 2 upward is a
kernel-path effect. Per-step cost grows roughly linearly with batch in the 2-8 range,
as though the batched quantized matmul falls back to a per-row path and then recovers
once the batch is large enough to select a tiled kernel.

The Ultra shows the same shape, much shallower, plus a second regression at 48 and
again at 96.

Because throughput per row keeps rising even where ms/step rises, x-realtime is the
metric to optimize, not ms/step.

## Why it happens, and why it cannot be fixed here

MLX dispatches quantized matmuls to one of several Metal kernels based on the leading
dimension, and the maintainers state the cause directly:

> "The drop from 4 to 8 is that we switch from batched qmv to the qmm."
> ml-explore/mlx discussion #1593

The threshold is `vector_limit`, computed by `get_qmv_batch_limit`. It is compile-time
C++ and hardware-specific: 10 on an M4 Pro for K,N>4096, 14 on an M4 Max at size 4352,
different again on an M2 Ultra 128GB. A related open issue (mlx#3553) documents the same
discontinuity at M=3 and reports that manually lowering `vector_limit` made things
substantially worse, with no fix landed.

## Experiments that did not work

**Reshaping to dodge the valley** (`scripts/benchmarks/probes/probe_batch_split.py`), M4 16GB,
nvfp4, ms/step:

| batch | plain | fold `[B,1,d]`->`[1,B,d]` | split into 2 | split into 4 |
|---|---|---|---|---|
| 4 | 50.3 | 50.3 | 50.1 | 80.9 |
| 8 | 101.6 | 102.1 | 102.8 | 106.3 |
| 12 | 84.1 | 78.1 | 143.4 | 152.0 |
| 16 | 80.6 | 80.3 | 226.8 | 216.5 |
| 32 | 103.2 | 99.8 | 176.0 | 436.6 |

`fold` is numerically bit-exact (max absolute difference 0.000000) and worth 3-7% at
best. Splitting is far worse, because each sub-batch pays full weight reads.

**Batching the encoder** (`scripts/benchmarks/probes/probe_encoder_batch.py`). mlx-audio's
encoder attention is batch-1 only, so giving it a batch axis is the obvious next move.
It is 0.84-0.91x, i.e. slightly slower:

| batch | s/chunk | vs per-chunk |
|---|---|---|
| per-chunk (stock) | 1.497 | 1.00x |
| 1 | 1.651 | 0.91x |
| 2 | 1.642 | 0.91x |
| 4 | 1.738 | 0.86x |
| 8 | 1.773 | 0.84x |

Arithmetic intensity explains it. Batching amortizes *weight reads*, so it only pays
when a stage is bandwidth-bound, and the encoder is not:

| stage | FLOP per chunk/step | bytes read | FLOP/byte | bound by |
|---|---|---|---|---|
| encoder, one 30s chunk | ~3270 GFLOP | 0.66GB | ~4950 | compute |
| decoder, one step | ~5 GFLOP | 2.5GB | ~2 | bandwidth |

At ~3270 GFLOP against an M4 GPU peak near 4 TFLOP/s, one chunk needs ~817ms of pure
math and measures 1.5s, so the encoder already runs at roughly half theoretical peak.
Its share of wall clock is a hard floor on this hardware, not an optimization
opportunity.

**`mx.compile` on the decode step**: 22.6 -> 23.2 ms/step at batch 1, 83.4 -> 89.8 at
batch 16, 98.6 -> 106.7 at 32. Equal or worse everywhere, which is the expected result
for a bandwidth-bound loop rather than a launch-bound one.

**A newer MLX with better small-batch kernels.** 0.32.0 was already the latest release;
nothing to upgrade to.

## Where the wall clock actually goes

Decode is not the whole story. Splitting wall clock on the 935s clip, 60s chunks:

| stage | M4 16GB | M2 Ultra 128GB |
|---|---|---|
| encode (16 chunks) | 47.3s | 10.8s |
| decode (816 steps) | 78.9s | 31.8s |
| total | 130.3s (7.2x) | 43.9s (21.3x) |

The encoder is 36% of M4's wall clock and grows as chunks get shorter, since each chunk
pays its own conv stem and 32-layer causal pass. That caps the short-chunk strategy:
going 60s -> 30s on the M4 cut decode from 78.9s to 48.5s but pushed encode from 47.3s
to 51.8s, a net 7.2x -> 8.3x. Mel and conv stem are noise (0.01s and 0.02s per chunk).

For calibration, antirez/voxtral.c, a hand-written C + Metal implementation with custom
attention/RoPE/KV kernels, reports 23.5-31.6 ms/step at batch 1 on an M3 Max. This
implementation does 22.4 ms/step at batch 1 on the M4, so the single-stream path is
already at parity with hand-tuned native code. The win here comes from batching, not
from kernel work.

## KV cache growth

steps/s decays as the cache fills, about 25% over 800 steps at batch 16 on the M4:

| kv_len | ms/step | peak GB |
|---|---|---|
| 138 | 78.4 | 4.28 |
| 438 | 89.0 | 4.94 |
| 838 | 112.6 | 6.41 |

At batch 16 and 838 positions the cache is 1.43GB *read per step*, comparable to the
2.5GB of weights, so it stops being free. Shorter chunks bound `kv_len`, which is a
second reason short chunks can win on a memory-poor machine despite more encode work.

`--kv-bits 8` halves those cache reads and was faster *and* no less accurate on both
machines, so it is on by default. See [quantization.md](quantization.md).

## How the default is chosen

Two tiers, in `mlx_asr/hardware.py`.

A machine listed in `mlx_asr/profiles.json`, matched on chip and RAM, uses its measured
numbers outright. That is the point of the file: the right batch is not predictable
from specs, because throughput is not monotonic, so a formula fitted to one machine
mispredicts the next.

For anything unbenchmarked, the batch is the smaller of two caps, then snapped onto the
sizes that measured well (1, 12, 16, 24, 32, 64, 128):

- a **memory cap**: half the GPU working set, minus model weights and ~0.6GB fixed
  overhead, divided by ~0.002 GB per row-second of chunk audio, which is the asymptote
  of both sweeps above.
- a **compute cap**: 3 rows per GPU core, since both sweeps reach 90% of peak
  throughput at 1.1-3.2 rows per core.

Anything landing in the 2-11 range falls back to 1, because the real choice there is
"12 or more" versus "1", not a point on a smooth curve.

Contributing a measured profile for a machine that is not listed is the most useful
contribution to this project, and needs no audio: see
[../../CONTRIBUTING.md](../../CONTRIBUTING.md).
