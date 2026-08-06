# Lever: how the audio is cut into chunks

**Voxtral only** for `--overlap-seconds`, `--vad`, `--compact-silence` and `--fast`;
`--chunk-seconds` also applies to `kotoba`, where it sets the independent-window length.
The `whisper-*` driver's 30s window is fixed by the model's positional encoding. Passing a
flag to an engine that cannot honour it is an error, not a warning.

**Conclusion first.** Chunk length has a clear mechanism and, on real material, no
measurable accuracy effect between 30s and 60s. The mechanism is chunk seams: a chunk's
first tokens are emitted with no left context, so errors concentrate at chunk *starts*, and
on a single clip longer chunks measurably win up to about 60s where the encoder's sliding
window ends the gains. **On the 20-file corpus, 60s versus 30s is +0.10 points with a CI of
[-1.89, +2.03], i.e. indistinguishable**, so pick between them on throughput and let
`profiles.json` decide per machine. Prefix overlap fixes the seam cost directly and won
1.4-1.8 points at 30s chunks on a single clip, **but reversed sign on a real corpus and is
therefore not on by default**. Cutting at energy minima beats cutting with a VAD, which is
the opposite of what the VAD literature predicts.

Chunk length is still worth understanding, because it is the largest lever on *throughput*
and because it does drive accuracy on dense single-speaker narration. It is just not a lever
to tune for accuracy on spontaneous conversational audio.

## Corpus

Two kinds of material, and they disagree, which is the main lesson of this document.
Single-clip work used one 935s Japanese prepared-narration recording with a complete
verbatim reference, so plain CER is meaningful there. Corpus work used 7 to 20
spontaneous multi-speaker recordings with editorial references, scored with the
coverage-aware metric. See [corpus.md](corpus.md).

## Method

`scripts/benchmarks/sweep_overlap.py` for overlap, `scripts/benchmarks/run_matrix.sh` for chunk length,
`scripts/benchmarks/probes/probe_seam_errors.py` for the seam analysis. Accuracy differences on
one clip are compared with a **paired** test over 40 regions of the same audio
(`scripts/benchmarks/compare_configs.py`), so shared difficulty cancels rather than inflating
both intervals. Absolute CER differences under about half a point are not resolvable on
one clip.

## Why seams cost anything: the mechanism

Locating every edit operation relative to the nearest chunk boundary, 30s chunks:

| region | subs | ins | del | total | share of edits | share of audio | enrichment |
|---|---|---|---|---|---|---|---|
| first 3s of a chunk | 33 | 24 | 28 | 85 | 22.3% | 9.9% | **2.24x** |
| last 3s of a chunk | 11 | 28 | 10 | 49 | 12.9% | 9.9% | 1.29x |
| elsewhere | 128 | 42 | 77 | 247 | 64.8% | 80.1% | 0.81x |

Errors concentrate at chunk *starts*, not ends, which is exactly what a causal model
predicts: at position 0 of a chunk it has no left context. This also tells you which
direction of overlap can possibly help. Both encoder and decoder are causal, so
appending audio *after* a chunk cannot change tokens already emitted; prepending audio
can.

## Experiment: chunk length

M2 Ultra 128GB, 4-bit, no overlap, single clip:

| chunk | batch | CER | x realtime |
|---|---|---|---|
| 20s | 48 | 12.46% | 25.6x |
| 30s | 32 | 9.13% | 31.0x |
| 60s | 16 | **7.37%** | 21.2x |
| 90s | 16 | 7.99% | 17.2x |
| 120s | 8 | 7.59% | 16.1x |
| 180s | 8 | 7.56% | 11.6x |

Accuracy improves up to 60s then flattens in the 7.5-8.0% band while speed falls away,
because chunks beyond ~60s exceed the encoder's 750-frame sliding window (a 60s chunk is
already ~1948 conv frames) and the batch has to shrink to fit memory. At 20s the loss is
mostly deletions, as short rows end early and drop text.

Paired testing is stricter than the point estimates suggest:

| comparison | diff | 95% CI | verdict |
|---|---|---|---|
| 60s beats 30s | 1.85 | [+0.71, +3.24] | significant |
| 60s beats 90s | 0.62 | [-0.36, +1.69] | **not supported** |

So **60s is the right default because it is simultaneously the fastest of the long
options and never measurably worse**, which is a weaker justification than "longer
chunks are worse" but the one the data supports. An earlier version of these docs
claimed the stronger thing; that was a correction, not a new measurement.

On the **7-file** corpus the 60s-versus-30s difference does not resolve either
(+1.67, CI [-1.22, +4.73]), and 30s/batch 32 was nominally better on every axis
including speed. Between-file variance dwarfs the effect. The shipped default is per
machine, from `profiles.json`.

### Re-run at n=20: still not resolvable, and the point estimate collapsed

The corpus later grew to 20 files, lowering the resolution floor from about 3.2 points to
about 1.6. A +1.67-point effect sat almost exactly on that floor, making this the one
small-corpus result that more audio might plausibly turn into a decision. It did not.

Both arms re-run on one machine (M4 16GB, 20 files, sequential, `--delay-ms 2400`, kv8;
60s at batch 16 and 30s at batch 32, each machine's profile for that chunk length):

| | 60s / b16 | 30s / b32 | paired difference | 95% CI |
|---|---|---|---|---|
| JP coverage CER, 17 files | 16.29% | **16.19%** | +0.10 | [-1.89, +2.03] |
| EN coverage WER, 3 files | 26.14% | **25.24%** | +0.90 | [-0.27, +1.69] |

Not resolvable on either unit, and the Japanese point estimate fell from +1.67 at n=7 to
+0.10 at n=17: the two chunk lengths are indistinguishable on this material. 30s won 6 of 17
Japanese files and 60s won 10, with a sign test at p=0.454. Per file the spread is enormous
and two-sided (one file 18.25 points better at 30s, another 10.02 points better at 60s),
which is the between-file variance that dominates every config effect in this project.

So this is now a *settled* negative rather than an open question, which is worth more than
the ambiguity it replaces: chunk length between 30s and 60s can be chosen purely on
throughput. That makes it a hardware decision, which is exactly what `profiles.json`
encodes. No accuracy claim should be attached to either value.

Speed is not comparable across those two rows here, since the machine was under unrelated
background load for part of the second arm and the harness flagged it; accuracy is unaffected
by load because decoding is greedy. The clean throughput comparison for these two configs is
in [decode-throughput.md](decode-throughput.md).

## Experiment: prefix overlap

`--overlap-seconds N` prepends N seconds of the preceding audio to each chunk and
discards the tokens produced from it, so the model warms up before the region that is
kept.

M2 Ultra 128GB, 30s chunks, batch 32, kv8, single clip:

| overlap | CER | delta | x realtime | extra audio decoded |
|---|---|---|---|---|
| 0s | 8.73% | - | 32.7x | 0s |
| 4s | 7.30% | -1.43 | 29.2x | +124s |
| 6s | 7.61% | -1.12 | 27.7x | +186s |
| 7s | 7.63% | -1.09 | 27.0x | +218s |
| 8s | **7.25%** | **-1.47** | 26.3x | +248s |
| 10s | 7.56% | -1.17 | 25.6x | +310s |
| 12s | 7.80% | -0.93 | 23.1x | +372s |
| 15s | 11.20% | +2.47 | 22.2x | +466s |

M4 16GB, 30s chunks, batch 32: 0s 9.23% -> 4s 7.97% -> 8s 7.42%, same shape.

Three things to read off this:

- **The win is real at short chunks and reproduces on both machines**, and it survives
  a paired test (+1.80, CI [+0.62, +3.20]).
- **The curve is noisy, not smooth.** Between 4s and 12s it wanders in the 7.25-7.80%
  band with no clean optimum, a spread smaller than the measurement noise. "4s or more
  helps by about 1.5 points" is defensible; "8s is optimal" is not. 4s buys most of the
  gain for half the extra decode.
- **It collapses at 15s** on a 30s chunk, where the warm-up region is half the chunk and
  rows start hitting EOS inside it.

At 60s chunks it stops paying, because seams are sparse:

| overlap | CER | delta |
|---|---|---|
| 0s | 7.37% | - |
| 4s | 7.59% | +0.21 |
| 8s | 8.06% | +0.69 |

Paired, that is -0.69 with CI [-1.47, +0.07], so the honest claim is **"no benefit at
long chunks", not "harmful"**. An earlier version said harmful; corrected.

### Why it is off by default anyway

On the 7-file corpus the effect **reversed sign**: -1.47 points, CI [-4.33, +2.36], with
no-overlap nominally better on 5 files to 2. English was worse by 4 points
(26.55% -> 30.77% coverage WER). Two plausible mechanisms: these recordings contain long
stretches of non-reference material, so a warm-up window often carries content the
reference cut; and the seam-error enrichment above was measured on dense narration and
may not hold for conversational audio with frequent long pauses.

So overlap is tied to `--fast`, which halves the chunk length and therefore creates the
dense-seam condition where it measurably helps. It is not a universal default. This is
the clearest case in the project of a significant single-clip result that did not
generalize.

## Experiment: where to cut, energy versus VAD

`--vad` uses Silero VAD (ONNX, no torch) to cut in the middle of the longest non-speech
run near each target, instead of at the quietest 50ms window. It never removes audio,
only chooses where to cut, so the chunks still cover the input exactly. VAD inference is
negligible: 2.2s for 935s of audio, 426x realtime.

M2 Ultra 128GB, 4-bit, kv8:

| config | CER | lenient CER | x realtime |
|---|---|---|---|
| 30s, energy | **8.73%** | 8.42% | 21.9x |
| 30s, VAD | 10.75% | 10.39% | 20.4x |
| 30s, energy, overlap 8s | **7.25%** | 7.04% | 18.7x |
| 30s, VAD, overlap 8s | 8.04% | 7.73% | 18.0x |
| 60s, energy | **7.37%** | 7.11% | 21.0x |
| 60s, VAD | 10.25% | 9.85% | 20.1x |

VAD loses by 0.8-3.0 points in every pairing, and this is one of the few results here
that clears significance: paired over 40 regions at 60s chunks, energy beats VAD by 3.00
points, CI [+0.74, +5.93], winning 21 regions to 7 (sign test p=0.013).

That is the opposite of what the VAD literature predicts, and the VAD cuts really are
cleaner by the obvious measure: speech probability in the 1s *after* a cut is 0.316 for
VAD versus 0.485 for energy, and only 10 of 30 VAD cuts start inside speech versus 17 of
31 energy cuts.

The likely explanation ties back to overlap. The energy splitter picks the quietest
*instant*, which lands mid-pause and hands the next chunk a run of leading silence to
warm up on. VAD picks the middle of a non-speech *run*, often a short inter-word gap
that satisfies the detector but leaves almost no silence before speech resumes.
**Warm-up room, not cut cleanliness, is what this model wants.** Kept as an opt-in flag
for noisy material where energy minima may mislead.

## Experiment: carrying context across seams instead of overlapping

The Voxtral paper notes the decoder reuses KV state as audio is appended, so a chunk
boundary is where this tool discards context. Two ways to give it back, using the
existing per-chunk prompt mechanism. M2 Ultra 128GB, 30s chunks, batch 32:

| variant | CER | delta | wall | x realtime | note |
|---|---|---|---|---|---|
| none | 8.73% | - | 27.6s | 33.9x | |
| carry_pair | 8.56% | -0.17 | 55.8s | 16.8x | 2 batched passes, pass 2 gets pass 1's tails |
| carry | 8.56% | -0.17 | 177.6s | 5.3x | strictly sequential, batch 1 |
| static keywords | 9.11% | +0.38 | 27.9s | 33.5x | domain prompt in every chunk |

Carrying context recovers 0.17 points against the ~1.5 that seams cost, and at best
doubles wall clock. Notably the strictly sequential version, which has true
left-to-right context, is **no better** than the cheap two-pass one, which says the
31-token prompt window is simply too small to carry meaningful context. Not worth it:
longer chunks recover the full amount for free.

## Experiment: dropping silence before decode

`--compact-silence` drops the middle of pauses longer than 400ms, keeping the first
240ms. Since decode cost is one step per 80ms frame, removing silence removes steps
one-for-one; on the reference clip it removed 12% of the audio. Timestamps are mapped
back to the original timeline, and the resulting chunk cuts were measurably *cleaner*
(no cut louder than -50dB, versus 3 cuts above -45dB without it).

The accuracy result nevertheless splits by quantization, which is the one result in this
project that does not transfer between machines:

| config | CER baseline | CER compacted | deletions |
|---|---|---|---|
| M4 16GB, nvfp4, 60s/B16 | 7.49% | 11.63% | 105 -> 222 |
| M4 16GB, nvfp4, 30s/B32 | 9.06% | 13.39% | 115 -> 338 |
| M2 Ultra 128GB, 4bit affine, 60s/B16 | 7.23% | 8.23% | 103 -> 118 |
| M2 Ultra 128GB, 4bit affine, 30s/B32 | 9.13% | 8.59% | 123 -> 109 |

On nvfp4 the deletions triple and the loss is concentrated rather than spread: one
two-minute stretch lost 45% of its text after only 4.8s of silence was removed there, so
this is not proportional information loss. The model appears to rely on pauses for its
own segmentation, and the more aggressively quantized weights tolerate their removal
much worse. The mechanism is a hypothesis, not a measurement. Off by default; if you
want the speed, verify on your own audio and model first.

## What ships

- Chunk length and overlap come from `mlx_asr/profiles.json` per machine, 60s and 0s on
  both benchmarked machines.
- `--fast` halves the chunk and enables 8s overlap, since that is the regime where
  overlap earns its cost back.
- Energy-based cut points, with `--vad` available as an opt-in.
- `--compact-silence` off.

## Related

[delay.md](delay.md) is a larger lever and free. [decode-throughput.md](decode-throughput.md)
explains why short chunks cost encoder time and why the batch that pairs with a chunk
length is not a free choice.
