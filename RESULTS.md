# Results

Findings live in [docs/benchmarks/](docs/benchmarks/), one document per lever, each with
its corpus, method, experiments and conclusion. This page is the index and the list of
what is still open.

## Headline

20 recordings, 8.01h[^r1] of Japanese and English, on an Apple M2 Ultra 128GB (Mac14,14),
re-measured 2026-08-19 after a reference-loading fix (see below):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmarks/img/picker-dark.svg">
  <img alt="Error against speed for every engine at its defaults, Japanese and English panels; filled points fit a 16GB Mac" src="docs/benchmarks/img/picker-light.svg">
</picture>

| engine | JP coverage CER | EN coverage WER | x realtime |
|---|---|---|---|
| voxtral (default, fp16 on this machine) | 15.20% | 21.21% | 23.9x |
| whisper-turbo, no-condition | **14.17%** ±0.28 | 18.65% ±0.49 | 18.5-23.2x |
| qwen3-asr (1.7B) | 18.62% | 24.72% | 24.6x |
| qwen3-asr-small (0.6B) | 22.31% | 22.95% | **35.5x** |
| voxtral-v1 3B (8bit) | 36.14% | 17.86% | 14.1x |
| voxtral-v1 24B (8bit) | 27.20% | **16.86%** | 3.4x |
| parakeet-ja (17 JP files)* | 25.68% | n/a, ja-only | **261.8x*** |
| reazon-k2 fp32 (17 JP files)* | 30.06% | n/a, ja-only | 52.9x*, CPU |

Source: [ultra-voxtral-precision-default.json](benchmarks/results/2026-10/ultra-voxtral-precision-default.json) runs `vox_wfp16_c30b128_kv8`, `vox_wfp16_c30b128_kv8_r2`, `vox_wfp16_c30b128_kv8_r3`; [ultra-whisper-sizes.json](benchmarks/results/2026-10/ultra-whisper-sizes.json) runs `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`, `whisper_turbo_nocond_r6`; [ultra-picker-speed.json](benchmarks/results/2026-10/ultra-picker-speed.json) runs `pk_qwen3_1.7B_8bit_c15_s1`, `pk_qwen3_1.7B_8bit_c15_s2`, `pk_qwen3_1.7B_8bit_c15_s3`, `pk_qwen3_0.6B_8bit_c15_s1`, `pk_qwen3_0.6B_8bit_c15_s2`, `pk_qwen3_0.6B_8bit_c15_s3`, `pk_v1_3B_8bit_c30_s1`, `pk_v1_3B_8bit_c30_s2`, `pk_v1_3B_8bit_c30_s3`, `pk_parakeet_default_s1`, `pk_parakeet_default_s2`, `pk_parakeet_default_s3`, `pk_reazon_fp32_default_s1`, `pk_reazon_fp32_default_s2`, `pk_reazon_fp32_default_s3`; [ultra-picker-speed-24b.json](benchmarks/results/2026-10/ultra-picker-speed-24b.json) runs `pk_v1_24B_8bit_c30_s1`, `pk_v1_24B_8bit_c30_s2`, `pk_v1_24B_8bit_c30_s3`

Voxtral's default precision depends on the machine: this one loads fp16, and a 16GB M4,[^r15]
where fp16 does not fit, loads 8-bit, at 15.12% Japanese and 21.17% English, 6.67x[^r15]
realtime. Both replaced 4-bit, which scored 15.79% and 21.50% at 33.6x here[^r16].

The two voxtral-v1 rows (Mistral's first-generation Voxtral, Mini 3B and Small 24B, 2507)
were added on 2026-09-25 and change no default. Japanese is not among the eight
languages their card lists, and it shows: both are behind every other multilingual engine
on Japanese, largely through repetition loops. The 24B's English figure is the lowest in
the table, on three files. Both write no subtitles. See
[voxtral-v1.md](docs/benchmarks/engines/voxtral-v1.md).

The two Japanese-specialized engines were added on 2026-08-23 and change no
default. Rows marked * differ from the rest of this table in one way, recorded in
their result files: they cover the 17 Japanese files rather than all 20 recordings
(the weights are Japanese-only). Parakeet is the fastest engine measured here, 4.9x[^r2] the next fastest
(reazon on the CPU, 52.9x[^r2]). See
[engines.md](docs/benchmarks/engines.md#experiment-the-japanese-only-engines-against-the-multilingual-defaults).

Whisper is about 1.0 point more accurate on Japanese than Voxtral's fp16 default[^r3], and
only one of the three tests this project uses resolves that lead. All 6 repeat runs beat
Voxtral, and the run-distribution interval [13.87, 14.47] lies entirely below its 15.20%[^r3].
The bootstrap over *files* gives +1.03 points[^r3] with CI [-0.11, +2.25], which includes
zero, and the sign test is even: Whisper wins 8 of 17 files, p=1.00[^r3]. The file-level tests
are the ones that speak to generalization, and both find a tie. Against the 4-bit build
Voxtral shipped before, the bootstrap did resolve the lead (+1.63 points, CI [+0.34, +3.12])[^r17],
though the sign test did not (12 of 17 files, p=0.14)[^r17].

Voxtral is ~1.1x faster on this machine[^r4], needs no language hint, has better-behaved timestamps, and
reproduces byte-identically on a given machine, which is why it is the default. Whisper's ±
is a run distribution, not a confidence interval on the audio: it samples, so it gets 6 runs
while Voxtral's reruns are identical.

The two Qwen3-ASR rows were added on 2026-08-19 and change no default; both run at their
15s window. The 1.7B is third of the eight rows on Japanese (18.62%)[^r5], behind Whisper and
Voxtral, and about as fast as Voxtral at fp16 (24.6x against 23.9x)[^r5]. The 0.6B runs at 35.5x in 2.07GB[^r5], the fastest multilingual
row, ahead of Voxtral's 23.9x[^r5]; that speed is why it ships. Both **write no subtitles**, since their
timestamps are decode-window boundaries. See
[qwen3-asr.md](docs/benchmarks/engines/qwen3-asr.md).

**The English figures changed on 2026-08-19** and every earlier English number in this repo
is superseded. Reference lines were being joined with no separator, fusing a word at every
line break on the word-level path; one subtitle-shaped reference carried many such fusions
and scored far worse than its correct figure of 3.29%[^r6]. Japanese was never affected, because it
has no word spaces. See [metrics.md](docs/benchmarks/reference/metrics.md), which also explains why
plain CER is meaningless on this editorial material.

## What was measured

| lever | conclusion | detail |
|---|---|---|
| transcription delay | `2400` is worth 9 points and free. The strongest result here. | [delay.md](docs/benchmarks/delay.md) |
| Qwen3-ASR | Neither alias beats an existing default on accuracy; the 0.6B (35.5x) is the fastest multilingual engine measured, ahead of Voxtral at fp16 (23.9x). Writes no subtitles. Ships a 15s window, which ties 30s on accuracy for both sizes and is faster. | [qwen3-asr.md](docs/benchmarks/engines/qwen3-asr.md) |
| Voxtral v1 | 3B and 24B added, no default changed. Weak on Japanese (outside the card's languages); the 24B is the best English row measured. The variant selector runs the 24B at bf16 where it fits and the 3B at 8bit on 16GB Macs; 3B at 4bit loses 8 points. Pass `--language`: detection sends Japanese windows out in other languages. | [voxtral-v1.md](docs/benchmarks/engines/voxtral-v1.md) |
| Japanese-specialized engines | parakeet 25.68% at a record 261.8x; reazon-k2 fp32 30.06% on CPU. Domain-matched training does not beat Whisper's breadth here; int8 is NOT near-parity for reazon on conversational audio. | [engines.md](docs/benchmarks/engines.md#experiment-the-japanese-only-engines-against-the-multilingual-defaults) |
| chunking | 30s vs 60s is not resolvable on the corpus (+0.10, CI [-1.92, +2.06]), so choose on speed: 30s is faster on both machines measured and ships on every machine. Overlap helps only where seams are dense. Energy cuts beat VAD. | [chunking.md](docs/benchmarks/chunking.md) |
| engine choice | Against Voxtral's fp16 default on the M2 Ultra, Whisper turbo + no-condition ties on Japanese (+1.03 points, CI [-0.11, +2.25]; 8 of 17 files, sign test p=1.00); Voxtral is somewhat faster (23.9x against 21.0x) and reproducible. | [engines.md](docs/benchmarks/engines.md) |
| batch size | Per-step time is not monotonic; 2-8 are slower per step than 1 and are never chosen. | [decode-throughput.md](docs/benchmarks/decode-throughput.md) |
| input level | Quiet input silently costs ~3.0 points; `--gain auto` fixes it. | [input-level.md](docs/benchmarks/input-level.md) |
| `--prompt` | Weak, except that an instruction there costs 1.4 points on Japanese and 14 on English. | [prompt.md](docs/benchmarks/prompt.md) |
| quantization | **On Voxtral**, costs accuracy monotonically in bit width; 4-bit is last of five, 0.94 behind 8-bit and 1.09 behind fp16. The default loads fp16 where it fits and a local 8-bit conversion on 16GB Macs. `qwen3-asr`'s ladder is separate: at its 15s window 8bit is the most accurate rung on both sizes, and 4bit costs 1.2 points on the 1.7B and 2.8 on the 0.6B. `--kv-bits 8` is close to free. | [quantization.md](docs/benchmarks/quantization.md) |
| timestamps | Voxtral holds timing, Whisper places cues better. | [timestamps.md](docs/benchmarks/timestamps.md) |
| cue grouping | Two sweeps run, neither adopted, deliberately. | [cue-layout.md](docs/benchmarks/cue-layout.md) |

Source: [ultra-voxtral-delay.json](benchmarks/results/2026-10/ultra-voxtral-delay.json) runs `delay7_c30b32_d480`; [ultra-voxtral-precision-default.json](benchmarks/results/2026-10/ultra-voxtral-precision-default.json) runs `vox_wfp16_c30b128_kv8`, `vox_wfp16_c30b128_kv8_r2`, `vox_wfp16_c30b128_kv8_r3`; [ultra-whisper-sizes.json](benchmarks/results/2026-10/ultra-whisper-sizes.json) runs `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`, `whisper_turbo_nocond_r6`; [ultra-picker-speed.json](benchmarks/results/2026-10/ultra-picker-speed.json) runs `pk_qwen3_0.6B_8bit_c30_s1`, `pk_qwen3_0.6B_8bit_c30_s2`, `pk_qwen3_0.6B_8bit_c30_s3`; [ultra-mincut-inputs.json](benchmarks/results/2026-10/ultra-mincut-inputs.json) runs `mincut7_vox_c30b32_kv8`; [ultra-voxtral-v1.json](benchmarks/results/2026-10/ultra-voxtral-v1.json) runs `3b_4bit_c30`, `3b_8bit_c30`; [ultra-picker-speed.json](benchmarks/results/2026-10/ultra-picker-speed.json) runs `pk_parakeet_default_s1`, `pk_parakeet_default_s2`, `pk_parakeet_default_s3`, `pk_reazon_fp32_default_s1`, `pk_reazon_fp32_default_s2`, `pk_reazon_fp32_default_s3`; [m4-voxtral-chunking.json](benchmarks/results/2026-10/m4-voxtral-chunking.json) runs `c30b32`, `c60b16`; [ultra-input-gain.json](benchmarks/results/2026-10/ultra-input-gain.json) runs `gain7_asrecorded_c30b32_kv8`, `gain7_m12db_c30b32_kv8`; [ultra-voxtral-prompt-trap.json](benchmarks/results/2026-10/ultra-voxtral-prompt-trap.json) runs `prompt_trap7_none`, `prompt_trap7_instruction`; [ultra-voxtral-precision.json](benchmarks/results/2026-10/ultra-voxtral-precision.json) runs `vox_w4bit_c60b16_kv8`, `vox_wnvfp4_c60b16_kv8`, `vox_w8bit_c60b16_kv8`, `vox_wmxfp8_c60b16_kv8`, `vox_wfp16_c60b16_kv8`; [ultra-qwen3-precision-c15.json](benchmarks/results/2026-10/ultra-qwen3-precision-c15.json) runs `qwen3_0.6B_4bit_c15`, `qwen3_0.6B_5bit_c15`, `qwen3_0.6B_6bit_c15`, `qwen3_0.6B_8bit_c15`, `qwen3_0.6B_bf16_c15`, `qwen3_1.7B_4bit_c15`, `qwen3_1.7B_5bit_c15`, `qwen3_1.7B_6bit_c15`, `qwen3_1.7B_8bit_c15`, `qwen3_1.7B_bf16_c15`; [ultra-voxtral-kv.json](benchmarks/results/2026-10/ultra-voxtral-kv.json) runs `vox_kv0_c60b16_w4bit`, `vox_kv8_c60b16_w4bit`; [ultra-engines-jp-only.json](benchmarks/results/2026-10/ultra-engines-jp-only.json) runs `jp_reazon_int8_c30_r1`, `jp_reazon_int8_c30_r2`, `jp_reazon_int8_c30_r3`; [ultra-mincut-inputs.json](benchmarks/results/2026-10/ultra-mincut-inputs.json) runs `mincut7_vox_c30b32_kv8_ov8`; [ultra-voxtral-clip.json](benchmarks/results/2026-10/ultra-voxtral-clip.json) runs `ov30_0s`, `ov30_8s`, `ov60_0s`, `ov60_8s`, `vad_c30b32_kv8_energy`, `vad_c30b32_kv8_vad`, `vad_c60b16_kv8_energy`, `vad_c60b16_kv8_vad`; [ultra-voxtral-chunking.json](benchmarks/results/2026-10/ultra-voxtral-chunking.json) runs `ch_vad_c60b16`; [ultra-synthetic-decode.json](benchmarks/results/2026-10/ultra-synthetic-decode.json) runs `synth_ultra_4bit_r1`, `synth_ultra_4bit_r2`, `synth_ultra_4bit_r3`; [m4-synthetic-decode.json](benchmarks/results/2026-10/m4-synthetic-decode.json) runs `synth_nvfp4_r1`, `synth_nvfp4_r2`, `synth_nvfp4_r3`; [ultra-input-gain-extra.json](benchmarks/results/2026-10/ultra-input-gain-extra.json) runs `gain7_peak_c30b32_kv8`, `gain7_rms_c30b32_kv8`; [ultra-timestamps.json](benchmarks/results/2026-10/ultra-timestamps.json) runs `ts_default_r1`, `ts_default_r2`, `ts_default_r3`, `ts_0732`; [ultra-cue-layout.json](benchmarks/results/2026-10/ultra-cue-layout.json) runs `cue_sweep_union`

Supporting: [corpus.md](docs/benchmarks/reference/corpus.md) (what the material is, how to build
your own), [metrics.md](docs/benchmarks/reference/metrics.md) (which number to trust),
[determinism.md](docs/benchmarks/reference/determinism.md) (what reproduces and what does not).
[JOURNAL.md](JOURNAL.md) is the chronological log the findings were distilled from,
including the runs that produced nothing and the conclusions that were later withdrawn.
Read it for history, not for current numbers.

## Ideas tested and rejected

Check here before proposing performance work. Each of these was measured, not reasoned
about.

| idea | result |
|---|---|
| Batching the encoder | 0.91-0.98x. It is compute-bound at ~4950 FLOP/byte, so batching cannot help. <!-- not-measured: ~4950 FLOP/byte is arithmetic, 3270 GFLOP over 0.66GB of weights --> |
| `mx.compile` on the decode step | At most 2% faster at any batch size, and slower at others. |
| Reshaping to dodge the batch valley | `fold` is bit-exact but worth at most 0.4%, and up to 3% slower on 4-bit; splitting is 1.0-4.1x worse. |
| Forcing MLX's `qmv_wide` kernel at the batch valley | 15-21% *slower* at B=2-8 on a gen-14 Ultra, reproducibly. MLX's gen-15 gate is right. |
| Prefix overlap at long chunks | No benefit once seams are sparse. |
| Carrying decoder context across seams | Recovers 0.12 of the ~1.5 points seams cost, at 2x wall clock. |
| Silero VAD cut points | 1.1-2.7 points worse than energy minima, significantly. |
| Lower sampling rate (8kHz) | No speed change: the step count depends on duration (816 steps either way, 43.6s against 43.2s on the narration clip). Costs 0.95 CER points (8.35% against 7.40%). |
| Time-stretching audio 1.25x | Shortens the audio by a fifth but not the decode in proportion: 13 chunks instead of 16, 806 steps against 816, 39.6s against 43.2s of wall clock on the narration clip. Costs 2.47 CER points (9.87% against 7.40%). |
| Raising `--delay-ms` past 2400 | 2400 is the maximum the model supports. |
| Chunks longer than 60s | Slower, and no better on accuracy. |
| Extending cue ends to the next cue's start | Makes break F1 worse. |

Source: [ultra-decode-probes.json](benchmarks/results/2026-10/ultra-decode-probes.json) runs `probe_encoder_batch_ultra_4bit`, `probe_encoder_batch_ultra_nvfp4`, `probe_batch_split_ultra_4bit`, `probe_batch_split_ultra_nvfp4`, `probe_compile_ultra_4bit`, `probe_compile_ultra_nvfp4`; [ultra-qmv-wide.json](benchmarks/results/2026-10/ultra-qmv-wide.json) runs `qmv_default_r1`, `qmv_default_r2`, `qmv_wide_r1`, `qmv_wide_r2`; [ultra-voxtral-clip.json](benchmarks/results/2026-10/ultra-voxtral-clip.json) runs `carry_none`, `carry_pair`, `carry_seq`, `vad_c30b32_kv8_energy`, `vad_c30b32_kv8_vad`, `vad_c30b32_kv8_ov8_energy`, `vad_c30b32_kv8_ov8_vad`, `vad_c60b16_kv8_energy`, `vad_c60b16_kv8_vad`, `ov60_0s`, `ov60_4s`, `ov60_8s`, `chunk_c30b32_kv0`, `chunk_c90b16_kv8`, `chunk_c120b8_kv8`, `chunk_c180b8_kv8`; [ultra-cue-layout.json](benchmarks/results/2026-10/ultra-cue-layout.json) runs `cue_sweep_union`; [ultra-voxtral-clip-cli.json](benchmarks/results/2026-10/ultra-voxtral-clip-cli.json) runs `hk_c60b16_kv0_4bit`, `hk_c60b16_kv0_8k16k`, `hk_c60b16_kv0_atempo125`, `score_clip`

## Open

**CLOSED: the headline is measured.** Both rows come from saved runs on an M2 Ultra,
arms run sequentially, and each result file records the machine and its state. The current
headline uses 3 Voxtral fp16 runs at 23.9x and 6 Whisper runs at 18.5-23.2x[^r4] for its
distribution. See
[engines.md](docs/benchmarks/engines.md).

**No x-realtime figure should be taken from a busy machine.** An earlier attempt on
2026-08-06 was discarded for this reason: the host was doing unrelated GPU work throughout.
Before any speed measurement, check GPU and memory use, and never run two benchmarks
concurrently. The two corpus runners now check this themselves: they read load average and
GPU memory in use before loading a model, warn when the host is not idle, and record the
machine and its state in the result JSON, which is what the voided run's output lacked.

**Timing quality is n=7** and needs more authored subtitle tracks. None of the 13
recordings added in the final corpus growth had them.

**Cue defaults have never been tested against a reference this project did not author.**
Until there is a second, independent set, "good cue segmentation" is not something this
corpus can measure, only agreement with one editor's conventions.

**The English side is n=3.** Every English conclusion, including the prompt-instruction
result and Whisper's English win, rests on three recordings. The direction is consistent;
the magnitudes should not be quoted as general.

**Competing-runner numbers are n=1.** That is why they support "no fastest claim survives"
and nothing stronger.

**Two quantization repos could not be evaluated** (`...-6bit`, `ellamind/...-8bit-mlx`):
both ship configs mlx-audio misroutes. Low value, since a local 8-bit conversion is measured
on the corpus instead (14.78% against 4-bit's 15.72%)[^r12].

**`--compact-silence` splits by quantization** and the mechanism is a hypothesis. Off by
default.

**Cross-machine output differs, but by far less than first reported.** Identical audio,
config and weights give different output on different chips, so a config comparison must
stay on one machine. Measured over 20 files rather than one, though, 11 score identically[^r8]
and 18 agree within 0.16 points; the 5.45-point case[^r8] that established this is the shortest
file in the corpus, and the effect scales inversely with reference length. The earlier
"~1 point per file floor" was an extrapolation from n=1 and is withdrawn. <!-- not-measured: quotes the withdrawn claim, not a current figure --> See
[determinism.md](docs/benchmarks/reference/determinism.md).

**CLOSED: chunk length 30s versus 60s is settled as a non-effect.** Re-run on one machine
over all 20 files: +0.10 points, CI [-1.92, +2.06][^r9], against +1.76 at n=7[^r11]. Both chunk lengths
are indistinguishable on this material, so the choice is purely throughput and belongs to
`profiles.json`. This was the strongest candidate for an effect that more audio might
resolve, and more audio dissolved it instead.

Deliberately not open: repeat runs of Voxtral on one machine (byte-identical, verified).

A corpus-wide quantization sweep *was* skipped on the same reasoning, that the effects were
0.11-0.38 points[^r7] against the 2.04 points the 7-file subset resolves[^r14]. That reasoning was wrong for fp16 versus
4-bit, which resolved cleanly at n=20 once measured (+1.09 points, CI [+0.32, +2.07])[^r10]. The
power estimate had extrapolated an effect size from a single narration clip where the pair
happens to tie. The full five-precision ladder has since run on the corpus at 60s/B16: 4-bit 15.72%, nvfp4 15.58%, mxfp8 15.36%, 8-bit 14.78%, fp16 14.63%[^r12],
so 4-bit is last of five, and 4-bit against 8-bit is +0.94 points, CI [+0.02, +2.05][^r12]. At the shipped 30s/B128, 8-bit
and fp16 tie (15.09% against 15.20%)[^r13], so the default loads fp16 where it fits and 8-bit
on 16GB Macs; 4-bit runs only where neither fits.

[^r1]: [corpus.json](benchmarks/results/corpus.json), durations of the 20 files the headline runs score
[^r2]: [ultra-picker-speed.json](benchmarks/results/2026-10/ultra-picker-speed.json) runs `pk_parakeet_default_s1`, `pk_parakeet_default_s2`, `pk_parakeet_default_s3`, `pk_kotoba_c10_s1`, `pk_kotoba_c10_s2`, `pk_kotoba_c10_s3`, `pk_reazon_fp32_default_s1`, `pk_reazon_fp32_default_s2`, `pk_reazon_fp32_default_s3` (the next fastest, reazon fp32 at 52.9x)
[^r3]: [ultra-voxtral-precision-default.json](benchmarks/results/2026-10/ultra-voxtral-precision-default.json) runs `vox_wfp16_c30b128_kv8`, `vox_wfp16_c30b128_kv8_r2`, `vox_wfp16_c30b128_kv8_r3`; [ultra-whisper-sizes.json](benchmarks/results/2026-10/ultra-whisper-sizes.json) runs `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`, `whisper_turbo_nocond_r6` (paired on the per-file mean of the six Whisper runs, bootstrap over the 17 Japanese files, 20000 resamples, seed 0; the interval is a one-sample t-interval on the six run aggregates)
[^r4]: [ultra-voxtral-precision-default.json](benchmarks/results/2026-10/ultra-voxtral-precision-default.json) runs `vox_wfp16_c30b128_kv8`, `vox_wfp16_c30b128_kv8_r2`, `vox_wfp16_c30b128_kv8_r3`; [ultra-whisper-sizes.json](benchmarks/results/2026-10/ultra-whisper-sizes.json) runs `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`, `whisper_turbo_nocond_r6`
[^r5]: [ultra-picker-speed.json](benchmarks/results/2026-10/ultra-picker-speed.json) runs `pk_qwen3_1.7B_8bit_c15_s1`, `pk_qwen3_1.7B_8bit_c15_s2`, `pk_qwen3_1.7B_8bit_c15_s3`, `pk_qwen3_0.6B_8bit_c15_s1`, `pk_qwen3_0.6B_8bit_c15_s2`, `pk_qwen3_0.6B_8bit_c15_s3`; [ultra-voxtral-precision-default.json](benchmarks/results/2026-10/ultra-voxtral-precision-default.json) runs `vox_wfp16_c30b128_kv8`, `vox_wfp16_c30b128_kv8_r2`, `vox_wfp16_c30b128_kv8_r3`
[^r6]: [ultra-voxtral-headline.json](benchmarks/results/2026-10/ultra-voxtral-headline.json) runs `vox_default_c30b128_kv8` (per-file row of the subtitle-shaped English reference)
[^r7]: [ultra-voxtral-clip.json](benchmarks/results/2026-10/ultra-voxtral-clip.json) runs `ladder_c60b16_kv0_4bit`, `ladder_c60b16_kv0_nvfp4`, `ladder_c60b16_kv0_mxfp8`, `ladder_c60b16_kv0_8bit`, `ladder_c60b16_kv0_fp16`
[^r8]: [m4-voxtral-chunking.json](benchmarks/results/2026-10/m4-voxtral-chunking.json) runs `c30b32`; [ultra-voxtral-headline.json](benchmarks/results/2026-10/ultra-voxtral-headline.json) runs `vox_default_c30b32_kv8`
[^r9]: [m4-voxtral-chunking.json](benchmarks/results/2026-10/m4-voxtral-chunking.json) runs `c60b16`, `c30b32`
[^r10]: [ultra-voxtral-precision.json](benchmarks/results/2026-10/ultra-voxtral-precision.json) runs `vox_w4bit_c60b16_kv8`, `vox_wfp16_c60b16_kv8`
[^r11]: [ultra-mincut-inputs.json](benchmarks/results/2026-10/ultra-mincut-inputs.json) runs `mincut7_vox_c60b16_kv8`, `mincut7_vox_c30b32_kv8` (Japanese files of the 7-file subset)
[^r12]: [ultra-voxtral-precision.json](benchmarks/results/2026-10/ultra-voxtral-precision.json) runs `vox_w4bit_c60b16_kv8`, `vox_wnvfp4_c60b16_kv8`, `vox_wmxfp8_c60b16_kv8`, `vox_w8bit_c60b16_kv8`, `vox_wfp16_c60b16_kv8`
[^r13]: [ultra-voxtral-precision-default.json](benchmarks/results/2026-10/ultra-voxtral-precision-default.json) runs `vox_w8bit_c30b128_kv8`, `vox_wfp16_c30b128_kv8`
[^r14]: [ultra-resolution-floor.json](benchmarks/results/2026-10/ultra-resolution-floor.json) runs `resolution_floor` (95% half-width at n=7 from the per-file paired SD on Japanese, 2.75 points, of [ultra-voxtral-headline.json](benchmarks/results/2026-10/ultra-voxtral-headline.json) runs `vox_default_c30b128_kv8`, `vox_default_c30b128_kv8_r2`, `vox_default_c30b128_kv8_r3` against [ultra-whisper-sizes.json](benchmarks/results/2026-10/ultra-whisper-sizes.json) runs `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`, `whisper_turbo_nocond_r6`)
[^r15]: [m4-voxtral-8bit.json](benchmarks/results/2026-10/m4-voxtral-8bit.json) runs `m4_c30b32_w8bit_r1`, `m4_c30b32_w8bit_r2`, `m4_c30b32_w8bit_r3`
[^r16]: [ultra-voxtral-headline.json](benchmarks/results/2026-10/ultra-voxtral-headline.json) runs `vox_default_c30b128_kv8`, `vox_default_c30b128_kv8_r2`, `vox_default_c30b128_kv8_r3`
[^r17]: [ultra-voxtral-headline.json](benchmarks/results/2026-10/ultra-voxtral-headline.json) runs `vox_default_c30b128_kv8`; [ultra-whisper-sizes.json](benchmarks/results/2026-10/ultra-whisper-sizes.json) runs `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`, `whisper_turbo_nocond_r6` (same method as note r3)
