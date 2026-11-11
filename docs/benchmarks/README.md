# Benchmark results

## Models

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/picker-dark.svg">
  <img alt="Error against speed for every engine at its defaults, Japanese and English panels; the best trade-offs are highlighted" src="img/picker-light.svg">
</picture>

Down and to the right is better: lower error, more speed. Highlighted points are the best trade-offs: no other engine is both faster and more accurate. The two panels disagree, so pick on the language you transcribe: on Japanese the Whisper sizes and Voxtral lead, on English the voxtral-v1 models do, though English rests on three files. A point whose label carries a note after the model name has a speed measured under different conditions from the rest; [engines.md](engines.md) and [MODELS.md](../MODELS.md) carry the numbers behind every point. Error is scored with a methodology built for Japanese and for edited rather than verbatim references ([metrics.md](reference/metrics.md)); the test material is described in [corpus.md](reference/corpus.md).

Each engine at the precision the M2 Ultra 128GB (Mac14,14) loads by default, over 20
recordings, 8.01h[^r1] of Japanese and English. Rows without English cover the 17 Japanese files,
since those engines are Japanese-only. Whisper samples, so its rows are means of 6 runs; the
greedy engines' reruns score identically.

| engine | precision | JP coverage CER | EN coverage WER | x realtime | runs |
|---|---|---|---|---|---|
| voxtral (default) | fp16 | 15.20% | 21.21% | 23.9x | 3 |
| whisper large-v3 | fp16 | 14.71% | 18.51% | 11.3x | 6 |
| whisper turbo | fp16 | 14.26% | 18.77% | 21.9x | 6 |
| qwen3-asr 1.7B | 8bit | 18.62% | 24.72% | 24.6x | 3 |
| qwen3-asr 0.6B | bf16 | 22.31% | 22.91% | 30.4x | 1 |
| voxtral-v1 24B | bf16 | 26.75% | 16.86% | 2.4x | 1 |
| voxtral-v1 3B | bf16 | 36.76% | 17.77% | 10.5x | 1 |
| kotoba | fp16 | 25.34% | n/a | 50.0x | 3 |
| parakeet | fp32 | 25.68% | n/a | 261.8x | 3 |
| reazon | fp32, CPU | 30.06% | n/a | 52.9x | 3 |

Source: [m2-ultra-voxtral-precision-default.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-precision-default.json) runs `vox_wfp16_c30b128_kv8`, `vox_wfp16_c30b128_kv8_r2`, `vox_wfp16_c30b128_kv8_r3`; [m2-ultra-whisper-sizes.json](../../benchmarks/results/2026-10/m2-ultra-whisper-sizes.json) runs `whisper_large-v3_nocond`, `whisper_large-v3_nocond_r2`, `whisper_large-v3_nocond_r3`, `whisper_large-v3_nocond_r5`, `whisper_large-v3_nocond_r6`, `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`; [m2-ultra-picker-speed.json](../../benchmarks/results/2026-10/m2-ultra-picker-speed.json) runs `pk_qwen3_1.7B_8bit_c15_s1`, `pk_qwen3_1.7B_8bit_c15_s2`, `pk_qwen3_1.7B_8bit_c15_s3`, `pk_parakeet_default_s1`, `pk_parakeet_default_s2`, `pk_parakeet_default_s3`, `pk_reazon_fp32_default_s1`, `pk_reazon_fp32_default_s2`, `pk_reazon_fp32_default_s3`; [m2-ultra-qwen3-precision-c15.json](../../benchmarks/results/2026-10/m2-ultra-qwen3-precision-c15.json) runs `qwen3_0.6B_bf16_c15`; [m2-ultra-headline-remeasure.json](../../benchmarks/results/2026-11/m2-ultra-headline-remeasure.json) runs `whisper_large-v3_nocond_r4`, `whisper_turbo_nocond_r6`, `24b_bf16_c30`, `3b_bf16_c30`; [m2-ultra-kotoba-engines.json](../../benchmarks/results/2026-10/m2-ultra-kotoba-engines.json) runs `k20_mlx_c15_r1`, `k20_mlx_c15_r2`, `k20_mlx_c15_r3`

## The levers

A lever is a thing you can change to affect a model's output or speed: the prompt,
quantization, model size, and so on. Each one has its own document, listed below with its
conclusion, largest effect first.

| document | lever | conclusion in one line |
|---|---|---|
| [delay.md](delay.md) | transcription delay | `2400` is worth 9 points and is free. The strongest result here. |
| [chunking.md](chunking.md) | chunk length, overlap, cut points | 30s vs 60s is indistinguishable at n=20, so it is a throughput choice. Overlap helps at short chunks on one clip but reversed on a corpus. VAD cut points and dropping silence both tie with the defaults. Throughput comes mostly from the batch. 30s/B32 is faster than 60s/B16 on both machines measured (44% on a 60-core GPU, 37% on a 10-core one), so 30s ships on both, and the composite `--fast` flag was removed. |
| [engines.md](engines.md) | which model | On the M2 Ultra, Whisper turbo + no-condition ties Voxtral's fp16 default on Japanese (+0.94 points, CI [-0.13, +2.08]); Voxtral is ~1.1x faster and reproducible, so it is the default. |
| [qwen3-asr.md](engines/qwen3-asr.md) | the Qwen3-ASR engine | The 1.7B is third of the eight engines on Japanese (18.62%); the 0.6B runs at 35.5x, ahead of Voxtral's 23.9x at fp16, and parakeet and reazon are faster still. Writes no subtitles (its timestamps are decode-window boundaries), and its library truncates long audio silently, which is why this project drives the chunk loop itself. Precision sensitivity differs by SIZE: the 1.7B ties across the whole ladder, the 0.6B does not. |
| [voxtral-v1.md](engines/voxtral-v1.md) | the Voxtral v1 engine (Mini 3B, Small 24B) | Behind every other multilingual engine on Japanese, which its card does not claim, largely through repetition loops; the 24B is the best English row measured. The variant selector runs the 24B at bf16 where it fits and the 3B at 8bit on 16GB Macs. Writes no subtitles. Leaving the language to the model switches Japanese windows into other languages, which coverage CER does not charge. |
| [decode-throughput.md](decode-throughput.md) | batch size | Not monotonic per step: batch 2-8 costs more per step than batch 1, though it still beats batch 1 in x realtime, and batch 12 and up beats both. The only lever anyone can reproduce without audio. |
| [qwen3-batch.md](qwen3-batch.md) | batch size on qwen3-asr | Batching whole chunks loses monotonically (2.7x slower by batch 8) for no accuracy gain, so `--max-batch` stays refused there. |
| [input-level.md](input-level.md) | `--gain` | Quiet input silently costs ~3.0 points. `auto` fixes it and is a no-op otherwise. |
| [prompt.md](prompt.md) | `--prompt` | No arm resolves a gain on either language; every arm costs English 2.0 to 75.9 WER points, and an *instruction* costs 6-14 points. |
| [quantization.md](quantization.md) | weight and KV precision | **Voxtral:** costs accuracy monotonically in bit width, reversing the earlier single-clip tie. 4-bit is last of five, 0.94 behind 8-bit and 1.09 behind fp16; 8-bit matches fp16 at 7.3GB and full speed, and the default loads fp16 where it fits and 8-bit on 16GB Macs. **`qwen3-asr`:** swept on the corpus at 15s windows; 8bit is the most accurate rung on both sizes, and 4bit costs 1.2 points on the 1.7B and 2.8 on the 0.6B. `--kv-bits 8` is close to free. |
| [timestamps.md](timestamps.md) | timestamp quality | Voxtral holds timing, Whisper places cues better. Different failure modes, reported separately. |
| [cue-layout.md](cue-layout.md) | subtitle grouping | Two sweeps run, neither adopted, on purpose. Costs 5.4 break-F1 points against the earlier published `0.7/32`. |

Source: [m2-ultra-voxtral-delay.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-delay.json) runs `delay7_c30b32_d480`; [m2-ultra-mincut-inputs.json](../../benchmarks/results/2026-10/m2-ultra-mincut-inputs.json) runs `mincut7_vox_c30b32_kv8`; [m2-ultra-voxtral-batch.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-batch.json) runs `c30b32`, `c30b32_r2`, `c30b32_r3`; [m2-ultra-voxtral-precision.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-precision.json) runs `vox_w4bit_c60b16_kv8`, `vox_w4bit_c60b16_kv8_r2`, `vox_w8bit_c60b16_kv8`, `vox_wfp16_c60b16_kv8`; [m2-ultra-voxtral-kv.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-kv.json) runs `vox_kv8_c60b16_w4bit`; [m2-ultra-voxtral-precision-default.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-precision-default.json) runs `vox_wfp16_c30b128_kv8`, `vox_wfp16_c30b128_kv8_r2`, `vox_wfp16_c30b128_kv8_r3`, `vox_w8bit_c30b128_kv8`; [m2-ultra-whisper-sizes.json](../../benchmarks/results/2026-10/m2-ultra-whisper-sizes.json) runs `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`; [m2-ultra-qwen3-batch.json](../../benchmarks/results/2026-10/m2-ultra-qwen3-batch.json) runs `qwen3batch_1.7B_8bit_c15_b1#batch 1`, `qwen3batch_1.7B_8bit_c15_b8#batch 8`; [m2-ultra-input-gain.json](../../benchmarks/results/2026-10/m2-ultra-input-gain.json) runs `gain7_asrecorded_c30b32_kv8#mode 0`, `gain7_m12db_c30b32_kv8#mode -12`; [m2-ultra-voxtral-prompt-trap.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-prompt-trap.json) runs `prompt_trap7_none`, `prompt_trap7_instruction`; [m2-ultra-qwen3-precision-c15.json](../../benchmarks/results/2026-10/m2-ultra-qwen3-precision-c15.json) runs `qwen3_1.7B_4bit_c15`, `qwen3_1.7B_8bit_c15`, `qwen3_0.6B_4bit_c15`, `qwen3_0.6B_8bit_c15`; [m2-ultra-timestamps.json](../../benchmarks/results/2026-10/m2-ultra-timestamps.json) runs `ts_0732#engine voxtral`, `ts_default_r1#engine voxtral`, `ts_default_r2#engine voxtral`, `ts_default_r3#engine voxtral`; [m4-voxtral-chunking.json](../../benchmarks/results/2026-10/m4-voxtral-chunking.json) runs `c30b32`, `c60b16`; [m2-ultra-picker-speed.json](../../benchmarks/results/2026-10/m2-ultra-picker-speed.json) runs `pk_qwen3_1.7B_8bit_c15_s1`, `pk_qwen3_1.7B_8bit_c15_s2`, `pk_qwen3_1.7B_8bit_c15_s3`, `pk_qwen3_0.6B_8bit_c15_s1`, `pk_qwen3_0.6B_8bit_c15_s2`, `pk_qwen3_0.6B_8bit_c15_s3`, `pk_parakeet_default_s1`, `pk_parakeet_default_s2`, `pk_parakeet_default_s3`, `pk_reazon_fp32_default_s1`, `pk_reazon_fp32_default_s2`, `pk_reazon_fp32_default_s3`; [m4-synthetic-decode.json](../../benchmarks/results/2026-10/m4-synthetic-decode.json) runs `synth_nvfp4_r1`, `synth_nvfp4_r2`, `synth_nvfp4_r3`; [m2-ultra-synthetic-decode.json](../../benchmarks/results/2026-10/m2-ultra-synthetic-decode.json) runs `synth_m2_ultra_4bit_r1`, `synth_m2_ultra_4bit_r2`, `synth_m2_ultra_4bit_r3`; [m2-ultra-voxtral-prompt.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-prompt.json) runs `prompt20_c30b32_kv8#none -`, `prompt20_c30b32_kv8#instruction en`, `prompt20_c30b32_kv8#instruction ja`, `prompt20_c30b32_kv8#description en`, `prompt20_c30b32_kv8#description ja`; [m2-ultra-voxtral-prompt-derived.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-prompt-derived.json) runs `prompt20_derived_freq`, `prompt20_derived_names`; [m2-ultra-headline-remeasure.json](../../benchmarks/results/2026-11/m2-ultra-headline-remeasure.json) runs `whisper_turbo_nocond_r6`

## Supporting documents

| document | what it covers |
|---|---|
| [corpus.md](reference/corpus.md) | what the test material is, in general terms, and how to build your own |
| [metrics.md](reference/metrics.md) | why coverage CER exists, which metric to trust, how to compare two configs |
| [determinism.md](reference/determinism.md) | what reproduces (one machine) and what does not (across machines, and Whisper at all) |
| [peak-memory.md](reference/peak-memory.md) | what each model costs in GPU memory, and why Whisper's figure depends on how long your audio is |

[JOURNAL.md](../../JOURNAL.md) is the chronological log the findings were distilled from,
including the runs that produced nothing and the conclusions that were later withdrawn.
Read it for history, not for current numbers.

## Read this before quoting a number

**Every English figure measured before 2026-08-19 is superseded, and only the engine
comparison has been re-measured.** Reference lines were joined with no separator, so on the
word-level path the last word of each line fused to the first word of the next. Japanese was
never affected, because it has no word spaces and the scorer strips whitespace anyway. The
damage depended entirely on where a reference broke its lines: the two plain transcripts end
every line with punctuation and had **zero** fused words, while the one subtitle-shaped
reference had 131 and scored 20.09% where the truth is 3.29%[^b2].

Corrected where re-measured: Voxtral (4-bit) English 25.24% -> **21.50%**[^b1], Whisper turbo-nocond
22.81% -> **18.77%**[^b1] (see [engines.md](engines.md)). **Not** re-measured, so read the English
column of these as wrong by about as much as the re-measured engine rows moved, direction
intact:
[delay.md](delay.md), [input-level.md](input-level.md), [chunking.md](chunking.md),
[prompt.md](prompt.md), [determinism.md](reference/determinism.md).

Those are left rather than re-run because each would cost a full corpus pass per arm, and in
every case the English side is n=3 and was already labelled directional. The offset is
shared across arms within a comparison, so paired conclusions survive; absolute English
levels do not. Japanese figures throughout these documents are unaffected and stand.


**Between-file variance here is larger than most of the config effects being measured.**
Per-file coverage error spans 6-24%[^b3] for a single unchanged config, so a 1-point
difference between two configs is a tie unless a paired test says otherwise. The per-file
paired SD of Voxtral at 4-bit against Whisper turbo-nocond (6 runs) on the Japanese files is
2.57 points[^b4], so the resolution floor, the 95% half-width of a paired difference, depends on
how much audio a given experiment used: about **1.91 points at n=7** and **1.13 at n=20**.[^b4]

**The samples are small and uneven, and they differ between findings.** These findings use
all 20 files: the engine comparison in [engines.md](engines.md), the prompt language crossing
in [prompt.md](prompt.md), both precision ladders and the KV comparison in
[quantization.md](quantization.md), the batch sweep in [qwen3-batch.md](qwen3-batch.md), and in
[chunking.md](chunking.md) the 30s-versus-60s comparison, VAD cut points, silence
compaction, the `--fast` decomposition and both batch sweeps. **The rest of the lever
results here are n=7 or a single clip** ([delay.md](delay.md), [input-level.md](input-level.md),
the timing pages, the overlap and chunk-length sweeps), because those sweeps predate the
corpus growth and re-running one costs a full corpus pass per arm. So a finding marked "not resolvable" may mean "smaller than 1.91 points"[^b4] rather than "smaller
than 1.13".[^b4] Where that matters the doc says so. Of 20 files only 3 are English, and timing results are n=7 because
only those files have authored subtitle tracks. Several single-clip findings in this project
reversed sign when a real corpus arrived, which is why single-clip and corpus results are
always labelled as such.

**A significant result on one clip means "real on this clip", not "real".** Two different
methods answer two different questions; see the comparison section of
[metrics.md](reference/metrics.md).

**Speed numbers depend on the machine's state.** Check GPU and memory use before starting,
never run two benchmarks at once, and treat any x-realtime figure without a stated machine
state as unreliable.

`run_corpus.py` and `run_whisper.py` now read the machine before loading a model, print a
warning when it is not idle, and record chip, enclosure, load average, GPU memory already
in use, **power source, Low Power Mode, and the swapout rate** into their result JSON
(`scripts/benchmarks/machine_state.py`).

Two of those were added after they were needed. **Connect a laptop to mains before an
eval**: on battery, and in Low Power Mode, macOS caps sustained clocks, so the figure
describes a power policy rather than a config. And the **swapout rate** matters where the
resident swap total does not: a host was found swapping out heavily during a run while its
swap total looked no different from an idle machine's. Free memory was low during the run
and recovered after it. The rate is a
property of now; the total is a property of uptime.
`summarize_engines.py` marks such a run `(busy)` in the throughput column itself rather
than in a footnote, since the table is what gets pasted into a findings document. Only the
timing column is affected: greedy decoding does not depend on machine state, so the accuracy
columns of such a run remain valid. Resident swap deliberately does not trigger the
warning, because macOS never reclaims it and a host that paged days ago still reports
gigabytes while idle.

## Ideas tested and rejected

Check here before proposing performance work. Each of these was measured, not reasoned
about.

| idea | result |
|---|---|
| Batching the encoder | 0.91-0.98x. It is compute-bound at ~4950 FLOP/byte, so batching cannot help. <!-- not-measured: ~4950 FLOP/byte is arithmetic, 3270 GFLOP over 0.66GB of weights --> |
| `mx.compile` on the decode step | At most 2% faster at any batch size, and slower at others. |
| Reshaping to dodge the batch valley | `fold` is bit-exact but worth at most 0.4%, and up to 3% slower on 4-bit; splitting is 1.0-4.1x worse. |
| Forcing MLX's `qmv_wide` kernel at the batch valley | 15-21% *slower* at B=2-8 on a gen-14 M2 Ultra, reproducibly. MLX's gen-15 gate is right. |
| Prefix overlap at long chunks | No benefit once seams are sparse. |
| Carrying decoder context across seams | Recovers 0.12 of the ~1.5 points seams cost, at 2x wall clock. |
| Silero VAD cut points | 1.1-2.7 points worse than energy minima, significantly. |
| Lower sampling rate (8kHz) | No speed change: the step count depends on duration (816 steps either way, 43.6s against 43.2s on the narration clip). Costs 0.95 CER points (8.35% against 7.40%). |
| Time-stretching audio 1.25x | Shortens the audio by a fifth but not the decode in proportion: 13 chunks instead of 16, 806 steps against 816, 39.6s against 43.2s of wall clock on the narration clip. Costs 2.47 CER points (9.87% against 7.40%). |
| Raising `--delay-ms` past 2400 | 2400 is the maximum the model supports. |
| Chunks longer than 60s | Slower, and no better on accuracy. |
| Extending cue ends to the next cue's start | Makes break F1 worse. |

Source: [m2-ultra-decode-probes.json](../../benchmarks/results/2026-10/m2-ultra-decode-probes.json) runs `probe_encoder_batch_m2_ultra_4bit`, `probe_encoder_batch_m2_ultra_nvfp4`, `probe_batch_split_m2_ultra_4bit`, `probe_batch_split_m2_ultra_nvfp4`, `probe_compile_m2_ultra_4bit`, `probe_compile_m2_ultra_nvfp4`; [m2-ultra-qmv-wide.json](../../benchmarks/results/2026-10/m2-ultra-qmv-wide.json) runs `qmv_default_r1`, `qmv_default_r2`, `qmv_wide_r1`, `qmv_wide_r2`; [m2-ultra-voxtral-clip.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-clip.json) runs `carry_none`, `carry_pair`, `carry_seq`, `vad_c30b32_kv8_energy`, `vad_c30b32_kv8_vad`, `vad_c30b32_kv8_ov8_energy`, `vad_c30b32_kv8_ov8_vad`, `vad_c60b16_kv8_energy`, `vad_c60b16_kv8_vad`, `ov60_0s`, `ov60_4s`, `ov60_8s`, `chunk_c30b32_kv0`, `chunk_c90b16_kv8`, `chunk_c120b8_kv8`, `chunk_c180b8_kv8`; [m2-ultra-cue-layout.json](../../benchmarks/results/2026-10/m2-ultra-cue-layout.json) runs `cue_sweep_union`; [m2-ultra-voxtral-clip-cli.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-clip-cli.json) runs `hk_c60b16_kv0_4bit`, `hk_c60b16_kv0_8k16k`, `hk_c60b16_kv0_atempo125`, `score_clip`

## Open

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
[determinism.md](reference/determinism.md).

## Scope

This is a Japanese-first evaluation. 17 of the 20 corpus files are Japanese, the defaults
were chosen against Japanese audio, and the metrics are character-level with a leniency
rule for kanji/kana respelling. English is measured on 3 files and reported separately,
never averaged in. No other language is tested here.

## Reproducing

The scripts are in `scripts/benchmarks/`, the scorers in `scripts/metrics/`. Both need the eval extra:

```bash
uv sync --extra whisper --extra eval
```

Corpus-wide accuracy, one config:

```bash
uv run python scripts/benchmarks/run_corpus.py --corpus DIR --json out.json
uv run python scripts/benchmarks/run_whisper.py --corpus DIR \
    --model mlx-community/whisper-large-v3-turbo --no-condition --json out.json
uv run python scripts/benchmarks/summarize_engines.py out.json ...
```

Per-lever sweeps:

```bash
uv run python scripts/benchmarks/sweep_overlap.py  --audio A --reference R --overlaps 0,4,8
uv run python scripts/benchmarks/sweep_precision.py --audio A --reference R      # needs >=32GB
uv run python scripts/benchmarks/sweep_gain.py     --corpus DIR
uv run python scripts/benchmarks/sweep_cues.py     --corpus DIR   # timed references only
uv run python scripts/benchmarks/sweep_min_cut.py  --corpus DIR --hyp-dir bench_out/corpus
uv run python scripts/benchmarks/ab_prompt.py      --audio A --reference R --terms "a,b,c"
sh scripts/benchmarks/run_matrix.sh A R                            # chunk x batch x kv x silence
```

Timing quality, and every alias end to end:

```bash
uv run python scripts/benchmarks/run_timing_sweep.py --corpus DIR --json out.json
uv run python scripts/benchmarks/matrix_models.py --audio A --reference R --json out.json
```

Comparisons and significance:

```bash
uv run python scripts/benchmarks/compare_configs.py a.srt b.srt --reference R  # paired, one clip
uv run python scripts/benchmarks/compare_engines.py a.json b.json             # paired, over files
uv run python scripts/benchmarks/repeat_distribution.py --runs r*.json --baseline vox.json
uv run python scripts/benchmarks/rescore_all.py --reference R bench_out/*.srt
```

Plotting the evidence behind a comparison, since an interval in a sentence hides its
shape. Needs the `eval` extra for matplotlib; files are labelled by duration, never by
name:

```bash
uv run python scripts/benchmarks/plot_evidence.py paired A.json B.json --out p.png
uv run python scripts/benchmarks/plot_evidence.py bootstrap A.json B.json --out b.png
uv run python scripts/benchmarks/plot_evidence.py runs --baseline vox.json \
    --runs w1.json w2.json w3.json --out r.png
uv run python scripts/benchmarks/plot_evidence.py sensitivity --hyp-dir DIR \
    --corpus DIR --out s.png
```

`paired` is the one to reach for first: it shows whether an aggregate is broad agreement
or two files pulling against each other, which the point estimate cannot distinguish and
which decides whether more audio would help.

The charts embedded in these documents are drawn from the published tables, not from
result files: `scripts/docs/chart_data.py` holds the figures with the document each came
from, and `tests/test_charts.py` fails if a figure no longer appears in its document.
After changing a table, update the figure there and regenerate:

```bash
uv run --extra eval python scripts/docs/gen_charts.py
```

Decode throughput, **no audio needed**, and the basis of the contributed-profile flow:

```bash
uv run mlx-asr-bench
```

Lower-level probes (encoder batching, KV length, seam-error location, step breakdown,
batch reshaping, `mx.compile`) are in `scripts/benchmarks/probes/`. Most recorded a negative
result, which is why they are kept: check
[decode-throughput.md](decode-throughput.md) before proposing performance work.

[^b1]: [m2-ultra-voxtral-headline.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-headline.json) runs `vox_default_c30b128_kv8`, `vox_default_c30b128_kv8_r2`, `vox_default_c30b128_kv8_r3`; [m2-ultra-whisper-sizes.json](../../benchmarks/results/2026-10/m2-ultra-whisper-sizes.json) runs `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`; [m2-ultra-headline-remeasure.json](../../benchmarks/results/2026-11/m2-ultra-headline-remeasure.json) runs `whisper_turbo_nocond_r6`
[^b2]: [m2-ultra-voxtral-headline.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-headline.json) runs `vox_default_c30b128_kv8` (per-file row of the subtitle-shaped English reference)
[^b3]: [m2-ultra-voxtral-headline.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-headline.json) runs `vox_default_c30b128_kv8` (Japanese per-file rows)
[^b4]: [m2-ultra-resolution-floor-remeasure.json](../../benchmarks/results/2026-11/m2-ultra-resolution-floor-remeasure.json) runs `resolution_floor`
[^r1]: [corpus.json](../../benchmarks/results/corpus.json), durations of the 20 files the headline runs score
[^r8]: [m4-voxtral-chunking.json](../../benchmarks/results/2026-10/m4-voxtral-chunking.json) runs `c30b32`; [m2-ultra-voxtral-headline.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-headline.json) runs `vox_default_c30b32_kv8`
[^r12]: [m2-ultra-voxtral-precision.json](../../benchmarks/results/2026-10/m2-ultra-voxtral-precision.json) runs `vox_w4bit_c60b16_kv8`, `vox_wnvfp4_c60b16_kv8`, `vox_wmxfp8_c60b16_kv8`, `vox_w8bit_c60b16_kv8`, `vox_wfp16_c60b16_kv8`
