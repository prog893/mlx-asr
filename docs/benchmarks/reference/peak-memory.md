# Reference: peak GPU memory

The peak GPU memory a model reaches during a transcription, as reported by
`mx.get_peak_memory()`. It is the number that decides whether a model runs on a given
machine, and it is published instead of download size because the download does not predict
it: whisper `tiny` downloads 0.07GB <!-- not-measured: hub file size --> and peaks at 3.98GB.[^p1] Each figure is the maximum over the
20-file corpus on one M2 Ultra, at that model's default configuration for the machine.

**Setup:** [20-file corpus](corpus.md#the-20-file-corpus), M2 Ultra 128GB. Counter reset per
file, CLI run at its default per-machine config, cell = maximum over the 20 files.

## Method

Every cell in the [MODELS.md](../../MODELS.md) tables is the maximum of
`mx.get_peak_memory()` over the same 20-file corpus, reset per file, driven through the CLI
at its default per-machine config.

    M2 Ultra 128GB (Mac14,14), 60 GPU cores, macOS 26.4.1, mlx 0.32.0
    20 cells x 20 files = 400 decodes, zero failures
    2026-08-21

All cells share one method, which is the reason for publishing a column: cells that came
from different bases cannot be compared to each other, and a reader who compares them
anyway gets a wrong answer about what fits.

This replaced a table that mixed three bases (the corpus for some cells, one 93-minute
file for the intermediate quantization rungs, three files for kotoba) while the prose
claimed two. The re-run reproduced **the earlier cells except kotoba**, so the earlier numbers
were right about the models and wrong about their provenance. Only kotoba moved, for the
reason below.

The six `voxtral-v1` cells were added on 2026-09-25 from the engine's corpus runs
(`run_qwen3.py`) rather than a CLI-per-file sweep. The basis is the same: the counter is
reset before the model loads, as the CLI does before `run()`, and the cell is the maximum
over the 20 files. The one file measured both ways read 10.91GB in each.[^p10] Like qwen3-asr its
peak is flat across the corpus, because a fixed window fixes the working set.

## What the number is sensitive to

**Audio length, on the Whisper family.** Its peak grows monotonically with duration and
does not plateau inside this corpus, whose files run from 1.9 to 93 minutes:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/peak-memory-whisper-duration-dark.svg">
  <img alt="whisper turbo peak GPU memory rises steadily from 2.54GB on a 1.9-minute file to 5.53GB on a 93.2-minute file" src="../img/peak-memory-whisper-duration-light.svg">
</picture>

**Table:** whisper turbo peak GPU memory on six corpus files of increasing length.

| `whisper --size turbo` | peak |
|---|---|
| 1.9 min | 2.54GB |
| 13.1 min | 2.76GB |
| 35.4 min | 3.51GB |
| 52.0 min | 4.03GB |
| 69.6 min | 4.54GB |
| 93.2 min | 5.53GB |

Source: [ultra-whisper-sizes.json](../../../benchmarks/results/2026-10/ultra-whisper-sizes.json) runs `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`, `whisper_turbo_nocond_r6`; [corpus.json](../../../benchmarks/results/corpus.json)

The published whisper figures therefore describe a 93-minute file rather than whisper in
general. Every size behaves this way, with a spread of 2.9 to 3.6GB[^p3] between the shortest and
longest file. On a memory-tight machine, a 20-minute recording costs Whisper substantially
less than this table implies. The mechanism is the transcript accumulating across the
sequential 30s window loop rather than any single window.

**Barely anything, on the other three.** Voxtral rises to a plateau and stays there (5.38
to 6.77GB,[^p4] flat above ~35 minutes) because its working set is the batch, which is full
once the file is long enough to fill it. qwen3-asr moves 0.01GB[^p5] across the whole
corpus, and kotoba is flat to the last decimal:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/peak-memory-range-dark.svg">
  <img alt="per-model smallest and largest per-file peak GPU memory over the corpus: kotoba 2.38 to 2.38GB, qwen3-asr 0.6B 2.06 to 2.07GB, qwen3-asr 1.7B 3.73 to 3.74GB, voxtral 4bit 5.38 to 6.77GB, whisper turbo 2.54 to 5.53GB" src="../img/peak-memory-range-light.svg">
</picture>

**Table:** smallest and largest per-file peak GPU memory over the 20 files, per model, and the gap between them.

| model | min | max | spread |
|---|---|---|---|
| `kotoba` | 2.38GB | 2.38GB | 0.00 |
| `qwen3-asr 0.6B/8bit` | 2.06GB | 2.07GB | 0.01 |
| `qwen3-asr 1.7B/8bit` | 3.73GB | 3.74GB | 0.01 |
| `voxtral 4bit` | 5.38GB | 6.77GB | 1.39 |
| `whisper turbo` | 2.54GB | 5.53GB | 2.99 |

Source: [ultra-peak-memory-kotoba.json](../../../benchmarks/results/2026-10/ultra-peak-memory-kotoba.json) runs `pm_kotoba_7f8150fdef0e`, `pm_kotoba_c18b2f5aa2a2`, `pm_kotoba_9aefd8f2352e`, `pm_kotoba_07fd66556725`, `pm_kotoba_c29657df52e9`, `pm_kotoba_cbeb19543902`, `pm_kotoba_b88caf8c0924`, `pm_kotoba_bbdabfabfb4f`, `pm_kotoba_1672083b2499`, `pm_kotoba_5b44e078553d`, `pm_kotoba_a73e94506f13`, `pm_kotoba_bb5cf5d5b247`, `pm_kotoba_0c193bf7ced2`, `pm_kotoba_57d841fe39b5`, `pm_kotoba_a773a315fb42`, `pm_kotoba_71bb5eba3baa`, `pm_kotoba_a1372c4d2900`, `pm_kotoba_c47ad127bc83`, `pm_kotoba_1da58e1b2b69`, `pm_kotoba_b3432b40c079`; [ultra-qwen3-precision-c15.json](../../../benchmarks/results/2026-10/ultra-qwen3-precision-c15.json) runs `qwen3_0.6B_8bit_c15`, `qwen3_1.7B_8bit_c15`; [ultra-voxtral-precision.json](../../../benchmarks/results/2026-10/ultra-voxtral-precision.json) runs `vox_w4bit_c60b16_kv8`; [ultra-whisper-sizes.json](../../../benchmarks/results/2026-10/ultra-whisper-sizes.json) runs `whisper_turbo_nocond`, `whisper_turbo_nocond_r2`, `whisper_turbo_nocond_r3`, `whisper_turbo_nocond_r4`, `whisper_turbo_nocond_r5`, `whisper_turbo_nocond_r6`

A fixed decode window (10s for kotoba, 15s for qwen3-asr) is what makes a model's memory
predictable, and it is worth more than a small weight file when the question is whether
something fits.

## kotoba: the old figure measured the first-use conversion

This cell previously read 3.03GB[^p11] and now reads 2.38GB.[^p6] The old number was measured
correctly, but it measured the conversion rather than the decode.

kotoba is published in transformers format, so the first `--model kotoba` on a machine
converts it to MLX. That conversion `mx.load`s the entire checkpoint and writes an fp16
copy, all through MLX, so `get_peak_memory()` counts it. The original measurement ran on a
machine where the conversion cache was cold; every run afterwards is cheaper.

Reproduced deliberately by moving the cache aside, on the 112s file through the CLI:[^p11]

**Table:** kotoba peak GPU memory on that file with the conversion cache cold and warm.

| conversion cache | peak | note |
|---|---|---|
| cold | 3.03GB | the old published figure |
| warm | 2.38GB | every subsequent run |

Source: [ultra-kotoba-conversion.json](../../../benchmarks/results/2026-10/ultra-kotoba-conversion.json) runs `pm_kotoba_cold_r112`, `pm_kotoba_warm_r112`

Both are real. 2.38GB[^p6] is published because it is what a user sees for all but one run in
the life of the machine, and the 3.03GB[^p11] one-off is noted in MODELS.md where the conversion
is described. As a side effect: converting twice produces byte-identical output, so the
conversion is deterministic.

## Caveats

- **One machine.** Peak memory depends on the allocator and the GPU working set, and
  [cross-machine nondeterminism](determinism.md) is established for output, so treat these
  as an M2 Ultra measurement rather than a constant. The M4 16GB has a much smaller
  recommended working set (12.71GB), which is close enough <!-- not-measured: device constant from mx.device_info(), not a run result --> to the voxtral fp16 figure
  (12.98GB)[^p7] that fp16 is not usable there.
- **Default config at measurement time.** Voxtral's figures were measured at the earlier
  60s/B16 profile; the current Ultra default is fp16 at 30s/B128 ([chunking.md](../chunking.md#experiment-batch-size-end-to-end)), which peaks at 17.14GB[^p14] (11.33GB[^p8] on 4-bit), and the current M4 16GB default is 8bit at 30s/B32, which peaked at 6.78GB[^p15] there (6.87GB[^p13] on 4-bit). A
  different `--chunk-seconds`/`--max-batch` pair moves them, and so does the KV setting:
  `quantization.md` reports 9.36GB[^p9] and 15.28GB[^p12] at the same 60s/B16 pair with
  unquantized KV, for the same two builds on a different clip.
  Both are correct for what they measured.
- **n=1 per cell.** These are maxima over 20 files rather than repeated runs of one file,
  so they carry no rerun spread. Voxtral and qwen3-asr are deterministic, so a rerun would
  be identical; whisper samples, and its allocation pattern can therefore vary run to run
  by an amount not measured here.

## Reproducing

`scripts/docs/gen_model_matrix.py` holds the table as data and prints the markdown; it
does not measure. The sweep is a separate script kept out of the repo because it needs the
corpus, which is not distributable ([corpus.md](corpus.md)). It is 100 lines: for each
`(model, size, quantization)`, run the CLI over every file with `--stats-json`, take the
max of `peak_memory_gb`.

Any single run reports its own figure, which is the one that matters on your hardware:

```bash
mlx-asr audio.wav --stats-json stats.json      # peak_memory_gb in the JSON
```

## Related

- [MODELS.md](../../MODELS.md): the per-model table these figures populate.
- [quantization.md](../quantization.md): peak memory per Voxtral precision, at a fixed config.
- [chunking.md](../chunking.md): chunk length per chip and batch sized from memory.
- [determinism.md](determinism.md): how far results move across machines.
- [corpus.md](corpus.md): the corpus every cell is measured over.

[^p1]: [ultra-whisper-sizes.json](../../../benchmarks/results/2026-10/ultra-whisper-sizes.json) runs `whisper_tiny_libdef`, `whisper_tiny_libdef_r2`, `whisper_tiny_libdef_r3`
[^p3]: [ultra-whisper-sizes.json](../../../benchmarks/results/2026-10/ultra-whisper-sizes.json) runs `whisper_tiny_libdef`, `whisper_base_libdef`, `whisper_small_nocond`, `whisper_medium_nocond`, `whisper_large-v2_nocond`, `whisper_large-v3_nocond`, `whisper_turbo_nocond`
[^p4]: [ultra-voxtral-precision.json](../../../benchmarks/results/2026-10/ultra-voxtral-precision.json) runs `vox_w4bit_c60b16_kv8`
[^p5]: [ultra-qwen3-precision-c15.json](../../../benchmarks/results/2026-10/ultra-qwen3-precision-c15.json) runs `qwen3_0.6B_8bit_c15`, `qwen3_1.7B_8bit_c15`
[^p6]: [ultra-peak-memory-kotoba.json](../../../benchmarks/results/2026-10/ultra-peak-memory-kotoba.json) runs `pm_kotoba_7f8150fdef0e`, `pm_kotoba_c18b2f5aa2a2`, `pm_kotoba_9aefd8f2352e`, `pm_kotoba_07fd66556725`, `pm_kotoba_c29657df52e9`, `pm_kotoba_cbeb19543902`, `pm_kotoba_b88caf8c0924`, `pm_kotoba_bbdabfabfb4f`, `pm_kotoba_1672083b2499`, `pm_kotoba_5b44e078553d`, `pm_kotoba_a73e94506f13`, `pm_kotoba_bb5cf5d5b247`, `pm_kotoba_0c193bf7ced2`, `pm_kotoba_57d841fe39b5`, `pm_kotoba_a773a315fb42`, `pm_kotoba_71bb5eba3baa`, `pm_kotoba_a1372c4d2900`, `pm_kotoba_c47ad127bc83`, `pm_kotoba_1da58e1b2b69`, `pm_kotoba_b3432b40c079`
[^p7]: [ultra-voxtral-precision.json](../../../benchmarks/results/2026-10/ultra-voxtral-precision.json) runs `vox_wfp16_c60b16_kv8`
[^p8]: [ultra-voxtral-headline.json](../../../benchmarks/results/2026-10/ultra-voxtral-headline.json) runs `vox_default_c30b128_kv8`, `vox_default_c30b128_kv8_r2`, `vox_default_c30b128_kv8_r3`
[^p9]: [ultra-voxtral-clip.json](../../../benchmarks/results/2026-10/ultra-voxtral-clip.json) runs `compact_c60b16_kv0_off`
[^p10]: [ultra-voxtral-v1.json](../../../benchmarks/results/2026-10/ultra-voxtral-v1.json) runs `3b_bf16_c30` (the corpus-run way; the CLI-per-file way has no saved run)
[^p11]: [ultra-kotoba-conversion.json](../../../benchmarks/results/2026-10/ultra-kotoba-conversion.json) runs `pm_kotoba_cold_r112`, `pm_kotoba_warm_r112`
[^p12]: [ultra-voxtral-clip-cli.json](../../../benchmarks/results/2026-10/ultra-voxtral-clip-cli.json) runs `hk_c60b16_kv0_fp16`
[^p13]: [m4-voxtral-chunking.json](../../../benchmarks/results/2026-10/m4-voxtral-chunking.json) runs `c30b32`, `c30b32_r2`, `c30b32_r3` (6.87GB on all three)
[^p14]: [ultra-voxtral-precision-default.json](../../../benchmarks/results/2026-10/ultra-voxtral-precision-default.json) runs `vox_wfp16_c30b128_kv8`, `vox_wfp16_c30b128_kv8_r2`, `vox_wfp16_c30b128_kv8_r3`
[^p15]: [m4-voxtral-8bit.json](../../../benchmarks/results/2026-10/m4-voxtral-8bit.json) runs `m4_c30b32_w8bit_r1`, `m4_c30b32_w8bit_r2`, `m4_c30b32_w8bit_r3`
