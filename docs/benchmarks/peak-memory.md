# Peak GPU memory per model

The number that decides whether a model runs on a given machine. Measured because the
download size does not predict it: whisper `tiny` downloads 0.07GB and peaks at 3.99GB.

## Method

Every cell in the [MODELS.md](../MODELS.md) tables is the maximum of
`mx.get_peak_memory()` over the same 20-file corpus (7.95h, 1.9 to 93 minutes per file),
reset per file, driven through the CLI at its shipped per-machine config.

    M2 Ultra 128GB (Mac14,14), 60 GPU cores, macOS 26.4.1, mlx 0.32.0
    20 cells x 20 files = 400 decodes, 8.2h wall, zero failures
    load average 0.24/core at launch, 0 swapouts/s, GPU otherwise idle
    2026-08-24

One method for all of them, which is the point of publishing a column: cells that came
from different bases cannot be compared to each other, and a reader who compares them
anyway gets a wrong answer about what fits.

This replaced a table that mixed three bases (the corpus for some cells, one 93-minute
file for the intermediate quantization rungs, three files for kotoba) while the prose
claimed two. The re-run reproduced **19 of 20 cells within 0.02GB**, so the earlier numbers
were right about the models and wrong about their provenance. Only kotoba moved, for the
reason below.

## What the number is sensitive to

**Audio length, on the Whisper family.** Its peak grows monotonically with duration and
does not plateau inside this corpus:

| `whisper --size turbo` | peak |
|---|---|
| 1.9 min | 2.56GB |
| 9.3 min | 2.74GB |
| 35.4 min | 3.46GB |
| 52.0 min | 4.05GB |
| 69.6 min | 4.54GB |
| 93.2 min | 5.52GB |

So the published whisper figures are "at a 93-minute file", not "for whisper". Every size
behaves this way, with a spread of 2.9 to 3.6GB between the shortest and longest file. On
a memory-tight machine, a 20-minute recording costs Whisper substantially less than this
table implies. The mechanism is the transcript accumulating across the sequential 30s
window loop rather than any single window.

**Barely anything, on the other three.** Voxtral rises to a plateau and stays there (5.38
to 6.76GB, flat above ~35 minutes) because its working set is the batch, which is full
once the file is long enough to fill it. qwen3-asr moves 0.08 to 0.10GB across the whole
corpus, and kotoba is flat to the last decimal:

| model | min | max | spread |
|---|---|---|---|
| `kotoba` | 2.38GB | 2.38GB | 0.00 |
| `qwen3-asr 0.6B/8bit` | 2.28GB | 2.36GB | 0.08 |
| `qwen3-asr 1.7B/8bit` | 3.95GB | 4.05GB | 0.10 |
| `voxtral 4bit` | 5.38GB | 6.77GB | 1.39 |
| `whisper turbo` | 2.56GB | 5.52GB | 2.96 |

A fixed decode window (10s for kotoba, 30s for qwen3-asr) is what makes a model's memory
predictable, and it is worth more than a small weight file when the question is whether
something fits.

## kotoba: 3.03GB was the conversion, not the decode

This cell previously read 3.03GB and now reads 2.38GB, and the old number was measured
correctly but measured the wrong thing.

kotoba ships in transformers format, so the first `--model kotoba` on a machine converts it
to MLX. That conversion `mx.load`s the entire checkpoint and writes an fp16 copy, all
through MLX, so `get_peak_memory()` counts it. The original measurement ran on a machine
where the conversion cache was cold; every run afterwards is cheaper.

Reproduced deliberately by moving the cache aside:

    conversion cache cold   3.03GB      (the old published figure)
    conversion cache warm   2.38GB      (every subsequent run)

Both are real. 2.38GB is published because it is what a user sees for all but one run in
the life of the machine, and the 3.03GB one-off is noted in MODELS.md where the conversion
is described. As a side effect: converting twice produces byte-identical output, so the
conversion is deterministic.

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

## Caveats

- **One machine.** Peak memory depends on the allocator and the GPU working set, and
  [cross-machine nondeterminism](determinism.md) is established for output, so treat these
  as an M2 Ultra measurement rather than a constant. The M4 16GB has a much smaller
  recommended working set (12.71GB), which is close enough to the voxtral fp16 figure
  (12.98GB) that fp16 is not usable there.
- **Shipped config only.** Voxtral's figures assume the Ultra's profile (60s chunks, batch
  16). A different `--chunk-seconds`/`--max-batch` pair moves them: `quantization.md`
  reports 9.36GB and 15.28GB for the same two builds at a different pair, on a different
  clip. Both are correct for what they measured.
- **n=1 per cell.** These are maxima over 20 files rather than repeated runs of one file,
  so they carry no rerun spread. Voxtral and qwen3-asr are deterministic, so a rerun would
  be identical; whisper samples, and its allocation pattern can therefore vary run to run
  by an amount not measured here.
