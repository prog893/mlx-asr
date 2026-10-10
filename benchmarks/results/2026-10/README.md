# Run results, October 2026

Every usable run behind the October 2026 benchmark pages, kept so intervals, paired tests
and new comparisons can be recomputed without re-running anything. Each JSON is the
runner's own output with files named by their stable corpus id (`rec-NN`); the
stem-to-id map stays with the private corpus. No transcripts are kept.

Corpus: the [20-file corpus](../../../docs/benchmarks/reference/corpus.md#the-20-file-corpus)
(17 Japanese files scored by coverage CER, 3 English by coverage WER, 7.95h), or the
[7-file subset](../../../docs/benchmarks/reference/corpus.md#the-7-file-subset) where a
name starts `gain7_`, `mincut7_` or `qwen3w7_`. Each JSON records the machine and its
state when the run started. Speed is the run's own `x_realtime`; accuracy is
deterministic per machine except Whisper, which samples on fallback.

To recompute an interval: `uv run python scripts/benchmarks/ci_from_run.py FILE.json`.
To archive a new run: `scripts/benchmarks/archive_results.py --ids <map> --out <folder> RUN.json`.

Runs that were measured while other GPU work was resident and then re-measured are not
kept; only the re-measurement is.

## ultra-rerun

M2 Ultra 128GB (Mac14,14), mlx 0.32.0, mlx-audio 0.4.5, 2026-10-07 to 2026-10-11.
Re-measurement of the published tables, one arm at a time, from
`scripts/benchmarks/run_corpus.py`, `run_whisper.py`, `run_qwen3.py`, `sweep_gain.py`
and `sweep_qwen3_batch.py`.

| files | what | feeds |
|---|---|---|
| `whisper_<size>_<libdef or nocond>.json`, `_r2`, `_r3` | every Whisper size at its default config, three runs each | engines/whisper.md sizes table |
| `vox_w{4bit,8bit,fp16,mxfp8,nvfp4}_c60b16_kv8.json` | Voxtral weight ladder, 60s / batch 16 / kv8 / delay 2400 | quantization.md ladder |
| `vox_kv{0,4,8}_c60b16_w4bit.json`, `vox_w8bit_kv0_c60b16.json` | KV cache precision with 4-bit weights, and unquantized KV with 8-bit weights | quantization.md KV table |
| `vox_default_c30b32_kv8.json` | the former headline config, 30s / batch 32 | repeat of the batch sweep's B32 |
| `vox_default_c30b128_kv8.json`, `_r2`, `_r3` | the headline config, 30s / batch 128, three runs | RESULTS.md headline, picker chart |
| `gain7_*_c30b32_kv8.json` | input gain -20 / -12 / 0 / +6 dB, per-file rows in `results.<mode>.per_file` | input-level.md |
| `qwen3_{1.7B,0.6B}_{4bit,5bit,6bit,8bit,bf16}_c30.json` | qwen3-asr precision ladder, 30s windows | engines/qwen3-asr.md ladder |
| `qwen3w7_1.7B_8bit_c{15,30,60,120,300}.json` | qwen3-asr window length | engines/qwen3-asr.md window table |
| `qwen3batch_1.7B_8bit_c15_b{1,2,4,8}.json` | qwen3-asr decoder batch, 15s windows, per-file rows in `arms.<b>.per_file` | qwen3-batch.md |
| `mincut7_*.json` | inputs to the min_cut scoring tables | reference/metrics.md |

## ultra-batch

M2 Ultra 128GB, 2026-10-07 to 2026-10-10. Voxtral end to end at 30s chunks, 4-bit, kv8,
delay 2400, one file per max batch (`run_corpus.py --chunk-seconds 30 --max-batch B`).
Feeds the batch experiment in chunking.md and the Ultra batch default.

## m4-batch-r1

M4 16GB (Mac16,1), on AC power, 2026-10-08. Voxtral end to end at 60s chunks, B16 / B24 /
B32, the first clean sweep. Feeds chunking.md's M4 batch table.

## m4-batch-repeat

M4 16GB, 2026-10-09 to 2026-10-10. B32 and B24 runs 2 to 4, interleaved
(B32, B24, B32, ...) so drift spreads over both arms.

## m4-batch-explore

M4 16GB, 2026-10-10. One exploratory run each of B48 and B64 at 60s chunks.

## m4-worstcase

M4 16GB, 2026-10-09. B32 at 60s on `worst32`: a 1949s cut of a corpus file whose 32 rows
all pad to an 89.4s last chunk (2860 padded row-seconds, 99% of the theoretical worst).
Only `peak_memory_gb` is meaningful; the reference is the uncut file's, so the score is
not. Feeds the padded-worst-case note in chunking.md and `mlx_asr/hardware.py`.
