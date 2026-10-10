# Run results, October 2026

Raw results behind the October 2026 benchmark pages, so the published metrics can be
derived again, and later runs added, without re-running what is here. One JSON per
experiment run-group; each lists its runs.

## Format

A run-group JSON has `group`, `question` (what it measures), `docs` (pages it feeds) and
`runs`. Each run has:

- `run`: the run's name; `command`: the runner invocation, rebuilt from the recorded config
- `params`: every input parameter the runner recorded
- `machine`: chip, model id, RAM, GPU cores, macOS and mlx versions, and the machine's
  state when the run started (load, GPU memory in use, swap, power)
- `aggregate`: the run's own aggregates (`char` = Japanese coverage CER, `word` = English
  coverage WER, length-weighted); `measures`: run-level results such as `x_realtime` and
  peak memory
- `files`: one row per corpus file with every per-file measure the runner wrote
- `missing`: what the run did not record. A non-empty list marks a rerun candidate.
- `arm`: for a sweep that writes several arms into one file, which arm this is; the
  `command` selects it by its own argument, so it reruns exactly that arm
- `caveats`: recorded conditions that qualify the run's timing, e.g. other GPU work
  already resident when it started (taken from `machine`)

Files are named by `sha256(source id)[:12]`, with any derived suffix kept after it. The
source id is the filename stem for a recording and the video id for a downloaded public
video. [`../corpus.json`](../corpus.json) maps each id to the only metadata that may be
published (language, duration, codec, sample rate, bit depth, bitrate, reference length) and records every
processing step between the corpus copy and the model, including how the derived
worst-case inputs were cut. No transcripts and no other corpus details are kept.

Accuracy is deterministic per machine for every engine except Whisper, which samples on
fallback; speed is the run's own measurement. Runs measured while other GPU work was
resident and then re-measured are not kept; only the re-measurement is.

Per-file rows keep the reference length each scorer used (`ref_chars`, and
`kana_ref_chars` / `lenient_ref_chars` where a run reports those scorers), so every
length-weighted aggregate can be recomputed from the rows.

## Superseded audio and invalid rows

[`../revisions.json`](../revisions.json) lists corpus files whose audio was replaced, with
the reason and how the new copy was made. A per-file row that used a superseded copy
(matched by its recorded `duration_s`) carries `"invalid"`, and its run lists the file in
`invalid_files`. A run's `aggregate` and `measures` are exactly what the runner wrote, so
they include invalid rows; recompute from the rows without an `invalid` key.

So far one file: `71bb5eba3baa`. Every run that contains it used its previous copy (the
7-file groups do not contain it), which held only the
first 558.2s of a 783.5s source while the reference runs to 661.2s. Leaving it out lowers
each run's JP aggregate by about 0.2 points (Voxtral default: 16.29% to 16.07%). The runs
that contain it are rerun candidates on the new copy.

## Run-groups

| group | machine | what |
|---|---|---|
| `ultra-whisper-sizes` | M2 Ultra 128GB | every Whisper size at its default config, 3 runs each; large-v3 and turbo 6 runs each |
| `ultra-voxtral-precision` | M2 Ultra 128GB | Voxtral weight precision at 60s / batch 16 / kv8 |
| `ultra-voxtral-kv` | M2 Ultra 128GB | KV cache precision with 4-bit weights, and unquantized KV with 8-bit weights |
| `ultra-voxtral-headline` | M2 Ultra 128GB | the default config, 30s / batch 128, 3 runs; and the former 30s / batch 32 |
| `ultra-input-gain` | M2 Ultra 128GB | input gain on the 7-file subset (no machine state or timing recorded: rerun candidate) |
| `ultra-qwen3-precision` | M2 Ultra 128GB | qwen3-asr precision ladder, both sizes, 30s windows |
| `ultra-qwen3-window` | M2 Ultra 128GB | qwen3-asr window length, 7-file subset |
| `ultra-qwen3-batch` | M2 Ultra 128GB | qwen3-asr decoder batch, 15s windows |
| `ultra-mincut-inputs` | M2 Ultra 128GB | run outputs that the min_cut scoring tables rescore, 7-file subset |
| `ultra-voxtral-batch` | M2 Ultra 128GB | Voxtral end to end by max batch at 30s chunks; B32 and B64 3 runs each |
| `ultra-voxtral-precision-default` | M2 Ultra 128GB | fp16 and 8-bit weights at the default 30s / batch 128 / kv8, with the three 4-bit default runs |
| `m4-voxtral-batch` | M4 16GB | Voxtral end to end by max batch at 60s chunks: first sweep, interleaved repeats of B24/B32, exploratory and repeated B48/B64 |
| `m4-voxtral-worstcase` | M4 16GB | peak memory with every row padded to a 1.5x-target last chunk, at B32 (`worst32`) and B64 (`worst64`), on cut inputs |

## Tools

- Recompute an interval: `uv run python scripts/benchmarks/ci_from_run.py` on a runner's
  own JSON; the run-group rows carry the same per-file fields.
- Archive new runs: `scripts/benchmarks/archive_results.py` (one run-group) and
  `scripts/benchmarks/archive_corpus.py` (corpus metadata). Both need the local corpus;
  they refuse to write anything that still contains a filename or an absolute path.
