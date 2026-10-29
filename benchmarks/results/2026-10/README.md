# Run results, October 2026

Raw results behind the October 2026 benchmark pages, so the published metrics can be
derived again, and later runs added, without re-running what is here. One JSON per
experiment run-group; each lists its runs.

## Format

A run-group JSON has `group`, `question` (what it measures), `docs` (pages it feeds) and
`runs`. Each run has:

- `run`: the run's name; `command`: the runner invocation, rebuilt from the recorded config,
  ending with a relative `--json` output. Paths relative to the repo root (such as
  `bench_out/corpus_all`) replay as written; `<local>/` stands for a machine-local parent
  directory that was removed, so point it at your own copy before replaying
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
- `supplements`: later reruns of the same config on a few files, used to add or replace a
  corpus file without rerunning the whole corpus. Each has its own `command`, `machine`
  and `measures`, a `canary` (files the run already held, reproduced exactly) and the
  files it `added`; an added row carries `supplement`, the index of its entry. Its timing
  comes from a different session than the rest of the run. Whisper samples on fallback,
  so its canary may differ; the difference is kept in `canary_drift`
- `caveats`: recorded conditions that qualify the run's timing, e.g. other GPU work
  already resident when it started (taken from `machine`)

Files are named by `sha256(source id)[:12]`, with any derived suffix kept after it. The
source id is a private identifier kept with the local corpus, usually the filename stem.
[`../corpus.json`](../corpus.json) maps each id to the only metadata that may be
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

So far one file: `71bb5eba3baa`. Runs made before its revision (2026-10-11) used the previous
copy (the 7-file groups do not contain it), which held only the first 558.2s of a 783.5s
source while the reference runs to 661.2s; their rows for it are flagged `invalid`, and
leaving them out lowers such a run's JP aggregate by about 0.2 points (Voxtral at 4-bit:
16.29% to 16.07%[^r1]). Runs made after the revision, whose rows record the file at 783.5s,
used the current copy and are valid. Pre-revision runs that contain it are rerun
candidates on the new copy.

## Run-groups

| group | machine | what |
|---|---|---|
| `ultra-whisper-sizes` | M2 Ultra 128GB | every Whisper size at its default config, 3 runs each; large-v3 and turbo 6 runs each |
| `ultra-voxtral-precision` | M2 Ultra 128GB | Voxtral weight precision at 60s / batch 16 / kv8; 4-bit twice (`_r2`, 2026-10-27) |
| `ultra-voxtral-kv` | M2 Ultra 128GB | KV cache precision with 4-bit weights, and unquantized KV with 8-bit weights |
| `ultra-voxtral-headline` | M2 Ultra 128GB | the default config, 30s / batch 128, 3 runs; and the former 30s / batch 32 |
| `ultra-input-gain` | M2 Ultra 128GB | input gain on the 7-file subset (no machine state or timing recorded: rerun candidate) |
| `ultra-qwen3-precision` | M2 Ultra 128GB | qwen3-asr precision ladder, both sizes, 30s windows |
| `ultra-qwen3-precision-c15` | M2 Ultra 128GB | qwen3-asr precision ladder, both sizes, 15s windows (the default since 2026-10-27), same flags as the 30s ladder |
| `ultra-qwen3-window` | M2 Ultra 128GB | qwen3-asr window length, 7-file subset |
| `ultra-qwen3-batch` | M2 Ultra 128GB | qwen3-asr decoder batch, 15s windows |
| `ultra-mincut-inputs` | M2 Ultra 128GB | run outputs that the min_cut scoring tables rescore, 7-file subset |
| `ultra-voxtral-batch` | M2 Ultra 128GB | Voxtral end to end by max batch at 30s chunks; B32 and B64 3 runs each |
| `ultra-voxtral-precision-default` | M2 Ultra 128GB | fp16 (3 runs, `_r2` and `_r3` from 2026-10-28) and 8-bit weights at 30s / batch 128 / kv8, with the three 4-bit runs; fp16 is the Ultra default since 2026-10-28 |
| `m4-voxtral-batch` | M4 16GB | Voxtral end to end by max batch at 60s chunks: first sweep, interleaved repeats of B24/B32, exploratory and repeated B48/B64 |
| `m4-voxtral-worstcase` | M4 16GB | peak memory with every row padded to a 1.5x-target last chunk, at B32 (`worst32`) and B64 (`worst64`), on cut inputs |
| `m4-voxtral-8bit` | M4 16GB | Voxtral 8bit (converted on first use) at 30s / batch 32 / kv8 over the 20-file corpus, 3 runs; the 16GB default since 2026-10-28 |
| `m4-voxtral-8bit-conversion` | M4 16GB | the first-use 8bit conversion compared tensor by tensor with the CLI-recipe build |
| `m4-voxtral-chunking` | M4 16GB | 30s / batch 32 (3 runs, `_r2` and `_r3` from 2026-10-27) against 60s / batch 16 over the 20-file corpus (the 60s run with its supplement) |
| `m4-synthetic-decode` | M4 16GB | synthetic decode throughput and peak memory by batch (`mlx-asr-bench`, nvfp4, 400 steps), 3 runs; per-batch `rows`, no corpus files |
| `ultra-picker-speed` | M2 Ultra 128GB | each picker engine at its default config, 3 fresh runs on the corrected corpus: qwen3-asr 1.7B and 0.6B 8-bit (at 30s, and at 15s from 2026-10-27), voxtral-v1 3B 8-bit, parakeet, reazon fp32, kotoba at 10s |
| `ultra-picker-speed-24b` | M2 Ultra 128GB | voxtral-v1 24B 8-bit at 30s windows, 3 fresh runs on the corrected corpus |
| `ultra-voxtral-v1` | M2 Ultra 128GB | voxtral-v1 3B and 24B at 4-bit, 8-bit and bf16, 20-file corpus, 30s windows, and 3B bf16 with auto language; measured on the previous copy of 71bb5eba3baa (its rows are invalid, no supplement) |
| `ultra-voxtral-v1-window` | M2 Ultra 128GB | voxtral-v1 3B bf16 by window (15 / 60 / 120s) and 3B 8-bit determinism, 7-file subset |
| `ultra-parakeet-window` | M2 Ultra 128GB | parakeet at its default window and at 300s, 20-file corpus; August 2026 runs recovered from the committed legacy results (legacy ids mapped to the saved-results ids); and at 60s (October 2026) |
| `ultra-reazon-precision` | M2 Ultra 128GB | reazon-k2 fp32 and int8, 20-file corpus; August 2026 runs recovered from the committed legacy results |
| `ultra-voxtral-chunking` | M2 Ultra 128GB | VAD boundaries, silence compaction at four weight precisions, and chunk / batch / overlap combinations, 20-file corpus |
| `ultra-voxtral-delay` | M2 Ultra 128GB | streaming delay 480 and 960ms at 30s / batch 32, 7-file subset |
| `ultra-whisper-condition` | M2 Ultra 128GB | large-v3 and turbo, condition on previous text on and off, runs r2 and r3, 7-file subset |
| `ultra-voxtral-prompt` | M2 Ultra 128GB | prompt content and language arms that run without topic or term text, 20-file corpus |
| `ultra-qwen3-loops` | M2 Ultra 128GB | qwen3-asr 1.7B bf16 on the 7-file subset; the batch-1 per-window token budget; the per-window budget on the 1553s file, and the library's whole-file budget (8192, 19950) on it <!-- not-measured: 8192 and 19950 are token-budget settings --> |
| `ultra-kotoba-windows` | M2 Ultra 128GB | kotoba-whisper on the MLX chunked driver at 10s (3 runs), 20s and 30s windows (3 runs each); kotoba samples on fallback like Whisper |
| `ultra-whisper-runners` | M2 Ultra 128GB | competing Whisper runners on the 935s clip, 3 runs each (whisper.cpp 1.9.5 turbo and large-v3, faster-whisper 1.2.1 on CPU, batched and unbatched), plain CER |
| `ultra-voxtral-clip` | M2 Ultra 128GB | the 935s clip: chunk length, overlap, VAD, carry-context, silence compaction, weight-precision ladder; plain CER (the clip's reference is complete); the clip runners record no machine state |
| `ultra-voxtral-prompt-trap` | M2 Ultra 128GB | no prompt against an instruction prompt, 7-file subset, 30s / batch 32 / kv8 |
| `ultra-voxtral-prompt-overlap-clip` | M2 Ultra 128GB | prompt and overlap on the 935s clip and on its first 180s (`clip180` in corpus.json `derived`) |
| `ultra-input-gain-extra` | M2 Ultra 128GB | gain modes peak, rms and -6dB, 7-file subset (no machine state or timing recorded: rerun candidate for speed) |
| `ultra-whisper-language` | M2 Ultra 128GB | turbo at library defaults: autodetect on the 7-file subset, forced Japanese on its two English files, 3 runs each |
| `ultra-engines-narration` | M2 Ultra 128GB | the 935s clip: Voxtral 60s / batch 16 / kv8, every Whisper size, kotoba at 10s and 20s windows, 3 runs each |
| `ultra-engines-jp-only` | M2 Ultra 128GB | reazon int8 (3 runs) and fp32 over the 20-file corpus, scored on the 17 Japanese files |
| `ultra-timestamps` | M2 Ultra 128GB | cue timing against the authored subtitles on the 7 timed files: shipped cues (3 runs) and 0.7s / 32 chars |
| `ultra-cue-layout` | M2 Ultra 128GB | cue layout grid over the 7 timed files from one Voxtral decode; grid points under `rows` |
| `ultra-synthetic-decode` | M2 Ultra 128GB | synthetic decode throughput and peak memory by batch (`mlx-asr-bench`, 4-bit, 400 steps, batches 1-128), 3 runs |
| `ultra-text-vs-srt` | M2 Ultra 128GB | token text against joined SRT text on the 935s clip for six CLI configs; scores under `rows` |
| `ultra-decode-probes` | M2 Ultra 128GB | decode-throughput probes (KV length, batch split, encoder batching, `mx.compile`), nvfp4 and 4-bit; the probes print tables, kept under `stdout` |
| `m4-decode-probes` | M4 16GB | the same probes on the M4, nvfp4 (KV length at batch 16, batch split, encoder batching, `mx.compile`); tables under `stdout` |
| `m4-stage-timing` | M4 16GB | CLI encode / decode / load split on the 935s clip at 60s / batch 16, no KV quantization, nvfp4, 3 runs |
| `m4-voxtral-clip-cli` | M4 16GB | CLI decodes of the 935s clip at 60s / batch 16, KV unquantized and kv8, 4-bit and nvfp4, with `--stats-json`; CER from `m4_score_clip` |
| `m4-voxtral-clip` | M4 16GB | the 935s clip: overlap 0 / 4 / 8s at 30s / batch 32 (4-bit), silence compaction off and on at 60s / batch 16 and 30s / batch 32 (nvfp4 and 4-bit) |
| `m4-text-vs-srt` | M4 16GB | token text against joined SRT text on the 935s clip for the six CLI configs of `ultra-text-vs-srt`, 4-bit; scores under `rows` |
| `m4-cross-machine-7file` | M4 16GB | Voxtral 30s / batch 32 / kv8 on the 7-file subset, 4-bit (the command of `ultra-mincut-inputs` `mincut7_vox_c30b32_kv8`) and nvfp4 |
| `ultra-stage-timing` | M2 Ultra 128GB | CLI encode / decode / load split on the 935s clip at 60s / batch 16, no KV quantization, 4-bit and nvfp4, 3 runs; `stdout` and the `--stats-json` |
| `ultra-peak-memory-kotoba` | M2 Ultra 128GB | kotoba peak memory per file through the CLI, 20-file corpus |
| `ultra-voxtral-prompt-derived` | M2 Ultra 128GB | topic and terms prompts derived from the references (`derive_prompt_terms.py`, variants freq and names), Japanese and English, 20-file corpus; the texts name corpus content and are not saved (configs hold a placeholder) |
| `ultra-qmv-wide` | M2 Ultra 128GB | decode steps/s at batch 1-8, default `qmv` against `qmv_wide` forced with `MLX_METAL_GPU_ARCH=applegpu_g15d`, two interleaved runs each |
| `ultra-kotoba-engines-clip` | M2 Ultra 128GB | kotoba on the 935s clip: MLX chunked driver against the authors' transformers pipeline, windows 10 / 15 / 20 / 30s, 3 runs each |
| `ultra-kotoba-engines` | M2 Ultra 128GB | kotoba over the 20-file corpus: MLX against the transformers pipeline at 15s (3 runs each), the pipeline at 10 / 20 / 30s, v2.0 in the pipeline |
| `ultra-profile-stages` | M2 Ultra 128GB | per-stage wall clock on the 935s clip, 4-bit at 60s, 3 runs; probe tables under `stdout` |
| `ultra-voxtral-clip-cli` | M2 Ultra 128GB | the 935s clip through the CLI: chunk length, overlap, KV precision, weight-precision ladder, fp16 at the tuned config, instruction prompt, delay, 8kHz round trip and 1.25x stretch (`clip8k`, `clipstretch125`), -14dB input (`clipattn14`) with gain none and auto; `--stats-json` per decode and the `score_clip` rescore (plain, kana, lenient CER) |
| `ultra-voxtral-clip-paired` | M2 Ultra 128GB | paired region comparisons (`compare_configs.py` stdout) of the `ultra-voxtral-clip-cli` decodes and of the `ultra-voxtral-clip` VAD hypotheses; edit-type counts, seam errors, the gain auto no-op |
| `ultra-derived-measures` | M2 Ultra 128GB | model-free counts (stdout JSON of the queue's derive scripts): VAD cost and cut placement, silence compaction, chunk peak levels, reference lengths, prompt-arm and voxtral-v1 text counts, clamped cues; and the min_cut sweep |
| `ultra-whisper-sizes-7file` | M2 Ultra 128GB | Whisper tiny, base, small, large-v2 at library defaults on the 7-file subset, 3 runs each; turbo greedy, 1 run; tiny autodetect, 3 runs |
| `ultra-kotoba-sequential` | M2 Ultra 128GB | kotoba through mlx-whisper's sequential driver, npz port and mlx_asr conversion, conditioning on and off, 5 Japanese files of the 7-file subset and the 935s clip, 3 runs each (one conditioned run hung and is not kept) |
| `ultra-kotoba-determinism` | M2 Ultra 128GB | kotoba MLX chunked driver at 10s, the 112s file decoded 5 times (no machine state recorded) |
| `ultra-kotoba-conversion` | M2 Ultra 128GB | kotoba CLI peak memory on the 112s file, cold (with first-use conversion) and warm; v2.0 against v2.2 tensor diff |
| `ultra-whisper-runners-cpu` | M2 Ultra 128GB | `time -p` totals of the unbatched faster-whisper runs in `ultra-whisper-runners` (CPU share) |
| `ultra-reazon-window` | M2 Ultra 128GB | reazon fp32 with one 120s window over the 112s file |
| `ultra-qwen3-determinism` | M2 Ultra 128GB | qwen3-asr 1.7B 8-bit at 30s, 7-file subset, every file decoded twice |
| `ultra-qwen3-batch-budget` | M2 Ultra 128GB | qwen3-asr 1.7B 8-bit, batch 4, 15s windows, the library's file-wide token budget, on the 883s file |
| `ultra-resolution-floor` | (computed) | per-file paired SD and 95% half-width by corpus size, from the saved Voxtral default and Whisper turbo-nocond runs (`scripts/metrics/resolution_floor.py`) |
| `ultra-voxtral-clip-recheck` | M2 Ultra 128GB | model-free rechecks on the 935s clip (2026-10-28): 30s against 60s with kv8 on both sides, and the auto-gain no-op by `cmp` exit status and SHA-256 |
| `ultra-worstcase-split` | M2 Ultra 128GB | the worst32 cut split at 60s: 32 chunks (canary), last and longest chunk against the 90s worst case |
| `ultra-voxtral-clip-batch1` | M2 Ultra 128GB | Voxtral CLI on the 935s clip at 30s / kv8, batch 1 against a same-session batch 32 |
| `ultra-voxtral-v1-conversion` | M2 Ultra 128GB | voxtral-v1 3B 8-bit, first-use conversion against convert_drop_source.py, tensor by tensor |
| `ultra-voxtral-prompt-overlap-corpus` | M2 Ultra 128GB | prompt x overlap on the 935s clip through run_corpus.py: none, fixed description, derived term list (redacted), at 0s and 4s overlap |

Source: [../corpus.json](../corpus.json) (file durations and derived cuts named in the table)

## Tools

- Recompute an interval: `uv run python scripts/benchmarks/ci_from_run.py` on a runner's
  own JSON; the run-group rows carry the same per-file fields.
- Save new runs: `scripts/benchmarks/archive_results.py` (one run-group) and
  `scripts/benchmarks/archive_corpus.py` (corpus metadata). Both need the local corpus;
  they refuse to write anything that still contains a filename or an absolute path.

[^r1]: [ultra-voxtral-headline.json](ultra-voxtral-headline.json) runs `vox_default_c30b128_kv8` (the aggregate over its rows with and without the invalid rows of `71bb5eba3baa`, before its supplement)
