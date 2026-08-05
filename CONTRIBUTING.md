# Contributing

## Hardware profiles

The most useful contribution is a measured profile for a Mac that is not yet in
`mlx_asr/profiles.json`. Run `uv run mlx-asr-bench` and open an issue
using the "Hardware profile" template. Maintainers turn a pasted bench block into
a registry entry by adding to `profiles.json`:

```json
{
  "match": {"chip": "Apple M3 Max", "ram_gb_min": 30, "ram_gb_max": 40},
  "batch": 32,
  "chunk_seconds": 60,
  "kv_bits": 8,
  "overlap_seconds": 0,
  "notes": "what is unusual about this machine's curve",
  "measured_x_realtime": 0.0,
  "peak_gb": 0.0,
  "source": "who measured it, and with which model/mlx version"
}
```

`match` is a substring test against `sysctl -n machdep.cpu.brand_string` plus a
RAM window. Keep `batch` out of the 2-8 range, and keep `peak_gb` under roughly
60% of the machine's GPU working set so a real run has room for the encoder.

## Before committing benchmark results

Benchmark JSON is committed so published numbers stay auditable, but it is
generated from real recordings and picks up hostnames, absolute paths, private
filenames, and prompt vocabulary. Scrub it first:

```bash
uv run python benchmarks/scrub_results.py benchmarks/*.json
uv run python benchmarks/scrub_results.py --check benchmarks/*.json
```

## Claiming an accuracy improvement

Accuracy differences in this project are usually smaller than the between-file
variance, and several single-clip findings did not survive a multi-file corpus
(see `RESULTS.md`). So a claim needs a paired test, not two overall numbers:

```bash
uv run python benchmarks/compare_configs.py a.srt b.srt --reference ref.srt
```

Report the difference with its confidence interval, and say how many files it was
measured over. Decoding is deterministic (greedy argmax), so repeating a run adds
no information; only more audio does.

Scoring notes: use `evals/eval_coverage.py` when a reference deliberately omits audio,
and prefer `evals/eval_cer_lenient.py` over `evals/eval_cer_kana.py` as the "fair" Japanese
number. Install the extras with `uv sync --extra eval`.
