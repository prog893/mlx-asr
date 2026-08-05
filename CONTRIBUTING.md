# Contributing

## Hardware profiles, which need no audio

The most useful contribution is a measured profile for a Mac that is not yet in
`mlx_asr/profiles.json`. Good batch sizes cannot be calculated: decode throughput is not
monotonic in batch size, and on every machine measured so far batch 2-8 is *slower per
step* than batch 1. So defaults come from measurements.

```bash
uv run mlx-asr-bench
```

It drives the real decoder with random embeddings, so it needs **no audio file and no
reference transcript**, takes a couple of minutes, and prints a ready-to-paste issue body:
a markdown table for humans plus one JSON blob in a `<details>` block. It deliberately
records chip, core count, memory and mlx version, and no hostname or file paths.

Open an issue with the "Hardware profile" template and paste that output. Mention anything
unusual about the machine's state, since it changes the numbers a lot: other heavy apps,
GPU work in another process, external display, low-power mode, thermal throttling, or a
non-default `iogpu.wired_limit_mb`. **Quiesce the machine first if you can.** A host doing
other GPU work can report several times lower throughput.

Maintainers turn a pasted bench block into a registry entry:

```json
{
  "match": {"chip": "Apple M3 Max", "ram_gb_min": 36, "ram_gb_max": 36},
  "model_id": "Mac15,8",
  "batch": 32,
  "chunk_seconds": 60,
  "kv_bits": 8,
  "overlap_seconds": 0,
  "notes": "what is unusual about this machine's curve",
  "measured_x_realtime": 0.0,
  "peak_gb": 0.0,
  "source": "Apple M3 Max, 36GB, 30 GPU cores (Mac15,8), mlx 0.32.0, 4bit"
}
```

`match` is a substring test against `sysctl -n machdep.cpu.brand_string` plus a RAM
window. **`ram_gb` is nameplate GiB**, the number on the spec sheet: a 16GB Mac reports
16, not the 17 you get from dividing bytes by 1e9. Keep `batch` out of the 2-8 range, and
keep `peak_gb` under roughly 60% of the machine's GPU working set so a real run has room
for the encoder.

`model_id` (`sysctl -n hw.model`) is recorded but not matched on. It identifies the
enclosure, and therefore the cooling: a fanless MacBook Air, a MacBook Pro and a Mac
Studio can share a chip and memory yet hold different clocks across a decode that runs for
minutes, and a Studio has no battery to throttle for. If two contributed profiles ever
disagree at equal chip and RAM, this is the field that explains why, and `match` can grow a
`model_id` key at that point.

Background on why this lever behaves the way it does:
[docs/benchmarks/decode-throughput.md](docs/benchmarks/decode-throughput.md).

## Accuracy work, which needs a corpus

Building one is the hard part, and the shape matters more than the size:
[docs/benchmarks/corpus.md](docs/benchmarks/corpus.md). Short version: audio paired by
stem with `<stem>.srt` or `<stem>.vtt` if you care about timings, or
`<stem>_transcript.txt` for text only. Vet that the spoken language matches the reference
before trusting any score; more than half of this project's candidate files failed that
check.

Then:

```bash
uv sync --extra whisper --extra eval
uv run python scripts/benchmarks/run_corpus.py --corpus DIR --json out.json
```

### Claiming an improvement

Accuracy differences here are usually smaller than the between-file variance, and several
single-clip findings did not survive a multi-file corpus. So a claim needs a paired test,
not two overall numbers:

```bash
uv run python scripts/benchmarks/compare_configs.py a.srt b.srt --reference ref.srt  # one clip
uv run python scripts/benchmarks/compare_engines.py a.json b.json                    # over files
```

Report the difference with its confidence interval and say how many files it covers. Note
the two scripts answer different questions: the first says "better on this clip", the
second says "better on this kind of audio", and only the second speaks to generalization.
At n=7 the corpus used here resolves about 3.2 points.

Decoding is deterministic on a given machine, so **repeating a run adds no information**;
only more audio does. It is *not* deterministic across machines, so do not compare a run
on one chip against a run on another and attribute the difference to your change. See
[docs/benchmarks/determinism.md](docs/benchmarks/determinism.md).

Scoring: use `scripts/metrics/eval_coverage.py` when a reference deliberately omits audio, and
prefer `scripts/metrics/eval_cer_lenient.py` over `scripts/metrics/eval_cer_kana.py` as the fair Japanese
number. [docs/benchmarks/metrics.md](docs/benchmarks/metrics.md) explains when each is the
wrong choice.

## Sharing results

**Result files are not committed to this repo**, and `scripts/benchmarks/*.json` is gitignored.
They are generated from private recordings, so they carry filenames and paths, and a file
of bare numbers with no statement of what was tested, which direction is better, or what
corpus it needed is not auditable by anyone, human or otherwise.

Put results in an issue or a pull request as a **markdown table with a sentence saying what
was tested and what you conclude**, optionally with raw JSON in a `<details>` block. If a
finding changes what the project believes, it belongs in the matching document under
`docs/benchmarks/`, next to the method that produced it. Check your paste for recording
filenames, absolute paths and hostnames before posting.

## Repository layout

| path | what it is |
|---|---|
| `mlx_asr/` | the shipped CLI. Nothing else is in the wheel. |
| `scripts/metrics/` | scoring functions: reference plus hypothesis in, a number out. No model, no GPU. |
| `scripts/benchmarks/` | experiments: drive an engine over material, then call `scripts/metrics/`. |
| `docs/benchmarks/` | what was measured and what it concluded, one document per lever. |
| `JOURNAL.md` | the chronological log, including withdrawn conclusions. History, not current numbers. |

`scripts/benchmarks/` imports `scripts/metrics/` and never the reverse.
