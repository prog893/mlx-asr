# Benchmarks: what was measured, and what it concluded

One document per **lever**: a thing you can change that might make the output better or
faster. Each states its corpus, its method, the experiments, and the conclusion, so a
number can be read without reconstructing the run that produced it.

Read [metrics.md](metrics.md) and [corpus.md](corpus.md) before quoting any figure. The
metrics here are unusual for a reason, and most of the material has editorial references
that make plain CER meaningless.

## The levers, largest effect first

| document | lever | conclusion in one line |
|---|---|---|
| [delay.md](delay.md) | transcription delay | `2400` is worth 9 points and is free. The strongest result here. |
| [chunking.md](chunking.md) | chunk length, overlap, cut points | 30s vs 60s is indistinguishable at n=20, so it is a throughput choice. Overlap helps at short chunks on one clip but reversed on a corpus. VAD cut points and dropping silence both tie with the defaults. Throughput comes from the BATCH rather than the chunk length, and the right pair reverses across hardware (46% faster on a 60-core GPU, slower on a 10-core one), so both live in the per-machine profile and the composite `--fast` flag was removed. |
| [engines.md](engines.md) | which model | Whisper turbo + no-condition is ~1.5 points more accurate; Voxtral is ~1.35x faster and reproducible, so it is the default. |
| [qwen3-asr.md](qwen3-asr.md) | the Qwen3-ASR engine | Last on accuracy of the four engines; the 0.6B is the fastest here. Writes no subtitles (its timestamps are decode-window boundaries), and its library truncates long audio silently, which is why this project drives the chunk loop itself. Precision sensitivity differs by SIZE: the 1.7B ties across the whole ladder, the 0.6B does not. |
| [decode-throughput.md](decode-throughput.md) | batch size | Not monotonic. Never use batch 2-8. The only lever anyone can reproduce without audio. |
| [qwen3-batch.md](qwen3-batch.md) | batch size on qwen3-asr | Batching whole chunks loses monotonically (2.3x slower by batch 8) for no accuracy gain, so `--max-batch` stays refused there. |
| [input-level.md](input-level.md) | `--gain` | Quiet input silently costs ~3.8 points. `auto` fixes it and is a no-op otherwise. |
| [prompt.md](prompt.md) | `--prompt` | Weak and unreliable, except that an *instruction* there costs 6-14 points. |
| [quantization.md](quantization.md) | weight and KV precision | **Voxtral:** costs accuracy monotonically in bit width, reversing the earlier single-clip tie. 4-bit is last of five, 1.07 behind 8-bit and 1.30 behind fp16; 8-bit matches fp16 at 7.3GB and full speed. **`qwen3-asr`: not swept**, one 7-file bf16 check only. `--kv-bits 8` is close to free. |
| [timestamps.md](timestamps.md) | timestamp quality | Voxtral holds timing, Whisper places cues better. Different failure modes, reported separately. |
| [cue-layout.md](cue-layout.md) | subtitle grouping | Two sweeps run, neither adopted, on purpose. Costs 5.4 break-F1 points. |

## Supporting documents

| document | what it covers |
|---|---|
| [corpus.md](corpus.md) | what the test material is, in general terms, and how to build your own |
| [metrics.md](metrics.md) | why coverage CER exists, which metric to trust, how to compare two configs |
| [determinism.md](determinism.md) | what reproduces (one machine) and what does not (across machines, and Whisper at all) |
| [peak-memory.md](peak-memory.md) | what each model costs in GPU memory, and why Whisper's figure depends on how long your audio is |

## Read this before quoting a number

**Every English figure measured before 2026-08-19 is superseded, and only the engine
comparison has been re-measured.** Reference lines were joined with no separator, so on the
word-level path the last word of each line fused to the first word of the next. Japanese was
never affected, because it has no word spaces and the scorer strips whitespace anyway. The
damage depended entirely on where a reference broke its lines: the two plain transcripts end
every line with punctuation and had **zero** fused words, while the one subtitle-shaped
reference had 131 and scored 20.09% where the truth is 3.29%.

Corrected where re-measured: Voxtral English 25.24% -> **21.50%**, Whisper turbo-nocond
22.81% -> **18.34%** (see [engines.md](engines.md)). **Not** re-measured, so read the English
column of these as "wrong by roughly 3 to 4 points, direction intact":
[delay.md](delay.md), [input-level.md](input-level.md), [chunking.md](chunking.md),
[prompt.md](prompt.md), [determinism.md](determinism.md).

Those are left rather than re-run because each would cost a full corpus pass per arm, and in
every case the English side is n=3 and was already labelled directional. The offset is
shared across arms within a comparison, so paired conclusions survive; absolute English
levels do not. Japanese figures throughout these documents are unaffected and stand.


**Between-file variance here is larger than most of the config effects being measured.**
Per-file coverage error spans 11-28% for a single unchanged config, so a 1-point
difference between two configs is a tie unless a paired test says otherwise. With a
per-file paired SD of 3.50 points, the resolution floor depends on how much audio a given
experiment used: roughly **3.2 points at n=7** and **1.6 at n=20**.

**The samples are small and uneven, and they differ between findings.** Three findings use
all 20 files: the engine comparison in [engines.md](engines.md), the prompt language crossing
in [prompt.md](prompt.md), and the 30s-versus-60s chunk comparison in
[chunking.md](chunking.md). **Every other lever doc here is n=7 or a single clip**, because
those sweeps predate the corpus growth and re-running one costs a full corpus pass per arm.
So a finding marked "not resolvable" may mean "smaller than 3.2 points" rather than "smaller
than 1.6". Where that matters the doc says so. Of 20 files only 3 are English, and timing results are n=7 because
only those files have authored subtitle tracks. Several single-clip findings in this project
reversed sign when a real corpus arrived, which is why single-clip and corpus results are
always labelled as such.

**A significant result on one clip means "real on this clip", not "real".** Two different
methods answer two different questions; see the comparison section of
[metrics.md](metrics.md).

**Speed numbers require a quiesced machine.** A host doing other GPU work reports several
times lower throughput, and one run here was voided for exactly that reason. Check GPU and
memory use before starting, never run two benchmarks at once, and treat any x-realtime
figure without a stated machine state as unreliable.

`run_corpus.py` and `run_whisper.py` now read the machine before loading a model, print a
warning when it is not idle, and record chip, enclosure, load average, GPU memory already
in use, **power source, Low Power Mode, and the swapout rate** into their result JSON
(`scripts/benchmarks/machine_state.py`).

Two of those were added after they were needed. **Connect a laptop to mains before an
eval**: on battery, and in Low Power Mode, macOS caps sustained clocks, so the figure
describes a power policy rather than a config. And the **swapout rate** matters where the
resident swap total does not: a 16GB host was found swapping out ~1.4GB/s during a run
while its resident total looked no different from an idle machine's. Free memory was at
10% and recovered to 63% the moment the competing process was killed. The rate is a
property of now; the total is a property of uptime.
`summarize_engines.py` marks such a run `(busy)` in the throughput column itself rather
than in a footnote, since the table is what gets pasted into a findings document. Only the
timing column is affected: greedy decoding is unaffected by contention, so the accuracy
columns of a loaded run remain valid. Resident swap deliberately does not trigger the
warning, because macOS never reclaims it and a host that paged days ago still reports
gigabytes while idle.

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

Decode throughput, **no audio needed**, and the basis of the contributed-profile flow:

```bash
uv run mlx-asr-bench
```

Lower-level probes (encoder batching, KV length, seam-error location, step breakdown,
batch reshaping, `mx.compile`) are in `scripts/benchmarks/probes/`. Most recorded a negative
result, which is why they are kept: check
[decode-throughput.md](decode-throughput.md) before proposing performance work.
