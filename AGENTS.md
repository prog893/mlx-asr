# mlx-asr: notes for coding agents

## Every benchmark run leaves its raw results in the repo

Any run whose numbers appear anywhere (docs, README, chart data, code comments,
model notes, profiles.json, CLI help) is saved as raw per-file rows, the way PR #14 did
for October 2026. A number without saved rows is not published.

- Save into `benchmarks/results/<yyyy-mm>/<group>.json`, one JSON per experiment
  run-group, with `scripts/benchmarks/archive_results.py`:

      uv run python scripts/benchmarks/archive_results.py --corpus DIR \
          --revisions benchmarks/results/revisions.json --keep-ref-lengths \
          --group NAME --question TEXT --doc PAGE [--doc PAGE ...] \
          --out benchmarks/results/<yyyy-mm>/NAME.json RUN.json [RUN.json ...]

  Keep the runner's own JSON for every run (per-file rows, command, params, machine
  state); do not run a benchmark in a mode that skips it.
- Add each new group to that month's `README.md` run-group table (a test fails otherwise),
  and any new corpus file to `benchmarks/results/corpus.json` with `archive_corpus.py`.
- A saved file is append-only. Each new batch of runs (a different date or purpose) is a new
  group in that month's directory, never added to an existing group. The only edit an existing
  group file accepts is marking a whole run, or a row, `invalid` with a reason; nothing in it is
  deleted or rewritten. Mark a run with `scripts/benchmarks/invalidate_run.py`, which adds that
  one line and refuses any other change.
- A row or run is usable when it has no `invalid` key and ignored when it has one. The tools
  skip invalid rows and runs, and a doc that cites an invalid run fails `tests/test_sources.py`.
- Replaced corpus audio goes in `benchmarks/results/revisions.json`; rows that used the old
  copy are flagged `invalid`. Existing `supplement` rows (a replaced file re-scored and merged
  into its old run) predate the append-only rule; a new re-score is saved as a new group.
- Save only valid runs. A run measured while other GPU work was resident and then re-measured
  is marked `invalid`, with the reason naming the re-measurement, and the re-measurement is
  cited instead.
- Save before publishing: the doc's table or chart names its source run-group file and
  runs (a `Source:` line or a `"source"` field).
- `uv run pytest tests/test_results_archive.py` must pass before committing.

## Every published number cites its run data

Every measured number anywhere (docs, README, chart specs, code comments,
profiles.json, CLI help) links to the saved run-group it is computed from, and the
value is computed from that group's rows with `scripts/metrics/archive_query.py`, never
copied from an older page.

- Markdown table: a `Source:` line directly under it; prose: a footnote reference after the
  number, its definition at the bottom of the page. Paths are relative to the page; several
  groups are separated by `;`:

```
Source: [group.json](../../benchmarks/results/2026-10/group.json) runs `run_a`, `run_b`
16.21%[^p1]
[^p1]: [group.json](../../benchmarks/results/2026-10/group.json) runs `run_a`
```

- Chart specs in scripts/docs: `"source": "group.json#run_a,run_b;group2.json#run"`; bare
  row lists get an entry in the module's `SOURCES` dict.
- Code comments and docstrings in mlx_asr/: `# source: group.json#run`; profiles.json:
  a `"runs"` field per profile.
- A number that is not a measurement (a design constant, an arithmetic example) carries
  `<!-- not-measured: reason -->` or `# not-measured: reason`. No placeholder sources.
- `tests/test_sources.py` resolves every marker and fails for a page in its `DONE` list
  that has an unsourced number. A number with no saved data is rerun or removed.

## Privacy (the repo is public)

- Corpus filenames, transcripts and any corpus content stay out of the repo, including in
  commands and commit messages. Files are named by `sha256(source id)[:12]`. The results scripts
  refuse to write output that still contains a filename or an absolute path; do not
  work around that.
- Never `git add -f` an ignored file (`tests/test_repo_hygiene.py` guards this).
