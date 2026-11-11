# Run results, November 2026

The file format is described in [2026-10/README.md](../2026-10/README.md#format). Saved files
are append-only: a new batch is a new group, and an existing group is edited only by marking
a row or a run `invalid` with a reason (see AGENTS.md).

## Run-groups

| group | machine | what |
|---|---|---|
| `m2-ultra-headline-remeasure` | M2 Ultra 128GB | re-measures four headline runs on an idle machine and the current corpus: whisper large-v3 r4 and turbo r6 (started with resident GPU memory) and voxtral-v1 3B and 24B at bf16 (run before a recording was replaced); the four originals are marked `invalid` |
| `m2-ultra-resolution-floor-remeasure` | none (computed from saved rows) | the resolution floor recomputed with the re-measured turbo r6; the October `m2-ultra-resolution-floor` run is marked `invalid` |
