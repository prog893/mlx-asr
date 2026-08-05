---
name: Hardware profile
about: Share measured batch sizes so your Mac gets good defaults
title: "Profile: <chip> <RAM>GB"
labels: profile
---

Good batch sizes cannot be calculated: decode throughput is not monotonic in
batch size, and on every machine measured so far batch 2-8 is *slower per step*
than batch 1. So defaults come from measurements contributed by users.

## How to generate

```bash
uv run mlx-asr-bench
```

Takes a couple of minutes and needs no audio file. Paste its output below; it
records chip, core count, memory and mlx version, and deliberately no hostname or
file paths.

## Paste the bench output here

```
(paste here)
```

## Anything unusual?

Other heavy apps running, external display, low-power mode, thermal throttling,
or a non-default `iogpu.wired_limit_mb` all move these numbers. Mention them if
they applied.
