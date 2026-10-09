"""Sweep charts for docs/benchmarks/engines/parakeet.md (parakeet)."""

SWEEPS = {
    "parakeet-window": {
        "doc": "docs/benchmarks/engines/parakeet.md",
        "title": "parakeet: decode window",
        "basis": "17 Japanese files, M2 Ultra with another GPU client resident, so "
                 "throughput is a floor. 60s was not run at corpus scale: it drops "
                 "content on a single file. 120s is the default because it has the "
                 "lowest error, the highest throughput and the lower peak memory of the "
                 "two windows measured.",
        "xlabel": "window length", "ylabel": "",
        "scale": "linear", "unit": "s",
        "series": ["JP coverage CER", "x realtime", "peak GPU GB"],
        "rows": [(120, "| **120s** |", ("26.19%", "244.6x", "4.77GB")),
                 (300, "| 300s |", ("32.60%", "204.4x", "8.5GB"))],
        "panels": [{"series": [0], "ylabel": "JP coverage CER %", "title": "accuracy"},
                   {"series": [1], "ylabel": "x realtime", "title": "throughput"},
                   {"series": [2], "ylabel": "peak GPU GB", "title": "memory", "zero": True}],
        "chosen": 120, "chosen_series": "JP coverage CER",
    },
}
