"""Sweep charts for docs/benchmarks/japanese-only.md (parakeet and reazon-k2)."""

SWEEPS = {
    "japanese-only-parakeet-window": {
        "doc": "docs/benchmarks/japanese-only.md",
        "title": "parakeet: decode window",
        "basis": "17 Japanese files. 60s was not run at corpus scale: it drops content "
                 "on a single file. Throughput is a floor: another GPU client was "
                 "resident.",
        "xlabel": "window length", "ylabel": "",
        "scale": "linear", "unit": "s",
        "series": ["JP coverage CER", "x realtime", "peak GPU GB"],
        "rows": [(120, "| **120s** |", ("26.19%", "244.6x", "4.77GB")),
                 (300, "| 300s |", ("32.60%", "204.4x", "8.5GB"))],
        "panels": [{"series": [0], "ylabel": "JP coverage CER %"},
                   {"series": [1], "ylabel": "x realtime"},
                   {"series": [2], "ylabel": "peak GPU GB", "zero": True}],
        "chosen": 120, "chosen_series": "JP coverage CER",
    },
}
