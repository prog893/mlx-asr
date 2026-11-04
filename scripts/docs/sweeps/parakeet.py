"""Sweep charts for docs/benchmarks/engines/parakeet.md (parakeet)."""

SWEEPS = {
    "parakeet-window": {
        "doc": "docs/benchmarks/engines/parakeet.md",
        "title": "parakeet: decode window",
        "basis": "16 scored Japanese files (60s: 17), M2 Ultra. 300s is worse on error, "
                 "throughput and memory; 60s ties 120s on error (paired -1.78, CI [-3.49, +0.44]) "
                 "but loses content on the 112s file, and 120s stays the default.",
        "xlabel": "window length", "ylabel": "",
        "scale": "linear", "unit": "s",
        "series": ["JP coverage CER", "x realtime", "peak GPU GB"],
        "rows": [(60, "| 60s |", ("24.06%", "269.8x", "4.56GB")),
                 (120, "| **120s** |", ("25.78%", "244.7x", "4.77GB")),
                 (300, "| 300s |", ("32.28%", "204.4x", "8.5GB"))],
        "source": "ultra-parakeet-window.json#parakeet_c120,parakeet_c300,parakeet_c60",
        "panels": [{"series": [0], "ylabel": "JP coverage CER %", "title": "accuracy"},
                   {"series": [1], "ylabel": "x realtime", "title": "throughput"},
                   {"series": [2], "ylabel": "peak GPU GB", "title": "memory", "zero": True}],
        "chosen": 120, "chosen_series": "JP coverage CER",
    },
}
