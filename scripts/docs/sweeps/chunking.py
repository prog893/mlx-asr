"""Sweep charts for docs/benchmarks/chunking.md."""

SWEEPS = {
    "chunking-length": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: chunk length (batch shrinks as chunks grow)",
        "basis": "One 935s clip, plain CER, M2 Ultra, 4-bit, no overlap; "
                 "batch changes with chunk length (shown on the x axis).",
        "xlabel": "chunk length / batch", "ylabel": "",
        "scale": "category",
        "series": ["x realtime", "CER"],
        "rows": [
            ("20s / B48", "| 20s | 48 |", ("25.6x", "12.46%")),
            ("30s / B32", "| 30s | 32 |", ("31.0x", "9.13%")),
            ("60s / B16", "| 60s | 16 |", ("21.2x", "7.37%")),
            ("90s / B16", "| 90s | 16 |", ("17.2x", "7.99%")),
            ("120s / B8", "| 120s | 8 |", ("16.1x", "7.59%")),
            ("180s / B8", "| 180s | 8 |", ("11.6x", "7.56%")),
        ],
        "panels": [
            {"series": [1], "ylabel": "CER % (lower is better)", "title": "accuracy"},
            {"series": [0], "ylabel": "x realtime (higher is faster)", "title": "speed"},
        ],
    },
    "chunking-overlap": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: prefix overlap between chunks",
        "basis": "One 935s clip, plain CER, M2 Ultra. 30s chunks at batch 32, kv8; "
                 "60s chunks measured at three overlaps only.",
        "xlabel": "overlap", "ylabel": "CER % (lower is better)",
        "scale": "linear",
        "unit": "s",
        "series": ["30s chunks", "60s chunks"],
        "rows": [
            (0, "| 0s |", ("8.73%", None)),
            (4, "| 4s |", ("7.30%", None)),
            (6, "| 6s |", ("7.61%", None)),
            (7, "| 7s |", ("7.63%", None)),
            (8, "| 8s |", ("7.25%", None)),
            (10, "| 10s |", ("7.56%", None)),
            (12, "| 12s |", ("7.80%", None)),
            (15, "| 15s |", ("11.20%", None)),
            (0, "| 0s |", (None, "7.37%")),
            (4, "| 4s |", (None, "7.59%")),
            (8, "| 8s |", (None, "8.06%")),
        ],
        "chosen": 0,
        "chosen_series": "30s chunks",
        "legend_loc": "upper center",
    },
}
