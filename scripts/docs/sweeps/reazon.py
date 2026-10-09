"""Sweep charts for docs/benchmarks/engines/reazon.md, copied verbatim from its tables."""

SWEEPS = {
    "reazon-precision": {
        "doc": "docs/benchmarks/engines/reazon.md",
        "title": "reazon-k2: int8 against fp32",
        "basis": "17 Japanese files of the 20-file corpus, M2 Ultra (not idle; throughput "
                 "is a floor), CPU, 30s windows. int8 is faster and smaller but drops whole "
                 "phrases on conversational audio, so the more accurate fp32 is the default.",
        "xlabel": "precision",
        "scale": "category", "unit": "",
        "connect": False,
        "series": ["Japanese CER", "x realtime", "peak RSS"],
        "panels": [
            {"series": [0], "ylabel": "coverage CER % (lower is better)", "title": "accuracy"},
            {"series": [1], "ylabel": "x realtime (higher is faster)", "zero": True,
             "title": "speed"},
            {"series": [2], "ylabel": "peak RSS, GB", "zero": True, "title": "memory"},
        ],
        "width": 10.0,
        "rows": [
            ("fp32", "| **fp32 (default)** |", ("30.45%", "51.6x", "3.31GB")),
            ("int8", "| int8 |", ("36.93%", "78.9x", "2.90GB")),
        ],
        "chosen": "fp32", "chosen_series": "Japanese CER",
    },
}
