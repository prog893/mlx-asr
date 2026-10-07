"""Sweep charts for docs/benchmarks/voxtral-v1.md, copied from its tables."""

_PRECISION_PANELS = [
    {"series": [0, 1], "ylabel": "error % (lower is better)", "title": "accuracy"},
    {"series": [2], "ylabel": "peak GPU memory (GB)", "zero": True, "title": "cost"},
]

SWEEPS = {
    "voxtral-v1-window": {
        "doc": "docs/benchmarks/voxtral-v1.md",
        "title": "voxtral-v1 3B: decode window",
        "basis": "7-file subset, bf16.",
        "xlabel": "window length", "ylabel": "error % (lower is better)",
        "scale": "log", "unit": "s",
        "series": ["Japanese CER", "English WER (2 files)"],
        "rows": [
            (15, "| 15s |", ("45.81%", "22.47%")),
            (30, "| **30s (default)** |", ("44.27%", "21.70%")),
            (60, "| 60s |", ("45.97%", "20.72%")),
            (120, "| 120s |", ("57.54%", "21.17%")),
        ],
        "chosen": 30, "chosen_series": "Japanese CER",
    },
    "voxtral-v1-precision-3b": {
        "doc": "docs/benchmarks/voxtral-v1.md",
        "title": "voxtral-v1 Mini 3B: precision",
        "basis": "20-file corpus, 30s windows.",
        "xlabel": "precision",
        "scale": "category",
        "series": ["Japanese CER", "English WER (3 files)", "peak GPU memory"],
        "rows": [
            ("4bit", "| 4bit |", ("44.54%", "18.08%", "5.25GB")),
            ("8bit", "| **8bit (default)** |", ("36.52%", "17.86%", "7.28GB")),
            ("bf16", "| bf16 |", ("37.16%", "17.77%", "10.91GB")),
        ],
        "panels": _PRECISION_PANELS,
        "chosen": "8bit", "chosen_series": "Japanese CER",
    },
    "voxtral-v1-precision-24b": {
        "doc": "docs/benchmarks/voxtral-v1.md",
        "title": "voxtral-v1 Small 24B: precision (Japanese rungs within about a point)",
        "basis": "20-file corpus, 30s windows.",
        "xlabel": "precision",
        "scale": "category",
        "series": ["Japanese CER", "English WER (3 files)", "peak GPU memory"],
        "rows": [
            ("4bit", "| 4bit |", ("28.14%", "17.89%", "16.27GB")),
            ("8bit", "| **8bit (default)** |", ("27.56%", "16.86%", "27.92GB")),
            ("bf16", "| bf16 |", ("27.10%", "16.86%", "50.08GB")),
        ],
        "panels": _PRECISION_PANELS,
        "chosen": "8bit", "chosen_series": "Japanese CER",
    },
}
