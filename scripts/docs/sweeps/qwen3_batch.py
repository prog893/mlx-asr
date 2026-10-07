"""Chart specs for docs/benchmarks/qwen3-batch.md. Values are verbatim from its table."""

SWEEPS = {
    "qwen3-batch-batch": {
        "doc": "docs/benchmarks/qwen3-batch.md",
        "title": "qwen3-asr 1.7B: decoder batch size",
        "basis": "20-file corpus, 15s windows; x realtime from a shared machine.",
        "xlabel": "batch size", "ylabel": "",
        "scale": "log", "unit": "",
        "series": ["x realtime", "Japanese CER", "English WER (3 files)"],
        "rows": [
            (1, "| **1 (default)** |", ("23.18x", "19.51%", "27.93%")),
            (2, "| 2 |", ("20.72x", "20.16%", "28.68%")),
            (4, "| 4 |", ("14.75x", "19.92%", "28.58%")),
            (8, "| 8 |", ("9.91x", "19.79%", "28.49%")),
        ],
        "panels": [
            {"series": [0], "ylabel": "x realtime", "title": "speed"},
            {"series": [1, 2], "ylabel": "error %", "title": "accuracy"},
        ],
        "chosen": 1, "chosen_series": "x realtime", "label_dy": -16,
    },
}
