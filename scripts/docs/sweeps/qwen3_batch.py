"""Chart specs for docs/benchmarks/qwen3-batch.md. Values are verbatim from its tables."""

SWEEPS = {
    "qwen3-batch-batch": {
        "doc": "docs/benchmarks/qwen3-batch.md",
        "title": "qwen3-asr 1.7B: decoder batch size",
        "basis": "20-file corpus, 15s windows; x realtime from a shared machine. "
                 "Batch 1 is the fastest arm and accuracy spans 0.65 CER points, "
                 "inside noise, so the fastest arm is the default.",
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
            {"series": [0], "ylabel": "x realtime", "title": "speed", "zero": True},
            {"series": [1, 2], "ylabel": "error %", "title": "accuracy", "min_span": 8},
        ],
        "chosen": 1, "chosen_series": "x realtime", "label_dy": -16,
    },
    "qwen3-batch-budget": {
        "doc": "docs/benchmarks/qwen3-batch.md",
        "title": "qwen3-asr 1.7B: token budget accounting at batch 4",
        "basis": "One 14.7-minute file. Batch 1 is the default: the per-group budget "
                 "only ties it on CER while running slower, and the file-wide budget "
                 "runs away.",
        "xlabel": "arm", "ylabel": "",
        "scale": "category", "connect": False, "unit": "",
        "series": ["CER", "x realtime", "peak GPU memory", "chars/s of audio"],
        "rows": [
            ("batch 1", "| batch 1 (default) |", ("17.56%", "36.2x", "3.73GB", "5.0")),
            ("batch 4\nper group", "| batch 4, budget per group |",
             ("17.40%", "26.2x", "3.77GB", "6.7")),
            ("batch 4\nfile-wide", "| batch 4, upstream's file-wide budget |",
             ("49.86%", "1.8x", "13.09GB", "37.8")),
        ],
        "panels": [
            {"series": [0], "ylabel": "CER %", "title": "accuracy", "zero": True},
            {"series": [1], "ylabel": "x realtime", "title": "speed", "zero": True},
            {"series": [2], "ylabel": "peak GPU GB", "title": "memory", "zero": True},
            {"series": [3], "ylabel": "chars/s of audio", "title": "output rate",
             "zero": True},
        ],
        "chosen": "batch 1", "chosen_series": "CER", "width": 12,
    },
}
