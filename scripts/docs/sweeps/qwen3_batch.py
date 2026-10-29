"""Chart specs for docs/benchmarks/qwen3-batch.md. Values are verbatim from its tables."""

SWEEPS = {
    "qwen3-batch-batch": {
        "doc": "docs/benchmarks/qwen3-batch.md",
        "title": "qwen3-asr 1.7B: decoder batch size",
        "basis": "20-file corpus, 15s windows. "
                 "Batch 1 is the fastest arm and accuracy spans 0.46 CER points, "
                 "inside noise, so the fastest arm is the default.",
        "xlabel": "batch size", "ylabel": "",
        "scale": "log", "unit": "",
        "series": ["x realtime", "Japanese CER", "English WER (3 files)"],
        "rows": [
            (1, "| **1 (default)** |", ("26.74x", "19.40%", "27.74%")),
            (2, "| 2 |", ("19.61x", "19.64%", "28.57%")),
            (4, "| 4 |", ("15.37x", "19.56%", "29.03%")),
            (8, "| 8 |", ("9.85x", "19.86%", "28.68%")),
        ],
        "source": "ultra-qwen3-batch.json#qwen3batch_1.7B_8bit_c15_b1#batch 1,"
                  "qwen3batch_1.7B_8bit_c15_b2#batch 2,qwen3batch_1.7B_8bit_c15_b4#batch 4,"
                  "qwen3batch_1.7B_8bit_c15_b8#batch 8",
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
            ("batch 1", "| batch 1 (default) |", ("17.65%", "35.1x", "3.73GB", "5.0")),
            ("batch 4\nper group", "| batch 4, budget per group |",
             ("17.43%", "25.6x", "3.82GB", "6.7")),
            ("batch 4\nfile-wide", "| batch 4, upstream's file-wide budget |",
             ("49.83%", "1.9x", "13.12GB", "37.8")),
        ],
        # The first two rows are the 883s file's rows of the batch 1 and batch 4 arms.
        "source": "ultra-qwen3-batch.json#qwen3batch_1.7B_8bit_c15_b1#batch 1,"
                  "qwen3batch_1.7B_8bit_c15_b4#batch 4;"
                  "ultra-qwen3-batch-budget.json#qwen3batch_wholefilebudget_b4_q883#batch 4",
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
