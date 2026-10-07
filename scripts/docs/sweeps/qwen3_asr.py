"""Sweep charts for docs/benchmarks/qwen3-asr.md. Values copied verbatim from its tables."""

SWEEPS = {
    "qwen3-asr-window": {
        "doc": "docs/benchmarks/qwen3-asr.md",
        "title": "qwen3-asr 1.7B: decode window",
        "basis": "7-file subset (5 JP, 2 EN files), 8bit, one run per arm.",
        "xlabel": "window length (log scale)",
        "scale": "log", "unit": "s",
        "series": ["Japanese CER", "English WER (2 files)", "x realtime", "peak GB"],
        "panels": [
            {"series": [0, 1], "ylabel": "error % (lower is better)",
             "title": "accuracy", "legend_loc": "upper left"},
            {"series": [2], "ylabel": "x realtime (higher is faster)",
             "title": "speed"},
            {"series": [3], "ylabel": "peak GPU memory, GB", "zero": True, "title": "memory"},
        ],
        "width": 11.0,
        "rows": [
            (15, "| 15s |", ("20.04%", "30.38%", "22.3x", "3.73")),
            (30, "| **30s** |", ("19.98%", "31.38%", "19.2x", "4.05")),
            (60, "| 60s |", ("21.42%", "29.86%", "16.7x", "4.10")),
            (120, "| 120s |", ("23.55%", "34.47%", "15.5x", "4.68")),
            (300, "| 300s |", ("62.47%", "37.24%", "9.4x", "5.77")),
        ],
        "chosen": 30, "chosen_series": "Japanese CER",
        "label_dy": 14,
    },
    # Both sizes on ONE y scale, so the 1.7B's sub-point spread reads as the tie it is
    # beside the 0.6B's 7-point slope. Each table line holds one size, so each row
    # carries a value for one series only.
    "qwen3-asr-precision": {
        "doc": "docs/benchmarks/qwen3-asr.md",
        "title": "qwen3-asr: precision ladder (1.7B flat within noise, 0.6B steep)",
        "basis": "20-file corpus, 30s window; 8bit is the default on both sizes. Every 1.7B "
                 "rung's paired CI against 8bit "
                 "spans zero.",
        "xlabel": "size and precision",
        "ylabel": "Japanese coverage CER % (lower is better)",
        "scale": "category",
        "series": ["1.7B", "0.6B"],
        "width": 7.6,
        "rows": [
            ("1.7B\n4bit", "| 1.7B | 4bit |", ("20.06%", None)),
            ("1.7B\n5bit", "| 1.7B | 5bit |", ("19.19%", None)),
            ("1.7B\n6bit", "| 1.7B | 6bit |", ("19.45%", None)),
            ("1.7B\n8bit", "| 1.7B | **8bit (default)** |", ("19.33%", None)),
            ("1.7B\nbf16", "| 1.7B | bf16 |", ("19.40%", None)),
            ("0.6B\n4bit", "| 0.6B | 4bit |", (None, "30.29%")),
            ("0.6B\n5bit", "| 0.6B | 5bit |", (None, "24.84%")),
            ("0.6B\n6bit", "| 0.6B | 6bit |", (None, "25.01%")),
            ("0.6B\n8bit", "| 0.6B | **8bit (default)** |", (None, "23.27%")),
            ("0.6B\nbf16", "| 0.6B | bf16 |", (None, "23.03%")),
        ],
        "chosen": "1.7B\n8bit", "chosen_series": "1.7B",
        "chosen_label": "default (8bit)",
        "label_dy": -16,
        "legend_loc": "upper right",
    },
}
