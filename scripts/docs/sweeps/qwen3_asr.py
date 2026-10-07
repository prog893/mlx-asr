"""Sweep charts for docs/benchmarks/engines/qwen3-asr.md. Values copied verbatim from its tables."""

SWEEPS = {
    "qwen3-asr-window": {
        "doc": "docs/benchmarks/engines/qwen3-asr.md",
        "title": "qwen3-asr 1.7B: decode window",
        "basis": "7-file subset (5 JP, 2 EN files), 8bit, one run per arm. 30s is the "
                 "Japanese CER optimum; 15s ties it within 0.06 points and is faster and "
                 "smaller, the memory-bound option.",
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
    # carries a value for one series only. The second panel is the paired difference
    # the table prints ("8bit minus rung", negative means 8bit is better) with its 95%
    # CI; the default rows have no paired figure, they are the zero line.
    "qwen3-asr-precision": {
        "doc": "docs/benchmarks/engines/qwen3-asr.md",
        "title": "qwen3-asr: precision ladder (1.7B flat within noise, 0.6B steep)",
        "basis": "20-file corpus, 30s window; 8bit is the default on both sizes. A rung "
                 "replaces 8bit only if its paired CI lies wholly above zero: every 1.7B "
                 "CI spans zero (a tie keeps the default), every lower 0.6B rung lies "
                 "below it (8bit wins).",
        "xlabel": "size and precision",
        "scale": "category",
        "series": ["1.7B", "0.6B", "1.7B, paired", "0.6B, paired"],
        "panels": [
            {"series": [0, 1], "ylabel": "Japanese coverage CER % (lower is better)",
             "title": "accuracy", "legend_loc": "upper right"},
            {"series": [2, 3], "ylabel": "8bit minus rung, CER points\n(below 0: 8bit better)",
             "title": "against the default (95% CI)", "zero_line": True,
             "legend_loc": "lower right"},
        ],
        "width": 12.0, "height": 3.9,
        "rows": [
            ("1.7B\n4bit", "| 1.7B | 4bit |", ("20.06%", None, "-0.73", None),
             (None, None, "[-1.76, +0.31]", None)),
            ("1.7B\n5bit", "| 1.7B | 5bit |", ("19.19%", None, "+0.15", None),
             (None, None, "[-0.37, +0.78]", None)),
            ("1.7B\n6bit", "| 1.7B | 6bit |", ("19.45%", None, "-0.11", None),
             (None, None, "[-0.53, +0.29]", None)),
            ("1.7B\n8bit", "| 1.7B | **8bit (default)** |", ("19.33%", None, None, None)),
            ("1.7B\nbf16", "| 1.7B | bf16 |", ("19.40%", None, "-0.07", None),
             (None, None, "[-0.23, +0.10]", None)),
            ("0.6B\n4bit", "| 0.6B | 4bit |", (None, "30.29%", None, "-7.02"),
             (None, None, None, "[-8.82, -5.30]")),
            ("0.6B\n5bit", "| 0.6B | 5bit |", (None, "24.84%", None, "-1.57"),
             (None, None, None, "[-2.16, -1.06]")),
            ("0.6B\n6bit", "| 0.6B | 6bit |", (None, "25.01%", None, "-1.74"),
             (None, None, None, "[-2.94, -0.86]")),
            ("0.6B\n8bit", "| 0.6B | **8bit (default)** |", (None, "23.27%", None, None)),
            ("0.6B\nbf16", "| 0.6B | bf16 |", (None, "23.03%", None, "-0.51"),
             (None, None, None, "[-1.01, +0.13]")),
        ],
        "chosen": "1.7B\n8bit", "chosen_series": "1.7B",
        "chosen_label": "default (8bit)",
        "label_dy": -16,
    },
    # Markers only: the two arms are different weights, not points on one axis.
    "qwen3-asr-loops": {
        "doc": "docs/benchmarks/engines/qwen3-asr.md",
        "title": "qwen3-asr 1.7B: looping windows per file, 8-bit against bf16",
        "basis": "7-file subset, 30s window, one run per arm. The default precision stays "
                 "because bf16 loops on the same files, about as often; where the "
                 "counts are equal the two markers coincide.",
        "xlabel": "file (shortest first)", "ylabel": "looping windows",
        "scale": "category", "connect": False, "zero": True,
        "series": ["8-bit (default)", "bf16"],
        "layout": "columns", "series_keys": ["| 8-bit |", "| bf16 |"],
        "rows": [
            (1, None, ("3", "2")), (2, None, ("0", "0")), (3, None, ("19", "18")),
            (4, None, ("0", "0")), (5, None, ("31", "31")), (6, None, ("40", "38")),
            (7, None, ("52", "56")),
        ],
        "legend_loc": "upper left",
    },
    # Markers only: unordered engines. No ring, since the engine default is decided on
    # engines.md; this chart only places both sizes against the two existing engines.
    "qwen3-asr-headline": {
        "doc": "docs/benchmarks/engines/qwen3-asr.md",
        "title": "qwen3-asr against the current headline engines",
        "basis": "20-file corpus (17 JP, 3 EN files), 8bit, 30s windows for qwen3-asr. "
                 "Neither size displaces an existing engine on Japanese CER; the 0.6B is "
                 "the fastest. Whisper speed is a range and Whisper/Voxtral peak is not "
                 "in this table, so those points are absent.",
        "xlabel": "engine",
        "scale": "category", "connect": False,
        "series": ["Japanese CER", "English WER (3 files)", "x realtime", "peak GB"],
        "panels": [
            {"series": [0, 1], "ylabel": "error % (lower is better)", "title": "accuracy",
             "legend_loc": "upper left"},
            {"series": [2], "ylabel": "x realtime (higher is faster)", "title": "speed",
             "zero": True},
            {"series": [3], "ylabel": "peak GPU memory, GB", "title": "memory",
             "zero": True},
        ],
        "width": 13.0, "height": 3.9,
        "rows": [
            ("whisper-turbo\nno-condition", "| whisper-turbo, no-condition |",
             ("14.49%", "18.34%", None, None)),
            ("voxtral", "| voxtral (default) |", ("16.22%", "21.50%", "29.6x", None)),
            ("qwen3-asr\n1.7B", "| qwen3-asr (1.7B) |",
             ("19.33%", "25.45%", "21.8x", "4.05")),
            ("qwen3-asr\n0.6B", "| qwen3-asr-small (0.6B) |",
             ("23.27%", "24.26%", "32.8x", "2.36")),
        ],
    },
    # Markers only: three budget configurations of the same engine on one file.
    "qwen3-asr-truncation": {
        "doc": "docs/benchmarks/engines/qwen3-asr.md",
        "title": "qwen3-asr: token budget on one 1553s Japanese file",
        "basis": "one 26-minute Japanese recording, 30s window. The budget is per window "
                 "because only that setting decodes the whole file.",
        "xlabel": "token budget",
        "scale": "category", "connect": False,
        "series": ["coverage CER", "audio covered"],
        "panels": [
            {"series": [0], "ylabel": "coverage CER % (lower is better)",
             "title": "accuracy", "zero": True},
            {"series": [1], "ylabel": "audio covered, %", "title": "coverage",
             "zero": True},
        ],
        "width": 10.0,
        "rows": [
            ("whole file\n8192 (library)", "| library default, whole-file budget of 8192 |",
             ("110.77%", "2%")),
            ("whole file\n19950 (scaled)", "| whole-file budget scaled to duration (19950) |",
             ("96.69%", "8%")),
            ("per window\n(default)", "| **per-window budget (default)** |",
             ("19.15%", "100%")),
        ],
        "chosen": "per window\n(default)", "chosen_series": "coverage CER",
    },
}
