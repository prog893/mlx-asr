"""Sweep charts for docs/benchmarks/engines/qwen3-asr.md. Values copied verbatim from its tables."""

SWEEPS = {
    "qwen3-asr-window": {
        "doc": "docs/benchmarks/engines/qwen3-asr.md",
        "title": "qwen3-asr 1.7B: decode window",
        "basis": "7-file subset (5 JP, 2 EN files), 8bit, one run per arm. 15s is the "
                 "default: 0.94 points lower on Japanese CER than 30s, a tie (CI [-2.87, +0.76]), "
                 "and faster and smaller.",
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
            (15, "| **15s** |", ("19.73%", "30.15%", "22.4x", "3.73")),
            (30, "| 30s |", ("20.67%", "31.45%", "18.1x", "4.05")),
            (60, "| 60s |", ("22.27%", "30.04%", "16.4x", "4.10")),
            (120, "| 120s |", ("23.55%", "34.62%", "16.2x", "4.68")),
            (300, "| 300s |", ("64.89%", "36.94%", "9.4x", "5.77")),
        ],
        "chosen": 15, "chosen_series": "Japanese CER",
        "label_dy": 14,
        "source": "ultra-qwen3-window.json#qwen3w7_1.7B_8bit_c15,qwen3w7_1.7B_8bit_c30,"
                  "qwen3w7_1.7B_8bit_c60,qwen3w7_1.7B_8bit_c120,qwen3w7_1.7B_8bit_c300",
    },
    # Both sizes on ONE y scale, so the 1.7B's narrower spread reads beside the 0.6B's
    # steeper slope. Each table line holds one size, so each row
    # carries a value for one series only. The second panel is the paired difference
    # the table prints ("8bit minus rung", negative means 8bit is better) with its 95%
    # CI; the default rows have no paired figure, they are the zero line.
    "qwen3-asr-precision": {
        "doc": "docs/benchmarks/engines/qwen3-asr.md",
        "title": "qwen3-asr: precision ladder (8bit best on the 1.7B, 0.6B steeper)",
        "basis": "20-file corpus, 15s window; 8bit is the default on both sizes. A rung "
                 "replaces 8bit only if its paired CI lies wholly above zero: on the 1.7B, "
                 "4bit, 6bit and bf16 lie below it (8bit wins) and 5bit spans it; on the "
                 "0.6B, 4bit and 5bit lie below it and 6bit and bf16 span it (a tie keeps "
                 "the default).",
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
            ("1.7B\n4bit", "| 1.7B | 4bit |", ("19.78%", None, "-1.16", None),
             (None, None, "[-1.49, -0.86]", None)),
            ("1.7B\n5bit", "| 1.7B | 5bit |", ("19.21%", None, "-0.59", None),
             (None, None, "[-1.34, +0.08]", None)),
            ("1.7B\n6bit", "| 1.7B | 6bit |", ("19.09%", None, "-0.47", None),
             (None, None, "[-0.88, -0.13]", None)),
            ("1.7B\n8bit", "| 1.7B | **8bit (default)** |", ("18.62%", None, None, None)),
            ("1.7B\nbf16", "| 1.7B | bf16 |", ("19.14%", None, "-0.52", None),
             (None, None, "[-0.89, -0.21]", None)),
            ("0.6B\n4bit", "| 0.6B | 4bit |", (None, "25.10%", None, "-2.79"),
             (None, None, None, "[-3.43, -2.19]")),
            ("0.6B\n5bit", "| 0.6B | 5bit |", (None, "23.23%", None, "-0.92"),
             (None, None, None, "[-1.63, -0.18]")),
            ("0.6B\n6bit", "| 0.6B | 6bit |", (None, "22.67%", None, "-0.36"),
             (None, None, None, "[-1.16, +0.53]")),
            ("0.6B\n8bit", "| 0.6B | **8bit** |", (None, "22.31%", None, None)),
            ("0.6B\nbf16", "| 0.6B | bf16 |", (None, "22.31%", None, "+0.01"),
             (None, None, None, "[-0.56, +0.60]")),
        ],
        "chosen": "1.7B\n8bit", "chosen_series": "1.7B",
        "chosen_label": "default (8bit)",
        "label_dy": -16,
        "source": "ultra-qwen3-precision-c15.json#qwen3_1.7B_4bit_c15,qwen3_1.7B_5bit_c15,"
                  "qwen3_1.7B_6bit_c15,qwen3_1.7B_8bit_c15,qwen3_1.7B_bf16_c15,"
                  "qwen3_0.6B_4bit_c15,qwen3_0.6B_5bit_c15,qwen3_0.6B_6bit_c15,"
                  "qwen3_0.6B_8bit_c15,qwen3_0.6B_bf16_c15",
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
            (1, None, ("3", "4")), (2, None, ("0", "0")), (3, None, ("18", "18")),
            (4, None, ("0", "0")), (5, None, ("32", "28")), (6, None, ("39", "36")),
            (7, None, ("51", "57")),
        ],
        "legend_loc": "upper left",
        "source": "ultra-qwen3-window.json#qwen3w7_1.7B_8bit_c30;"
                  "ultra-qwen3-loops.json#qwen3loop7_1.7B_bf16_c30",
    },
    # Markers only: unordered engines. No ring, since the engine default is decided on
    # engines.md; this chart only places both sizes against the two existing engines.
    "qwen3-asr-headline": {
        "doc": "docs/benchmarks/engines/qwen3-asr.md",
        "title": "qwen3-asr against the current headline engines",
        "basis": "20-file corpus (17 JP, 3 EN files), 8bit, 15s windows for qwen3-asr; "
                 "Voxtral at fp16, its M2 Ultra default. "
                 "Neither size displaces an existing engine on Japanese CER; the 0.6B is "
                 "faster than Voxtral. Whisper speed is a range and Whisper/Voxtral peak is not "
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
             ("14.17%", "18.65%", None, None)),
            ("voxtral", "| voxtral (default on 128GB, fp16) |", ("15.20%", "21.21%", "23.9x", None)),
            ("qwen3-asr\n1.7B", "| qwen3-asr (1.7B) |",
             ("18.62%", "24.72%", "23.0x", "3.74")),
            ("qwen3-asr\n0.6B", "| qwen3-asr-small (0.6B) |",
             ("22.31%", "22.95%", "36.4x", "2.07")),
        ],
        "source": "ultra-whisper-sizes.json#whisper_turbo_nocond,whisper_turbo_nocond_r2,"
                  "whisper_turbo_nocond_r3,whisper_turbo_nocond_r4,whisper_turbo_nocond_r5,"
                  "whisper_turbo_nocond_r6;ultra-voxtral-precision-default.json#vox_wfp16_c30b128_kv8,"
                  "vox_wfp16_c30b128_kv8_r2,vox_wfp16_c30b128_kv8_r3;"
                  "ultra-qwen3-precision-c15.json#qwen3_1.7B_8bit_c15,qwen3_0.6B_8bit_c15",
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
             ("96.67%", "2.1%")),
            ("whole file\n19950 (scaled)", "| whole-file budget scaled to duration (19950) |",
             ("96.71%", "2.1%")),
            ("per window\n(default)", "| **per-window budget (default)** |",
             ("17.29%", "100%")),
        ],
        "chosen": "per window\n(default)", "chosen_series": "coverage CER",
        "source": "ultra-qwen3-loops.json#q3budget_wholefile_8192,q3budget_wholefile_19950,"
                  "q3trunc_perwindow_1.7B_8bit_c30",
    },
}
