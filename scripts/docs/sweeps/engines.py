"""Sweep charts for docs/benchmarks/engines.md and engines/kotoba.md, copied from their tables.

Every engine comparison on engines.md is unordered, so those charts are markers only
(`connect: False`, `scale: "category"`). The Voxtral-against-Whisper table has metrics as
rows and engines as columns, so it uses `layout: "columns"`: each series is matched by
its row label. Its error bars are the 95% t-intervals the table prints for the Whisper
mean; the Whisper speed point is its middle run, with the bar spanning the three runs.
"""

SWEEPS = {
    "engines-voxtral-whisper": {
        "doc": "docs/benchmarks/engines.md",
        "title": "Voxtral against Whisper turbo-nocond, 20-file corpus",
        "basis": "Idle M2 Ultra; Voxtral is one deterministic run, Whisper the mean of 3 "
                 "runs with its printed 95% t-interval.\nThe speed bar spans Whisper's 3 "
                 "runs around the middle one. Voxtral is the default because it is faster "
                 "than every Whisper run\nand needs no language hint or stability flag, "
                 "while Whisper's accuracy lead is under 2 points.",
        "xlabel": "engine", "ylabel": "error %",
        "scale": "category", "unit": "",
        "connect": False,
        "layout": "columns",
        "series": ["JP coverage CER (17 files)", "EN coverage WER (3 files)", "x realtime"],
        "series_keys": ["| JP coverage CER, 17 files |", "| EN coverage WER, 3 files |",
                        "| x realtime |"],
        "rows": [
            ("voxtral", None, ("16.22%", "21.50%", "29.6x"), (None, None, None)),
            ("whisper\nturbo-nocond", None, ("14.49%", "18.34%", "21.6"),
             ("[13.82, 15.15]", "[16.62, 20.07]", "[18.0, 22.0]")),
        ],
        "chosen": "voxtral", "chosen_series": "JP coverage CER (17 files)",
        "panels": [
            {"series": [0, 1], "ylabel": "error % (lower is better)", "title": "accuracy"},
            {"series": [2], "ylabel": "x realtime (higher is faster)", "title": "speed",
             "zero": True},
        ],
        "width": 8.4, "height": 3.8,
    },
    "engines-generalization": {
        "doc": "docs/benchmarks/engines.md",
        "title": "Whisper's lead over Voxtral, bootstrapped over files",
        "basis": "17 Japanese files, the same runs as the Voxtral against Whisper chart; "
                 "95% CI from the bootstrap over files.\nThe whole interval is above zero, "
                 "so the Whisper lead also holds when the files are resampled.",
        "xlabel": "test", "ylabel": "Voxtral minus Whisper, JP CER points",
        "scale": "category", "unit": "",
        "connect": False,
        "series": ["Voxtral minus Whisper (positive: Whisper better)"],
        "rows": [
            ("bootstrap over 17 files", "| **bootstrap over 17 files** |", ("+1.85",),
             ("[+0.58, +3.33]",)),
        ],
        "chosen": None,
        "panels": [{"series": [0], "ylabel": "Voxtral minus Whisper, JP CER points",
                    "zero_line": True}],
        "width": 5.2, "height": 3.4,
    },
    "engines-corpus-size": {
        "doc": "docs/benchmarks/engines.md",
        "title": "Voxtral vs Whisper turbo-nocond as the corpus grew",
        "basis": "Japanese files only; Whisper is a mean of 6 or 3 runs, Voxtral one "
                 "deterministic run.\nThe 17-file Whisper point is the 3-run mean measured "
                 "2026-08-06 (14.29 / 14.37 / 14.79),\nthe same runs as the Voxtral "
                 "against Whisper table; an earlier session read 14.93%.",
        "xlabel": "Japanese files in the corpus", "ylabel": "JP coverage CER %",
        "scale": "linear",
        "unit": "",
        "series": ["voxtral", "whisper turbo-nocond (mean)"],
        "rows": [
            (5, "| original | 5 |", ("16.44%", "15.91%")),
            (12, "| grown | 12 |", ("16.08%", "14.07%")),
            (17, "| final | 17 |", ("16.22%", "14.49%")),
        ],
        "chosen": None,
        "legend_loc": "center right",
    },
    "engines-narration": {
        "doc": "docs/benchmarks/engines.md",
        "title": "Clean narration: Voxtral against the Whisper sizes and kotoba",
        "basis": "One narration clip (n=1), M2 Ultra. Voxtral has no coverage CER and "
                 "kotoba no throughput here.\nVoxtral has the lowest plain CER on this "
                 "clip; the default itself is decided on the 20-file corpus.",
        "xlabel": "engine", "ylabel": "CER %",
        "scale": "category", "unit": "",
        "connect": False,
        "series": ["coverage CER", "plain CER", "x realtime"],
        "rows": [
            ("voxtral", "| **voxtral (M2 Ultra 128GB, 60s b16 kv8)** |",
             (None, "7.28%", "21.2x")),
            ("turbo", "| whisper large-v3-turbo |", ("8.28%", "9.08%", "44.3x")),
            ("v3\nnocond", "| whisper large-v3, no-condition |",
             ("8.87%", "8.87%", "24.2x")),
            ("turbo\nnocond", "| whisper large-v3-turbo, no-condition |",
             ("10.42%", "10.42%", "73.5x")),
            ("v3", "| whisper large-v3 |", ("12.91%", "13.67%", "16.6x")),
            ("small", "| whisper small |", ("13.96%", "14.89%", "59.2x")),
            ("medium", "| whisper medium |", ("15.17%", "15.17%", "29.9x")),
            ("v2", "| whisper large-v2 |", ("15.20%", "16.96%", "23.0x")),
            ("kotoba\n20s", "| kotoba-whisper v2.2, chunk 20s |", ("16.55%", "16.55%", None)),
            ("base", "| whisper base |", ("22.73%", "25.97%", "90.5x")),
            ("tiny", "| whisper tiny |", ("34.51%", "36.27%", "101.8x")),
        ],
        "chosen": "voxtral", "chosen_series": "plain CER",
        "panels": [
            {"series": [0, 1], "ylabel": "CER % (lower is better)", "title": "accuracy"},
            {"series": [2], "ylabel": "x realtime (higher is faster)", "title": "speed",
             "zero": True},
        ],
        "width": 14.0, "height": 3.8,
    },
    "engines-japanese-only": {
        "doc": "docs/benchmarks/engines.md",
        "title": "Japanese-only engines against the multilingual defaults",
        "basis": "17 Japanese files, M2 Ultra; the Japanese-only rows ran with another GPU "
                 "client resident, so their speeds are floors. Whisper's 18.0-22.0x run "
                 "range is not drawn.\nNeither Japanese-only engine is more accurate than "
                 "Voxtral, so they do not change the default; parakeet wins only on speed.",
        "xlabel": "engine", "ylabel": "CER %",
        "scale": "category", "unit": "",
        "connect": False,
        "series": ["JP coverage CER", "kana CER", "x realtime"],
        "rows": [
            ("whisper\nturbo-nocond", "| whisper-turbo no-cond (3-run mean) |",
             ("14.49%", None, None)),
            ("voxtral", "| voxtral |", ("16.22%", None, "29.6x")),
            ("kotoba\n10s", "| kotoba chunk10s |", ("27.01%", None, "36.2x")),
            ("parakeet\nc120", "| **parakeet c120** |", ("26.19%", "23.35%", "244.6x")),
            ("reazon\nfp32", "| **reazon-k2 fp32 c30** |", ("30.45%", "27.73%", "51.6x")),
            ("reazon\nint8", "| reazon-k2 int8 c30 |", ("36.93%", None, "78.9x")),
        ],
        "chosen": "voxtral", "chosen_series": "JP coverage CER",
        "panels": [
            {"series": [0, 1], "ylabel": "CER % (lower is better)", "title": "accuracy"},
            {"series": [2], "ylabel": "x realtime (higher is faster)", "title": "speed",
             "zero": True},
        ],
        "width": 10.0, "height": 3.8,
    },
    "engines-kotoba-window": {
        "doc": "docs/benchmarks/engines/kotoba.md",
        "title": "kotoba: decode window (MLX chunked driver)",
        "basis": "17 Japanese files. 10s is the default as the lowest CER; longer windows trade accuracy for speed.",
        "xlabel": "window length", "ylabel": "JP coverage CER %",
        "scale": "linear",
        "unit": "s",
        "series": ["coverage CER", "x realtime"],
        "rows": [
            (10, "chunked, 10s windows", ("27.01%", "36.2x")),
            (20, "chunked, 20s windows", ("31.33%", "68.8x")),
            (30, "chunked, 30s windows", ("49.71%", "72.7x")),
        ],
        "chosen": 10,
        "panels": [
            {"series": [0], "ylabel": "JP coverage CER %", "title": "accuracy"},
            {"series": [1], "ylabel": "x realtime", "title": "speed"},
        ],
    },
    "engines-kotoba-authors-chunk": {
        "doc": "docs/benchmarks/engines/kotoba.md",
        "title": "kotoba: chunk_length_s in the authors' torch pipeline",
        "basis": "Corpus: the 5 Japanese files of the 7-file subset; narration: the single clip. "
                 "Reference driver only, not the CLI's.",
        "xlabel": "chunk_length_s", "ylabel": "CER %",
        "scale": "linear",
        "unit": "s",
        "series": ["corpus coverage CER", "narration CER"],
        "rows": [
            (10, "`chunk_length_s=10`", ("26.16%", "23.71%")),
            (15, "`chunk_length_s=15`", ("30.40%", "20.88%")),
            (20, "`chunk_length_s=20`", ("27.82%", "16.55%")),
            (30, "`chunk_length_s=30`", ("49.57%", "39.52%")),
        ],
        "chosen": None,
        "legend_loc": "upper left",
    },
}
