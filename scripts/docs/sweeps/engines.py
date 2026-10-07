"""Sweep charts for docs/benchmarks/engines.md. Values are copied verbatim from its tables."""

SWEEPS = {
    "engines-corpus-size": {
        "doc": "docs/benchmarks/engines.md",
        "title": "Voxtral vs Whisper turbo-nocond as the corpus grew",
        "basis": "Japanese files only; Whisper is a mean of 6 or 3 runs, Voxtral one "
                 "deterministic run.",
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
    "engines-kotoba-window": {
        "doc": "docs/benchmarks/engines.md",
        "title": "kotoba: decode window (MLX chunked driver)",
        "basis": "17 Japanese files.",
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
        "doc": "docs/benchmarks/engines.md",
        "title": "kotoba: chunk_length_s in the authors' torch pipeline",
        "basis": "Corpus: 17 Japanese files; narration: the single clip. "
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
