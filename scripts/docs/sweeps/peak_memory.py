"""Charts for docs/benchmarks/peak-memory.md. Data only; values are copied verbatim from
the page's tables, and tests/test_sweeps.py checks each row against them."""

SWEEPS = {
    "peak-memory-whisper-duration": {
        "doc": "docs/benchmarks/peak-memory.md",
        "title": "whisper turbo: peak GPU memory grows with audio length",
        "basis": "Six files of the 20-file corpus, M2 Ultra.",
        "xlabel": "audio length (minutes)",
        "ylabel": "peak GPU memory (GB)", "zero": True,
        "scale": "linear",
        "unit": "",
        "series": ["peak"],
        "rows": [
            (1.9, "| 1.9 min |", ("2.56GB",)),
            (9.3, "| 9.3 min |", ("2.74GB",)),
            (35.4, "| 35.4 min |", ("3.46GB",)),
            (52.0, "| 52.0 min |", ("4.05GB",)),
            (69.6, "| 69.6 min |", ("4.54GB",)),
            (93.2, "| 93.2 min |", ("5.52GB",)),
        ],
    },
}
