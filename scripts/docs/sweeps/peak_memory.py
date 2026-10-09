"""Charts for docs/benchmarks/reference/peak-memory.md. Data only; values are copied verbatim from
the page's tables, and tests/test_sweeps.py checks each row against them."""

SWEEPS = {
    "peak-memory-whisper-duration": {
        "doc": "docs/benchmarks/reference/peak-memory.md",
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
    "peak-memory-range": {
        "doc": "docs/benchmarks/reference/peak-memory.md",
        "title": "Peak GPU memory range across the corpus, per model",
        "basis": "20-file corpus, M2 Ultra. Overlapping marks: memory does not depend "
                 "on the file.",
        "xlabel": "model",
        "ylabel": "peak GPU memory (GB)", "zero": True,
        "scale": "category",
        "connect": False,
        "unit": "",
        "series": ["min over the 20 files", "max over the 20 files"],
        "rows": [
            ("kotoba", "| `kotoba` |", ("2.38GB", "2.38GB")),
            ("qwen3 0.6B", "| `qwen3-asr 0.6B/8bit` |", ("2.28GB", "2.36GB")),
            ("qwen3 1.7B", "| `qwen3-asr 1.7B/8bit` |", ("3.95GB", "4.05GB")),
            ("voxtral", "| `voxtral 4bit` |", ("5.38GB", "6.77GB")),
            ("whisper turbo", "| `whisper turbo` |", ("2.56GB", "5.52GB")),
        ],
        "chosen": None,
    },
}
