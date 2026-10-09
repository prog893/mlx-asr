"""Charts for docs/benchmarks/reference/determinism.md.

Both charts compare unordered things (two machines; six draws of one config), so they
are markers only. Neither has a default to ring.

`determinism-whisper-runs` plots each Whisper run as its difference from Voxtral, so
Voxtral is the zero line: a point below it is a run that beats Voxtral. The runs are in
the order the table prints them (sorted by score), and the last category is the mean of
the six with its 95% t-interval. The differences are printed on their own table row.
"""

_RUNS = "| Whisper run minus Voxtral (points) |"

SWEEPS = {
    "determinism-machines": {
        "doc": "docs/benchmarks/reference/determinism.md",
        "title": "Same audio, config and weights on two machines",
        "basis": "One 112s file, 30s chunks, batch 32, kv8, 2400ms. Each machine "
                 "repeats itself exactly; the two differ by 5.45 points.",
        "xlabel": "machine", "ylabel": "JP coverage CER %",
        "scale": "category", "unit": "", "connect": False,
        "series": ["coverage CER, one 112s file"],
        "rows": [
            ("M4 16GB", "coverage CER on one 112s file", ("12.56%",)),
            ("M2 Ultra 128GB", "coverage CER on one 112s file", ("18.01%",)),
        ],
        "zero": True, "width": 6.0,
        "chosen": None,
    },
    "determinism-whisper-runs": {
        "doc": "docs/benchmarks/reference/determinism.md",
        "title": "Six Whisper runs of one config, against deterministic Voxtral",
        "basis": "7-file subset, turbo-nocond. Zero is Voxtral; below it, Whisper "
                 "wins. JP runs straddle Voxtral and the mean's CI contains it.",
        "xlabel": "Whisper run (sorted by score), then the mean of the six",
        "ylabel": "Whisper minus Voxtral (points)\n0 = Voxtral, below = Whisper better",
        "scale": "category", "unit": "", "connect": False,
        "series": ["JP coverage CER", "EN coverage WER"],
        "rows": [
            ("run 1", _RUNS, ("-1.67", "-5.38")),
            ("run 2", _RUNS, ("-1.51", "-4.84")),
            ("run 3", _RUNS, ("-0.54", "-4.57")),
            ("run 4", _RUNS, ("-0.22", "-4.53")),
            ("run 5", _RUNS, ("-0.09", "-3.94")),
            ("run 6", _RUNS, ("+0.85", "-2.61")),
            ("mean", "| mean minus Voxtral, 95% t-interval (points) |",
             ("-0.53", "-4.31"), ("[-1.52, +0.46]", "[-5.31, -3.31]")),
        ],
        "panels": [{"series": [0, 1], "zero_line": True}], "width": 7.6,
        "chosen": None,
    },
}
