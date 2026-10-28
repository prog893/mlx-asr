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
        "source": ("m4-voxtral-chunking.json#c30b32;ultra-voxtral-batch.json#c30b32,c30b32_r2,"
                   "c30b32_r3"),
        "zero": True, "width": 6.0,
        "chosen": None,
    },
    "determinism-whisper-runs": {
        "doc": "docs/benchmarks/reference/determinism.md",
        "title": "Six Whisper runs of one config, against deterministic Voxtral",
        "basis": "7-file subset, turbo-nocond. Zero is Voxtral; below it, Whisper "
                 "wins. All six JP runs are below Voxtral and the mean's CI excludes it.",
        "xlabel": "Whisper run (sorted by score), then the mean of the six",
        "ylabel": "Whisper minus Voxtral (points)\n0 = Voxtral, below = Whisper better",
        "scale": "category", "unit": "", "connect": False,
        "series": ["JP coverage CER", "EN coverage WER"],
        "rows": [
            ("run 1", _RUNS, ("-1.46", "-4.58")),
            ("run 2", _RUNS, ("-1.35", "-4.41")),
            ("run 3", _RUNS, ("-1.00", "-4.29")),
            ("run 4", _RUNS, ("-0.90", "-4.02")),
            ("run 5", _RUNS, ("-0.77", "-3.19")),
            ("run 6", _RUNS, ("-0.01", "-3.14")),
            ("mean", "| mean minus Voxtral, 95% t-interval (points) |",
             ("-0.92", "-3.94"), ("[-1.46, -0.37]", "[-4.60, -3.28]")),
        ],
        "source": ("ultra-whisper-sizes.json#whisper_turbo_nocond,whisper_turbo_nocond_r2,"
                   "whisper_turbo_nocond_r3,whisper_turbo_nocond_r4,whisper_turbo_nocond_r5,"
                   "whisper_turbo_nocond_r6;ultra-mincut-inputs.json#mincut7_vox_c30b32_kv8"),
        "panels": [{"series": [0, 1], "zero_line": True}], "width": 7.6,
        "chosen": None,
    },
}
