"""Sweep charts for docs/benchmarks/input-level.md. Values are verbatim from its tables."""

SWEEPS = {
    "input-level-clamp": {
        "doc": "docs/benchmarks/input-level.md",
        "title": "Mel bins clamped, and error, by input gain",
        "basis": "Clamp share: one recording, peaking at -5.5 dBFS as recorded. Error: "
                 "7-file subset, Voxtral, coverage CER/WER (-6 dB not run).\n--gain auto "
                 "leaves a file already peaking above -6 dBFS alone, because 0 dB is the "
                 "Japanese optimum, +6 dB is marginally worse on Japanese\nand puts this "
                 "recording's peak at full scale; the English improvement rests on 2 files.",
        "xlabel": "gain applied (dB, 0 = as recorded)",
        "ylabel": "",
        "scale": "linear", "unit": "",
        "series": ["mel bins clamped", "Japanese coverage CER",
                   "English coverage WER (2 files)"],
        "rows": [
            (-20, "| -20dB |", ("65.5%", "23.76%", "36.65%")),
            (-12, "| -12dB |", ("41.5%", "19.42%", "34.23%")),
            (-6, "| -6dB |", ("24.0%", None, None)),
            (0, "| unity |", ("9.9%", "16.44%", "26.55%")),
            (6, "| +6dB |", ("2.8%", "17.09%", "23.96%")),
        ],
        "panels": [
            {"series": [0], "ylabel": "mel bins clamped (%)", "title": "clamped at the floor",
             "zero": True},
            {"series": [1, 2], "ylabel": "error %", "title": "accuracy"},
        ],
        "chosen": 0, "chosen_series": "Japanese coverage CER",
        "chosen_label": "auto (0 dB here)",
        "label_dy": 12,
        "height": 3.8,
    },
    "input-level-paired": {
        "doc": "docs/benchmarks/input-level.md",
        "title": "Paired difference against unity gain, with 95% CIs",
        "basis": "7-file subset, Voxtral, coverage CER/WER, paired across files; above "
                 "zero is worse than unity.\nAttenuation is the one resolved harm and "
                 "boosting is a wash overall, so auto lifts only quiet files and leaves "
                 "the rest at unity.",
        "xlabel": "arm against unity (as recorded)", "ylabel": "",
        "scale": "category", "connect": False, "unit": "",
        "series": ["all files", "English only (2 files)", "Japanese only (5 files)"],
        "rows": [
            ("-20 dB", "| -20dB vs unity |", ("+7.79", None, None),
             ("[+5.39, +12.08]", None, None)),
            ("-12 dB", "| -12dB vs unity |", ("+3.78", None, None),
             ("[+1.82, +7.83]", None, None)),
            ("+6 dB\nall", "| +6dB vs unity, all files |", ("+0.09", None, None),
             ("[-1.21, +0.94]", None, None)),
            ("+6 dB\nEnglish", "| +6dB vs unity, English only |", (None, "-2.59", None),
             (None, "[-3.19, -1.18]", None)),
            ("+6 dB\nJapanese", "| +6dB vs unity, Japanese only |", (None, None, "+0.65"),
             (None, None, "[+0.00, +1.08]")),
            ("peak to\n-1 dBFS", "| peak-normalize vs unity |", ("+0.21", None, None),
             ("[-0.19, +0.64]", None, None)),
        ],
        "panels": [
            {"series": [0, 1, 2], "ylabel": "difference vs unity (points)",
             "zero_line": True},
        ],
        "chosen": None,
        "width": 7.2, "height": 3.8,
    },
}
