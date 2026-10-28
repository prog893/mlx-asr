"""Charts for docs/benchmarks/cue-layout.md."""

SWEEPS = {
    "cue-layout-gap": {
        "doc": "docs/benchmarks/cue-layout.md",
        "title": "Cue break agreement against gap_s (max_chars 32)",
        "basis": ("7 timed reference files, all by one editor; cached Voxtral tokens "
                  "regrouped, machine-independent.\nThe default gap_s 1.2 is kept "
                  "although lower values score higher, because all seven references "
                  "share one editor's convention,\nand lower gap_s gains partly by "
                  "emitting more cues (right panel)."),
        "xlabel": "gap_s",
        "ylabel": "",
        "scale": "linear",
        "unit": "s",
        "series": ["break F1", "mid-phrase", "cues/ref"],
        "rows": [
            (0.6, "| 0.6 |", ("44.4%", "56.2%", "1.40")),
            (0.7, "| 0.7 |", ("42.6%", "56.2%", "1.29")),
            (0.8, "| 0.8 |", ("40.6%", "56.7%", "1.16")),
            (1.0, "| 1.0 |", ("36.4%", "59.4%", "1.02")),
            (1.2, "| 1.2 |", ("36.0%", "58.2%", "0.95")),
        ],
        "source": "ultra-cue-layout.json#cue_sweep_union",
        "chosen": 1.2, "chosen_series": "break F1",
        "chosen_label": "default gap_s",
        "panels": [
            {"series": [0, 1], "ylabel": "%",
             "title": "Break F1 and mid-phrase ends"},
            {"series": [2], "ylabel": "cues/ref",
             "title": "Cue count"},
        ],
        "width": 9.6,
        "height": 3.6,
    },
    "cue-layout-per-file": {
        "doc": "docs/benchmarks/cue-layout.md",
        "title": "Break F1 per file: default pair 1.2/28 against 0.7/32",
        "basis": ("7 timed reference files, all by one editor; cached Voxtral tokens "
                  "regrouped.\nThe default pair wins only on 632a295d981d, the file "
                  "it was fitted to, and loses on all six held-out files."),
        "xlabel": "file",
        "ylabel": "break F1 %",
        "scale": "category",
        "connect": False,
        "unit": "",
        "series": ["1.2/28 (default)", "0.7/32 (earlier published)"],
        "rows": [
            ("632a295d981d", "| 632a295d981d (", ("46.1%", "35.9%")),
            ("a773a315fb42", "| a773a315fb42 |", ("38.2%", "49.2%")),
            ("71bb5eba3baa", "| 71bb5eba3baa |", ("31.3%", "38.3%")),
            ("1da58e1b2b69", "| 1da58e1b2b69 |", ("32.4%", "41.2%")),
            ("a1372c4d2900", "| a1372c4d2900 |", ("33.3%", "41.2%")),
            ("c47ad127bc83", "| c47ad127bc83 |", ("41.9%", "48.8%")),
            ("b3432b40c079", "| b3432b40c079 |", ("38.8%", "43.6%")),
        ],
        "source": "ultra-cue-layout.json#cue_sweep_union",
        "chosen": None,
        "width": 7.6,
        "height": 3.6,
    },
    "cue-layout-end-to-end": {
        "doc": "docs/benchmarks/cue-layout.md",
        "title": "Default pair against 0.7/32, end to end",
        "basis": ("Voxtral end to end on the 7 timed files, written SRTs scored by "
                  "eval_timing.\nThe default 1.2/28 is kept at a cost of 5.4 break "
                  "points because lower gap_s fits one editor's convention,\nand the "
                  "default is slightly better on p95 drift."),
        "xlabel": "gap_s / max_chars",
        "ylabel": "",
        "scale": "category",
        "connect": False,
        "unit": "",
        "series": ["break F1", "mid-phrase", "median drift", "p95 drift"],
        "rows": [
            ("1.2/28", "`1.2/28` (default)", ("37.7%", "58.3%", "278", "865")),
            ("0.7/32", "`0.7/32` (earlier published config)",
             ("43.1%", "57.4%", "258", "976")),
        ],
        "source": ("ultra-timestamps.json#ts_0732#engine voxtral,ts_default_r1#engine voxtral,"
                   "ts_default_r2#engine voxtral,ts_default_r3#engine voxtral"),
        "chosen": "1.2/28", "chosen_series": "break F1",
        "panels": [
            {"series": [0, 1], "ylabel": "%", "title": "Break F1 and mid-phrase ends"},
            {"series": [2, 3], "ylabel": "ms", "title": "Timestamp drift", "zero": True},
        ],
        "width": 8.4,
        "height": 3.6,
    },
}
