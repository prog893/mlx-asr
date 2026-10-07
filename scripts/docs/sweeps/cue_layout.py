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
            (0.6, "| 0.6 |", ("44.5%", "55.7%", "1.36")),
            (0.7, "| 0.7 |", ("42.3%", "56.0%", "1.25")),
            (0.8, "| 0.8 |", ("40.2%", "56.5%", "1.11")),
            (1.0, "| 1.0 |", ("36.2%", "59.1%", "0.99")),
            (1.2, "| 1.2 |", ("35.9%", "57.8%", "0.92")),
        ],
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
        "title": "Break F1 per file: default pair 1.2/28 against the n=7 optimum 0.7/32",
        "basis": ("7 timed reference files, all by one editor; cached Voxtral tokens "
                  "regrouped.\nThe default pair wins only on narration-jp, the file "
                  "it was fitted to, and loses on all six held-out files."),
        "xlabel": "file",
        "ylabel": "break F1 %",
        "scale": "category",
        "connect": False,
        "unit": "",
        "series": ["1.2/28 (default)", "0.7/32 (n=7 optimum)"],
        "rows": [
            ("narration-jp", "| narration-jp (", ("46.1%", "35.9%")),
            ("rec-16", "| rec-16 |", ("38.2%", "49.2%")),
            ("rec-13", "| rec-13 |", ("28.6%", "36.5%")),
            ("rec-14", "| rec-14 |", ("32.4%", "41.2%")),
            ("rec-17", "| rec-17 |", ("33.3%", "41.2%")),
            ("rec-15", "| rec-15 |", ("41.9%", "48.8%")),
            ("rec-20", "| rec-20 |", ("38.8%", "43.6%")),
        ],
        "chosen": None,
        "width": 7.6,
        "height": 3.6,
    },
    "cue-layout-end-to-end": {
        "doc": "docs/benchmarks/cue-layout.md",
        "title": "Default pair against the n=7 optimum, end to end",
        "basis": ("Voxtral end to end on the 7 timed files, written SRTs scored by "
                  "eval_timing.\nThe default 1.2/28 is kept at a cost of 5.4 break "
                  "points because the optimum fits one editor's convention,\nand the "
                  "default is slightly better on p95 drift."),
        "xlabel": "gap_s / max_chars",
        "ylabel": "",
        "scale": "category",
        "connect": False,
        "unit": "",
        "series": ["break F1", "mid-phrase", "median drift", "p95 drift"],
        "rows": [
            ("1.2/28", "`1.2/28` (default)", ("37.4%", "58.1%", "278", "786")),
            ("0.7/32", "`0.7/32` (n=7 optimum)",
             ("42.9%", "57.0%", "258", "829")),
        ],
        "chosen": "1.2/28", "chosen_series": "break F1",
        "panels": [
            {"series": [0, 1], "ylabel": "%", "title": "Break F1 and mid-phrase ends"},
            {"series": [2, 3], "ylabel": "ms", "title": "Timestamp drift", "zero": True},
        ],
        "width": 8.4,
        "height": 3.6,
    },
}
