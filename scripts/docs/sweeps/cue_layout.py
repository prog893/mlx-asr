"""Charts for docs/benchmarks/cue-layout.md."""

SWEEPS = {
    "cue-layout-gap": {
        "doc": "docs/benchmarks/cue-layout.md",
        "title": "Cue break agreement against gap_s (max_chars 32)",
        "basis": ("7 timed reference files, all by one editor. The default pair "
                  "1.2s / 28 chars scores 37.0% break F1 and sits off this line; "
                  "the sweep optimum was not adopted."),
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
             "title": "Break F1 and mid-phrase ends", "legend_loc": "center right"},
            {"series": [2], "ylabel": "cues/ref",
             "title": "Cue count"},
        ],
        "width": 9.6,
        "height": 3.6,
    },
}
