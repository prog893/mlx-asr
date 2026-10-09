"""Charts for docs/benchmarks/reference/metrics.md.

`metrics-plain-vs-coverage` is a comparison of files, not a sweep: markers only, one
category per file, labelled by duration (never by file name).

The two min_cut tables lay configs out as rows and thresholds as columns, so these use
`layout: "columns"`: each series is one table row, matched by its row label, and every
value of that series has to sit on that row.
"""

_CONFIGS = ["30s b32 kv8", "30s b32 kv8 ov8", "60s b16 kv8"]
_ENGINES = ["turbo-nocond", "voxtral c30b32_kv8", "large-v3-nocond", "large-v3-turbo",
            "medium"]

SWEEPS = {
    "metrics-mincut-configs": {
        "doc": "docs/benchmarks/reference/metrics.md",
        "title": "Coverage CER by excusal threshold, three Voxtral configs",
        "basis": "7-file subset, Japanese files. Levels drift with the threshold; the "
                 "ranking holds at every value, so quote the threshold with any level.",
        "xlabel": "min_cut (characters)", "ylabel": "JP coverage CER %",
        "scale": "linear", "unit": "",
        "layout": "columns",
        "series": _CONFIGS,
        "series_keys": ["| 30s b32 kv8 |", "| 30s b32 kv8 ov8 |", "| 60s b16 kv8 |"],
        "rows": [
            (10, None, ("14.97%", "15.62%", "16.80%")),
            (20, None, ("15.89%", "16.67%", "17.95%")),
            (30, None, ("16.44%", "17.34%", "18.21%")),
            (50, None, ("17.81%", "18.12%", "19.32%")),
            (80, None, ("20.23%", "20.61%", "20.69%")),
        ],
        "chosen": 30, "chosen_series": "30s b32 kv8",
        "label_dy": -22,   # above the ring sits the ov8 line
        "legend_loc": "upper left",
    },
    "metrics-mincut-engines": {
        "doc": "docs/benchmarks/reference/metrics.md",
        "title": "Coverage CER by excusal threshold, across engines",
        "basis": "7-file subset, Japanese files. The two leaders keep their order across "
                 "the range; large-v3-nocond passes turbo-nocond only at 10.",
        "xlabel": "min_cut (characters)", "ylabel": "JP coverage CER %",
        "scale": "linear", "unit": "",
        "layout": "columns",
        "series": _ENGINES,
        "series_keys": ["| turbo-nocond |", "| voxtral c30b32_kv8 |",
                        "| large-v3-nocond |", "| large-v3-turbo |", "| medium |"],
        "rows": [
            (10, None, ("13.85%", "14.97%", "13.32%", "22.76%", "21.81%")),
            (20, None, ("14.78%", "15.89%", "15.53%", "23.99%", "26.23%")),
            (30, None, ("14.93%", "16.44%", "17.36%", "24.97%", "28.93%")),
            (50, None, ("16.03%", "17.81%", "20.10%", "26.58%", "32.18%")),
            (80, None, ("16.84%", "20.23%", "23.39%", "28.05%", "36.49%")),
        ],
        "chosen": 30, "chosen_series": "voxtral c30b32_kv8",
        "legend_loc": "upper left",
    },
    "metrics-plain-vs-coverage": {
        "doc": "docs/benchmarks/reference/metrics.md",
        "title": "Plain CER against coverage CER, per file",
        "basis": "20-file corpus, 7 Japanese files. Plain CER charges audio the "
                 "reference omits; rank configs on coverage CER.",
        "xlabel": "file, by duration", "ylabel": "JP CER % (lower is better)",
        "scale": "category", "unit": "", "connect": False, "zero": True,
        "series": ["plain CER", "coverage CER"],
        "rows": [
            ("1.9 min", "| 1.9 min |", ("37.4%", "18.0%")),
            ("5.5 min", "| 5.5 min |", ("99.3%", "5.5%")),
            ("13.3 min", "| 13.3 min |", ("64.4%", "11.7%")),
            ("25.9 min", "| 25.9 min |", ("148.2%", "19.9%")),
            ("52.0 min", "| 52.0 min |", ("150.7%", "17.4%")),
            ("69.6 min", "| 69.6 min |", ("134.1%", "14.6%")),
            ("93.2 min", "| 93.2 min |", ("144.5%", "17.1%")),
        ],
        "chosen": None,
    },
}
