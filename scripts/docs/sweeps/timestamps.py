"""Charts for docs/benchmarks/timestamps.md.

The engine table has metrics as rows and engines as columns, so this uses
`layout: "columns"`: each series is one metric row, matched by its row label. Values
are the numeric part of each cell ("250" from "**250 ms**"), which is what the table
prints verbatim. One panel per unit, since drift (ms, ms/min) and cue breaks (%) are
never combined.
"""

SWEEPS = {
    "timestamps-engines": {
        "doc": "docs/benchmarks/timestamps.md",
        "title": "Timing drift and cue breaks, Voxtral against Whisper",
        "basis": ("7 timed references, SRT at the default cue config (gap_s 1.2, "
                  "max_chars 28); the Whisper column is turbo-nocond, measured before "
                  "large-v3 became the Whisper default.\nVoxtral is the default because "
                  "drift (p95, slope) is the failure a user cannot fix, while cue "
                  "breaks are a tunable heuristic."),
        "xlabel": "engine", "ylabel": "",
        "scale": "category", "unit": "",
        "connect": False,
        "layout": "columns",
        "series": ["median timing error", "median p95 error", "worst drift slope",
                   "break F1", "mid-phrase splits"],
        "series_keys": ["| median timing error |", "| median p95 error |",
                        "| worst drift slope |", "| break F1 |",
                        "| mid-phrase splits |"],
        "rows": [
            ("Voxtral", None, ("278", "865", "33.0", "37.7%", "58.3%")),
            ("whisper\nturbo-nocond", None, ("244", "1994", "87.8", "56.9%", "41.4%")),
        ],
        "source": ("ultra-timestamps.json#ts_default_r1#engine voxtral,"
                   "ts_default_r2#engine voxtral,ts_default_r3#engine voxtral,"
                   "ts_default_r1#engine whisper-turbo-nocond,"
                   "ts_default_r2#engine whisper-turbo-nocond,"
                   "ts_default_r3#engine whisper-turbo-nocond"),
        "chosen": "Voxtral", "chosen_series": "worst drift slope",
        "panels": [
            {"series": [0, 1], "ylabel": "error at anchors (ms)",
             "title": "Timing error", "zero": True},
            {"series": [2], "ylabel": "drift slope (ms/min)",
             "title": "Worst drift slope", "zero": True},
            {"series": [3, 4], "ylabel": "%",
             "title": "Cue breaks", "zero": True},
        ],
        "width": 11.0,
        "height": 3.8,
    },
}
