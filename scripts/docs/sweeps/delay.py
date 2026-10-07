"""Sweep charts for docs/benchmarks/delay.md.

The error-by-delay chart (img/delay-*.svg) is drawn by shared code in gen_charts.py.
This module only carries the paired comparison against the default.
"""

SWEEPS = {
    "delay-paired": {
        "doc": "docs/benchmarks/delay.md",
        "title": "Paired difference against the 2400ms default, with 95% CIs",
        "basis": ("7-file subset, M2 Ultra 128GB. Zero is the 2400ms default; both "
                  "lower delays are worse, CIs clear of zero."),
        "xlabel": "--delay-ms compared with 2400ms",
        "ylabel": "vs 2400ms (points, + = worse)",
        "scale": "category", "unit": "ms",
        "connect": False,
        "series": ["paired difference"],
        "rows": [
            (480, "| 480ms vs 2400ms |", ("+9.07",), ("[+5.41, +14.25]",)),
            (960, "| 960ms vs 2400ms |", ("+4.02",), ("[+1.59, +8.84]",)),
        ],
        "chosen": None,
        "panels": [{"series": [0], "zero_line": True, "zero": True}],
    },
}
