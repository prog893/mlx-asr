"""Sweep charts for docs/benchmarks/input-level.md."""

SWEEPS = {
    "input-level-clamp": {
        "doc": "docs/benchmarks/input-level.md",
        "title": "Share of mel bins clamped at the floor, by input gain",
        "basis": "One recording, peaking at -5.5 dBFS as recorded.",
        "xlabel": "gain applied (dB, 0 = as recorded)",
        "ylabel": "mel bins clamped (%)",
        "scale": "linear", "unit": "",
        "series": ["mel bins clamped"],
        "rows": [
            (-20, "| -20dB |", ("65.5%",)),
            (-12, "| -12dB |", ("41.5%",)),
            (-6, "| -6dB |", ("24.0%",)),
            (0, "| unity |", ("9.9%",)),
            (6, "| +6dB |", ("2.8%",)),
        ],
        "chosen": 0,
        "chosen_label": "auto (0 dB on this recording)",
        "label_dy": -16,
    },
}
