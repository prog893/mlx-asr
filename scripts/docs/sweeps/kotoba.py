"""Sweep charts for docs/benchmarks/engines/kotoba.md, copied from its tables.

The page's other two charts (engines-kotoba-window, engines-kotoba-authors-chunk) are
defined in sweeps/engines.py.
"""

SWEEPS = {
    "kotoba-sequential": {
        "doc": "docs/benchmarks/engines/kotoba.md",
        "title": "kotoba under mlx-whisper's sequential driver",
        "basis": "5 JP files (7-file subset) and the narration clip, M2 Ultra. No default: "
                 "every config scores above 44%, so the CLI uses the chunked driver.",
        "xlabel": "weights and conditioning",
        "ylabel": "CER % (lower is better)",
        "scale": "category", "connect": False,
        "series": ["corpus coverage CER (5 files)", "narration CER"],
        "rows": [
            ("npz port", "| third-party npz port |", ("91.47%", "88.70%")),
            ("npz port\nno-condition", "| same, `--no-condition` | 53.20% |",
             ("53.20%", None)),
            ("own conversion", "| official weights, own MLX conversion |",
             ("94.23%", "79.88%")),
            ("own conversion\nno-condition", "| same, `--no-condition` | 53.53% |",
             ("53.53%", "44.78%")),
        ],
        "chosen": None,
        "width": 7.6,
        "zero": True,
    },
}
