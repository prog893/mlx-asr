"""Charts for docs/benchmarks/engines/voxtral.md, copied from its tables."""

SWEEPS = {
    "voxtral-corpus-growth": {
        "doc": "docs/benchmarks/engines/voxtral.md",
        "title": "Voxtral Realtime at its default configuration, as the corpus grew",
        "basis": ("Japanese files of three corpus versions, M2 Ultra, one run each. "
                  "All three are the default configuration, so none is ringed."),
        "xlabel": "corpus (Japanese files)", "ylabel": "JP coverage CER % (lower is better)",
        "scale": "category", "connect": False,
        "series": ["Japanese coverage CER"],
        "rows": [
            ("original (5)", "| original | 5 |", ("16.44%",)),
            ("grown (12)", "| grown | 12 |", ("16.08%",)),
            ("final (17)", "| final | 17 |", ("16.22%",)),
        ],
        "chosen": None,
        "min_span": 4,
    },
}
