"""Charts for docs/benchmarks/engines/voxtral.md, copied from its tables."""

SWEEPS = {
    "voxtral-corpus-growth": {
        "doc": "docs/benchmarks/engines/voxtral.md",
        "title": "Voxtral Realtime at its earlier 4-bit default, as the corpus grew",
        "basis": ("Japanese files of two corpus versions, M2 Ultra, one run each. "
                  "Both are the earlier 4-bit default configuration, so none is ringed."),
        "xlabel": "corpus (Japanese files)", "ylabel": "JP coverage CER % (lower is better)",
        "scale": "category", "connect": False,
        "series": ["Japanese coverage CER"],
        "rows": [
            ("original (5)", "| original | 5 |", ("16.44%",)),
            ("final (17)", "| final | 17 |", ("15.73%",)),
        ],
        "chosen": None,
        "min_span": 4,
        "source": "ultra-voxtral-prompt-trap.json#prompt_trap7_none;"
                  "ultra-voxtral-batch.json#c30b32,c30b32_r2,c30b32_r3;"
                  "ultra-voxtral-headline.json#vox_default_c30b32_kv8",
    },
}
