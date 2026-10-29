"""Sweep charts for docs/benchmarks/engines/reazon.md, copied verbatim from its tables."""

SWEEPS = {
    "reazon-precision": {
        "doc": "docs/benchmarks/engines/reazon.md",
        "title": "reazon-k2: int8 against fp32",
        "basis": "17 Japanese files of the 20-file corpus, M2 Ultra, CPU, "
                 "30s windows. int8 is faster and smaller but drops whole "
                 "phrases on conversational audio, so the more accurate fp32 is the default.",
        "xlabel": "precision",
        "scale": "category", "unit": "",
        "connect": False,
        "series": ["Japanese CER", "x realtime", "peak RSS"],
        "panels": [
            {"series": [0], "ylabel": "coverage CER % (lower is better)", "title": "accuracy"},
            {"series": [1], "ylabel": "x realtime (higher is faster)", "zero": True,
             "title": "speed"},
            {"series": [2], "ylabel": "peak RSS, GB", "zero": True, "title": "memory"},
        ],
        "width": 10.0,
        "rows": [
            ("fp32", "| **fp32 (default)** |", ("30.06%", "52.9x", "3.28GB")),
            ("int8", "| int8 |", ("36.55%", "79.9x", "2.72GB")),
        ],
        "source": "ultra-picker-speed.json#pk_reazon_fp32_default_s1,pk_reazon_fp32_default_s2,pk_reazon_fp32_default_s3;ultra-engines-jp-only.json#jp_reazon_int8_c30_r1,jp_reazon_int8_c30_r2,jp_reazon_int8_c30_r3",
        "chosen": "fp32", "chosen_series": "Japanese CER",
    },
}
