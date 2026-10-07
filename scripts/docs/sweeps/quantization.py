"""Charts for docs/benchmarks/quantization.md.

The weight-precision ladder is the shared `precision` chart (chart_data.PRECISION). The
KV-cache chart uses an 8-point minimum y span, the same floor as the precision panels,
because its three arms sit within 0.43 points and an auto-scaled axis drew that tie as a
steep slope. Its second panel is the paired difference the table prints (kv8 minus the
arm, negative means kv8 is better) with its 95% CI, which is what keeps kv8 the default
although kv4 is nominally lowest: both intervals span zero.
"""

SWEEPS = {
    "quantization-kv": {
        "doc": "docs/benchmarks/quantization.md",
        "title": "Voxtral: KV cache precision (a tie within noise)",
        "basis": "20-file corpus, Japanese files, M2 Ultra; the accuracy y axis spans 8 "
                 "points, so flat means a tie. kv8 stays the default because both paired "
                 "CIs span zero and a tie does not move a default.",
        "xlabel": "KV cache", "ylabel": "JP coverage CER %",
        "scale": "category", "unit": "",
        "connect": False,
        "series": ["JP coverage CER", "kv8 minus arm"],
        "panels": [
            {"series": [0], "ylabel": "JP coverage CER % (lower is better)",
             "title": "accuracy"},
            {"series": [1], "ylabel": "kv8 minus arm, CER points\n(below 0: kv8 better)",
             "title": "against the default (95% CI)", "zero_line": True},
        ],
        "rows": [("unquantized", "| unquantized KV |", ("16.38%", "-0.17"),
                  (None, "[-0.51, +0.12]")),
                 ("kv8", "| **kv8 (default)** |", ("16.21%", None)),
                 ("kv4", "| kv4 |", ("15.95%", "+0.27"), (None, "[-0.51, +1.27]"))],
        "chosen": "kv8", "chosen_series": "JP coverage CER",
        "min_span": 8,
    },
}
