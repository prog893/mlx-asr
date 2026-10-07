"""Charts for docs/benchmarks/quantization.md.

The weight-precision ladder is the shared `precision` chart (chart_data.PRECISION). The
KV-cache chart uses an 8-point minimum y span, the same floor as the precision panels,
because its three arms sit within 0.43 points and an auto-scaled axis drew that tie as a
steep slope.
"""

SWEEPS = {
    "quantization-kv": {
        "doc": "docs/benchmarks/quantization.md",
        "title": "Voxtral: KV cache precision (a tie within noise)",
        "basis": "20-file corpus, Japanese files, M2 Ultra. The y axis spans 8 points, so "
                 "flat means a tie.",
        "xlabel": "KV cache", "ylabel": "JP coverage CER %",
        "scale": "category", "unit": "",
        "series": ["JP coverage CER"],
        "rows": [("unquantized", "| unquantized KV |", ("16.38%",)),
                 ("kv8", "| **kv8 (default)** |", ("16.21%",)),
                 ("kv4", "| kv4 |", ("15.95%",))],
        "chosen": "kv8",
        "min_span": 8,
    },
}
