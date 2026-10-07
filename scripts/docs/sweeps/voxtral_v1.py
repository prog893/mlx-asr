"""Sweep charts for docs/benchmarks/engines/voxtral-v1.md, copied from its tables."""

_DOC = "docs/benchmarks/engines/voxtral-v1.md"
_HOST = "M2 Ultra 128GB (Mac14,14)"

SWEEPS = {
    "voxtral-v1-window": {
        "doc": _DOC,
        "title": "voxtral-v1 3B: decode window",
        "basis": f"7-file subset, {_HOST}, bf16. The default is the window with the "
                 "lowest Japanese CER; English prefers 60s by about a point.",
        "xlabel": "window length", "ylabel": "error % (lower is better)",
        "scale": "log", "unit": "s",
        "series": ["Japanese CER", "English WER (2 files)"],
        "rows": [
            (15, "| 15s |", ("45.81%", "22.47%")),
            (30, "| **30s (default)** |", ("44.27%", "21.70%")),
            (60, "| 60s |", ("45.97%", "20.72%")),
            (120, "| 120s |", ("57.54%", "21.17%")),
        ],
        "chosen": 30, "chosen_series": "Japanese CER",
    },
    "voxtral-v1-precision-3b": {
        "doc": _DOC,
        "title": "voxtral-v1 Mini 3B: precision",
        "basis": f"20-file corpus, {_HOST}, 30s windows. The default is the cheapest "
                 "build that ties bf16; 4bit loses 8 points on Japanese.",
        "xlabel": "precision",
        "scale": "category", "connect": False,
        "series": ["Japanese CER", "English WER (3 files)", "peak GPU memory"],
        "rows": [
            ("4bit", "| 4bit |", ("44.54%", "18.08%", "5.25GB")),
            ("8bit", "| **8bit (default)** |", ("36.52%", "17.86%", "7.28GB")),
            ("bf16", "| bf16 |", ("37.16%", "17.77%", "10.91GB")),
        ],
        "panels": [
            {"series": [0, 1], "ylabel": "error % (lower is better)", "title": "accuracy"},
            {"series": [2], "ylabel": "peak GPU memory (GB)", "zero": True, "title": "cost"},
        ],
        "chosen": "8bit", "chosen_series": "Japanese CER",
    },
    "voxtral-v1-precision-24b": {
        "doc": _DOC,
        "title": "voxtral-v1 Small 24B: precision",
        "basis": f"20-file corpus, {_HOST}, 30s windows; bars are 95% file-bootstrap CIs "
                 "of the paired difference. The default is the cheapest build with no "
                 "resolved loss against bf16: 4bit's English loss resolves, bf16's "
                 "Japanese gain does not.",
        "xlabel": "precision",
        "scale": "category", "connect": False,
        "series": ["Japanese CER", "English WER (3 files)",
                   "Japanese, vs 8bit", "English, vs 8bit", "peak GPU memory"],
        "rows": [
            ("4bit", "| 4bit |",
             ("28.14%", "17.89%", "+0.59", "+1.03", "16.27GB"),
             (None, None, "[-2.65, +3.83]", "[+0.06, +2.87]", None)),
            ("8bit", "| **8bit (default)** |",
             ("27.56%", "16.86%", "0.00", "0.00", "27.92GB")),
            ("bf16", "| bf16 |",
             ("27.10%", "16.86%", "-0.46", "0.00", "50.08GB"),
             (None, None, "[-1.56, +0.25]", None, None)),
        ],
        "panels": [
            {"series": [0, 1], "ylabel": "error % (lower is better)", "title": "accuracy"},
            {"series": [2, 3], "ylabel": "difference vs 8bit (points)",
             "title": "against the default", "zero_line": True},
            {"series": [4], "ylabel": "peak GPU memory (GB)", "zero": True, "title": "cost"},
        ],
        "chosen": "8bit", "chosen_series": "Japanese CER",
    },
    "voxtral-v1-language": {
        "doc": _DOC,
        "title": "voxtral-v1 3B: language forced or detected",
        "basis": f"20-file corpus, {_HOST}, bf16, 30s. Detection writes a quarter "
                 "of the Japanese output in Latin script; pass --language for Japanese.",
        "xlabel": "language",
        "scale": "category", "connect": False,
        "series": ["Japanese CER", "English WER (3 files)", "Latin-script share of JP output"],
        "rows": [
            ("forced", "| forced from the reference |", ("37.16%", "17.77%", "0.7%")),
            ("detected", "| detected by the model |", ("36.06%", "16.36%", "24.7%")),
        ],
        "panels": [
            {"series": [0, 1], "ylabel": "error % (lower is better)", "title": "accuracy"},
            {"series": [2], "ylabel": "Latin script in JP output (%)", "zero": True,
             "title": "wrong-script output"},
        ],
        "chosen": None, "width": 7.6,
    },
}
