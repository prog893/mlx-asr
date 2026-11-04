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
        "source": "ultra-voxtral-v1-window.json#sweep_3b_bf16_c15,sweep_3b_bf16_c60,"
                  "sweep_3b_bf16_c120;ultra-voxtral-v1.json#3b_bf16_c30",
    },
    "voxtral-v1-precision-3b": {
        "doc": _DOC,
        "title": "voxtral-v1 Mini 3B: precision",
        "basis": f"20-file corpus, {_HOST}, 30s windows. 8bit ties bf16, so bf16 comes "
                 "first where it fits (--size 3B on 128GB) and a 16GB Mac runs 8bit; "
                 "4bit loses 8 points on Japanese.",
        "xlabel": "precision",
        "scale": "category", "connect": False,
        "series": ["Japanese CER", "English WER (3 files)", "peak GPU memory"],
        "rows": [
            ("4bit", "| 4bit |", ("44.60%", "18.08%", "5.25GB")),
            ("8bit", "| **8bit (default on 16GB)** |", ("36.41%", "17.86%", "7.28GB")),
            ("bf16", "| **bf16 (default with `--size 3B` on 128GB)** |",
             ("36.84%", "17.77%", "10.91GB")),
        ],
        "panels": [
            {"series": [0, 1], "ylabel": "error % (lower is better)", "title": "accuracy"},
            {"series": [2], "ylabel": "peak GPU memory (GB)", "zero": True, "title": "cost"},
        ],
        "chosen": "8bit", "chosen_label": "default on 16GB", "chosen_series": "Japanese CER",
        "source": "ultra-voxtral-v1.json#3b_4bit_c30,3b_8bit_c30,3b_bf16_c30",
    },
    "voxtral-v1-precision-24b": {
        "doc": _DOC,
        "title": "voxtral-v1 Small 24B: precision",
        "basis": f"20-file corpus, {_HOST}, 30s windows; bars are 95% file-bootstrap CIs "
                 "of the paired difference. 8bit ties bf16, so bf16 comes first where its "
                 "50GB peak fits (128GB), then 8bit; 4bit's English loss resolves, "
                 "so it comes last.",
        "xlabel": "precision",
        "scale": "category", "connect": False,
        "series": ["Japanese CER", "English WER (3 files)",
                   "Japanese, vs 8bit", "English, vs 8bit", "peak GPU memory"],
        "rows": [
            ("4bit", "| 4bit |",
             ("28.06%", "17.89%", "+0.62", "+1.03", "16.27GB"),
             (None, None, "[-2.72, +3.99]", "[+0.06, +2.87]", None)),
            ("8bit", "| 8bit |",
             ("27.44%", "16.86%", "0.00", "0.00", "27.92GB")),
            ("bf16", "| **bf16 (default on 128GB)** |",
             ("26.98%", "16.86%", "-0.46", "0.00", "50.08GB"),
             (None, None, "[-1.60, +0.27]", None, None)),
        ],
        "panels": [
            {"series": [0, 1], "ylabel": "error % (lower is better)", "title": "accuracy"},
            {"series": [2, 3], "ylabel": "difference vs 8bit (points)",
             "title": "against 8bit", "zero_line": True},
            {"series": [4], "ylabel": "peak GPU memory (GB)", "zero": True, "title": "cost"},
        ],
        "chosen": "bf16", "chosen_label": "default on 128GB", "chosen_series": "Japanese CER",
        "source": "ultra-voxtral-v1.json#24b_4bit_c30,24b_8bit_c30,24b_bf16_c30",
    },
    "voxtral-v1-language": {
        "doc": _DOC,
        "title": "voxtral-v1 3B: language forced or detected",
        "basis": f"20-file corpus, {_HOST}, bf16, 30s. Detection writes 30.7% "
                 "of the Japanese output in Latin script (12 recordings counted); pass --language for Japanese.",
        "xlabel": "language",
        "scale": "category", "connect": False,
        "series": ["Japanese CER", "English WER (3 files)", "Latin-script share of JP output"],
        "rows": [
            ("forced", "| forced from the reference |", ("36.84%", "17.77%", "0.6%")),
            ("detected", "| detected by the model |", ("35.70%", "16.36%", "30.7%")),
        ],
        "panels": [
            {"series": [0, 1], "ylabel": "error % (lower is better)", "title": "accuracy"},
            {"series": [2], "ylabel": "Latin script in JP output (%)", "zero": True,
             "title": "wrong-script output"},
        ],
        "chosen": None, "width": 7.6,
        "source": "ultra-voxtral-v1.json#3b_bf16_c30,3b_bf16_c30_auto;ultra-derived-measures.json#derive_v1latin",
    },
}
