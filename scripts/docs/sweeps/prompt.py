"""Charts for docs/benchmarks/prompt.md. Values are verbatim from its tables.

All three are comparisons of unordered configs, so markers only on a category axis.
The language experiment prints no per-arm CIs, so it has no error bars; the instruction
trap prints CIs for two of its three rows, and the Japanese row's "CI spans zero" has no
numbers to draw.
"""

_ARMS = [
    ("none", "| none | - | 0 |"),
    ("instr\nen", "| instruction | en |"),
    ("instr\nja", "| instruction | ja |"),
    ("descr\nen", "| description | en |"),
    ("descr\nja", "| description | ja |"),
    ("topic\nen", "| topic | en |"),
    ("topic\nja", "| topic | ja |"),
    ("terms\nen", "| terms | en |"),
    ("terms\nja", "| terms | ja |"),
]
# (JP coverage CER, EN coverage WER, JP vs none, EN vs none), as printed; the chart
# draws EN WER as its difference against none only, which is the deciding number
_VALUES = [
    ("16.22%", "25.24%", None, None),
    ("17.62%", "39.54%", "+1.40", "+14.30"),
    ("16.48%", "82.05%", "+0.26", "+56.81"),
    ("18.22%", "95.33%", "+2.00", "+70.09"),
    ("16.39%", "71.79%", "+0.17", "+46.55"),
    ("19.53%", "96.48%", "+3.31", "+71.25"),
    ("16.20%", "78.99%", "-0.02", "+53.76"),
    ("18.50%", "97.60%", "+2.28", "+72.37"),
    ("15.97%", "82.47%", "-0.25", "+57.23"),
]

SWEEPS = {
    "prompt-language": {
        "doc": "docs/benchmarks/prompt.md",
        "title": "Voxtral: prompt content and language against no prompt",
        "basis": "20-file corpus (17 Japanese, 3 English), M2 Ultra, 30s chunks, batch 32, "
                 "kv8. No prompt is the default because every arm costs English 14 to 72 "
                 "WER points and the best Japanese arm gains 0.25 CER points, inside a "
                 "resolution floor of about 1.6.",
        "xlabel": "prompt arm (content, language)", "ylabel": "",
        "scale": "category", "unit": "", "connect": False,
        "series": ["Japanese CER (17 files)", "Japanese, vs none",
                   "English WER, vs none (3 files)"],
        "rows": [(x, key, (v[0], v[2], v[3])) for (x, key), v in zip(_ARMS, _VALUES)],
        "chosen": "none", "chosen_series": "Japanese CER (17 files)",
        "panels": [
            {"series": [0], "ylabel": "JP coverage CER %", "title": "Japanese audio",
             "min_span": 8},
            {"series": [1], "ylabel": "CER points vs none", "title": "Japanese, against none",
             "zero_line": True, "min_span": 8},
            {"series": [2], "ylabel": "WER points vs none", "title": "English, against none",
             "zero_line": True},
        ],
        "width": 16, "height": 3.8,
    },
    "prompt-instruction": {
        "doc": "docs/benchmarks/prompt.md",
        "title": "Voxtral: an English instruction prompt against no prompt",
        "basis": "7-file subset, M2 Ultra; English in WER points, Japanese in CER points "
                 "(its CI spans zero, not printed). Zero is no prompt, the default; an "
                 "interval clear of zero is a resolved cost.",
        "width": 8.5,
        "xlabel": "files scored", "ylabel": "instruction minus none (points)",
        "scale": "category", "unit": "", "connect": False,
        "series": ["instruction vs none"],
        "rows": [
            ("English (2 files)", "| English (2 files) |", ("+13.77",), ("[+10.34, +15.25]",)),
            ("Japanese (5 files)", "| Japanese (5 files) |", ("+1.41",), (None,)),
            ("all 7, pooled", "| all 7, pooled |", ("+3.53",), ("[+0.41, +9.82]",)),
        ],
        "panels": [{"series": [0], "zero_line": True}],
    },
    "prompt-overlap": {
        "doc": "docs/benchmarks/prompt.md",
        "title": "Voxtral: prompt and overlap together",
        "basis": "One clip, plain CER, M2 Ultra, 30s chunks, batch 32, kv8. Ringed: the "
                 "default (no prompt, 0s overlap); overlap lost on the corpus (chunking.md), "
                 "and with both flags the CLI drops the prompt.",
        "width": 8.5,
        "xlabel": "config", "ylabel": "CER % (lower is better)",
        "scale": "category", "unit": "", "connect": False,
        "series": ["CER"],
        "rows": [
            ("no prompt, no overlap", "| no prompt, no overlap |", ("9.04%",)),
            ("prompt only", "| prompt only |", ("9.04%",)),
            ("overlap 4s only", "| overlap 4s only |", ("7.16%",)),
            ("prompt + overlap 4s", "| **prompt + overlap 4s** |", ("18.64%",)),
        ],
        "chosen": "no prompt, no overlap", "zero": True,
    },
}
