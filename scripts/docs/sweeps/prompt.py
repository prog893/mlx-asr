"""Charts for docs/benchmarks/prompt.md. Values are verbatim from its tables.

All three are comparisons of unordered configs, so markers only on a category axis.
The language experiment draws its paired CIs on the two "vs none" panels; the instruction
trap prints CIs for two of its three rows, and the Japanese row's "CI spans zero" has no
numbers to draw.
"""

_ARMS = [
    ("none", "| none | - | 0 |"),
    ("instr\nen", "| instruction | en |"),
    ("instr\nja", "| instruction | ja |"),
    ("descr\nen", "| description | en |"),
    ("descr\nja", "| description | ja |"),
    ("topic\nfreq en", "| topic, freq | en |"),
    ("topic\nfreq ja", "| topic, freq | ja |"),
    ("topic\nnames en", "| topic, names | en |"),
    ("topic\nnames ja", "| topic, names | ja |"),
    ("terms\nfreq en", "| terms, freq | en |"),
    ("terms\nfreq ja", "| terms, freq | ja |"),
    ("terms\nnames en", "| terms, names | en |"),
    ("terms\nnames ja", "| terms, names | ja |"),
]
# (JP coverage CER, EN coverage WER, JP vs none, EN vs none, JP CI, EN CI), as printed;
# the chart draws EN WER as its difference against none only, which is the deciding number
_VALUES = [
    ("15.73%", "21.50%", None, None, None, None),
    ("17.13%", "36.68%", "+1.40", "+15.18", "[-0.19, +3.35]", "[+10.34, +20.26]"),
    ("15.99%", "80.53%", "+0.26", "+59.03", "[-0.77, +1.24]", "[+54.69, +63.82]"),
    ("17.73%", "95.21%", "+2.00", "+73.71", "[+0.43, +3.69]", "[+67.87, +94.14]"),
    ("15.90%", "69.50%", "+0.17", "+48.00", "[-0.79, +0.83]", "[+45.58, +49.36]"),
    ("18.13%", "97.45%", "+2.40", "+75.94", "[+0.68, +4.28]", "[+69.03, +94.26]"),
    ("15.58%", "86.66%", "-0.15", "+65.16", "[-1.01, +0.55]", "[+61.61, +70.04]"),
    ("17.92%", "96.44%", "+2.20", "+74.94", "[+0.37, +4.20]", "[+68.92, +94.02]"),
    ("15.63%", "85.44%", "-0.10", "+63.94", "[-0.94, +0.58]", "[+56.44, +70.04]"),
    ("20.46%", "24.10%", "+4.73", "+2.59", "[+2.09, +7.85]", "[+0.06, +3.89]"),
    ("16.67%", "25.72%", "+0.94", "+4.22", "[+0.36, +1.53]", "[+0.12, +6.57]"),
    ("18.30%", "23.54%", "+2.57", "+2.04", "[+0.92, +4.48]", "[+0.12, +3.14]"),
    ("15.80%", "34.66%", "+0.08", "+13.15", "[-0.74, +0.76]", "[+9.40, +16.12]"),
]
_DERIVED = ",".join(f"prompt20_derived_{v}#{c} {lang}" for v in ("freq", "names")
                    for c in ("topic", "terms") for lang in ("en", "ja"))

SWEEPS = {
    "prompt-language": {
        "doc": "docs/benchmarks/prompt.md",
        "title": "Voxtral: prompt content and language against no prompt",
        "basis": "20-file corpus (17 Japanese, 3 English), M2 Ultra, 30s chunks, batch 32, "
                 "kv8; topic and terms derived from the references (freq and names). No prompt "
                 "is the default because every arm costs English 2.0 to 75.9 WER points and "
                 "no arm resolves a Japanese gain; the best, -0.15, has a CI spanning zero.",
        "xlabel": "prompt arm (content, language)", "ylabel": "",
        "scale": "category", "unit": "", "connect": False,
        "series": ["Japanese CER (17 files)", "Japanese, vs none",
                   "English WER, vs none (3 files)"],
        "rows": [(x, key, (v[0], v[2], v[3]), (None, v[4], v[5]))
                 for (x, key), v in zip(_ARMS, _VALUES)],
        "chosen": "none", "chosen_series": "Japanese CER (17 files)",
        "panels": [
            {"series": [0], "ylabel": "JP coverage CER %", "title": "Japanese audio",
             "min_span": 8},
            {"series": [1], "ylabel": "CER points vs none", "title": "Japanese, against none",
             "zero_line": True, "min_span": 8},
            {"series": [2], "ylabel": "WER points vs none", "title": "English, against none",
             "zero_line": True},
        ],
        "width": 20, "height": 3.8,
        "source": "ultra-voxtral-prompt.json#prompt20_c30b32_kv8#none -,"
                  "prompt20_c30b32_kv8#instruction en,prompt20_c30b32_kv8#instruction ja,"
                  "prompt20_c30b32_kv8#description en,prompt20_c30b32_kv8#description ja;"
                  "ultra-voxtral-prompt-derived.json#" + _DERIVED,
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
            ("all 7, pooled", "| all 7, pooled |", ("+3.53",), ("[+0.41, +9.81]",)),
        ],
        "panels": [{"series": [0], "zero_line": True}],
        "source": "ultra-voxtral-prompt-trap.json#prompt_trap7_instruction,prompt_trap7_none",
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
            ("no prompt, no overlap", "| no prompt, no overlap |", ("8.63%",)),
            ("terms only", "| prompt only (term list) |", ("9.32%",)),
            ("description only", "| prompt only (description) |", ("8.73%",)),
            ("overlap 4s only", "| overlap 4s only |", ("7.16%",)),
            ("terms + overlap 4s", "| **prompt + overlap 4s** (term list) |", ("18.64%",)),
            ("description + overlap 4s", "| prompt + overlap 4s (description) |", ("12.60%",)),
        ],
        "chosen": "no prompt, no overlap", "zero": True,
        "source": "ultra-voxtral-prompt-overlap-corpus.json#po_none_ov0,po_terms_ov0,po_desc_ov0,po_none_ov4,po_terms_ov4,po_desc_ov4",
    },
}
