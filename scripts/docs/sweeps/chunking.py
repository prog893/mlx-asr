"""Sweep and comparison charts for docs/benchmarks/chunking.md.

The chunk/batch default is per machine (30s/B128 on the M2 Ultra, 60s/B32 on the M4), so
a chart drawn from one machine rings that machine's default and says which machine.
Difference panels plot the paired differences and 95% CIs exactly as the page prints
them, with the sign convention the page uses for that table named in the axis label.
"""

SWEEPS = {
    "chunking-length": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: chunk length (batch shrinks as chunks grow)",
        "basis": "One 935s clip, plain CER, M2 Ultra, 4-bit, no overlap; batch changes "
                 "with chunk length. The M2 Ultra uses 30s chunks because the corpus "
                 "ties 30s and 60s, so throughput decides, and 30s is fastest here.",
        "xlabel": "chunk length / batch", "ylabel": "",
        "scale": "category",
        "series": ["x realtime", "CER"],
        "rows": [
            ("20s / B48", "| 20s | 48 |", ("25.6x", "12.46%")),
            ("30s / B32", "| 30s | 32 |", ("31.0x", "9.13%")),
            ("60s / B16", "| 60s | 16 |", ("21.2x", "7.37%")),
            ("90s / B16", "| 90s | 16 |", ("17.2x", "7.99%")),
            ("120s / B8", "| 120s | 8 |", ("16.1x", "7.59%")),
            ("180s / B8", "| 180s | 8 |", ("11.6x", "7.56%")),
        ],
        "chosen": "30s / B32", "chosen_series": "x realtime",
        "chosen_label": "30s chosen (M2 Ultra)",
        "width": 10.4,
        "panels": [
            {"series": [1], "ylabel": "CER % (lower is better)", "title": "accuracy"},
            {"series": [0], "ylabel": "x realtime (higher is faster)", "title": "speed"},
        ],
    },
    "chunking-length-paired": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: paired chunk-length differences on one clip",
        "basis": "One 935s clip, M2 Ultra, paired over 40 regions; positive means 60s has "
                 "the lower CER. A CI clear of zero is a resolved difference.",
        "xlabel": "comparison", "ylabel": "CER points saved by 60s",
        "scale": "category", "connect": False,
        "series": ["60s minus the other arm"],
        "rows": [
            ("60s vs 30s", "| 60s beats 30s |", ("1.85",), ("[+0.71, +3.24]",)),
            ("60s vs 90s", "| 60s beats 90s |", ("0.62",), ("[-0.36, +1.69]",)),
        ],
        "panels": [{"series": [0], "ylabel": "CER points saved by 60s",
                    "title": "paired difference, 95% CI", "zero_line": True}],
        "width": 6.0,
    },
    "chunking-30-60": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: 30s/B32 against 60s/B16 on the corpus (M4)",
        "basis": "20-file corpus, M4 16GB, sequential, delay 2400ms, kv8. Both paired CIs "
                 "span zero, so the M4 chunk length (60s) is set by throughput on that machine.",
        "xlabel": "config", "ylabel": "",
        "scale": "category", "connect": False,
        "series": ["JP coverage CER", "EN coverage WER (3 files)",
                   "JP: 60s minus 30s", "EN: 60s minus 30s"],
        "rows": [
            ("60s / B16", "| 60s / b16 (M4 chunk length) |", ("16.29%", "26.14%", None, None)),
            ("30s / B32", "| 30s / b32 |", ("16.19%", "25.24%", "+0.10", "+0.90"),
             (None, None, "[-1.89, +2.03]", "[-0.27, +1.69]")),
        ],
        "chosen": "60s / B16", "chosen_series": "JP coverage CER",
        "chosen_label": "60s chosen (M4)",
        "width": 9.0,
        "panels": [
            {"series": [0, 1], "ylabel": "error % (lower is better)", "title": "accuracy",
             "min_span": 12},
            {"series": [2, 3], "ylabel": "60s minus 30s (points)",
             "title": "paired difference, 95% CI (above 0: 30s better)", "zero_line": True},
        ],
    },
    "chunking-overlap": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: prefix overlap between chunks",
        "basis": "One 935s clip, plain CER, M2 Ultra. 30s chunks at batch 32, kv8; "
                 "60s chunks measured at three overlaps only. The default stays 0s because "
                 "the clip win reversed sign on the 7-file subset (next chart).",
        "xlabel": "overlap", "ylabel": "CER % (lower is better)",
        "scale": "linear",
        "unit": "s",
        "series": ["30s chunks", "60s chunks"],
        "rows": [
            (0, "| 0s | 8.73% |", ("8.73%", None)),
            (4, "| 4s | 7.30% |", ("7.30%", None)),
            (6, "| 6s |", ("7.61%", None)),
            (7, "| 7s |", ("7.63%", None)),
            (8, "| 8s | **7.25%** |", ("7.25%", None)),
            (10, "| 10s |", ("7.56%", None)),
            (12, "| 12s |", ("7.80%", None)),
            (15, "| 15s |", ("11.20%", None)),
            (0, "| 0s | 7.37% |", (None, "7.37%")),
            (4, "| 4s | 7.59% |", (None, "7.59%")),
            (8, "| 8s | 8.06% |", (None, "8.06%")),
        ],
        "chosen": 0,
        "chosen_series": "30s chunks",
    },
    "chunking-overlap-paired": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: does prefix overlap help, paired",
        "basis": "Paired differences as printed; positive means overlap lowered the error. "
                 "Overlap defaults to 0s because the one resolved win is on one clip and "
                 "the 7-file subset puts the estimate on the losing side.",
        "xlabel": "material", "ylabel": "error points saved by overlap",
        "scale": "category", "connect": False,
        "series": ["overlap minus none"],
        "rows": [
            ("one clip,\n30s chunks", "| one clip, 30s chunks |", ("+1.80",), ("[+0.62, +3.20]",)),
            ("one clip,\n60s chunks", "| one clip, 60s chunks |", ("-0.69",), ("[-1.47, +0.07]",)),
            ("7-file\nsubset", "| 7-file subset |", ("-1.47",), ("[-4.33, +2.36]",)),
        ],
        "panels": [{"series": [0], "ylabel": "error points saved by overlap",
                    "title": "paired difference, 95% CI", "zero_line": True}],
        "width": 6.6,
    },
    "chunking-vad-clip": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: energy against VAD cut points on one clip",
        "basis": "One 935s clip, M2 Ultra, 4-bit, kv8. Every VAD arm has a higher CER than "
                 "its energy pair, so energy is the default cut rule.",
        "xlabel": "config", "ylabel": "",
        "scale": "category", "connect": False,
        "series": ["CER", "lenient CER", "x realtime"],
        "rows": [
            ("30s\nenergy", "| 30s, energy |", ("8.73%", "8.42%", "21.9x")),
            ("30s\nVAD", "| 30s, VAD |", ("10.75%", "10.39%", "20.4x")),
            ("30s, energy\nov 8s", "| 30s, energy, overlap 8s |", ("7.25%", "7.04%", "18.7x")),
            ("30s, VAD\nov 8s", "| 30s, VAD, overlap 8s |", ("8.04%", "7.73%", "18.0x")),
            ("60s\nenergy", "| 60s, energy |", ("7.37%", "7.11%", "21.0x")),
            ("60s\nVAD", "| 60s, VAD |", ("10.25%", "9.85%", "20.1x")),
        ],
        "chosen": "30s\nenergy", "chosen_series": "CER",
        "chosen_label": "default (M2 Ultra)",
        "width": 11.0,
        "panels": [
            {"series": [0, 1], "ylabel": "CER % (lower is better)", "title": "accuracy"},
            {"series": [2], "ylabel": "x realtime (higher is faster)", "title": "speed"},
        ],
    },
    "chunking-vad-corpus": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: energy against VAD cut points on the corpus",
        "basis": "20-file corpus, idle M2 Ultra, 60s/B16, 2026-08-24. Energy is never "
                 "behind and VAD costs a dependency, so a marginal VAD result keeps it off.",
        "xlabel": "cut points", "ylabel": "",
        "scale": "category", "connect": False,
        "series": ["JP coverage CER", "EN coverage WER (3 files)",
                   "JP: VAD minus energy", "EN: VAD minus energy", "x realtime"],
        "rows": [
            ("energy", "| energy (default) |", ("16.21%", "22.43%", None, None, "19.8x")),
            ("VAD", "| VAD |", ("16.68%", "25.95%", "+0.47", "+3.52", "19.3x"),
             (None, None, "[-0.84, +2.04]", "[+0.06, +5.62]", None)),
        ],
        "chosen": "energy", "chosen_series": "JP coverage CER",
        "width": 12.0,
        "panels": [
            {"series": [0, 1], "ylabel": "error % (lower is better)", "title": "accuracy",
             "min_span": 12},
            {"series": [2, 3], "ylabel": "VAD minus energy (points)",
             "title": "paired diff., 95% CI (above 0: energy better)", "zero_line": True},
            {"series": [4], "ylabel": "x realtime (higher is faster)", "title": "speed",
             "min_span": 4},
        ],
    },
    "chunking-carry": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: carrying context across seams",
        "basis": "One 935s clip, M2 Ultra, 30s chunks, batch 32. Carrying context saves "
                 "0.17 points at half the speed or less, so the default carries none.",
        "xlabel": "variant", "ylabel": "",
        "scale": "category", "connect": False,
        "series": ["CER", "x realtime"],
        "rows": [
            ("none", "| none |", ("8.73%", "33.9x")),
            ("carry_pair", "| carry_pair |", ("8.56%", "16.8x")),
            ("carry", "| carry |", ("8.56%", "5.3x")),
            ("static\nkeywords", "| static keywords |", ("9.11%", "33.5x")),
        ],
        "chosen": "none", "chosen_series": "x realtime",
        "width": 9.2,
        "panels": [
            {"series": [0], "ylabel": "CER % (lower is better)", "title": "accuracy",
             "min_span": 2},
            {"series": [1], "ylabel": "x realtime (higher is faster)", "title": "speed",
             "zero": True},
        ],
    },
    "chunking-silence": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: dropping silence before decode",
        "basis": "20-file corpus, idle M2 Ultra, 4-bit, 2026-08-24. The paired CI spans "
                 "zero, and a flag that discards input stays opt-in on a tie.",
        "xlabel": "--compact-silence", "ylabel": "",
        "scale": "category", "connect": False,
        "series": ["JP coverage CER", "EN coverage WER (3 files)",
                   "JP: on minus off", "x realtime"],
        "rows": [
            ("off", "| off (default) |", ("16.21%", "22.43%", None, "19.8x")),
            ("on", "| `--compact-silence` |", ("16.00%", "22.43%", "-0.21", "20.5x"),
             (None, None, "[-0.70, +0.22]", None)),
        ],
        "chosen": "off", "chosen_series": "JP coverage CER",
        "width": 12.0,
        "panels": [
            {"series": [0, 1], "ylabel": "error % (lower is better)", "title": "accuracy",
             "min_span": 12},
            {"series": [2], "ylabel": "on minus off (points)",
             "title": "paired diff., 95% CI (below 0: on better)", "zero_line": True},
            {"series": [3], "ylabel": "x realtime (higher is faster)", "title": "speed",
             "min_span": 4},
        ],
    },
    "chunking-silence-precision": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: silence compaction at four precisions",
        "basis": "20-file corpus, M2 Ultra, JP coverage CER. Every precision is a tie and "
                 "every one is faster on; the default stays off because removing input "
                 "should be opted into.",
        "xlabel": "precision", "ylabel": "",
        "scale": "category", "connect": False,
        "series": ["compaction off", "compaction on", "off minus on",
                   "x realtime, off", "x realtime, on"],
        "rows": [
            ("4-bit", "| 4-bit (default) |",
             ("16.21%", "16.00%", "+0.21", "19.8x", "20.5x"),
             (None, None, "[-0.22, +0.70]", None, None)),
            ("8-bit", "| 8-bit |", ("15.27%", "15.30%", "-0.03", "19.8x", "20.4x"),
             (None, None, "[-0.63, +0.42]", None, None)),
            ("mxfp8", "| mxfp8 |", ("15.86%", "15.78%", "+0.08", "19.4x", "20.7x"),
             (None, None, "[-0.63, +0.65]", None, None)),
            ("nvfp4", "| nvfp4 |", ("16.07%", "16.05%", "+0.02", "19.4x", "20.2x"),
             (None, None, "[-0.97, +0.78]", None, None)),
        ],
        "chosen": "4-bit", "chosen_series": "compaction off",
        "width": 13.0,
        "panels": [
            {"series": [0, 1], "ylabel": "JP coverage CER % (lower is better)",
             "title": "accuracy", "min_span": 4},
            {"series": [2], "ylabel": "off minus on (points)",
             "title": "paired diff., 95% CI (above 0: on better)", "zero_line": True},
            {"series": [3, 4], "ylabel": "x realtime (higher is faster)", "title": "speed",
             "min_span": 4},
        ],
    },
    "chunking-composite": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: the --fast bundle and its parts",
        "basis": "20-file corpus, idle M2 Ultra, JP coverage CER. Every arm ties on "
                 "accuracy (every CI spans zero), so the fastest arm, 30s/B32/ov0, set the "
                 "M2 Ultra chunk length; its batch was later raised to 128.",
        "xlabel": "chunk / batch / overlap", "ylabel": "",
        "scale": "category", "connect": False,
        "series": ["JP coverage CER", "x realtime"],
        "rows": [
            ("60s/B16\nov0", "| 60s / B16 / ov0 (the old default) |", ("16.21%", "19.8x")),
            ("30s/B32\nov8", "| 30s / B32 / ov8 (what", ("16.79%", "23.5x")),
            ("30s/B16\nov0", "| 30s / B16 / ov0 |", ("16.28%", "19.7x")),
            ("60s/B32\nov0", "| 60s / B32 / ov0 |", ("16.25%", "24.9x")),
            ("60s/B16\nov8", "| 60s / B16 / ov8 |", ("16.32%", "17.7x")),
            ("30s/B32\nov0", "| **30s / B32 / ov0** |", ("16.22%", "28.9x")),
        ],
        "chosen": "30s/B32\nov0", "chosen_series": "x realtime",
        "chosen_label": "30s chosen (M2 Ultra)",
        "width": 11.0,
        "panels": [
            {"series": [0], "ylabel": "JP coverage CER % (lower is better)",
             "title": "accuracy", "min_span": 4},
            {"series": [1], "ylabel": "x realtime (higher is faster)", "title": "speed",
             "zero": True},
        ],
    },
    "chunking-machines": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: the faster chunk/batch pair depends on the machine",
        "basis": "M2 Ultra from the 20-file corpus; M4 printed as a range, drawn as its "
                 "two ends. Each machine takes the chunk length of its faster pair: 30s "
                 "on the Ultra, 60s on the M4.",
        "xlabel": "chunk / batch", "ylabel": "x realtime",
        "scale": "category", "connect": False,
        "layout": "columns",
        "series": ["M2 Ultra 128GB", "M4 16GB, low end of range",
                   "M4 16GB, high end of range"],
        "series_keys": ["| M2 Ultra 128GB |", "| M4 16GB |", "| M4 16GB |"],
        "rows": [
            ("60s / B16", "", ("19.8x", "1.9", "2.0x")),
            ("30s / B32", "", ("28.9x", "1.5", "1.9x")),
        ],
        "chosen": "30s / B32", "chosen_series": "M2 Ultra 128GB",
        "chosen_label": "30s chosen (M2 Ultra)",
        "width": 9.0,
        "panels": [
            {"series": [0], "ylabel": "x realtime (higher is faster)",
             "title": "M2 Ultra, 60 GPU cores", "zero": True},
            {"series": [1, 2], "ylabel": "x realtime (higher is faster)",
             "title": "M4, 10 GPU cores (60s chosen)", "min_span": 1.5},
        ],
    },
    "chunking-seams": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: where the edit operations fall relative to chunk boundaries",
        "basis": "One 935s clip, 30s chunks. A region whose share of edits exceeds its "
                 "share of audio is enriched; the first 3s of a chunk is 2.24x.",
        "xlabel": "region", "ylabel": "share %",
        "scale": "category", "connect": False,
        "series": ["share of edits", "share of audio"],
        "rows": [
            ("first 3s", "| first 3s of a chunk |", ("22.3%", "9.9%")),
            ("last 3s", "| last 3s of a chunk |", ("12.9%", "9.9%")),
            ("elsewhere", "| elsewhere |", ("64.8%", "80.1%")),
        ],
        "zero": True,
        "width": 6.6,
    },
    "chunking-batch": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: batch size end to end at 30s chunks (M2 Ultra)",
        "basis": "20-file corpus, M2 Ultra 128GB, 30s chunks, kv8, delay 2400ms, one run "
                 "per batch. A file of 32 chunks or fewer fits in one batch from B32 up, "
                 "so only the 8 longer files can gain; accuracy moves under 0.1 point.",
        "xlabel": "max batch (log scale)", "ylabel": "",
        "scale": "log",
        "series": ["all 20 files", "12 files of 32 chunks or fewer",
                   "8 files over 32 chunks", "peak memory"],
        "rows": [
            (16, "| 16 |", ("19.3x", "19.7x", "19.2x", "6.56GB")),
            (32, "| 32 |", ("28.0x", "27.9x", "28.0x", "7.23GB")),
            (48, "| 48 |", ("26.4x", "24.2x", "27.4x", "8.14GB")),
            (64, "| 64 |", ("31.9x", "25.7x", "35.2x", "9.17GB")),
            (128, "| **128** |", ("33.6x", "28.1x", "36.3x", "11.33GB")),
        ],
        "chosen": 128, "chosen_series": "all 20 files",
        "chosen_label": "default (M2 Ultra)",
        "width": 10.4,
        "panels": [
            {"series": [0, 1, 2], "ylabel": "x realtime (higher is faster)",
             "title": "speed", "zero": True},
            {"series": [3], "ylabel": "peak GB", "title": "memory", "zero": True},
        ],
    },
    "chunking-batch-m4": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: batch size end to end at 60s chunks (M4)",
        "basis": "20-file corpus, M4 16GB, 60s chunks, kv8, delay 2400ms, one run per "
                 "batch on AC power. A file of 16 chunks or fewer fits in one batch from "
                 "B16 up; B48 exceeds the M4's ~8.4GB KV wall, so B32 is the largest batch "
                 "that runs reliably.",
        "xlabel": "max batch", "ylabel": "",
        "scale": "category",
        "series": ["all 20 files", "12 files of 16 chunks or fewer",
                   "8 files over 16 chunks", "peak memory"],
        "rows": [
            ("16", "| 16 |", ("4.66x", "4.14x", "4.89x", "6.60GB")),
            ("24", "| 24 |", ("5.62x", "4.27x", "6.42x", "7.34GB")),
            ("32", "| **32** |", ("5.46x", "4.08x", "6.30x", "8.08GB")),
        ],
        "chosen": "32", "chosen_series": "all 20 files",
        "chosen_label": "default (M4)",
        "width": 10.4,
        "panels": [
            {"series": [0, 1, 2], "ylabel": "x realtime (higher is faster)",
             "title": "speed", "zero": True},
            {"series": [3], "ylabel": "peak GB", "title": "memory", "zero": True},
        ],
    },
}
