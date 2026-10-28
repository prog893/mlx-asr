"""Sweep and comparison charts for docs/benchmarks/chunking.md.

The chunk/batch default is per machine (30s/B128 on the M2 Ultra, 30s/B32 on the M4), so
a chart drawn from one machine rings that machine's default and says which machine.
Difference panels plot the paired differences and 95% CIs exactly as the page prints
them, with the sign convention the page uses for that table named in the axis label.
"""

SWEEPS = {
    "chunking-length": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: chunk length (batch shrinks as chunks grow)",
        "basis": "One 935s clip, plain CER, M2 Ultra, 4-bit, no overlap, KV unquantized at 20s "
                 "and 30s and kv8 from 60s up; batch changes with chunk length. The M2 Ultra uses 30s chunks because the corpus "
                 "ties 30s and 60s, so throughput decides, and 30s is fastest here.",
        "xlabel": "chunk length / batch", "ylabel": "",
        "scale": "category",
        "series": ["x realtime", "CER"],
        "rows": [
            ("20s / B48", "| 20s | 48 |", ("28.0x", "11.49%")),
            ("30s / B32", "| 30s | 32 |", ("32.5x", "8.66%")),
            ("60s / B16", "| 60s | 16 |", ("22.0x", "7.28%")),
            ("90s / B16", "| 90s | 16 |", ("18.4x", "7.94%")),
            ("120s / B8", "| 120s | 8 |", ("17.1x", "7.44%")),
            ("180s / B8", "| 180s | 8 |", ("12.4x", "7.49%")),
        ],
        "source": "ultra-voxtral-clip.json#chunk_c20b48_kv0#overlap 0,chunk_c30b32_kv0#overlap 0,ov60_0s#overlap 0,chunk_c90b16_kv8#overlap 0,chunk_c120b8_kv8#overlap 0,chunk_c180b8_kv8#overlap 0",
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
            ("60s vs 30s", "| 60s beats 30s |", ("1.37",), ("[+0.12, +2.89]",)),
            ("60s vs 90s", "| 60s beats 90s |", ("0.67",), ("[-0.33, +1.76]",)),
        ],
        "source": "ultra-voxtral-clip-paired.json#cmp_len_30v60,cmp_len_90v60",
        "panels": [{"series": [0], "ylabel": "CER points saved by 60s",
                    "title": "paired difference, 95% CI", "zero_line": True}],
        "width": 6.0,
    },
    "chunking-30-60": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: 30s/B32 against 60s/B16 on the corpus (M4)",
        "basis": "20-file corpus, M4 16GB, sequential, delay 2400ms, kv8. Both paired CIs "
                 "span zero, so chunk length is a throughput choice; these runs give 30s/B32 "
                 "6.37x (mean of 3 runs) against 4.66x, so the M4 ships 30s/B32.",
        "xlabel": "config", "ylabel": "",
        "scale": "category", "connect": False,
        "series": ["JP coverage CER", "EN coverage WER (3 files)",
                   "JP: 60s minus 30s", "EN: 60s minus 30s"],
        "rows": [
            ("60s / B16", "| 60s / b16 |", ("15.79%", "22.43%", None, None)),
            ("30s / B32", "| 30s / b32 |", ("15.70%", "21.50%", "+0.10", "+0.93"),
             (None, None, "[-1.92, +2.06]", "[-0.06, +1.69]")),
        ],
        "source": "m4-voxtral-chunking.json#c60b16,c30b32,c30b32_r2,c30b32_r3",
        "chosen": "30s / B32", "chosen_series": "JP coverage CER",
        "chosen_label": "30s chosen (M4)",
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
            (0, "| 0s | 8.63% |", ("8.63%", None)),
            (4, "| 4s | 7.16% |", ("7.16%", None)),
            (6, "| 6s |", ("7.44%", None)),
            (7, "| 7s |", ("7.47%", None)),
            (8, "| 8s | **7.13%** |", ("7.13%", None)),
            (10, "| 10s |", ("7.40%", None)),
            (12, "| 12s |", ("7.61%", None)),
            (15, "| 15s |", ("11.03%", None)),
            (0, "| 0s | 7.28% |", (None, "7.28%")),
            (4, "| 4s | 7.42% |", (None, "7.42%")),
            (8, "| 8s | 7.94% |", (None, "7.94%")),
        ],
        "source": "ultra-voxtral-clip.json#ov30_0s#overlap 0,ov30_4s#overlap 4,ov30_6s#overlap 6,ov30_7s#overlap 7,ov30_8s#overlap 8,ov30_10s#overlap 10,ov30_12s#overlap 12,ov30_15s#overlap 15,ov60_0s#overlap 0,ov60_4s#overlap 4,ov60_8s#overlap 8",
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
            ("one clip,\n30s chunks", "| one clip, 30s chunks |", ("+1.49",), ("[+0.36, +2.96]",)),
            ("one clip,\n60s chunks", "| one clip, 60s chunks |", ("-0.66",), ("[-1.44, +0.10]",)),
            ("7-file\nsubset", "| 7-file subset |", ("-1.47",), ("[-4.34, +2.33]",)),
        ],
        "source": ("ultra-mincut-inputs.json#mincut7_vox_c30b32_kv8_ov8,mincut7_vox_c30b32_kv8;"
                   "ultra-voxtral-clip-paired.json#cmp_ov30_0v8,cmp_ov60_0v8"),
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
            ("30s\nenergy", "| 30s, energy |", ("8.63%", "8.32%", "33.7x")),
            ("30s\nVAD", "| 30s, VAD |", ("10.34%", "9.99%", "34.1x")),
            ("30s, energy\nov 8s", "| 30s, energy, overlap 8s |", ("7.13%", "6.92%", "27.2x")),
            ("30s, VAD\nov 8s", "| 30s, VAD, overlap 8s |", ("8.20%", "7.94%", "26.7x")),
            ("60s\nenergy", "| 60s, energy |", ("7.28%", "7.02%", "22.6x")),
            ("60s\nVAD", "| 60s, VAD |", ("10.01%", "9.56%", "23.0x")),
        ],
        # CER and x realtime from the decode runs; lenient CER from score_vad
        "source": ("ultra-voxtral-clip.json#vad_c30b32_kv8_energy,vad_c30b32_kv8_vad,"
                   "vad_c30b32_kv8_ov8_energy,vad_c30b32_kv8_ov8_vad,vad_c60b16_kv8_energy,"
                   "vad_c60b16_kv8_vad;ultra-voxtral-clip-paired.json#score_vad"),
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
        "basis": "20-file corpus, M2 Ultra, 60s/B16, run on or after 2026-10-11. Energy is never "
                 "behind and VAD costs a dependency, so a marginal VAD result keeps it off.",
        "xlabel": "cut points", "ylabel": "",
        "scale": "category", "connect": False,
        "series": ["JP coverage CER", "EN coverage WER (3 files)",
                   "JP: VAD minus energy", "EN: VAD minus energy", "x realtime"],
        "rows": [
            ("energy", "| energy (default) |", ("15.72%", "22.43%", None, None, "19.6x")),
            ("VAD", "| VAD |", ("16.18%", "25.95%", "+0.46", "+3.52", "20.3x"),
             (None, None, "[-0.86, +2.09]", "[+0.06, +5.62]", None)),
        ],
        "source": "ultra-voxtral-precision.json#vox_w4bit_c60b16_kv8;ultra-voxtral-chunking.json#ch_vad_c60b16",
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
                 "0.12 points at half the speed or less, so the default carries none.",
        "xlabel": "variant", "ylabel": "",
        "scale": "category", "connect": False,
        "series": ["CER", "x realtime"],
        "rows": [
            ("none", "| none |", ("8.63%", "33.8x")),
            ("carry_pair", "| carry_pair |", ("8.51%", "16.7x")),
            ("carry", "| carry |", ("8.51%", "5.1x")),
        ],
        "source": ("ultra-voxtral-clip.json#carry_none#variant none,"
                   "carry_pair#variant carry_pair,carry_seq#variant carry"),
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
        "basis": "20-file corpus, M2 Ultra, 4-bit, run on or after 2026-10-11. The paired CI spans "
                 "zero, and a flag that discards input stays opt-in on a tie.",
        "xlabel": "--compact-silence", "ylabel": "",
        "scale": "category", "connect": False,
        "series": ["JP coverage CER", "EN coverage WER (3 files)",
                   "JP: on minus off", "x realtime"],
        "rows": [
            ("off", "| off (default) |", ("15.72%", "22.43%", None, "19.6x")),
            ("on", "| `--compact-silence` |", ("15.51%", "22.43%", "-0.21", "20.8x"),
             (None, None, "[-0.69, +0.22]", None)),
        ],
        "source": "ultra-voxtral-precision.json#vox_w4bit_c60b16_kv8;ultra-voxtral-chunking.json#ch_compact_c60b16_w4bit",
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
            ("4-bit", "| 4-bit |",
             ("15.72%", "15.51%", "+0.21", "19.6x", "20.8x"),
             (None, None, "[-0.22, +0.69]", None, None)),
            ("8-bit", "| 8-bit |", ("14.78%", "14.81%", "-0.03", "20.0x", "21.4x"),
             (None, None, "[-0.62, +0.41]", None, None)),
            ("mxfp8", "| mxfp8 |", ("15.36%", "15.28%", "+0.08", "19.9x", "21.1x"),
             (None, None, "[-0.61, +0.66]", None, None)),
            ("nvfp4", "| nvfp4 |", ("15.58%", "15.56%", "+0.02", "20.1x", "20.9x"),
             (None, None, "[-0.95, +0.77]", None, None)),
        ],
        "source": "ultra-voxtral-precision.json#vox_w4bit_c60b16_kv8,vox_w8bit_c60b16_kv8,vox_wmxfp8_c60b16_kv8,vox_wnvfp4_c60b16_kv8;ultra-voxtral-chunking.json#ch_compact_c60b16_w4bit,ch_compact_c60b16_w8bit,ch_compact_c60b16_wmxfp8,ch_compact_c60b16_wnvfp4",
        "chosen": None,
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
        "basis": "20-file corpus, M2 Ultra, JP coverage CER. Every arm ties on "
                 "accuracy (every CI spans zero), so the fastest arm, 30s/B32/ov0, set the "
                 "M2 Ultra chunk length; its batch was later raised to 128.",
        "xlabel": "chunk / batch / overlap", "ylabel": "",
        "scale": "category", "connect": False,
        "series": ["JP coverage CER", "x realtime"],
        "rows": [
            ("60s/B16\nov0", "| 60s / B16 / ov0 (the old default) |", ("15.72%", "20.4x")),
            ("30s/B32\nov8", "| 30s / B32 / ov8 (what", ("16.30%", "24.6x")),
            ("30s/B16\nov0", "| 30s / B16 / ov0 |", ("15.78%", "20.3x")),
            ("60s/B32\nov0", "| 60s / B32 / ov0 |", ("15.76%", "26.0x")),
            ("60s/B16\nov8", "| 60s / B16 / ov8 |", ("15.83%", "18.4x")),
            ("30s/B32\nov0", "| **30s / B32 / ov0** |", ("15.73%", "29.3x")),
        ],
        "source": "ultra-voxtral-precision.json#vox_w4bit_c60b16_kv8,vox_w4bit_c60b16_kv8_r2;ultra-voxtral-kv.json#vox_kv8_c60b16_w4bit;ultra-voxtral-chunking.json#ch_c30b32_ov8,ch_c30b16_ov0,ch_c60b32_ov0,ch_c60b16_ov8;ultra-voxtral-batch.json#c30b32,c30b32_r2,c30b32_r3",
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
        "title": "Voxtral: 30s/B32 against 60s/B16 on two machines",
        "basis": "20-file corpus, 4-bit; means of 3 runs except the M4 60s/B16, one run. "
                 "30s/B32 is the faster pair on "
                 "both machines, and both profiles ship 30s chunks.",
        "xlabel": "chunk / batch", "ylabel": "x realtime",
        "scale": "category", "connect": False,
        "layout": "columns",
        "series": ["M2 Ultra 128GB", "M4 16GB"],
        "series_keys": ["| M2 Ultra 128GB |", "| M4 16GB |"],
        "rows": [
            ("60s / B16", "", ("20.4x", "4.66x")),
            ("30s / B32", "", ("29.3x", "6.37x")),
        ],
        "source": ("ultra-voxtral-precision.json#vox_w4bit_c60b16_kv8,vox_w4bit_c60b16_kv8_r2;"
                   "ultra-voxtral-kv.json#vox_kv8_c60b16_w4bit;"
                   "ultra-voxtral-batch.json#c30b32,c30b32_r2,c30b32_r3;"
                   "m4-voxtral-chunking.json#c60b16,c30b32,c30b32_r2,c30b32_r3"),
        "chosen": "30s / B32", "chosen_series": "M2 Ultra 128GB",
        "chosen_label": "30s chosen (M2 Ultra)",
        "width": 9.0,
        "panels": [
            {"series": [0], "ylabel": "x realtime (higher is faster)",
             "title": "M2 Ultra, 60 GPU cores", "zero": True},
            {"series": [1], "ylabel": "x realtime (higher is faster)",
             "title": "M4, 10 GPU cores (ships 30s)", "zero": True},
        ],
    },
    "chunking-seams": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: where the edit operations fall relative to chunk boundaries",
        "basis": "One 935s clip, 30s chunks. A region whose share of edits exceeds its "
                 "share of audio is enriched; the first 3s of a chunk is 1.99x.",
        "xlabel": "region", "ylabel": "share %",
        "scale": "category", "connect": False,
        "series": ["share of edits", "share of audio"],
        "rows": [
            ("first 3s", "| first 3s of a chunk |", ("19.8%", "9.9%")),
            ("last 3s", "| last 3s of a chunk |", ("13.5%", "9.9%")),
            ("elsewhere", "| elsewhere |", ("66.7%", "80.1%")),
        ],
        "source": "ultra-voxtral-clip-paired.json#seam_c30",
        "zero": True,
        "width": 6.6,
    },
    "chunking-batch": {
        "doc": "docs/benchmarks/chunking.md",
        "title": "Voxtral: batch size end to end at 30s chunks (M2 Ultra)",
        "basis": "20-file corpus, M2 Ultra 128GB, 30s chunks, kv8, delay 2400ms; B32 and "
                 "B64 are means of 3 runs. A file of 32 chunks or fewer fits in one batch from B32 up, "
                 "so only the 8 longer files can gain; accuracy moves under 0.1 point.",
        "xlabel": "max batch (log scale)", "ylabel": "",
        "scale": "log",
        "series": ["all 20 files", "12 files of 32 chunks or fewer",
                   "8 files over 32 chunks", "peak memory"],
        "rows": [
            (16, "| 16 |", ("19.3x", "19.7x", "19.2x", "6.56GB")),
            (32, "| 32 |", ("29.3x", "28.4x", "29.9x", "7.23GB")),
            (48, "| 48 |", ("30.7x", "28.6x", "31.9x", "8.14GB")),
            (64, "| 64 |", ("32.9x", "28.3x", "35.3x", "9.17GB")),
            (128, "| **128** |", ("33.6x", "28.6x", "36.3x", "11.33GB")),
        ],
        "source": "ultra-voxtral-batch.json#c30b16,c30b32,c30b32_r2,c30b32_r3,c30b48,c30b64,c30b64_r2,c30b64_r3,c30b128",
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
        "basis": "20-file corpus, M4 16GB, 60s chunks, kv8, delay 2400ms, on AC power; "
                 "B24 and B32 are means of 4 and 7 runs. A file of 16 chunks or fewer fits in one batch from "
                 "B16 up. B48 and B64 (not drawn) also completed the corpus, at 5.82x and "
                 "6.07x mean. Not ringed: the M4 default is 30s/B32, which this 60s "
                 "sweep does not contain.",
        "xlabel": "max batch", "ylabel": "",
        "scale": "category",
        "series": ["all 20 files", "12 files of 16 chunks or fewer",
                   "8 files over 16 chunks", "peak memory"],
        "rows": [
            ("16", "| 16 |", ("4.66x", "4.36x", "4.89x", "6.60GB")),
            ("24", "| 24 |", ("5.56x", "4.47x", "6.33x", "7.34GB")),
            ("32", "| **32** |", ("5.80x", "4.35x", "6.89x", "8.08GB")),
        ],
        "source": "m4-voxtral-batch.json#c60b16,c60b24,c60b24_r2,c60b24_r3,c60b24_r4,c60b32,c60b32_r1,batch_m4_repeat_20261009__c60b32_r2,batch_m4_b64_20261010__c60b32_r2,batch_m4_repeat_20261009__c60b32_r3,batch_m4_b64_20261010__c60b32_r3,c60b32_r4,"
                  "c60b48,c60b48_r1,c60b48_r2,c60b48_r3,c60b64,c60b64_r1,c60b64_r2,c60b64_r3",
        "chosen": None,
        "width": 10.4,
        "panels": [
            {"series": [0, 1, 2], "ylabel": "x realtime (higher is faster)",
             "title": "speed", "zero": True},
            {"series": [3], "ylabel": "peak GB", "title": "memory", "zero": True},
        ],
    },
}
