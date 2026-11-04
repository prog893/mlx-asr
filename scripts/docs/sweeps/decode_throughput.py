"""Sweep charts for docs/benchmarks/decode-throughput.md.

Every value is copied verbatim from a table row on that page; tests/test_sweeps.py
checks each row's key and values against the doc. The batch ladder chart is not here:
it predates this file and lives in chart_data.BATCH.
"""

DOC = "docs/benchmarks/decode-throughput.md"

SWEEPS = {
    "decode-throughput-reshape": {
        "doc": DOC,
        "title": "Reshaping the decode batch: ms/step by method",
        "basis": "M4 16GB, nvfp4, synthetic inputs (probe_batch_split.py). "
                 "Lower is faster. Fold is within 3.3% of plain; splitting is far worse "
                 "from batch 12 up.",
        "xlabel": "batch size",
        "ylabel": "ms/step (lower is faster)",
        "scale": "linear",
        "unit": "",
        "series": ["plain", "fold", "split into 2", "split into 4"],
        "rows": [
            (4, "| 4 | 51.7 |", ("51.7", "53.4", "55.2", "83.4")),
            (8, "| 8 | 103.4 |", ("103.4", "104.0", "99.6", "101.1")),
            (12, "| 12 | 80.8 |", ("80.8", "81.0", "147.2", "142.1")),
            (16, "| 16 | 81.8 |", ("81.8", "83.8", "207.1", "195.8")),
            (24, "| 24 | 86.6 |", ("86.6", "89.0", "160.2", "307.0")),
            (32, "| 32 | 93.3 |", ("93.3", "92.9", "163.6", "412.9")),
        ],
        "source": "m4-decode-probes.json#m4_probe_batch_split_nvfp4",
        # Same unit in both panels; split so plain and fold (within a few percent of
        # each other) are not drawn on the 0-440 scale the 4-way split needs.
        "panels": [{"series": [0, 1], "title": "plain vs fold",
                    "ylabel": "ms/step (lower is faster)", "legend_loc": "lower right"},
                   {"series": [0, 2, 3], "title": "plain vs split",
                    "ylabel": "ms/step (lower is faster)", "legend_loc": "upper left"}],
    },
    "decode-throughput-encoder-batch": {
        "doc": DOC,
        "title": "Batching the encoder: seconds per chunk",
        "basis": "M4, nvfp4, 30s audio chunks (probe_encoder_batch.py). Lower is faster; "
                 "every batched variant is slower than the per-chunk default.",
        "xlabel": "encoder batch",
        "ylabel": "s/chunk (lower is faster)",
        "scale": "category",
        "unit": "",
        "series": ["s/chunk"],
        "rows": [
            ("per-chunk", "| per-chunk (stock) |", ("1.576",)),
            ("1", "| 1 | 1.763 |", ("1.763",)),
            ("2", "| 2 | 1.769 |", ("1.769",)),
            ("4", "| 4 | 1.826 |", ("1.826",)),
            ("8", "| 8 | 1.749 |", ("1.749",)),
        ],
        "source": "m4-decode-probes.json#m4_probe_encoder_batch_nvfp4",
        "chosen": "per-chunk",
        "chosen_label": "default",
    },
    "decode-throughput-kvlen": {
        "doc": DOC,
        "title": "KV cache growth at batch 16",
        "basis": "M4, nvfp4, batch 16, synthetic inputs. Step cost and peak memory both "
                 "rise as the cache fills.",
        "xlabel": "kv_len (decoded positions)",
        "ylabel": "",
        "scale": "linear",
        "unit": "",
        "series": ["ms/step", "peak GB"],
        "rows": [
            (138, "| 138 |", ("78.5", "4.28")),
            (238, "| 238 |", ("81.4", "4.28")),
            (338, "| 338 |", ("87.3", "4.94")),
            (438, "| 438 |", ("90.3", "4.94")),
            (538, "| 538 |", ("101.2", "5.79")),
            (638, "| 638 |", ("100.5", "5.79")),
            (738, "| 738 |", ("104.6", "5.79")),
            (838, "| 838 |", ("133.1", "6.58")),
        ],
        "source": "m4-decode-probes.json#m4_probe_kvlen_nvfp4_b16",
        "panels": [{"series": [0], "ylabel": "ms/step (lower is faster)"},
                   {"series": [1], "ylabel": "peak GB", "zero": True}],
    },
    "decode-throughput-qmv-wide": {
        "doc": DOC,
        "title": "Forcing qmv_wide on the M2 Ultra: change in decode steps/s",
        "basis": "M2 Ultra, 4-bit affine, synthetic inputs, two interleaved runs per "
                 "arm. Hypothesis refuted: batch 1 is the control, 2-8 all got slower.",
        "xlabel": "batch size",
        "ylabel": "change vs default qmv (%)",
        "scale": "linear",
        "unit": "",
        "series": ["forced qmv_wide vs qmv"],
        "rows": [
            (1, "| 1 | 90.3 / 90.9 |", ("-0.2%",)),
            (2, "| 2 | 77.9 / 77.8 |", ("-17.6%",)),
            (4, "| 4 | 61.3 / 61.4 |", ("-14.6%",)),
            (8, "| 8 | 40.5 / 40.5 |", ("-21.0%",)),
        ],
        "source": "ultra-qmv-wide.json#qmv_default_r1,qmv_default_r2,qmv_wide_r1,qmv_wide_r2",
    },
    "decode-throughput-stages": {
        "doc": DOC,
        "title": "Wall clock by stage on the 935s clip, 60s chunks",
        "basis": "935s clip in 60s chunks, M4 16GB (nvfp4) and M2 Ultra 128GB (4-bit affine). "
                 "Encode is 35% of the M4's wall clock, so decode speed alone does not set it.",
        "xlabel": "stage",
        "ylabel": "seconds (lower is faster)",
        "scale": "category",
        "connect": False,
        "unit": "s",
        "series": ["M4 16GB", "M2 Ultra 128GB"],
        "rows": [
            ("encode", "| encode (16 chunks) |", ("47.6", "10.6")),
            ("decode", "| decode (816 steps) |", ("80.0", "31.2")),
            ("total", "| total |", ("137.2", "44.4")),
        ],
        "source": ("m4-stage-timing.json#m4_stages_nvfp4_c60b16_r1,m4_stages_nvfp4_c60b16_r2,"
                   "m4_stages_nvfp4_c60b16_r3;"
                   "ultra-stage-timing.json#stages_ultra_4bit_c60b16_r1,"
                   "stages_ultra_4bit_c60b16_r2,stages_ultra_4bit_c60b16_r3"),
        "zero": True,
    },
}
