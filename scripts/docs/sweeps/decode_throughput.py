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
                 "Lower is faster. Fold gains 3-7% at best; splitting is far worse.",
        "xlabel": "batch size",
        "ylabel": "ms/step (lower is faster)",
        "scale": "linear",
        "unit": "",
        "series": ["plain", "fold", "split into 2", "split into 4"],
        "rows": [
            (4, "| 4 | 50.3 |", ("50.3", "50.3", "50.1", "80.9")),
            (8, "| 8 | 101.6 |", ("101.6", "102.1", "102.8", "106.3")),
            (12, "| 12 | 84.1 |", ("84.1", "78.1", "143.4", "152.0")),
            (16, "| 16 | 80.6 |", ("80.6", "80.3", "226.8", "216.5")),
            (32, "| 32 | 103.2 |", ("103.2", "99.8", "176.0", "436.6")),
        ],
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
        "basis": "M4, audio chunks (probe_encoder_batch.py). Lower is faster; "
                 "every batched variant is slower than the per-chunk default.",
        "xlabel": "encoder batch",
        "ylabel": "s/chunk (lower is faster)",
        "scale": "category",
        "unit": "",
        "series": ["s/chunk"],
        "rows": [
            ("per-chunk", "| per-chunk (stock) |", ("1.497",)),
            ("1", "| 1 | 1.651 |", ("1.651",)),
            ("2", "| 2 | 1.642 |", ("1.642",)),
            ("4", "| 4 | 1.738 |", ("1.738",)),
            ("8", "| 8 | 1.773 |", ("1.773",)),
        ],
        "chosen": "per-chunk",
        "chosen_label": "default",
    },
    "decode-throughput-kvlen": {
        "doc": DOC,
        "title": "KV cache growth at batch 16",
        "basis": "M4, batch 16, synthetic inputs. Step cost and peak memory both "
                 "rise as the cache fills.",
        "xlabel": "kv_len (decoded positions)",
        "ylabel": "",
        "scale": "linear",
        "unit": "",
        "series": ["ms/step", "peak GB"],
        "rows": [
            (138, "| 138 |", ("78.4", "4.28")),
            (438, "| 438 |", ("89.0", "4.94")),
            (838, "| 838 |", ("112.6", "6.41")),
        ],
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
            (1, "| 1 | 89.4 / 90.2 |", ("-0.5%",)),
            (2, "| 2 | 73.0 / 77.3 |", ("-14.8%",)),
            (4, "| 4 | 61.3 / 61.2 |", ("-14.5%",)),
            (8, "| 8 | 40.5 / 40.6 |", ("-26.8%",)),
        ],
    },
}
