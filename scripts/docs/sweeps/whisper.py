"""Comparison charts for docs/benchmarks/engines/whisper.md. Values copied verbatim from its tables.

The size sweep on that page ("whisper-sizes") is drawn by shared code in gen_charts.py.
"""

SWEEPS = {
    # One model per table row; both arms and the printed change sit on that row.
    "whisper-condition": {
        "doc": "docs/benchmarks/engines/whisper.md",
        "title": "Whisper: condition_on_previous_text, library default against False",
        "basis": "Japanese files of the 7-file subset, M2 Ultra. False is "
                 "lower on every model, by 10 to 38 points, so it is the default on small "
                 "and larger.",
        "xlabel": "model",
        "scale": "category",
        "connect": False,
        "series": ["library defaults", "no-condition (False)", "change"],
        "panels": [
            {"series": [0, 1], "ylabel": "JP coverage CER % (lower is better)",
             "title": "accuracy", "zero": True},
            {"series": [2], "ylabel": "change with False (points)",
             "title": "False against the library default", "zero_line": True},
        ],
        "width": 9.6,
        "rows": [
            ("large-v3", "| large-v3 | 39.91% | 17.36% |", ("39.91%", "17.36%", "-22.6")),
            ("large-v3-turbo", "| large-v3-turbo | 24.97% | 14.93% |",
             ("24.97%", "14.93%", "-10.0")),
            ("kotoba-whisper\nv2.0", "| kotoba-whisper v2.0 |", ("91.47%", "53.20%", "-38.3")),
        ],
        "chosen": "large-v3", "chosen_series": "no-condition (False)",
        "chosen_label": "default",
        "label_dy": 14,
    },
    "whisper-language": {
        "doc": "docs/benchmarks/engines/whisper.md",
        "title": "Whisper: three ways of supplying the language",
        "basis": "7-file subset, M2 Ultra, large-v3-turbo at library defaults. Forcing ja "
                 "has no CER point because the English files score about 100% WER.",
        "xlabel": "how the language is supplied",
        "ylabel": "JP coverage CER % (lower is better)",
        "scale": "category",
        "connect": False,
        "zero": True,
        "series": ["Japanese coverage CER"],
        "width": 6.6,
        "rows": [
            ("per-file,\nfrom the reference", "| per-file, from the reference |", ("24.97%",)),
            ("Whisper's own\n30s autodetect", "| Whisper's own 30s autodetect |", ("50.14%",)),
            ("force ja\nfor every file\n(unusable)", "| force `ja` for every file |", (None,)),
        ],
        "chosen": None,
    },
    # Unordered runner configs: markers only, no default (the page names none).
    "whisper-runners": {
        "doc": "docs/benchmarks/engines/whisper.md",
        "title": "Whisper runners on Apple Silicon, one clip",
        "basis": "One 935s clip (n=1, noise band about 1.3 CER points), M2 Ultra, "
                 "language set to ja, load and warm-up excluded, runs serialized. "
                 "mlx = mlx-whisper, cpp = whisper.cpp 1.9.1 (Metal), ifw = "
                 "insanely-fast-whisper (torch/MPS), fw = faster-whisper 1.2.1 (CPU only), "
                 "v3 = large-v3.",
        "xlabel": "runner, model, quantization",
        "scale": "category",
        "connect": False,
        "series": ["x realtime", "plain CER"],
        "panels": [
            {"series": [0], "ylabel": "x realtime (higher is faster)", "title": "speed",
             "zero": True},
            {"series": [1], "ylabel": "plain CER, fraction (lower is better)",
             "title": "accuracy", "zero": True},
        ],
        "width": 13.0,
        "height": 4.2,
        "rows": [
            ("mlx\nturbo\nfp16", "| mlx-whisper | large-v3-turbo fp16 |", ("44.3x", "0.0908")),
            ("cpp\nturbo\nfp16 -t8", "| large-v3-turbo fp16 (`-t 8`) |", ("41.7x", "0.0832")),
            ("cpp\nturbo\nfp16 -t4", "| large-v3-turbo fp16 (`-t 4`) |", ("41.0x", "0.0832")),
            ("cpp\nturbo\nq5_0", "| large-v3-turbo q5_0 |", ("30.0x", "0.0830")),
            ("mlx\nv3\nfp16", "| mlx-whisper | large-v3 fp16 |", ("16.6x", "0.1367")),
            ("cpp\nv3\nq5_0", "| large-v3 q5_0 |", ("11.7x", "0.1641")),
            ("ifw\nturbo\nfp16", "| insanely-fast-whisper |", ("10.2x", "0.2495")),
            ("fw\nturbo\nint8\nbatch16", "| large-v3-turbo int8, batched 16 |",
             ("6.6x", "0.0828")),
            ("fw\nturbo\nint8", "| large-v3-turbo int8 | no, CPU |", ("3.5x", "0.0804")),
            ("fw\nv3\nint8", "| large-v3 int8 |", ("1.1x", "0.1175")),
        ],
        "chosen": None,
    },
}
