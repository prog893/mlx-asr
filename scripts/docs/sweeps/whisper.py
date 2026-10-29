"""Comparison charts for docs/benchmarks/engines/whisper.md. Values copied verbatim from its tables.

The size sweep on that page ("whisper-sizes") is drawn by shared code in gen_charts.py.
"""

SWEEPS = {
    # One model per table row; both arms and the printed change sit on that row.
    "whisper-condition": {
        "doc": "docs/benchmarks/engines/whisper.md",
        "title": "Whisper: condition_on_previous_text, library default against False",
        "basis": "Japanese files of the 7-file subset, M2 Ultra. False is "
                 "lower on every model, by 16 to 38 points, so it is the default on small "
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
            ("large-v3", "| large-v3 | 33.40% | 17.73% |", ("33.40%", "17.73%", "-15.7")),
            ("large-v3-turbo", "| large-v3-turbo | 34.56% | 15.46% |",
             ("34.56%", "15.46%", "-19.1")),
            ("kotoba-whisper\nv2.0", "| kotoba-whisper v2.0 |", ("91.31%", "53.26%", "-38.0")),
        ],
        "source": "ultra-mincut-inputs.json#mincut7_whisper_large-v3_libdef,mincut7_whisper_large-v3_nocond,mincut7_whisper_turbo_libdef,mincut7_whisper_turbo_nocond;ultra-whisper-condition.json#cond7_whisper_large-v3_libdef_r2,cond7_whisper_large-v3_libdef_r3,cond7_whisper_large-v3_nocond_r2,cond7_whisper_large-v3_nocond_r3,cond7_whisper_turbo_libdef_r2,cond7_whisper_turbo_libdef_r3,cond7_whisper_turbo_nocond_r2,cond7_whisper_turbo_nocond_r3;ultra-kotoba-sequential.json#kseq_npz_cond_r1,kseq_npz_cond_r2,kseq_npz_cond_r3,kseq_npz_nocond_r1,kseq_npz_nocond_r2,kseq_npz_nocond_r3",
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
            ("per-file,\nfrom the reference", "| per-file, from the reference |", ("34.56%",)),
            ("Whisper's own\n30s autodetect", "| Whisper's own 30s autodetect |", ("59.21%",)),
            ("force ja\nfor every file\n(unusable)", "| force `ja` for every file |", (None,)),
        ],
        "source": "ultra-mincut-inputs.json#mincut7_whisper_turbo_libdef;ultra-whisper-condition.json#cond7_whisper_turbo_libdef_r2,cond7_whisper_turbo_libdef_r3;ultra-whisper-language.json#lang7_turbo_autodetect_r1,lang7_turbo_autodetect_r2,lang7_turbo_autodetect_r3,lang7_turbo_forceja_en_r1,lang7_turbo_forceja_en_r2,lang7_turbo_forceja_en_r3",
        "chosen": None,
    },
    # Unordered runner configs: markers only, no default (the page names none).
    "whisper-runners": {
        "doc": "docs/benchmarks/engines/whisper.md",
        "title": "Whisper runners on Apple Silicon, one clip",
        "basis": "One 935s clip (n=1), M2 Ultra, "
                 "language set to ja, load and warm-up excluded, runs serialized. "
                 "mlx = mlx-whisper, cpp = whisper.cpp 1.9.5 (Metal), "
                 "fw = faster-whisper 1.2.1 (CPU only), "
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
            ("mlx\nturbo\nfp16", "| mlx-whisper | large-v3-turbo fp16 |", ("42.5x", "0.0881")),
            ("cpp\nturbo\nfp16 -t8", "| large-v3-turbo fp16 (`-t 8`) |", ("44.0x", "0.0832")),
            ("cpp\nturbo\nfp16 -t4", "| large-v3-turbo fp16 (`-t 4`) |", ("41.0x", "0.0832")),
            ("cpp\nturbo\nq5_0", "| large-v3-turbo q5_0 |", ("34.7x", "0.0806")),
            ("mlx\nv3\nfp16", "| mlx-whisper | large-v3 fp16 |", ("15.7x", "0.1317")),
            ("cpp\nv3\nq5_0", "| large-v3 q5_0 |", ("12.9x", "0.1175")),
            ("fw\nturbo\nint8\nbatch16", "| large-v3-turbo int8, batched 16 |",
             ("6.7x", "0.0828")),
            ("fw\nturbo\nint8", "| large-v3-turbo int8 | no, CPU |", ("3.6x", "0.0804")),
            ("fw\nv3\nint8", "| large-v3 int8 |", ("1.1x", "0.1318")),
        ],
        "source": "ultra-whisper-runners.json#rn_mlx_turbo_r1,rn_mlx_turbo_r2,rn_mlx_turbo_r3,rn_cpp_turbo_fp16_t8_r1,rn_cpp_turbo_fp16_t8_r2,rn_cpp_turbo_fp16_t8_r3,rn_cpp_turbo_fp16_t4_r1,rn_cpp_turbo_fp16_t4_r2,rn_cpp_turbo_fp16_t4_r3,rn_cpp_turbo_q5_0_t4_r1,rn_cpp_turbo_q5_0_t4_r2,rn_cpp_turbo_q5_0_t4_r3,rn_mlx_large-v3_r1,rn_mlx_large-v3_r2,rn_mlx_large-v3_r3,rn_fw_turbo_int8_b16_r1,rn_fw_turbo_int8_b16_r2,rn_fw_turbo_int8_b16_r3,rn_cpp_large-v3_q5_0_t4_r1,rn_cpp_large-v3_q5_0_t4_r2,rn_cpp_large-v3_q5_0_t4_r3,rn_fw_turbo_int8_b1_r1,rn_fw_turbo_int8_b1_r2,rn_fw_turbo_int8_b1_r3,rn_fw_large-v3_int8_b1_r1,rn_fw_large-v3_int8_b1_r2,rn_fw_large-v3_int8_b1_r3",
        "chosen": None,
    },
}
