"""Sweep charts for docs/benchmarks/engines.md and engines/kotoba.md, copied from their tables.

Every engine comparison on engines.md is unordered, so those charts are markers only
(`connect: False`, `scale: "category"`). The Voxtral-against-Whisper table has metrics as
rows and engines as columns, so it uses `layout: "columns"`: each series is matched by
its row label. Its error bars are the 95% t-intervals the table prints for the Whisper
mean; the Whisper speed point is the mean of its 6 runs, with the bar spanning their range.
"""

SWEEPS = {
    "engines-voxtral-whisper": {
        "doc": "docs/benchmarks/engines.md",
        "title": "Voxtral (fp16 default) against Whisper turbo-nocond, 20-file corpus",
        "basis": "M2 Ultra; Voxtral at fp16 is deterministic in accuracy (speed the mean of 3 runs), Whisper the mean of 6 "
                 "runs with its printed 95% t-interval.\nThe speed bar spans Whisper's 6 "
                 "runs around their mean. Voxtral is the default because it is faster "
                 "than every Whisper run\nand needs no language hint or stability flag, "
                 "while Whisper's Japanese lead is not resolved over files.",
        "xlabel": "engine", "ylabel": "error %",
        "scale": "category", "unit": "",
        "connect": False,
        "layout": "columns",
        "series": ["JP coverage CER (17 files)", "EN coverage WER (3 files)", "x realtime"],
        "series_keys": ["| JP coverage CER, 17 files |", "| EN coverage WER, 3 files |",
                        "| x realtime |"],
        "rows": [
            ("voxtral", None, ("15.20%", "21.21%", "23.9x"), (None, None, None)),
            ("whisper\nturbo-nocond", None, ("14.17%", "18.65%", "21.0"),
             ("[13.87, 14.47]", "[18.13, 19.17]", "[18.5, 23.2]")),
        ],
        "source": "ultra-voxtral-precision-default.json#vox_wfp16_c30b128_kv8,vox_wfp16_c30b128_kv8_r2,vox_wfp16_c30b128_kv8_r3;ultra-whisper-sizes.json#whisper_turbo_nocond,whisper_turbo_nocond_r2,whisper_turbo_nocond_r3,whisper_turbo_nocond_r4,whisper_turbo_nocond_r5,whisper_turbo_nocond_r6",
        "chosen": "voxtral", "chosen_series": "JP coverage CER (17 files)",
        "panels": [
            {"series": [0, 1], "ylabel": "error % (lower is better)", "title": "accuracy"},
            {"series": [2], "ylabel": "x realtime (higher is faster)", "title": "speed",
             "zero": True},
        ],
        "width": 8.4, "height": 3.8,
    },
    "engines-generalization": {
        "doc": "docs/benchmarks/engines.md",
        "title": "Whisper's lead over Voxtral, bootstrapped over files",
        "basis": "17 Japanese files, the same runs as the Voxtral against Whisper chart; "
                 "95% CI from the bootstrap over files.\nThe interval includes zero, "
                 "so the Whisper lead does not hold when the files are resampled.",
        "xlabel": "test", "ylabel": "Voxtral minus Whisper, JP CER points",
        "scale": "category", "unit": "",
        "connect": False,
        "series": ["Voxtral minus Whisper (positive: Whisper better)"],
        "rows": [
            ("bootstrap over 17 files", "| **bootstrap over 17 files** |", ("+1.03",),
             ("[-0.11, +2.25]",)),
        ],
        "source": "ultra-voxtral-precision-default.json#vox_wfp16_c30b128_kv8,vox_wfp16_c30b128_kv8_r2,vox_wfp16_c30b128_kv8_r3;ultra-whisper-sizes.json#whisper_turbo_nocond,whisper_turbo_nocond_r2,whisper_turbo_nocond_r3,whisper_turbo_nocond_r4,whisper_turbo_nocond_r5,whisper_turbo_nocond_r6",
        "chosen": None,
        "panels": [{"series": [0], "ylabel": "Voxtral minus Whisper, JP CER points",
                    "zero_line": True}],
        "width": 5.2, "height": 3.4,
    },
    "engines-corpus-size": {
        "doc": "docs/benchmarks/engines.md",
        "title": "Voxtral vs Whisper turbo-nocond as the corpus grew",
        "basis": "Japanese files only; Whisper is a mean of 3 or 6 runs, Voxtral one "
                 "deterministic 4-bit run.\nThe 17-file Whisper point is the 6-run mean "
                 "(14.22 / 13.92 / 14.20 / 14.69 / 13.96 / 14.02),\nthe same runs as the Voxtral "
                 "against Whisper table.",
        "xlabel": "Japanese files in the corpus", "ylabel": "JP coverage CER %",
        "scale": "linear",
        "unit": "",
        "series": ["voxtral", "whisper turbo-nocond (mean)"],
        "rows": [
            (5, "| original | 5 |", ("16.44%", "15.46%")),
            (17, "| final | 17 |", ("15.73%", "14.17%")),
        ],
        "source": "ultra-mincut-inputs.json#mincut7_vox_c30b32_kv8,mincut7_whisper_turbo_nocond;ultra-whisper-condition.json#cond7_whisper_turbo_nocond_r2,cond7_whisper_turbo_nocond_r3;ultra-voxtral-batch.json#c30b32,c30b32_r2,c30b32_r3;ultra-voxtral-headline.json#vox_default_c30b32_kv8;ultra-whisper-sizes.json#whisper_turbo_nocond,whisper_turbo_nocond_r2,whisper_turbo_nocond_r3,whisper_turbo_nocond_r4,whisper_turbo_nocond_r5,whisper_turbo_nocond_r6",
        "chosen": None,
        "legend_loc": "center right",
    },
    "engines-narration": {
        "doc": "docs/benchmarks/engines.md",
        "title": "Clean narration: Voxtral against the Whisper sizes and kotoba",
        "basis": "One narration clip (n=1), M2 Ultra, means of 3 runs.\nVoxtral has the lowest plain CER on this "
                 "clip; the default itself is decided on the 20-file corpus.",
        "xlabel": "engine", "ylabel": "CER %",
        "scale": "category", "unit": "",
        "connect": False,
        "series": ["coverage CER", "plain CER", "x realtime"],
        "rows": [
            ("voxtral", "| **voxtral (M2 Ultra 128GB, 60s b16 kv8)** |",
             ("7.28%", "7.28%", "22.0x")),
            ("turbo", "| whisper large-v3-turbo |", ("8.81%", "8.81%", "42.5x")),
            ("v3\nnocond", "| whisper large-v3, no-condition |",
             ("8.87%", "8.87%", "23.8x")),
            ("turbo\nnocond", "| whisper large-v3-turbo, no-condition |",
             ("10.42%", "10.42%", "71.2x")),
            ("v3", "| whisper large-v3 |", ("13.17%", "13.17%", "15.7x")),
            ("small", "| whisper small |", ("15.16%", "16.68%", "47.6x")),
            ("medium", "| whisper medium |", ("11.66%", "11.66%", "31.2x")),
            ("v2", "| whisper large-v2 |", ("14.09%", "15.26%", "21.4x")),
            ("kotoba\n20s", "| kotoba-whisper v2.2, chunk 20s |", ("16.55%", "16.55%", "29.7x")),
            ("base", "| whisper base |", ("24.33%", "26.11%", "90.2x")),
            ("tiny", "| whisper tiny |", ("33.94%", "35.25%", "88.5x")),
        ],
        "source": "ultra-engines-narration.json#clip_vox_c60b16_kv8_r1,clip_vox_c60b16_kv8_r2,clip_vox_c60b16_kv8_r3,clip_whisper_large-v3_nocond_r1,clip_whisper_large-v3_nocond_r2,clip_whisper_large-v3_nocond_r3,clip_whisper_turbo_nocond_r1,clip_whisper_turbo_nocond_r2,clip_whisper_turbo_nocond_r3,clip_whisper_small_libdef_r1,clip_whisper_small_libdef_r2,clip_whisper_small_libdef_r3,clip_whisper_medium_libdef_r1,clip_whisper_medium_libdef_r2,clip_whisper_medium_libdef_r3,clip_whisper_large-v2_libdef_r1,clip_whisper_large-v2_libdef_r2,clip_whisper_large-v2_libdef_r3,clip_whisper_base_libdef_r1,clip_whisper_base_libdef_r2,clip_whisper_base_libdef_r3,clip_whisper_tiny_libdef_r1,clip_whisper_tiny_libdef_r2,clip_whisper_tiny_libdef_r3;ultra-whisper-runners.json#rn_mlx_turbo_r1,rn_mlx_turbo_r2,rn_mlx_turbo_r3,rn_mlx_large-v3_r1,rn_mlx_large-v3_r2,rn_mlx_large-v3_r3;ultra-kotoba-engines-clip.json#kclip_hf22_c20_r1,kclip_hf22_c20_r2,kclip_hf22_c20_r3",
        "chosen": "voxtral", "chosen_series": "plain CER",
        "panels": [
            {"series": [0, 1], "ylabel": "CER % (lower is better)", "title": "accuracy"},
            {"series": [2], "ylabel": "x realtime (higher is faster)", "title": "speed",
             "zero": True},
        ],
        "width": 14.0, "height": 3.8,
    },
    "engines-japanese-only": {
        "doc": "docs/benchmarks/engines.md",
        "title": "Japanese-only engines against the multilingual defaults",
        "basis": "17 Japanese files, M2 Ultra. Whisper's 18.5-23.2x run "
                 "range is not drawn.\nNeither Japanese-only engine is more accurate than "
                 "Voxtral, so they do not change the default; parakeet wins only on speed.",
        "xlabel": "engine", "ylabel": "CER %",
        "scale": "category", "unit": "",
        "connect": False,
        "series": ["JP coverage CER", "kana CER", "x realtime"],
        "rows": [
            ("whisper\nturbo-nocond", "| whisper-turbo no-cond (6-run mean) |",
             ("14.17%", None, None)),
            ("voxtral", "| voxtral |", ("15.20%", None, "23.9x")),
            ("kotoba\n15s", "| kotoba chunk15s |", ("25.34%", None, "50.0x")),
            ("parakeet\nc120", "| **parakeet c120** |", ("25.68%", "22.79%", "261.8x")),
            ("reazon\nfp32", "| **reazon-k2 fp32 c30** |", ("30.06%", "27.29%", "52.9x")),
            ("reazon\nint8", "| reazon-k2 int8 c30 |", ("36.55%", None, "79.9x")),
        ],
        "source": "ultra-whisper-sizes.json#whisper_turbo_nocond,whisper_turbo_nocond_r2,whisper_turbo_nocond_r3,whisper_turbo_nocond_r4,whisper_turbo_nocond_r5,whisper_turbo_nocond_r6;ultra-voxtral-precision-default.json#vox_wfp16_c30b128_kv8,vox_wfp16_c30b128_kv8_r2,vox_wfp16_c30b128_kv8_r3;ultra-kotoba-engines.json#k20_mlx_c15_r1,k20_mlx_c15_r2,k20_mlx_c15_r3;ultra-picker-speed.json#pk_parakeet_default_s1,pk_parakeet_default_s2,pk_parakeet_default_s3,pk_reazon_fp32_default_s1,pk_reazon_fp32_default_s2,pk_reazon_fp32_default_s3;ultra-engines-jp-only.json#jp_reazon_int8_c30_r1,jp_reazon_int8_c30_r2,jp_reazon_int8_c30_r3",
        "chosen": "voxtral", "chosen_series": "JP coverage CER",
        "panels": [
            {"series": [0, 1], "ylabel": "CER % (lower is better)", "title": "accuracy"},
            {"series": [2], "ylabel": "x realtime (higher is faster)", "title": "speed",
             "zero": True},
        ],
        "width": 10.0, "height": 3.8,
    },
    "engines-kotoba-window": {
        "doc": "docs/benchmarks/engines/kotoba.md",
        "title": "kotoba: decode window (MLX chunked driver)",
        "basis": "17 Japanese files, MLX chunked driver, means of 3 runs. 15s is the default: lower CER than 10s "
                 "(paired -1.47, CI [-2.75, -0.13]) and faster.\nAbove 15s, longer windows trade accuracy for speed.",
        "xlabel": "window length", "ylabel": "JP coverage CER %",
        "scale": "linear",
        "unit": "s",
        "series": ["coverage CER", "x realtime"],
        "rows": [
            (10, "chunked, 10s windows", ("26.81%", "36.8x")),
            (15, "chunked, 15s windows", ("25.34%", "50.0x")),
            (20, "chunked, 20s windows", ("31.02%", "67.8x")),
            (30, "chunked, 30s windows", ("49.42%", "77.9x")),
        ],
        "source": "ultra-kotoba-windows.json#pk_kotoba_c10_s1,pk_kotoba_c10_s2,pk_kotoba_c10_s3,kw_kotoba_c20,kw_kotoba_c20_r2,kw_kotoba_c20_r3,kw_kotoba_c30,kw_kotoba_c30_r2,kw_kotoba_c30_r3;ultra-kotoba-engines.json#k20_mlx_c15_r1,k20_mlx_c15_r2,k20_mlx_c15_r3",
        "chosen": 15,
        "panels": [
            {"series": [0], "ylabel": "JP coverage CER %", "title": "accuracy"},
            {"series": [1], "ylabel": "x realtime", "title": "speed"},
        ],
    },
    "engines-kotoba-authors-chunk": {
        "doc": "docs/benchmarks/engines/kotoba.md",
        "title": "kotoba: chunk_length_s in the authors' torch pipeline",
        "basis": "Corpus: each run's rows for the 5 Japanese files of the 7-file subset; narration: the single clip. "
                 "Reference driver only, not the CLI's.",
        "xlabel": "chunk_length_s", "ylabel": "CER %",
        "scale": "linear",
        "unit": "s",
        "series": ["corpus coverage CER", "narration CER"],
        "rows": [
            (10, "`chunk_length_s=10`", ("26.15%", "23.71%")),
            (15, "`chunk_length_s=15`", ("30.40%", "20.88%")),
            (20, "`chunk_length_s=20`", ("27.82%", "16.55%")),
            (30, "`chunk_length_s=30`", ("49.57%", "39.52%")),
        ],
        "source": "ultra-kotoba-engines.json#k20_hf22_c10_r1,k20_hf22_c15_r1,k20_hf22_c15_r2,k20_hf22_c15_r3,k20_hf22_c20_r1,k20_hf22_c30_r1;ultra-kotoba-engines-clip.json#kclip_hf22_c10_r1,kclip_hf22_c10_r2,kclip_hf22_c10_r3,kclip_hf22_c15_r1,kclip_hf22_c15_r2,kclip_hf22_c15_r3,kclip_hf22_c20_r1,kclip_hf22_c20_r2,kclip_hf22_c20_r3,kclip_hf22_c30_r1,kclip_hf22_c30_r2,kclip_hf22_c30_r3",
        "chosen": None,
        "legend_loc": "upper left",
    },
}
