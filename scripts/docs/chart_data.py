"""The published figures the doc charts are drawn from, with the doc each came from.

Data only, no matplotlib, so `tests/test_charts.py` can check every value here against
the table it was copied from: a chart that disagrees with its own doc is worse than no
chart, because the picture is what people will remember. Every number is an aggregate
already published in docs/; nothing per file, nothing private.

A value is written exactly as its doc prints it (a string), and parsed for plotting.
"""

# --- the picker: one point per engine/size at its defaults ------------------------------
#
# Defaults on the M2 Ultra 128GB, where the speeds were measured; Voxtral's is fp16 there
# (a 16GB Mac loads 8-bit instead, docs/DEFAULTS.md "How a variant is chosen").
# JP and EN from the 20-file corpus (JP: the 17 Japanese files; EN: the 3 English
# files). Speed is the run's own x realtime; `speed_note` is a qualifier the chart
# prints beside the label (reazon decodes on the CPU). `peak` is GPU memory from
# MODELS.md, where a 16GB Mac's working set (12.7GB) is the line that matters.
PICKER_SOURCE = {
    "jp": "docs/benchmarks/engines.md",
    "peak": "docs/MODELS.md",
}
M16_WORKING_SET_GB = 12.7  # source: m4-synthetic-decode.json#synth_nvfp4_r1 (machine gpu_working_set_gb)

PICKER = [
    # name, JP CER, EN WER, x realtime, peak, speed_note, source of the JP/EN/speed row
    ("voxtral", "15.20%", "21.21%", "23.9x", "12.98GB", "", "RESULTS.md"),
    ("whisper large-v3", "14.51%", "18.59%", "11.5x", "7.00GB", "",
     "docs/benchmarks/engines/whisper.md"),
    ("whisper turbo", "14.17%", "18.65%", "21.0x", "5.53GB", "",
     "docs/benchmarks/engines/whisper.md"),
    ("qwen3-asr 1.7B", "18.62%", "24.72%", "24.6x", "3.74GB", "", "RESULTS.md"),
    ("qwen3-asr 0.6B", "22.31%", "22.95%", "35.5x", "2.07GB", "", "RESULTS.md"),
    ("voxtral-v1 3B", "36.14%", "17.86%", "14.1x", "7.28GB", "",
     "RESULTS.md"),
    ("voxtral-v1 24B", "27.20%", "16.86%", "3.4x", "27.92GB", "",
     "RESULTS.md"),
    ("kotoba", "25.34%", None, "50.0x", "2.38GB", "",
     "docs/benchmarks/engines.md"),
    ("parakeet", "25.68%", None, "261.8x", "4.77GB", "",
     "docs/benchmarks/engines.md"),
    ("reazon", "30.06%", None, "52.9x", None, "CPU",
     "docs/benchmarks/engines.md"),
]

# --- sweeps: (doc, title, x label, rows, chosen x) ---------------------------------
#
# rows: (x, JP, EN) with EN None where the sweep has no English column. x is either a
# number (plotted on a log scale where the doc's steps are multiplicative) or a label
# for an ordinal axis. `basis` is printed under the chart so a 7-file sweep is never
# read as a 20-file one.
WHISPER_SIZES = {
    "doc": "docs/benchmarks/engines/whisper.md",
    "basis": "20 files, default config (no-condition on small and larger)",
    "rows": [("tiny", "51.00%", "31.84%"), ("base", "30.67%", "25.93%"),
             ("small", "21.45%", "21.71%"), ("medium", "21.37%", "18.62%"),
             ("large-v2", "17.74%", "17.72%"), ("large-v3", "14.51%", "18.59%"),
             ("turbo", "14.17%", "18.65%")],
    "chosen": "large-v3",
    "source": ("ultra-whisper-sizes.json#whisper_large-v2_nocond_r2,"
                "whisper_large-v2_nocond_r3,whisper_large-v2_nocond,whisper_large-v3_nocond_r2,"
                "whisper_large-v3_nocond_r3,whisper_large-v3_nocond,whisper_medium_nocond_r2,"
                "whisper_medium_nocond_r3,whisper_medium_nocond,whisper_small_nocond_r2,"
                "whisper_small_nocond_r3,whisper_small_nocond,whisper_turbo_nocond_r2,"
                "whisper_turbo_nocond_r3,whisper_turbo_nocond,whisper_base_libdef_r2,"
                "whisper_base_libdef_r3,whisper_base_libdef,whisper_tiny_libdef_r2,"
                "whisper_tiny_libdef_r3,whisper_tiny_libdef,whisper_large-v3_nocond_r4,"
                "whisper_large-v3_nocond_r5,whisper_large-v3_nocond_r6,whisper_turbo_nocond_r4,"
                "whisper_turbo_nocond_r5,whisper_turbo_nocond_r6"),
}

WINDOWS = [
    {"name": "kotoba", "doc": "docs/benchmarks/engines/kotoba.md",
     "basis": "17 JP files",
     "rows": [(10, "26.81%", None), (15, "25.34%", None), (20, "31.02%", None),
              (30, "49.42%", None)],
     "chosen": 15},
    {"name": "qwen3-asr 1.7B", "doc": "docs/benchmarks/engines/qwen3-asr.md",
     "basis": "7-file subset",
     "rows": [(15, "19.73%", "30.15%"), (30, "20.67%", "31.45%"),
              (60, "22.27%", "30.04%"), (120, "23.55%", "34.62%"),
              (300, "64.89%", "36.94%")],
     "chosen": 15},
    {"name": "voxtral-v1 3B", "doc": "docs/benchmarks/engines/voxtral-v1.md",
     "basis": "7-file subset, bf16",
     "rows": [(15, "45.81%", "22.47%"), (30, "44.27%", "21.70%"),
              (60, "45.97%", "20.72%"), (120, "57.54%", "21.17%")],
     "chosen": 30},
    {"name": "parakeet", "doc": "docs/benchmarks/engines/parakeet.md",
     "basis": "17 JP files",
     "rows": [(120, "25.78%", None), (300, "32.28%", None)],
     "chosen": 120},
]

# Precision ladders, JP coverage CER on the 20-file corpus, ordered by bits. The tick
# carries the peak memory, which is the cost side of the trade, without a second axis.
PRECISION = [
    {"name": "voxtral", "doc": "docs/benchmarks/quantization.md",
     "rows": [("4-bit", "15.72%", "6.77GB"), ("nvfp4", "15.58%", "5.09GB"),
              ("8-bit", "14.78%", "7.29GB"), ("mxfp8", "15.36%", "7.14GB"),
              ("fp16", "14.63%", "12.98GB")],
     "chosen": "fp16"},   # the M2 Ultra default; 16GB Macs get 8-bit
    {"name": "qwen3-asr 1.7B", "doc": "docs/benchmarks/engines/qwen3-asr.md",
     "rows": [("4bit", "19.78%", "2.88GB"), ("5bit", "19.21%", "3.09GB"),
              ("6bit", "19.09%", "3.31GB"), ("8bit", "18.62%", "3.74GB"),
              ("bf16", "19.14%", "5.35GB")],
     "chosen": "8bit"},
    {"name": "qwen3-asr 0.6B", "doc": "docs/benchmarks/engines/qwen3-asr.md",
     "rows": [("4bit", "25.10%", "1.77GB"), ("5bit", "23.23%", "1.85GB"),
              ("6bit", "22.67%", "1.92GB"), ("8bit", "22.31%", "2.07GB"),
              ("bf16", "22.31%", "2.63GB")],
     "chosen": "8bit"},
    {"name": "voxtral-v1 3B", "doc": "docs/benchmarks/engines/voxtral-v1.md",
     "rows": [("4bit", "44.60%", "5.25GB"), ("8bit", "36.41%", "7.28GB"),
              ("bf16", "36.84%", "10.91GB")],
     "chosen": "8bit"},
    {"name": "voxtral-v1 24B", "doc": "docs/benchmarks/engines/voxtral-v1.md",
     "rows": [("4bit", "28.06%", "16.27GB"), ("8bit", "27.44%", "27.92GB"),
              ("bf16", "26.98%", "50.08GB")],
     "chosen": "8bit"},
]

# Voxtral decode throughput by batch size, synthetic inputs (mlx-asr-bench), so no
# audio is involved. The two machines differ ~4x in scale, hence two panels.
BATCH = [
    {"name": "M2 Ultra, 60 GPU cores", "doc": "docs/benchmarks/decode-throughput.md",
     "rows": [(1, "7.2"), (2, "12.5"), (4, "19.6"), (8, "25.9"), (12, "30.9"),
              (16, "38.2"), (24, "55.5"), (32, "75.2"), (48, "71.8"), (64, "95.9"),
              (96, "99.7"), (128, "103.9")],
     "chosen": 128},
    {"name": "M4, 10 GPU cores", "doc": "docs/benchmarks/decode-throughput.md",
     "rows": [(1, "3.8"), (2, "6.2"), (4, "6.3"), (8, "6.2"), (12, "11.9"),
              (16, "15.1"), (24, "20.8"), (32, "25.8"), (48, "21.2")],
     "chosen": 32},
]

DELAY = {
    "doc": "docs/benchmarks/delay.md", "basis": "7-file subset",
    "rows": [(480, "25.62%", "35.06%"), (960, "20.51%", "30.36%"),
             (2400, "16.44%", "26.55%")],
    "chosen": 2400,
    "source": "ultra-voxtral-delay.json#delay7_c30b32_d480,delay7_c30b32_d960;"
              "ultra-mincut-inputs.json#mincut7_vox_c30b32_kv8",
}

# Input level: the four fixed-gain arms on a dB axis. `auto` is the default and is a no-op on
# this corpus (every file already peaks above -6 dBFS), so it sits on the 0 dB point.
GAIN = {
    "doc": "docs/benchmarks/input-level.md", "basis": "7-file subset",
    "rows": [(-20, "23.76%", "36.65%"), (-12, "19.42%", "34.23%"),
             (0, "16.44%", "26.55%"), (6, "17.09%", "23.96%")],
    "chosen": 0,
    "source": "ultra-input-gain.json#gain7_m20db_c30b32_kv8,gain7_m12db_c30b32_kv8,"
              "gain7_asrecorded_c30b32_kv8,gain7_p6db_c30b32_kv8",
}

# Saved run data behind each bare row list above: "group.json#run,run;group2.json#run".
SOURCES = {
    "PICKER": ("ultra-voxtral-precision-default.json#vox_wfp16_c30b128_kv8,"
               "vox_wfp16_c30b128_kv8_r2,vox_wfp16_c30b128_kv8_r3;"
               "ultra-whisper-sizes.json#whisper_large-v3_nocond,whisper_large-v3_nocond_r2,"
               "whisper_large-v3_nocond_r3,whisper_large-v3_nocond_r4,"
               "whisper_large-v3_nocond_r5,whisper_large-v3_nocond_r6,whisper_turbo_nocond,"
               "whisper_turbo_nocond_r2,whisper_turbo_nocond_r3,whisper_turbo_nocond_r4,"
               "whisper_turbo_nocond_r5,whisper_turbo_nocond_r6;"
               "ultra-picker-speed.json#pk_qwen3_1.7B_8bit_c15_s1,pk_qwen3_1.7B_8bit_c15_s2,"
               "pk_qwen3_1.7B_8bit_c15_s3,pk_qwen3_0.6B_8bit_c15_s1,pk_qwen3_0.6B_8bit_c15_s2,"
               "pk_qwen3_0.6B_8bit_c15_s3,pk_v1_3B_8bit_c30_s1,pk_v1_3B_8bit_c30_s2,"
               "pk_v1_3B_8bit_c30_s3;ultra-kotoba-engines.json#k20_mlx_c15_r1,k20_mlx_c15_r2,k20_mlx_c15_r3;"
               "ultra-picker-speed.json#"
               "pk_parakeet_default_s1,pk_parakeet_default_s2,pk_parakeet_default_s3,"
               "pk_reazon_fp32_default_s1,pk_reazon_fp32_default_s2,pk_reazon_fp32_default_s3;"
               "ultra-picker-speed-24b.json#pk_v1_24B_8bit_c30_s1,pk_v1_24B_8bit_c30_s2,"
               "pk_v1_24B_8bit_c30_s3;ultra-voxtral-precision.json#vox_wfp16_c60b16_kv8;"
               "ultra-qwen3-precision-c15.json#qwen3_1.7B_8bit_c15,qwen3_0.6B_8bit_c15;"
               "ultra-voxtral-v1.json#3b_8bit_c30,24b_8bit_c30;"
               "ultra-peak-memory-kotoba.json#pm_kotoba_7f8150fdef0e,pm_kotoba_c18b2f5aa2a2,"
               "pm_kotoba_9aefd8f2352e,pm_kotoba_07fd66556725,pm_kotoba_c29657df52e9,"
               "pm_kotoba_cbeb19543902,pm_kotoba_b88caf8c0924,pm_kotoba_bbdabfabfb4f,"
               "pm_kotoba_1672083b2499,pm_kotoba_5b44e078553d,pm_kotoba_a73e94506f13,"
               "pm_kotoba_bb5cf5d5b247,pm_kotoba_0c193bf7ced2,pm_kotoba_57d841fe39b5,"
               "pm_kotoba_a773a315fb42,pm_kotoba_71bb5eba3baa,pm_kotoba_a1372c4d2900,"
               "pm_kotoba_c47ad127bc83,pm_kotoba_1da58e1b2b69,pm_kotoba_b3432b40c079"),
    "WINDOWS": ("ultra-kotoba-windows.json#pk_kotoba_c10_s1,pk_kotoba_c10_s2,pk_kotoba_c10_s3,"
                "kw_kotoba_c20,kw_kotoba_c30,kw_kotoba_c20_r2,kw_kotoba_c20_r3,"
                "kw_kotoba_c30_r2,kw_kotoba_c30_r3;"
                "ultra-kotoba-engines.json#k20_mlx_c15_r1,k20_mlx_c15_r2,k20_mlx_c15_r3;"
                "ultra-qwen3-window.json#qwen3w7_1.7B_8bit_c120,qwen3w7_1.7B_8bit_c15,"
                "qwen3w7_1.7B_8bit_c30,qwen3w7_1.7B_8bit_c300,qwen3w7_1.7B_8bit_c60;"
                "ultra-voxtral-v1-window.json#sweep_3b_bf16_c15,sweep_3b_bf16_c60,"
                "sweep_3b_bf16_c120;ultra-voxtral-v1.json#3b_bf16_c30;"
                "ultra-parakeet-window.json#parakeet_c120,parakeet_c300"),
    "PRECISION": ("ultra-voxtral-precision.json#vox_w4bit_c60b16_kv8,vox_wnvfp4_c60b16_kv8,"
                  "vox_w8bit_c60b16_kv8,vox_wmxfp8_c60b16_kv8,vox_wfp16_c60b16_kv8;"
                  "ultra-qwen3-precision-c15.json#qwen3_0.6B_4bit_c15,qwen3_0.6B_5bit_c15,"
                  "qwen3_0.6B_6bit_c15,qwen3_0.6B_8bit_c15,qwen3_0.6B_bf16_c15,"
                  "qwen3_1.7B_4bit_c15,qwen3_1.7B_5bit_c15,qwen3_1.7B_6bit_c15,"
                  "qwen3_1.7B_8bit_c15,qwen3_1.7B_bf16_c15;ultra-voxtral-v1.json#3b_4bit_c30,"
                  "3b_8bit_c30,3b_bf16_c30,24b_4bit_c30,24b_8bit_c30,24b_bf16_c30"),
    "BATCH": ("ultra-synthetic-decode.json#synth_ultra_4bit_r1,synth_ultra_4bit_r2,"
              "synth_ultra_4bit_r3;m4-synthetic-decode.json#synth_nvfp4_r1,synth_nvfp4_r2,"
              "synth_nvfp4_r3"),
}


def pct(s: str) -> float:
    return float(s.rstrip("%"))


def num(s: str) -> float:
    return float(s.rstrip("xGB"))


# --- generic single-lever sweeps ----------------------------------------------------
#
# One chart each, embedded in its own doc. A spec is data only:
#   doc      the doc the table lives in (and that the chart is embedded in)
#   title, basis, xlabel, ylabel
#   scale    "log", "linear" or "category"
#   unit     suffix for tick labels ("s", "ms", "")
#   series   ordered series labels; each row gives one value per series, or None
#   rows     (x, key, values): `key` is text unique to that table row in the doc,
#            which the test uses to match the row's values to the right line
#   chosen   the x the doc names as the default (ringed), on series `chosen_series`
SWEEPS: dict = {}


def _load_sweeps():
    """Collect SWEEPS from scripts/docs/sweeps/<doc>.py, one module per doc page.

    One file per page so the page's charts are edited alongside it and two pages never
    touch the same file. Chart names must start with the page's stem.
    """
    import importlib.util
    from pathlib import Path

    for path in sorted((Path(__file__).parent / "sweeps").glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"sweeps_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name, sweep in getattr(mod, "SWEEPS", {}).items():
            assert name.startswith(path.stem.replace("_", "-")), (path.name, name)
            assert name not in SWEEPS, name
            SWEEPS[name] = sweep


def ci(s: str) -> tuple[float, float]:
    """A printed interval "[15.10, 17.40]" (units like % allowed) to (lo, hi)."""
    lo, hi = (float(x.strip().rstrip("%x").replace("+", "")) for x in s.strip("[] ").split(","))
    return lo, hi


def value(s: str) -> float:
    """A doc cell as printed ("44.27%", "29.6x", "7.28GB", "0.428") to a float."""
    import re
    s = s.strip("*").replace(",", "").strip()
    s = re.sub(r"\s*(ms/min|ms|min|GB|s|x|%)$", "", s)   # printed unit suffixes
    return float(s)


_load_sweeps()
