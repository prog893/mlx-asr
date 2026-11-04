"""Charts for docs/benchmarks/reference/peak-memory.md. Data only; values are copied verbatim from
the page's tables, and tests/test_sweeps.py checks each row against them."""

SWEEPS = {
    "peak-memory-whisper-duration": {
        "doc": "docs/benchmarks/reference/peak-memory.md",
        "title": "whisper turbo: peak GPU memory grows with audio length",
        "basis": "Six files of the 20-file corpus, M2 Ultra.",
        "xlabel": "audio length (minutes)",
        "ylabel": "peak GPU memory (GB)", "zero": True,
        "scale": "linear",
        "unit": "",
        "series": ["peak"],
        "rows": [
            (1.9, "| 1.9 min |", ("2.54GB",)),
            (13.1, "| 13.1 min |", ("2.76GB",)),
            (35.4, "| 35.4 min |", ("3.51GB",)),
            (52.0, "| 52.0 min |", ("4.03GB",)),
            (69.6, "| 69.6 min |", ("4.54GB",)),
            (93.2, "| 93.2 min |", ("5.53GB",)),
        ],
        "source": ("ultra-whisper-sizes.json#whisper_turbo_nocond,whisper_turbo_nocond_r2,"
                   "whisper_turbo_nocond_r3,whisper_turbo_nocond_r4,whisper_turbo_nocond_r5,"
                   "whisper_turbo_nocond_r6;corpus.json"),
    },
    "peak-memory-range": {
        "doc": "docs/benchmarks/reference/peak-memory.md",
        "title": "Peak GPU memory range across the corpus, per model",
        "basis": "20-file corpus, M2 Ultra. Overlapping marks: memory does not depend "
                 "on the file.",
        "xlabel": "model",
        "ylabel": "peak GPU memory (GB)", "zero": True,
        "scale": "category",
        "connect": False,
        "unit": "",
        "series": ["min over the 20 files", "max over the 20 files"],
        "rows": [
            ("kotoba", "| `kotoba` |", ("2.38GB", "2.38GB")),
            ("qwen3 0.6B", "| `qwen3-asr 0.6B/8bit` |", ("2.06GB", "2.07GB")),
            ("qwen3 1.7B", "| `qwen3-asr 1.7B/8bit` |", ("3.73GB", "3.74GB")),
            ("voxtral", "| `voxtral 4bit` |", ("5.38GB", "6.77GB")),
            ("whisper turbo", "| `whisper turbo` |", ("2.54GB", "5.53GB")),
        ],
        "source": ("ultra-peak-memory-kotoba.json#pm_kotoba_7f8150fdef0e,pm_kotoba_c18b2f5aa2a2,"
                   "pm_kotoba_9aefd8f2352e,pm_kotoba_07fd66556725,pm_kotoba_c29657df52e9,"
                   "pm_kotoba_cbeb19543902,pm_kotoba_b88caf8c0924,pm_kotoba_bbdabfabfb4f,"
                   "pm_kotoba_1672083b2499,pm_kotoba_5b44e078553d,pm_kotoba_a73e94506f13,"
                   "pm_kotoba_bb5cf5d5b247,pm_kotoba_0c193bf7ced2,pm_kotoba_57d841fe39b5,"
                   "pm_kotoba_a773a315fb42,pm_kotoba_71bb5eba3baa,pm_kotoba_a1372c4d2900,"
                   "pm_kotoba_c47ad127bc83,pm_kotoba_1da58e1b2b69,pm_kotoba_b3432b40c079;"
                   "ultra-qwen3-precision-c15.json#qwen3_0.6B_8bit_c15,qwen3_1.7B_8bit_c15;"
                   "ultra-voxtral-precision.json#vox_w4bit_c60b16_kv8;"
                   "ultra-whisper-sizes.json#whisper_turbo_nocond,whisper_turbo_nocond_r2,"
                   "whisper_turbo_nocond_r3,whisper_turbo_nocond_r4,whisper_turbo_nocond_r5,"
                   "whisper_turbo_nocond_r6"),
        "chosen": None,
    },
}
