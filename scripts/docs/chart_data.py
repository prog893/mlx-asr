"""The published figures the doc charts are drawn from, with the doc each came from.

Data only, no matplotlib, so `tests/test_charts.py` can check every value here against
the table it was copied from: a chart that disagrees with its own doc is worse than no
chart, because the picture is what people will remember. Every number is an aggregate
already published in docs/; nothing per file, nothing private.

A value is written exactly as its doc prints it (a string), and parsed for plotting.
"""

# --- the picker: one point per engine/size at its defaults ------------------------------
#
# JP and EN from the 20-file corpus (JP: the 17 Japanese files; EN: the 3 English
# files). Speed is the run's own x realtime; `speed_note` marks a figure that is not
# from an idle GPU, which the chart prints beside the label. `peak` is GPU memory from
# MODELS.md, where a 16GB Mac's working set (12.7GB) is the line that matters.
PICKER_SOURCE = {
    "jp": "docs/benchmarks/engines.md",
    "peak": "docs/MODELS.md",
}
M16_WORKING_SET_GB = 12.7

PICKER = [
    # name, JP CER, EN WER, x realtime, peak, speed_note, source of the JP/EN/speed row
    ("voxtral", "16.22%", "21.50%", "29.6x", "6.77GB", "", "RESULTS.md"),
    ("whisper large-v3", "14.55%", "18.26%", "11.4x", "6.99GB", "",
     "docs/benchmarks/engines/whisper.md"),
    ("whisper turbo", "14.68%", "18.31%", "23.7x", "5.52GB", "floor",
     "docs/benchmarks/engines/whisper.md"),
    ("qwen3-asr 1.7B", "19.68%", "25.53%", "21.1x", "4.05GB", "", "RESULTS.md"),
    ("qwen3-asr 0.6B", "23.35%", "24.27%", "31.7x", "2.36GB", "", "RESULTS.md"),
    ("voxtral-v1 3B", "36.52%", "17.86%", "12.9x", "7.28GB", "shared GPU",
     "RESULTS.md"),
    ("voxtral-v1 24B", "27.56%", "16.86%", "3.1x", "27.92GB", "shared GPU",
     "RESULTS.md"),
    ("kotoba", "27.01%", None, "36.2x", "2.38GB", "",
     "docs/benchmarks/engines.md"),
    ("parakeet", "26.19%", None, "244.6x", "4.77GB", "floor",
     "docs/benchmarks/engines.md"),
    ("reazon", "30.45%", None, "51.6x", None, "CPU, floor",
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
    "rows": [("tiny", "51.28%", "32.17%"), ("base", "29.93%", "27.14%"),
             ("small", "21.33%", "22.68%"), ("medium", "21.63%", "18.23%"),
             ("large-v2", "17.87%", "17.68%"), ("large-v3", "14.55%", "18.26%"),
             ("turbo", "14.68%", "18.31%")],
    "chosen": "large-v3",
}

WINDOWS = [
    {"name": "kotoba", "doc": "docs/benchmarks/engines/kotoba.md",
     "basis": "17 JP files",
     "rows": [(10, "27.01%", None), (20, "31.33%", None), (30, "49.71%", None)],
     "chosen": 10},
    {"name": "qwen3-asr 1.7B", "doc": "docs/benchmarks/engines/qwen3-asr.md",
     "basis": "7-file subset",
     "rows": [(15, "20.04%", "30.38%"), (30, "19.98%", "31.38%"),
              (60, "21.42%", "29.86%"), (120, "23.55%", "34.47%"),
              (300, "62.47%", "37.24%")],
     "chosen": 30},
    {"name": "voxtral-v1 3B", "doc": "docs/benchmarks/engines/voxtral-v1.md",
     "basis": "7-file subset, bf16",
     "rows": [(15, "45.81%", "22.47%"), (30, "44.27%", "21.70%"),
              (60, "45.97%", "20.72%"), (120, "57.54%", "21.17%")],
     "chosen": 30},
    {"name": "parakeet", "doc": "docs/benchmarks/engines/parakeet.md",
     "basis": "17 JP files",
     "rows": [(120, "26.19%", None), (300, "32.60%", None)],
     "chosen": 120},
]

# Precision ladders, JP coverage CER on the 20-file corpus, ordered by bits. The tick
# carries the peak memory, which is the cost side of the trade, without a second axis.
PRECISION = [
    {"name": "voxtral", "doc": "docs/benchmarks/quantization.md",
     "rows": [("4-bit", "16.34%", "6.77GB"), ("nvfp4", "16.07%", "5.09GB"),
              ("8-bit", "15.27%", "7.29GB"), ("mxfp8", "15.86%", "7.14GB"),
              ("fp16", "15.04%", "12.98GB")],
     "chosen": "4-bit"},
    {"name": "qwen3-asr 1.7B", "doc": "docs/benchmarks/engines/qwen3-asr.md",
     "rows": [("4bit", "20.03%", "3.19GB"), ("5bit", "19.44%", "3.40GB"),
              ("6bit", "19.63%", "3.62GB"), ("8bit", "19.68%", "4.05GB"),
              ("bf16", "19.51%", "5.66GB")],
     "chosen": "8bit"},
    {"name": "qwen3-asr 0.6B", "doc": "docs/benchmarks/engines/qwen3-asr.md",
     "rows": [("4bit", "28.36%", "2.06GB"), ("5bit", "24.11%", "2.14GB"),
              ("6bit", "24.33%", "2.21GB"), ("8bit", "23.35%", "2.36GB"),
              ("bf16", "23.40%", "2.92GB")],
     "chosen": "8bit"},
    {"name": "voxtral-v1 3B", "doc": "docs/benchmarks/engines/voxtral-v1.md",
     "rows": [("4bit", "44.54%", "5.25GB"), ("8bit", "36.52%", "7.28GB"),
              ("bf16", "37.16%", "10.91GB")],
     "chosen": "8bit"},
    {"name": "voxtral-v1 24B", "doc": "docs/benchmarks/engines/voxtral-v1.md",
     "rows": [("4bit", "28.14%", "16.27GB"), ("8bit", "27.56%", "27.92GB"),
              ("bf16", "27.10%", "50.08GB")],
     "chosen": "8bit"},
]

# Voxtral decode throughput by batch size, synthetic inputs (mlx-asr-bench), so no
# audio is involved. The two machines differ ~4x in scale, hence two panels.
BATCH = [
    {"name": "M2 Ultra, 60 GPU cores", "doc": "docs/benchmarks/decode-throughput.md",
     "rows": [(1, "7.4"), (2, "12.5"), (4, "19.7"), (8, "26.0"), (12, "30.8"),
              (16, "38.1"), (24, "55.9"), (32, "75.6"), (48, "72.2"), (64, "96.5"),
              (96, "92.8"), (128, "103.5")],
     "chosen": 32},
    {"name": "M4, 10 GPU cores", "doc": "docs/benchmarks/decode-throughput.md",
     "rows": [(1, "3.6"), (2, "6.1"), (4, "6.2"), (8, "5.8"), (12, "10.8"),
              (16, "14.2"), (24, "19.7"), (32, "24.7"), (48, "20.1")],
     "chosen": 16},
]

DELAY = {
    "doc": "docs/benchmarks/delay.md", "basis": "7-file subset",
    "rows": [(480, "25.62%", "35.06%"), (960, "20.51%", "30.36%"),
             (2400, "16.44%", "26.55%")],
    "chosen": 2400,
}

# Input level: the four fixed-gain arms on a dB axis. `auto` is the default and is a no-op on
# this corpus (every file already peaks above -6 dBFS), so it sits on the 0 dB point.
GAIN = {
    "doc": "docs/benchmarks/input-level.md", "basis": "7-file subset",
    "rows": [(-20, "23.76%", "36.65%"), (-12, "19.42%", "34.23%"),
             (0, "16.44%", "26.55%"), (6, "17.09%", "23.96%")],
    "chosen": 0,
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
