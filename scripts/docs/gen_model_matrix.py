"""Generate the model tables for docs/MODELS.md, one per family.

    uv run python scripts/docs/gen_model_matrix.py

Generated rather than hand-written because it is 20 repo ids across 4 families, and a
table of ids maintained by hand drifts. A drifted table is worse than none: it reads as
authoritative. `tests/test_models.py` fails if the registry gains a variant the docs
lack.

Also verifies every id against the Hugging Face API, so a renamed or deleted repo shows
up here rather than as a 404 for a user. `--offline` skips that.
"""
import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mlx_asr.models import (  # noqa: E402
    CONVERT_SEP,
    DEFAULT_SIZE,
    _quant_sort_key,
    families,
)
from mlx_asr.variants import PRIORITY  # noqa: E402

HF = "https://huggingface.co"

# Peak GPU memory, GB, keyed by (internal alias, quantization).
#
# Published instead of download size because the working set is what decides whether a
# machine can run something; a 3GB download that peaks at 15GB does not fit 16GB.
#
# ONE method for every cell, which is the point: each is the max of
# mx.get_peak_memory() (reset per file) over the corpus on an M2 Ultra, as recorded in
# the saved run-group SOURCE names for the family (benchmarks/results/). An earlier
# version of this table mixed three bases (corpus / one 93-minute file / 3 files), which
# cannot be read down a column, which is the only reason to publish a column. An
# earlier CLI re-run (not saved) agreed with these cells except kotoba, and the note
# there says why.
PEAK = {
    # whisper. These barely track the download: tiny downloads 0.07GB and peaks at
    # 3.98, because the working set is dominated by the 30s mel window and decoder
    # activations rather than by weights.
    # source: ultra-whisper-sizes.json#whisper_tiny_libdef,whisper_base_libdef,whisper_small_nocond,whisper_medium_nocond,whisper_large-v2_nocond,whisper_large-v3_nocond,whisper_turbo_nocond
    ("whisper-tiny", None): "3.98GB",
    ("whisper-base", None): "4.07GB",
    ("whisper-small", None): "4.37GB",
    ("whisper-medium", None): "5.43GB",
    ("whisper-large-v2", None): "6.97GB",
    ("whisper-large-v3", None): "7.00GB",
    ("whisper-turbo", None): "5.53GB",
    # qwen3-asr, 15s windows (the shipped window). large-v2 and large-v3 land within 0.03GB of each other,
    # and several rungs here are close too: the window and the weights set the working
    # set, and the two checkpoints are the same size.
    # source: ultra-qwen3-precision-c15.json#qwen3_0.6B_4bit_c15,qwen3_0.6B_5bit_c15,qwen3_0.6B_6bit_c15,qwen3_0.6B_8bit_c15,qwen3_0.6B_bf16_c15,qwen3_1.7B_4bit_c15,qwen3_1.7B_5bit_c15,qwen3_1.7B_6bit_c15,qwen3_1.7B_8bit_c15,qwen3_1.7B_bf16_c15
    ("qwen3-asr", "4bit"): "2.88GB",
    ("qwen3-asr", "5bit"): "3.09GB",
    ("qwen3-asr", "6bit"): "3.31GB",
    ("qwen3-asr", "8bit"): "3.74GB",
    ("qwen3-asr", "bf16"): "5.35GB",
    ("qwen3-asr-small", "4bit"): "1.77GB",
    ("qwen3-asr-small", "5bit"): "1.85GB",
    ("qwen3-asr-small", "6bit"): "1.92GB",
    ("qwen3-asr-small", "8bit"): "2.07GB",
    ("qwen3-asr-small", "bf16"): "2.63GB",
    # voxtral at 60s/B16, the config of the saved precision sweep, so both precisions
    # share one config. The Ultra's shipped config (30s/B128) peaks higher because its
    # batch is sized from memory. Lower than the 9.36/15.28GB in quantization.md, which
    # used the narration clip at a different chunk/batch pair.
    # source: ultra-voxtral-precision.json#vox_w4bit_c60b16_kv8,vox_wfp16_c60b16_kv8
    # source: ultra-voxtral-precision.json#vox_w8bit_c60b16_kv8
    ("voxtral", "4bit"): "6.77GB",
    ("voxtral", "8bit"): "7.29GB",
    ("voxtral", "fp16"): "12.98GB",
    # kotoba is FLAT at 2.38GB across all 20 files, from a 1.9-minute clip to a
    # 93-minute one, because its 10s window fixes the working set.
    #
    # This cell used to read 3.03GB, and that number was real but measured the wrong
    # thing: it came from a run that ALSO performed the one-time transformers-to-MLX
    # conversion, which mx.load()s the whole fp32 checkpoint and writes an fp16 copy,
    # all through MLX and so all counted by get_peak_memory(). Reproduced deliberately
    # by moving the conversion cache aside: 3.03GB on the converting run, 2.38GB on
    # every run after. 2.38GB is what a user sees for all but the first decode, so it is
    # what belongs in a table about whether a model fits.
    # source: ultra-kotoba-conversion.json#pm_kotoba_cold_r112,pm_kotoba_warm_r112
    # source: ultra-peak-memory-kotoba.json (one run per corpus file, max over the runs)
    ("kotoba", None): "2.38GB",
    # voxtral-v1, 30s windows, one window at a time. Same basis (max over the 20 files,
    # load included) from run_qwen3.py, which resets the counter before the load as the
    # CLI does before run(); the one file measured both ways read 10.91GB in each.
    # Flat across the corpus, like qwen3-asr, because the window fixes the working set.
    # source: ultra-voxtral-v1.json#3b_4bit_c30,3b_8bit_c30,3b_bf16_c30,24b_4bit_c30,24b_8bit_c30,24b_bf16_c30
    ("voxtral-v1-3b", "4bit"): "5.25GB",
    ("voxtral-v1-3b", "8bit"): "7.28GB",
    ("voxtral-v1-3b", "bf16"): "10.91GB",
    ("voxtral-v1-24b", "4bit"): "16.27GB",
    ("voxtral-v1-24b", "8bit"): "27.92GB",
    ("voxtral-v1-24b", "bf16"): "50.08GB",
}

# What the first `--model kotoba` costs, before the conversion is cached. Separate from
# PEAK because it is a one-off on a fresh machine rather than a property of decoding, and
# folding it into the table would overstate the model by 0.65GB forever.
# source: ultra-kotoba-conversion.json#pm_kotoba_cold_r112,pm_kotoba_warm_r112
KOTOBA_FIRST_USE_PEAK = "3.03GB"

# Peak cells that are not a bare figure, so tests/test_models.py does not pin them
# against PEAK: parakeet covers the 17 Japanese files only, and reazon decodes on CPU.
# source: ultra-picker-speed.json#pk_parakeet_default_s1,pk_parakeet_default_s2,pk_parakeet_default_s3
PEAK_NOTE = {
    ("parakeet", None): "4.77GB (17-file corpus)",
    ("reazon-k2", None): "n/a, CPU",
}

_R = "../benchmarks/results/2026-10"


def _src(group: str, *runs: str) -> str:
    return f"[{group}.json]({_R}/{group}.json) runs " + ", ".join(f"`{r}`" for r in runs)


_HUB = " (peak; download is the hub repo size)"
_KOTOBA_RUNS = (
    "pm_kotoba_7f8150fdef0e", "pm_kotoba_c18b2f5aa2a2", "pm_kotoba_9aefd8f2352e",
    "pm_kotoba_07fd66556725", "pm_kotoba_c29657df52e9", "pm_kotoba_cbeb19543902",
    "pm_kotoba_b88caf8c0924", "pm_kotoba_bbdabfabfb4f", "pm_kotoba_1672083b2499",
    "pm_kotoba_5b44e078553d", "pm_kotoba_a73e94506f13", "pm_kotoba_bb5cf5d5b247",
    "pm_kotoba_0c193bf7ced2", "pm_kotoba_57d841fe39b5", "pm_kotoba_a773a315fb42",
    "pm_kotoba_71bb5eba3baa", "pm_kotoba_a1372c4d2900", "pm_kotoba_c47ad127bc83",
    "pm_kotoba_1da58e1b2b69", "pm_kotoba_b3432b40c079")

# The line printed under each family's table: the saved run-group its peaks come
# from (a `Source:` line, see AGENTS.md), or the reason the table has no measurement.
SOURCE = {
    "voxtral": "Source: " + _src("ultra-voxtral-precision", "vox_w4bit_c60b16_kv8",
                                 "vox_w8bit_c60b16_kv8", "vox_wfp16_c60b16_kv8") + _HUB,
    "voxtral-v1": "Source: " + _src("ultra-voxtral-v1", "3b_4bit_c30", "3b_8bit_c30",
                                    "3b_bf16_c30", "24b_4bit_c30", "24b_8bit_c30",
                                    "24b_bf16_c30") + _HUB,
    "whisper": "Source: " + _src("ultra-whisper-sizes", "whisper_tiny_libdef",
                                 "whisper_base_libdef", "whisper_small_nocond",
                                 "whisper_medium_nocond", "whisper_large-v2_nocond",
                                 "whisper_large-v3_nocond", "whisper_turbo_nocond") + _HUB,
    "kotoba": "Source: " + _src("ultra-peak-memory-kotoba", *_KOTOBA_RUNS)
              + " (peak, maximum over the files, at the 10s window shipped then; download is the hub repo size)",
    "qwen3-asr": "Source: " + _src("ultra-qwen3-precision-c15", *(
        f"qwen3_{z}_{q}_c15" for z in ("0.6B", "1.7B")
        for q in ("4bit", "5bit", "6bit", "8bit", "bf16")))
                 + " (peak at the shipped 15s window; download is the hub repo size)",
    "parakeet": "Source: " + _src("ultra-picker-speed", "pk_parakeet_default_s1",
                                  "pk_parakeet_default_s2", "pk_parakeet_default_s3") + _HUB,
    "reazon": "<!-- not-measured: hub repo sizes only; reazon decodes on CPU and has no "
              "GPU peak -->",
}


def hub_size_gb(repo: str) -> tuple[bool, float]:
    """(exists, total repo size in GB) from the hub API.

    Skips `consolidated*` files, which the voxtral-v1 loader never downloads (the
    mistralai repos carry a second, Mistral-native copy of every weight), so the figure
    is what a user actually fetches.
    """
    try:
        with urllib.request.urlopen(
                f"{HF}/api/models/{repo}?blobs=true", timeout=30) as r:
            d = json.load(r)
    except urllib.error.HTTPError:
        return False, 0.0
    except Exception:
        return True, 0.0          # network trouble is not a missing repo
    return True, sum(s.get("size") or 0 for s in d.get("siblings", [])
                     if not s["rfilename"].startswith("consolidated")) / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true")
    a = p.parse_args()

    missing = []
    for fam, entries in families().items():
        multi_size = any(m.size for m in entries)
        default_size = DEFAULT_SIZE.get(fam)
        # A family with a priority list has no single default: the first entry that
        # fits the machine runs (mlx_asr/variants.py), so the table shows the order.
        rank = {(s, q): i + 1 for i, (s, q, _) in enumerate(PRIORITY.get(fam, []))}
        print(f"### `--model {fam}`\n")

        cols = ((["priority"] if rank else []) + (["`--size`"] if multi_size else [])
                + ["`--quantization`", "weights", "download", "peak GPU memory"])
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))

        for m in entries:
            quants = (sorted(m.quant_repos, key=_quant_sort_key)
                      if m.quant_repos else [None])
            for q in quants:
                repo = m.quant_repos[q] if q else m.repo
                # `source:precision` is converted on first use from `source`, so the
                # link and the download are the source's.
                source = repo.split(CONVERT_SEP)[0]
                exists, gb = (True, 0.0) if a.offline else hub_size_gb(source)
                if not exists:
                    missing.append(repo)
                cells = []
                if rank:
                    cells.append(str(rank.get((m.size, q or ""), "-")))
                if multi_size:
                    # Marked on the first row of the group only: repeating it down every
                    # precision row reads as though each were separately default.
                    mark = (" **default**" if not rank and m.size == default_size
                            and q == quants[0] else "")
                    cells.append(f"`{m.size}`{mark}")
                cells.append((f"`{q}`" + (" **default**" if not rank and repo == m.repo
                                          else "")) if q else "-")
                # The repo id is the link, so there is no separate link column.
                if source != repo:
                    cells.append(f"[{source}]({HF}/{source}), converted on first use")
                else:
                    cells.append(f"[{repo}]({HF}/{repo})")
                cells.append(f"{gb:.2f}GB" if gb else "?")
                # Measured working set, which is what limits a machine. Only exists for
                # variants that have been run; the rest say so rather than borrowing a
                # number from a sibling.
                cells.append(PEAK.get((m.alias, q))
                             or PEAK_NOTE.get((m.alias, q), "not measured"))
                print("| " + " | ".join(cells) + " |")
        # Trailing blank line per table. Without it GitHub swallows the next paragraph
        # into the final row as if it were more table.
        print()
        if fam in SOURCE:
            print(SOURCE[fam] + "\n")

    print(f"{sum(len(v) for v in families().values())} entries checked.",
          file=sys.stderr)
    if missing:
        print(f"MISSING on the hub: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
