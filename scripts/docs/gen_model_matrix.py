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
    DEFAULT_SIZE,
    _quant_sort_key,
    families,
)

HF = "https://huggingface.co"

# Peak GPU memory, GB, keyed by (internal alias, quantization). Measured on an M2 Ultra
# over the 20-file corpus with mx.get_peak_memory(), reset per file.
#
# Published instead of download size because the working set is what decides whether a
# machine can run something; a 3GB download that peaks at 15GB does not fit 16GB.
PEAK = {
    # whisper, shipped config, 20 files. Notice these barely track the download: tiny
    # downloads 0.07GB and peaks at 3.98, because the working set is dominated by the
    # 30s mel window and decoder activations rather than by weights.
    ("whisper-tiny", None): "3.98GB",
    ("whisper-base", None): "4.07GB",
    ("whisper-small", None): "4.37GB",
    ("whisper-medium", None): "5.43GB",
    ("whisper-large-v2", None): "6.97GB",
    ("whisper-large-v3", None): "7.00GB",
    ("whisper-turbo", None): "5.53GB",
    # qwen3-asr at its default 8bit, 30s windows, 20 files.
    ("qwen3-asr", "8bit"): "4.05GB",
    ("qwen3-asr-small", "8bit"): "2.36GB",
    ("qwen3-asr", "bf16"): "5.66GB",
    ("qwen3-asr-small", "bf16"): "2.92GB",
    # voxtral, from the precision sweep on the narration clip at 60s/B16.
    ("voxtral", "4bit"): "9.36GB",
    ("voxtral", "fp16"): "15.28GB",
}


def hub_size_gb(repo: str) -> tuple[bool, float]:
    """(exists, total repo size in GB) from the hub API."""
    try:
        with urllib.request.urlopen(
                f"{HF}/api/models/{repo}?blobs=true", timeout=30) as r:
            d = json.load(r)
    except urllib.error.HTTPError:
        return False, 0.0
    except Exception:
        return True, 0.0          # network trouble is not a missing repo
    return True, sum(s.get("size") or 0 for s in d.get("siblings", [])) / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true")
    a = p.parse_args()

    missing = []
    for fam, entries in families().items():
        multi_size = any(m.size for m in entries)
        default_size = DEFAULT_SIZE.get(fam)
        print(f"### `--model {fam}`\n")

        cols = ((["`--size`"] if multi_size else [])
                + ["`--quantization`", "weights", "download", "peak GPU memory"])
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))

        for m in entries:
            quants = (sorted(m.quant_repos, key=_quant_sort_key)
                      if m.quant_repos else [None])
            for q in quants:
                repo = m.quant_repos[q] if q else m.repo
                exists, gb = (True, 0.0) if a.offline else hub_size_gb(repo)
                if not exists:
                    missing.append(repo)
                cells = []
                if multi_size:
                    # Marked on the first row of the group only: repeating it down every
                    # precision row reads as though each were separately default.
                    mark = (" **default**"
                            if m.size == default_size and q == quants[0] else "")
                    cells.append(f"`{m.size}`{mark}")
                cells.append((f"`{q}`" + (" **default**" if repo == m.repo else ""))
                             if q else "-")
                # The repo id is the link, so there is no separate link column.
                cells.append(f"[{repo}]({HF}/{repo})")
                cells.append(f"{gb:.2f}GB" if gb else "?")
                # Measured working set, which is what limits a machine. Only exists for
                # variants that have been run; the rest say so rather than borrowing a
                # number from a sibling.
                cells.append(PEAK.get((m.alias, q), "not measured"))
                print("| " + " | ".join(cells) + " |")
        print()

    print(f"{sum(len(v) for v in families().values())} entries checked.",
          file=sys.stderr)
    if missing:
        print(f"MISSING on the hub: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
