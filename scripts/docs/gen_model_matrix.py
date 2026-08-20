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

# Plain CER and x-realtime on the 180s Japanese narration clip, keyed by internal alias.
# Measured through the installed binary (scripts/benchmarks/matrix_models.py), one clip,
# so it says "this loads and roughly how fast", not which model is better. Kept here
# rather than in the registry because it is a benchmark result, not a shipped default,
# and only exists for the default precision of each size.
CLIP = {
    "voxtral":          ("11.76%", "1.7x"),
    "whisper-turbo":    ("13.67%", "15.1x"),
    "whisper-large-v3": ("14.41%", "6.1x"),
    "whisper-large-v2": ("13.45%", "4.0x"),
    "whisper-medium":   ("16.42%", "10.8x"),
    "whisper-small":    ("24.47%", "21.7x"),
    "whisper-base":     ("35.59%", "20.9x"),
    "whisper-tiny":     ("32.10%", "51.4x"),
    "kotoba":           ("22.67%", "9.8x"),
    "qwen3-asr":        ("19.70%", "19.6x"),
    "qwen3-asr-small":  ("18.86%", "25.4x"),
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
                + ["`--quantization`", "weights", "on disk", "clip CER", "x-rt"])
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
                    cells.append(f"`{m.size}`"
                                 + (" **default**" if m.size == default_size else ""))
                cells.append((f"`{q}`" + (" **default**" if repo == m.repo else ""))
                             if q else "only one")
                # The repo id is the link, so there is no separate link column.
                cells.append(f"[{repo}]({HF}/{repo})")
                cells.append(f"{gb:.2f}GB" if gb else "?")
                # Only the default precision was measured, so the others say so rather
                # than borrowing a number they did not earn.
                cer, xrt = CLIP.get(m.alias, ("", "")) if repo == m.repo else ("", "")
                cells += [cer or "not run", xrt or ""]
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
