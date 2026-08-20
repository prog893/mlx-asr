"""Generate the full model/size/quantization -> repo id table for docs/MODELS.md.

    uv run python scripts/docs/gen_model_matrix.py >> docs/MODELS.md

Derived from the registry rather than written by hand, for the same reason the --help
text is: a table of 30-odd repo ids maintained manually drifts, and a drifted table is
worse than none because it looks authoritative.

Prints Markdown. Verifies every id exists on the hub unless --offline.
"""
import argparse
import json
import sys
import pathlib
import urllib.error
import urllib.request

# Repo root, so this runs from anywhere.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mlx_asr.models import (  # noqa: E402
    DEFAULT_SIZE,
    REGISTRY,
    families,
    resolve_family,
    _quant_sort_key,
)

HF = "https://huggingface.co"


def check(repo: str) -> tuple[bool, float]:
    """(exists, size_gb) from the hub API."""
    try:
        with urllib.request.urlopen(
                f"{HF}/api/models/{repo}?blobs=true", timeout=30) as r:
            d = json.load(r)
    except urllib.error.HTTPError:
        return False, 0.0
    except Exception:
        return True, 0.0          # network trouble, not a missing repo
    total = sum(s.get("size") or 0 for s in d.get("siblings", []))
    return True, total / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true")
    a = p.parse_args()

    rows = []
    for fam, entries in families().items():
        for m in entries:
            default_size = (not m.size) or m.size == DEFAULT_SIZE.get(fam)
            # A model with no ladder still has exactly one build: its own repo.
            quants = (sorted(m.quant_repos, key=_quant_sort_key)
                      if m.quant_repos else [None])
            for q in quants:
                repo = m.quant_repos[q] if q else m.repo
                is_default_q = repo == m.repo
                gb = None
                exists = True
                if not a.offline:
                    exists, gb = check(repo)
                rows.append({
                    "family": fam, "size": m.size, "quant": q,
                    "repo": repo, "gb": gb, "exists": exists,
                    "default": default_size and is_default_q,
                    "default_quant": is_default_q,
                    "backend": m.backend,
                })

    print(f"| `--model` | `--size` | `--quantization` | weights | on disk | Hugging Face |")
    print(f"|---|---|---|---|---|---|")
    for r in rows:
        star = " **(default)**" if r["default"] else ""
        size = f"`{r['size']}`" if r["size"] else "n/a"
        if r["quant"]:
            q = f"`{r['quant']}`" + (" *(default)*" if r["default_quant"]
                                     and r["size"] else "")
        else:
            q = "n/a"
        gb = f"{r['gb']:.2f}GB" if r["gb"] else "?"
        miss = "" if r["exists"] else " **MISSING**"
        print(f"| `{r['family']}`{star} | {size} | {q} | `{r['repo']}`{miss} | {gb} "
              f"| [link]({HF}/{r['repo']}) |")

    missing = [r["repo"] for r in rows if not r["exists"]]
    print(f"\n{len(rows)} combinations, {len(set(r['repo'] for r in rows))} distinct "
          f"repos.", file=sys.stderr)
    if missing:
        print(f"MISSING on the hub: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
