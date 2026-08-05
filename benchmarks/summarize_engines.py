"""Collect every corpus/narration run JSON into the comparison tables.

Takes the run JSONs produced by run_corpus.py (Voxtral) and run_whisper.py
(Whisper) and emits the markdown that goes into RESULTS.md: accuracy per engine
on both scoring units, throughput, and the per-file spread that decides whether
any of it is resolvable.

`extra_ratio` is carried into the table on purpose. It is the tell for
hallucination loops on this corpus: a model that pads a 2535-character reference
into 14000 characters of repeated text can still post a flattering coverage CER,
because coverage excuses long insertion runs as omitted-from-reference audio.
Reading the two columns together is what keeps that from passing as accuracy.

    uv run python benchmarks/summarize_engines.py benchmarks/*.json --md out.md
"""

import argparse
import json
import sys
from pathlib import Path


def load(path):
    try:
        d = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None
    if "results" not in d or "aggregate" not in d:
        return None
    rows = [r for r in d["results"] if "error" not in r]
    if not rows:
        return None
    engine = d.get("engine", "voxtral")
    return {
        "path": path,
        "label": d.get("label") or Path(path).stem,
        "engine": engine,
        "char": d["aggregate"].get("char"),
        "word": d["aggregate"].get("word"),
        "x_realtime": d.get("x_realtime"),
        "n_files": len(rows),
        "languages": d.get("languages"),
        "skipped": d.get("skipped_files") or [],
        "extra_char": _mean(rows, "char", "extra_ratio"),
        "extra_word": _mean(rows, "word", "extra_ratio"),
        "cov_char": _mean(rows, "char", "coverage"),
        "rows": rows,
        "config": d.get("config", {}),
    }


def _mean(rows, unit, key):
    vals = [r[key] for r in rows if r.get("unit") == unit and key in r]
    return sum(vals) / len(vals) if vals else None


def pct(v, nd=2):
    return "-" if v is None else f"{v*100:.{nd}f}%"


def ratio(v):
    return "-" if v is None else f"{v:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json", nargs="+")
    p.add_argument("--sort", default="char", choices=["char", "word", "speed"])
    p.add_argument("--md", help="write the markdown tables here")
    a = p.parse_args()

    runs = [r for r in (load(x) for x in a.json) if r]
    if not runs:
        print("no usable run JSONs", file=sys.stderr)
        return 2

    corpus = [r for r in runs if r["n_files"] > 1]
    single = [r for r in runs if r["n_files"] == 1]

    def key(r):
        if a.sort == "speed":
            return -(r["x_realtime"] or 0)
        return r[a.sort] if r[a.sort] is not None else 9e9

    out = []
    if corpus:
        out.append("### Corpus: 7 recordings, 5.18h, mixed Japanese/English\n")
        out.append("| engine / config | JP coverageCER | EN coverageWER | "
                   "x realtime | JP extra | EN extra | files |")
        out.append("|---|---|---|---|---|---|---|")
        for r in sorted(corpus, key=key):
            note = f" ({len(r['skipped'])} EN skipped)" if r["skipped"] else ""
            out.append(
                f"| {r['label']}{note} | {pct(r['char'])} | {pct(r['word'])} | "
                f"{ratio(r['x_realtime'])}x | {ratio(r['extra_char'])} | "
                f"{ratio(r['extra_word'])} | {r['n_files']} |")
        out.append("")

    if single:
        out.append("### Single narration clip (15.6 min, complete reference)\n")
        out.append("Plain CER is meaningful here: `extra_ratio` is near 1, so "
                   "the reference is not an edited one.\n")
        out.append("| engine / config | coverageCER | plain CER | coverage | "
                   "extra | x realtime |")
        out.append("|---|---|---|---|---|---|")
        for r in sorted(single, key=key):
            row = r["rows"][0]
            out.append(
                f"| {r['label']} | {pct(r['char'])} | {pct(row.get('cer'))} | "
                f"{pct(row.get('coverage'), 1)} | "
                f"{ratio(row.get('extra_ratio'))} | "
                f"{ratio(r['x_realtime'])}x |")
        out.append("")

    if corpus:
        out.append("### Per-file coverage error (corpus)\n")
        durs = sorted({round(x["duration_s"]) for r in corpus for x in r["rows"]})
        hdr = " | ".join(f"{d//60}m" for d in durs)
        out.append(f"| engine / config | {hdr} |")
        out.append("|---" * (len(durs) + 1) + "|")
        for r in sorted(corpus, key=key):
            by = {round(x["duration_s"]): x for x in r["rows"]}
            cells = " | ".join(
                pct(by[d]["coverage_cer"]) + (
                    "" if by[d]["unit"] == "char" else " (w)")
                if d in by else "-" for d in durs)
            out.append(f"| {r['label']} | {cells} |")
        out.append("")

    text = "\n".join(out)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")
        print(f"[saved] {a.md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
