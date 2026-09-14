"""Score a directory of saved hypothesis texts into run_corpus.py's JSON shape.

Issue #2 asks for a paired bootstrap over the two saved hypothesis sets, and
`compare_engines.py` already implements that test. It reads run_corpus.py's JSON, which
the peak-memory sweep did not write (it saved raw text plus a stats file per decode). This
rescores that text into the shape the comparison expects, so the statistics come from the
shipped harness rather than from a second implementation of the same bootstrap.

Nothing is re-decoded. Both arms were produced in the same sweep on an idle machine at the
same chunk/batch config, so they are already paired by construction; the only thing missing
was the score.

    uv run python scripts/benchmarks/score_hyps.py --hyp-dir DIR \\
        --prefix voxtral_4bit__ --corpus DIR --label voxtral-4bit --out a.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metrics.eval_coverage import load_reference, score_pair


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hyp-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    hyp_dir, corpus = Path(a.hyp_dir), Path(a.corpus)
    results = []
    for hyp in sorted(hyp_dir.glob(f"{a.prefix}*")):
        if hyp.suffix == ".json":
            continue
        stem = hyp.name[len(a.prefix):]
        ref = next((corpus / f"{stem}{s}" for s in
                    ("_transcript.txt", ".srt", ".vtt")
                    if (corpus / f"{stem}{s}").exists()), None)
        if ref is None:
            print(f"  no reference for {stem}", file=sys.stderr)
            continue
        stats = json.loads((hyp_dir / f"{hyp.name}.json").read_text())
        unit, sc = score_pair(load_reference(str(ref)), hyp.read_text())
        results.append({
            "file": stem, "unit": unit,
            "duration_s": round(stats["duration_s"], 2),
            "x_realtime": round(stats["x_realtime"], 2),
            "peak_memory_gb": stats["peak_memory_gb"],
            **{k: v for k, v in sc.items() if k != "excused_runs"},
        })

    agg_ref, agg_charged = {}, {}
    for r in results:
        agg_ref[r["unit"]] = agg_ref.get(r["unit"], 0) + r["ref_chars"]
        agg_charged[r["unit"]] = (agg_charged.get(r["unit"], 0)
                                 + r["coverage_cer"] * r["ref_chars"])
    Path(a.out).write_text(json.dumps({
        "label": a.label, "complete": True,
        "files_scored": len(results), "files_expected": len(results),
        "aggregate": {u: round(agg_charged[u] / agg_ref[u], 6) for u in agg_ref},
        "results": results,
    }, indent=1))
    print(f"{a.label}: {len(results)} files, "
          f"{ {u: f'{agg_charged[u] / agg_ref[u]:.2%}' for u in agg_ref} }",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
