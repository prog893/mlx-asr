"""How much do corpus results depend on the cut-detection threshold?

`eval_coverage.py` excuses an insertion run as omitted-from-reference audio when
it is at least ``min_cut`` units long, and charges it as an error otherwise. That
threshold is a judgment call, and every corpus number depends on it, so the
honest thing is to measure how sensitive the conclusions are to it.

Two things to look for:

  absolute drift   the aggregate error rate necessarily falls as the threshold
                   drops (more insertions get excused). Large drift means the
                   absolute numbers should be quoted with the threshold attached.
  rank stability   whether configs change order across thresholds. If the ranking
                   holds, comparisons between configs are safe even though the
                   absolute level is threshold-dependent. That is the property
                   the tuning conclusions actually rely on.

Operates on saved hypotheses, so no re-decoding is needed.

    uv run python benchmarks/sweep_min_cut.py --corpus DIR \\
        --hyp-dir bench_out/corpus --thresholds 10,20,30,50,80
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.eval_coverage import (
    coverage_score,
    is_space_delimited,
    load_reference,
    normalize,
    tokenize,
)
from benchmarks.run_corpus import find_pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--hyp-dir", required=True,
                   help="directory of <stem>.<label>.txt hypotheses")
    p.add_argument("--thresholds", default="10,20,30,50,80",
                   help="character-unit thresholds; the word-unit path uses /5")
    p.add_argument("--json")
    a = p.parse_args()

    corpus = Path(a.corpus).expanduser()
    refs = {audio.stem: ref for audio, ref in find_pairs(corpus)}
    hyps = {}
    for f in sorted(Path(a.hyp_dir).glob("*.txt")):
        m = re.match(r"(.+?)\.(.+)\.txt$", f.name)
        if not m:
            continue
        stem, label = m.group(1), m.group(2)
        if stem in refs:
            hyps.setdefault(label, {})[stem] = f

    if not hyps:
        print(f"no hypotheses found in {a.hyp_dir}", file=sys.stderr)
        return 2

    thresholds = [int(x) for x in a.thresholds.split(",")]
    labels = sorted(hyps)
    print(f"{len(labels)} configs, thresholds {thresholds} "
          f"(word-unit threshold = chars/5)\n")

    table = {}
    for unit in ("char", "word"):
        header = f"{'config':>22} " + " ".join(f"{'mc'+str(t):>9}" for t in thresholds)
        rows = []
        for label in labels:
            vals = []
            for t in thresholds:
                tot = charged = 0
                for stem, hf in hyps[label].items():
                    ref_text = load_reference(str(refs[stem]))
                    if (unit == "word") != is_space_delimited(ref_text):
                        continue
                    hyp_text = hf.read_text(encoding="utf-8")
                    if unit == "word":
                        r, h, mc = tokenize(ref_text), tokenize(hyp_text), max(1, t // 5)
                    else:
                        r, h, mc = normalize(ref_text), normalize(hyp_text), t
                    s = coverage_score(r, h, mc)
                    tot += s["ref_chars"]
                    charged += s["coverage_cer"] * s["ref_chars"]
                vals.append(charged / tot if tot else None)
            if any(v is not None for v in vals):
                rows.append((label, vals))
        if not rows:
            continue
        m = "coverageCER" if unit == "char" else "coverageWER"
        print(f"--- {m} ({unit} unit) ---")
        print(header)
        for label, vals in rows:
            print(f"{label:>22} " + " ".join(
                f"{v*100:>8.2f}%" if v is not None else f"{'-':>9}" for v in vals))
        # rank stability across thresholds
        orders = []
        for i in range(len(thresholds)):
            order = tuple(l for l, v in sorted(rows, key=lambda x: x[1][i]))
            orders.append(order)
        stable = len(set(orders)) == 1
        spread = max(v[0] for _, v in rows) - min(v[-1] for _, v in rows)
        print(f"  best config at each threshold: "
              f"{[o[0] for o in orders]}")
        print(f"  ranking identical across all thresholds: {stable}")
        print(f"  absolute drift from tightest to loosest: "
              f"{(rows[0][1][0]-rows[0][1][-1])*100:+.2f} points (first config)\n")
        table[unit] = {"thresholds": thresholds,
                       "rows": {l: [round(v, 5) if v else None for v in vs]
                                for l, vs in rows},
                       "rank_stable": stable}

    print("Interpretation: absolute levels move with the threshold, so quote it "
          "alongside any number.\nWhat matters for tuning is whether the ranking "
          "holds; see 'ranking identical' above.")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(table, f, indent=2)
        print(f"[saved] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
