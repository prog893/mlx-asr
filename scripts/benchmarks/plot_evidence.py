"""Plot the evidence behind a comparison, because prose hides the shape of it.

Every conclusion in docs/benchmarks/ is a point estimate and an interval in a sentence.
That format is honest but it conceals the thing that actually governs this project: the
between-file spread is larger than almost every effect being measured, and it is
two-sided. A reader cannot tell from "+0.10, CI [-1.89, +2.03]" whether that is twenty
files agreeing on nothing much or two files pulling hard in opposite directions. Those
call for different follow-up work, so the difference matters.

Four plots, each answering a question text answers badly:

  paired      per-file difference against zero, sorted, with the aggregate. Shows whether
              a result is broad or driven by a few files, and how wide the spread is
              relative to the effect.
  bootstrap   the resampled distribution the CI comes from, with zero marked. Shows
              skew, and shows how much of the mass sits on the wrong side of zero.
  runs        a sampling engine's run distribution against a deterministic engine's
              fixed value, which is the comparison repeat_distribution.py computes.
  sensitivity aggregate against the `min_cut` excusal threshold for several configs at
              once. The claim that rankings are stable across thresholds is a line
              crossing question, and lines are the right medium for it.

Files are labelled by DURATION, never by name: the corpus filenames are private
recording identifiers. That matches summarize_engines.py, so a plot and a table can be
read side by side.

matplotlib is in the `eval` extra, not in the shipped package.

    uv run python scripts/benchmarks/plot_evidence.py paired A.json B.json --out p.png
    uv run python scripts/benchmarks/plot_evidence.py bootstrap A.json B.json --out b.png
    uv run python scripts/benchmarks/plot_evidence.py runs --baseline vox.json \\
        --runs w1.json w2.json w3.json --out r.png
    uv run python scripts/benchmarks/plot_evidence.py sensitivity a.json b.json --out s.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

UNIT_NAME = {"char": "coverage CER", "word": "coverage WER"}


def _die(msg):
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def load(path):
    """Read a result file, refusing incomplete ones.

    `run_whisper.py` rewrites its JSON after every file, so a partial run on disk is
    indistinguishable from a finished one except for this flag. Plotting a partial run
    would produce a figure that looks authoritative over a subset of the corpus.
    """
    d = json.load(open(path, encoding="utf-8"))
    if d.get("complete") is False:
        _die(f"{path} is an incomplete run "
             f"({d.get('files_scored')}/{d.get('files_expected')} files); refusing to plot")
    label = d.get("label") or Path(path).stem
    rows = [r for r in d.get("results", []) if "error" not in r and "unit" in r]
    if not rows:
        _die(f"{path} has no scored rows")
    return label, d, rows


def pair(rows_a, rows_b, unit):
    """Match files across two runs on (rounded duration, unit).

    Duration rather than filename, so a scrubbed run can be compared against a local
    one; unit as well, because the corpus contains one recording dubbed into two
    languages at identical length, and a duration-only key silently collapses them.
    """
    ka = {(round(r["duration_s"]), r["unit"]): r for r in rows_a if r["unit"] == unit}
    kb = {(round(r["duration_s"]), r["unit"]): r for r in rows_b if r["unit"] == unit}
    keys = sorted(set(ka) & set(kb), key=lambda k: k[0])
    ref = np.array([ka[k]["ref_chars"] for k in keys], float)
    ea = np.array([ka[k]["coverage_cer"] for k in keys], float)
    eb = np.array([kb[k]["coverage_cer"] for k in keys], float)
    mins = [k[0] / 60 for k in keys]
    return ref, ea, eb, mins


def agg(err, ref):
    return float((err * ref).sum() / ref.sum()) if ref.sum() else float("nan")


def boot_diff(ref, ea, eb, n_boot, seed=0):
    """Length-weighted PAIRED bootstrap, matching compare_engines.py exactly.

    One index draw applied to both arms, so the pairing survives resampling. Drawing
    twice would compare two independent samples and inflate the interval.
    """
    rng = np.random.default_rng(seed)
    n = len(ref)
    idx = rng.integers(0, n, size=(n_boot, n))
    w = ref[idx]
    return (ea[idx] * w).sum(1) / w.sum(1) - (eb[idx] * w).sum(1) / w.sum(1)


def _fig(nrows=1, ncols=1, w=9.0, h=5.0):
    import matplotlib
    matplotlib.use("Agg")           # no display on a benchmark host
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt, plt.subplots(nrows, ncols, figsize=(w, h))


def cmd_paired(a):
    la, _, ra = load(a.a)
    lb, _, rb = load(a.b)
    units = [u for u in ("char", "word")
             if any(r["unit"] == u for r in ra) and any(r["unit"] == u for r in rb)]
    if not units:
        _die("no shared scoring unit between the two runs")

    plt, (fig, axes) = _fig(1, len(units), w=5.6 * len(units), h=5.2)
    axes = np.atleast_1d(axes)
    for ax, unit in zip(axes, units):
        ref, ea, eb, mins = pair(ra, rb, unit)
        d = (ea - eb) * 100
        order = np.argsort(d)
        y = np.arange(len(d))
        colors = ["#c0392b" if v > 0 else "#2471a3" for v in d[order]]
        ax.barh(y, d[order], color=colors, height=0.7)
        ax.axvline(0, color="black", lw=1)
        overall = (agg(ea, ref) - agg(eb, ref)) * 100
        ax.axvline(overall, color="#f39c12", lw=2, ls="--",
                   label=f"aggregate {overall:+.2f}")
        lo, hi = np.percentile(boot_diff(ref, ea, eb, a.n_boot), [2.5, 97.5]) * 100
        ax.axvspan(lo, hi, color="#f39c12", alpha=0.12,
                   label=f"95% CI [{lo:+.2f}, {hi:+.2f}]")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{mins[i]:.0f}m" for i in order], fontsize=7)
        ax.set_xlabel(f"{UNIT_NAME[unit]} difference, points\n"
                      f"left = {lb} better, right = {la} better")
        ax.set_title(f"{UNIT_NAME[unit]}, n={len(d)}")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.suptitle(f"Per-file paired difference: {la} minus {lb}", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[saved] {a.out}")


def cmd_bootstrap(a):
    la, _, ra = load(a.a)
    lb, _, rb = load(a.b)
    units = [u for u in ("char", "word")
             if any(r["unit"] == u for r in ra) and any(r["unit"] == u for r in rb)]
    plt, (fig, axes) = _fig(1, len(units), w=5.6 * len(units), h=4.4)
    axes = np.atleast_1d(axes)
    for ax, unit in zip(axes, units):
        ref, ea, eb, _ = pair(ra, rb, unit)
        boot = boot_diff(ref, ea, eb, a.n_boot) * 100
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax.hist(boot, bins=60, color="#5499c7", edgecolor="none")
        ax.axvline(0, color="black", lw=1.5, label="no difference")
        ax.axvline(lo, color="#c0392b", lw=1.2, ls="--")
        ax.axvline(hi, color="#c0392b", lw=1.2, ls="--",
                   label=f"95% CI [{lo:+.2f}, {hi:+.2f}]")
        # The fraction on the wrong side of zero is the honest read of "how sure":
        # a CI that merely touches zero and one that straddles it evenly are both
        # reported as "not resolvable", and they are not the same evidence.
        frac = float((boot > 0).mean())
        ax.set_title(f"{UNIT_NAME[unit]}, n={len(ref)} files\n"
                     f"{frac*100:.0f}% of resamples favour {lb}")
        ax.set_xlabel(f"{la} minus {lb}, points")
        ax.set_ylabel("resamples")
        ax.legend(fontsize=7)
    fig.suptitle(f"Bootstrap distribution ({a.n_boot} resamples over files)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[saved] {a.out}")


def cmd_runs(a):
    """A sampling engine's spread against a deterministic engine's single value."""
    lb, _, rows_b = load(a.baseline)
    runs = [load(p) for p in a.runs]
    units = ["char", "word"]
    plt, (fig, axes) = _fig(1, 2, w=11.0, h=4.4)
    for ax, unit in zip(np.atleast_1d(axes), units):
        base_rows = [r for r in rows_b if r["unit"] == unit]
        if not base_rows:
            ax.set_visible(False)
            continue
        ref = np.array([r["ref_chars"] for r in base_rows], float)
        base = agg(np.array([r["coverage_cer"] for r in base_rows]), ref) * 100
        vals = []
        for _, d, rows in runs:
            rr = [r for r in rows if r["unit"] == unit]
            w = np.array([r["ref_chars"] for r in rr], float)
            vals.append(agg(np.array([r["coverage_cer"] for r in rr]), w) * 100)
        vals = np.array(vals)
        ax.scatter(vals, np.zeros_like(vals), s=70, color="#2471a3", zorder=3,
                   label=f"{len(vals)} sampled runs")
        mean = vals.mean()
        ax.axvline(mean, color="#2471a3", lw=1.4, label=f"mean {mean:.2f}")
        if len(vals) >= 2:
            # One-sample t interval on the sampled mean, with the deterministic engine
            # as a constant: only one side of this comparison has sampling error.
            sd = vals.std(ddof=1)
            # Keyed on DEGREES OF FREEDOM (n-1), not n. Reuses repeat_distribution.py's
            # table so the plot and the printed interval cannot disagree.
            from benchmarks.repeat_distribution import t_crit
            half = t_crit(len(vals) - 1) * sd / np.sqrt(len(vals))
            ax.axvspan(mean - half, mean + half, color="#2471a3", alpha=0.15,
                       label=f"95% t-interval +/-{half:.2f}")
        ax.axvline(base, color="#c0392b", lw=2,
                   label=f"{lb} (deterministic) {base:.2f}")
        ax.set_yticks([])
        ax.set_xlabel(f"{UNIT_NAME[unit]}, %")
        ax.set_title(UNIT_NAME[unit])
        ax.legend(fontsize=7, loc="upper center")
    fig.suptitle("Sampling engine run distribution vs a deterministic baseline",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[saved] {a.out}")


def cmd_sensitivity(a):
    """Aggregate vs the min_cut excusal threshold, rescored from saved hypotheses.

    Absolute levels drift several points across the threshold range, so the question
    that matters is whether the LINES CROSS. If they do not, the excusal rule is
    changing every config by a similar amount and the comparisons survive.
    """
    from metrics.eval_coverage import load_reference, score_pair

    if not a.hyp_dir or not a.corpus:
        _die("sensitivity needs --hyp-dir and --corpus (it rescores raw hypotheses)")
    hypdir, corpus = Path(a.hyp_dir).expanduser(), Path(a.corpus).expanduser()
    refs = {}
    for p in sorted(corpus.iterdir()):
        if p.suffix == ".wav":
            continue
        refs[p.stem.replace("_transcript", "")] = load_reference(str(p))
    if not refs:
        _die(f"no references found in {corpus}")

    cuts = [int(c) for c in a.cuts.split(",")]
    configs = {}
    for hyp in sorted(hypdir.glob("*.txt")):
        parts = hyp.stem.split(".")
        stem, label = parts[0], ".".join(parts[1:]) or "hyp"
        if stem in refs:
            configs.setdefault(label, []).append((stem, hyp))
    if not configs:
        _die(f"no hypotheses in {hypdir} matched a reference stem")

    plt, (fig, ax) = _fig(w=8.0, h=4.8)
    for label, files in sorted(configs.items()):
        ys = []
        for cut in cuts:
            num = den = 0.0
            for stem, hyp in files:
                unit, s = score_pair(refs[stem],
                                     hyp.read_text(encoding="utf-8"), cut, 6)
                if unit != "char":
                    continue
                num += s["coverage_cer"] * s["ref_chars"]
                den += s["ref_chars"]
            ys.append(num / den * 100 if den else float("nan"))
        ax.plot(cuts, ys, marker="o", label=f"{label} ({len(files)} files)")
    ax.axvline(30, color="black", lw=1, ls=":", label="shipped min_cut = 30")
    ax.set_xlabel("min_cut: insertion-run length excused as omitted-from-reference audio")
    ax.set_ylabel("aggregate coverage CER, %")
    ax.set_title("Threshold sensitivity: absolute level moves, ranking should not")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[saved] {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("paired", "bootstrap"):
        s = sub.add_parser(name)
        s.add_argument("a")
        s.add_argument("b")
        s.add_argument("--out", required=True)
        s.add_argument("--n-boot", type=int, default=20000)

    s = sub.add_parser("runs")
    s.add_argument("--baseline", required=True)
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True)

    s = sub.add_parser("sensitivity")
    s.add_argument("--hyp-dir", required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--cuts", default="10,20,30,50,80")
    s.add_argument("--out", required=True)

    a = p.parse_args()
    return {"paired": cmd_paired, "bootstrap": cmd_bootstrap,
            "runs": cmd_runs, "sensitivity": cmd_sensitivity}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
