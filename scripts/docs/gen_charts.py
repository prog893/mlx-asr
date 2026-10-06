"""Render the doc charts, light and dark, from the published figures in chart_data.py.

    uv run --extra eval python scripts/docs/gen_charts.py

Writes docs/benchmarks/img/<chart>-{light,dark}.svg. Draws nothing it was not given:
every value comes from a doc table, and tests/test_charts.py checks each one against
that table. The charts illustrate the tables; the tables stay the reference.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

import chart_data as cd  # noqa: E402
from chartstyle import THEMES, legend, mark_chosen, save  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "benchmarks" / "img"


def _style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    ax.grid(True, color=t["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=t["ink2"], labelcolor=t["ink2"], length=0)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    ax.title.set_color(t["ink"])
    # Room for the end rings and the "ships" label above the top point.
    ax.margins(x=0.07, y=0.16)


def _grid(t, nrows, ncols, width, height):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    fig.patch.set_facecolor(t["surface"])
    flat = [ax for row in axes for ax in row]
    for ax in flat:
        _style_axes(ax, t)
    return fig, flat


def _caption(fig, t, text):
    fig.text(0.01, -0.02, text, ha="left", va="top", color=t["muted"], fontsize=8.5)


def _log_x(ax, xs, fmt=lambda x: f"{x:g}"):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([fmt(x) for x in xs])


def _jp_en_lines(ax, t, xs, jp, en, ordinal=False, en_files=2):
    pos = list(range(len(xs))) if ordinal else xs
    ax.plot(pos, jp, color=t["s1"], marker="o", markersize=6, label="Japanese CER",
            markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)
    if any(v is not None for v in en):
        pts = [(p, v) for p, v in zip(pos, en) if v is not None]
        ax.plot([p for p, _ in pts], [v for _, v in pts], color=t["s2"], marker="s",
                markersize=6, label=f"English WER ({en_files} files)",
                markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)
    return pos


# --- the picker ---------------------------------------------------------------------

# Label offsets in points, per engine, to keep names off each other. Hand-set because
# there are ten fixed points and an automatic placer moves them on every data change.
_PICKER_OFFSETS = {
    "jp": {"voxtral": (8, 2), "whisper large-v3": (-8, 3), "whisper turbo": (8, 4),
           "qwen3-asr 1.7B": (8, 2), "qwen3-asr 0.6B": (8, -2),
           "voxtral-v1 3B": (8, -2), "voxtral-v1 24B": (8, 2), "kotoba": (8, -10),
           "parakeet": (-8, 8), "reazon": (8, 2)},
    "en": {"voxtral": (8, 2), "whisper large-v3": (-8, 6), "whisper turbo": (8, 6),
           "qwen3-asr 1.7B": (8, 2), "qwen3-asr 0.6B": (8, -2),
           "voxtral-v1 3B": (8, -9), "voxtral-v1 24B": (8, 2)},
}


def picker(t):
    fig, (ax_jp, ax_en) = _grid(t, 1, 2, 10.0, 4.4)
    for ax, key, idx, title in ((ax_jp, "jp", 1, "Japanese: 17 files"),
                                (ax_en, "en", 2, "English: 3 files only")):
        for row in cd.PICKER:
            name, value, speed, peak, note = row[0], row[idx], row[3], row[4], row[5]
            if value is None:
                continue
            x, y = cd.num(speed), cd.pct(value)
            if peak is None:
                # A CPU engine has no GPU peak, so it is neither a measured fit nor a
                # measured miss; its own marker says so rather than borrowing "fits".
                ax.scatter([x], [y], s=70, zorder=4, marker="D", facecolors="none",
                           edgecolors=t["s1"], linewidths=2)
            elif cd.num(peak) <= cd.M16_WORKING_SET_GB:
                ax.scatter([x], [y], s=70, zorder=4, facecolors=t["s1"],
                           edgecolors=t["surface"], linewidths=2)
            else:
                ax.scatter([x], [y], s=70, zorder=4, facecolors="none",
                           edgecolors=t["s1"], linewidths=2)
            label = name + (f" ({note})" if note else "")
            dx, dy = _PICKER_OFFSETS[key].get(name, (8, 2))
            ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx >= 0 else "right", va="center",
                        color=t["ink"], fontsize=8.5)
        _log_x(ax, [2, 5, 10, 20, 50, 100, 250], lambda x: f"{x:g}x")
        ax.set_xlim(2, 600 if key == "jp" else 120)
        ax.set_xlabel("speed, x realtime (log scale, right is faster)")
        ax.set_ylabel("error, coverage CER/WER % (lower is better)")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=t["ink"])
        ax.annotate("better", xy=(0.97, 0.03), xytext=(0.84, 0.13),
                    xycoords="axes fraction", textcoords="axes fraction",
                    color=t["muted"], fontsize=8.5, ha="center",
                    arrowprops=dict(arrowstyle="->", color=t["muted"], lw=1))
    ax_jp.set_ylim(6, 40)
    ax_en.set_ylim(15, 27)
    # Fill legend: the one encoding the labels do not carry.
    h_fit = ax_jp.scatter([], [], s=70, facecolors=t["s1"], edgecolors=t["surface"])
    h_big = ax_jp.scatter([], [], s=70, facecolors="none", edgecolors=t["s1"],
                          linewidths=2)
    h_cpu = ax_jp.scatter([], [], s=70, marker="D", facecolors="none",
                          edgecolors=t["s1"], linewidths=2)
    ax_jp.legend([h_fit, h_big, h_cpu], ["fits a 16GB Mac (GPU peak under 12.7GB)",
                                         "needs more GPU memory",
                                         "runs on the CPU, no GPU peak"],
                 frameon=False, labelcolor=t["ink2"], fontsize=8.5, loc="lower left")
    _caption(fig, t, "M2 Ultra. Speeds marked floor or shared GPU were measured with "
                     "other GPU work resident; CPU runs on the CPU. Each shipped default "
                     "precision. Source: RESULTS.md, engines.md, MODELS.md.")
    fig.tight_layout()
    return fig


# --- sweeps -------------------------------------------------------------------------

def whisper_sizes(t):
    fig, (ax,) = _grid(t, 1, 1, 7.2, 3.6)
    d = cd.WHISPER_SIZES
    xs = [r[0] for r in d["rows"]]
    pos = _jp_en_lines(ax, t, xs, [cd.pct(r[1]) for r in d["rows"]],
                       [cd.pct(r[2]) for r in d["rows"]], ordinal=True, en_files=3)
    ax.set_xticks(pos)
    ax.set_xticklabels(xs)
    i = xs.index(d["chosen"])
    mark_chosen(ax, i, cd.pct(d["rows"][i][1]), t, dy=-12)
    ax.set_ylabel("error % (lower is better)")
    ax.set_title("Whisper size, shipped config", loc="left", fontsize=10,
                 fontweight="bold", color=t["ink"])
    legend(ax, t, loc="upper right")
    _caption(fig, t, f"{d['basis']}. large-v3 and turbo tie; the tie goes to the full "
                     f"decoder. Source: engines.md.")
    fig.tight_layout()
    return fig


def windows(t):
    fig, axes = _grid(t, 2, 2, 9.0, 6.2)
    for ax, p in zip(axes, cd.WINDOWS):
        xs = [r[0] for r in p["rows"]]
        jp = [cd.pct(r[1]) for r in p["rows"]]
        en = [cd.pct(r[2]) if r[2] else None for r in p["rows"]]
        _jp_en_lines(ax, t, xs, jp, en)
        _log_x(ax, xs, lambda x: f"{x:g}s")
        mark_chosen(ax, p["chosen"], jp[xs.index(p["chosen"])], t, dy=12)
        ax.set_title(f"{p['name']} ({p['basis']})", loc="left", fontsize=9.5,
                     fontweight="bold", color=t["ink"])
        ax.set_ylabel("error %")
    legend(axes[1], t, loc="upper left")
    _caption(fig, t, "Decode window length per engine. Longer windows let a repetition "
                     "loop burn a bigger token budget, which is why most curves turn up. "
                     "Sources: engines.md, qwen3-asr.md, voxtral-v1.md, japanese-only.md.")
    fig.tight_layout()
    return fig


def precision(t):
    fig, axes = _grid(t, 2, 3, 10.0, 6.0)
    for ax, p in zip(axes, cd.PRECISION):
        names = [r[0] for r in p["rows"]]
        jp = [cd.pct(r[1]) for r in p["rows"]]
        pos = list(range(len(names)))
        ax.plot(pos, jp, color=t["s1"], marker="o", markersize=6,
                markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{n}\n{r[2]}" for n, r in zip(names, p["rows"])],
                           fontsize=8)
        i = names.index(p["chosen"])
        mark_chosen(ax, i, jp[i], t, dy=12)
        # Every panel spans at least 8 points, so a tie looks like a tie. Zooming each
        # panel to its own range made voxtral's 1.3-point ladder look as steep as the
        # 0.6B's 7-point drop.
        lo, hi = min(jp), max(jp)
        span = max(8.0, (hi - lo) * 1.3)
        mid = (lo + hi) / 2
        ax.set_ylim(mid - span / 2, mid + span / 2)
        ax.set_title(p["name"], loc="left", fontsize=9.5, fontweight="bold", color=t["ink"])
        ax.set_ylabel("JP CER %")
    axes[-1].set_visible(False)
    _caption(fig, t, "Japanese coverage CER by precision, 20-file corpus; each tick "
                     "shows peak GPU memory. Every panel spans at least 8 points, so "
                     "flat means a tie. Sources: quantization.md, qwen3-asr.md, "
                     "voxtral-v1.md.")
    fig.tight_layout()
    return fig


def batch(t):
    fig, axes = _grid(t, 1, 2, 9.0, 3.6)
    for ax, p in zip(axes, cd.BATCH):
        xs = [r[0] for r in p["rows"]]
        ys = [cd.num(r[1]) for r in p["rows"]]
        ax.plot(xs, ys, color=t["s1"], marker="o", markersize=6,
                markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)
        _log_x(ax, [x for x in xs if x in (1, 2, 4, 8, 16, 32, 64, 128) or x == xs[-1]])
        mark_chosen(ax, p["chosen"], ys[xs.index(p["chosen"])], t, dy=12)
        ax.set_title(p["name"], loc="left", fontsize=9.5, fontweight="bold", color=t["ink"])
        ax.set_xlabel("batch size (log scale)")
        ax.set_ylabel("x realtime")
    _caption(fig, t, "Voxtral decode throughput by batch, synthetic inputs. Not "
                     "monotonic: on the M4, throughput stalls from batch 2 to 8, so the "
                     "profiles skip that range. Source: decode-throughput.md.")
    fig.tight_layout()
    return fig


def _simple_sweep(t, d, title, xlabel, xfmt, log, chosen_label="ships"):
    fig, (ax,) = _grid(t, 1, 1, 6.4, 3.4)
    xs = [r[0] for r in d["rows"]]
    jp = [cd.pct(r[1]) for r in d["rows"]]
    _jp_en_lines(ax, t, xs, jp, [cd.pct(r[2]) for r in d["rows"]])
    if log:
        _log_x(ax, xs, xfmt)
    else:
        ax.set_xticks(xs)
        ax.set_xticklabels([xfmt(x) for x in xs])
    mark_chosen(ax, d["chosen"], jp[xs.index(d["chosen"])], t, label=chosen_label,
                dy=12)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=t["ink"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("error % (lower is better)")
    legend(ax, t, loc="upper right")
    return fig, ax


def delay(t):
    d = cd.DELAY
    fig, _ = _simple_sweep(t, d, "Voxtral transcription delay", "--delay-ms",
                           lambda x: f"{x:g}", log=True)
    _caption(fig, t, f"{d['basis']}. 2400ms is also the model's maximum, and costs no "
                     f"speed. Source: delay.md.")
    fig.tight_layout()
    return fig


def gain(t):
    d = cd.GAIN
    fig, _ = _simple_sweep(t, d, "Voxtral input level", "gain applied (dB)",
                           lambda x: f"{x:+g}" if x else "0", log=False,
                           chosen_label="auto (a no-op here)")
    _caption(fig, t, f"{d['basis']}. Attenuation hurts; boosting audio that is already loud "
                     f"is slightly worse on Japanese and better on English. --gain auto "
                     f"boosts only quiet input. Source: "
                     f"input-level.md.")
    fig.tight_layout()
    return fig


CHARTS = {"picker": picker, "whisper-sizes": whisper_sizes, "windows": windows,
          "precision": precision, "batch": batch, "delay": delay, "gain": gain}


def render_all(out: Path = OUT) -> list[Path]:
    return [save(fn(theme), out, name, mode)
            for name, fn in CHARTS.items() for mode, theme in THEMES.items()]


def main():
    for path in render_all():
        print(path.relative_to(OUT.parents[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
