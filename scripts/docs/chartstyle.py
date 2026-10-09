"""Shared styling for the doc charts: one light and one dark SVG per chart.

Colors are the validated reference palette (slots 1 to 5 in fixed order, checked for
colorblind and normal-vision separation in both modes; slots 3 to 5 sit under 3:1 on the
light surface, so every chart carries its table directly below it and, from four series
up, direct end labels), with text in ink tokens rather than series
colors. Output is deterministic (no date metadata, a fixed hash salt, text kept as
text), so regenerating an unchanged chart produces no diff.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100",
              "s5": "#e87ba4"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "s4": "#c98500",
             "s5": "#d55181"},
}

matplotlib.rcParams.update({
    "svg.fonttype": "none",
    "svg.hashsalt": "mlx-asr",
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})


def mark_chosen(ax, x, y, theme, label="default", dx=0, dy=-14):
    """Ring the default value and say so; the one direct label a sweep needs."""
    ax.scatter([x], [y], s=150, facecolors="none", edgecolors=theme["ink"],
               linewidths=1.5, zorder=5)
    ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                ha="center", va="top" if dy < 0 else "bottom", color=theme["ink"],
                fontsize=9, fontweight="bold")


def legend(ax, theme, ncol=None, **_ignored):
    """Legend in a row ABOVE the plot area, never over the data.

    The panel's left title is lifted to sit above the legend rows, so title, legend and
    plot stack without touching. Any `loc` a caller passes is ignored on purpose: a
    legend placed inside the axes is exactly what this exists to prevent.
    """
    import math

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    ncol = ncol or min(len(labels), 3)
    rows = math.ceil(len(labels) / ncol)
    leg = ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.02),
                    ncol=ncol, frameon=False, labelcolor=theme["ink2"], fontsize=8.5,
                    borderaxespad=0, handlelength=1.8, columnspacing=1.2)
    lift_title(ax, rows)
    ax._legend_rows = rows
    return leg


def lift_title(ax, rows):
    """Raise the panel's left title to clear `rows` legend rows."""
    title = ax.get_title(loc="left")
    if title:
        old = ax._left_title
        ax.set_title(title, loc="left", pad=8 + 15 * rows,
                     fontsize=old.get_fontsize(), fontweight=old.get_fontweight(),
                     color=old.get_color())


def align_panels(axes):
    """Give every panel the title space of the tallest legend, so plot areas line up."""
    rows = max((getattr(ax, "_legend_rows", 0) for ax in axes), default=0)
    if rows:
        for ax in axes:
            if getattr(ax, "_legend_rows", 0) != rows:
                lift_title(ax, rows)


def save(fig, out_dir: Path, name: str, mode: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}-{mode}.svg"
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None},
                facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def picture(name: str, alt: str, rel: str = "img") -> str:
    """The markdown block a doc embeds: light/dark sources, alt text for the rest."""
    return (f'<picture>\n'
            f'  <source media="(prefers-color-scheme: dark)" srcset="{rel}/{name}-dark.svg">\n'
            f'  <img alt="{alt}" src="{rel}/{name}-light.svg">\n'
            f'</picture>')
