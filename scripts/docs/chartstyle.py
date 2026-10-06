"""Shared styling for the doc charts: one light and one dark SVG per chart.

Colors are the validated reference palette (slots 1 and 2, checked for colorblind and
normal-vision separation in both modes), with text in ink tokens rather than series
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
              "s1": "#2a78d6", "s2": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "s1": "#3987e5", "s2": "#d95926"},
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


def mark_chosen(ax, x, y, theme, label="ships", dx=0, dy=-14):
    """Ring the shipped value and say so; the one direct label a sweep needs."""
    ax.scatter([x], [y], s=150, facecolors="none", edgecolors=theme["ink"],
               linewidths=1.5, zorder=5)
    ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                ha="center", va="top" if dy < 0 else "bottom", color=theme["ink"],
                fontsize=9, fontweight="bold")


def legend(ax, theme, **kw):
    leg = ax.legend(frameon=False, labelcolor=theme["ink2"], fontsize=9, **kw)
    return leg


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
