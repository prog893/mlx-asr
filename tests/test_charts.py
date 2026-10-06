"""The doc charts must agree with the doc tables they illustrate.

The figures in `scripts/docs/chart_data.py` are copied from published tables. A copy
drifts, and a chart that disagrees with its own table is worse than no chart, because
the picture is what a reader takes away. Three checks:

- every chart row's values sit together on ONE table row of the doc they are
  attributed to, next to that row's own key, so swapping two rows' values fails;
- every sweep rings a value it actually has;
- the committed SVGs are exactly what the generator produces from the current data,
  so a table edit cannot leave a stale picture behind.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "docs"))

import chart_data as cd  # noqa: E402

CHARTS = ["picker", "whisper-sizes", "windows", "precision", "batch", "delay", "gain"]
IMG = ROOT / "docs" / "benchmarks" / "img"

# The MODELS.md row each picker point takes its peak from: a repo id or row prefix
# unique to that row.
PEAK_ROW = {
    "voxtral": "Voxtral-Mini-4B-Realtime-2602-4bit)",
    "whisper large-v3": "whisper-large-v3-mlx)",
    "whisper turbo": "whisper-large-v3-turbo)",
    "qwen3-asr 1.7B": "Qwen3-ASR-1.7B-8bit)",
    "qwen3-asr 0.6B": "Qwen3-ASR-0.6B-8bit)",
    "voxtral-v1 3B": "| `3B` | `8bit`",
    "voxtral-v1 24B": "| `24B` | `8bit`",
    "kotoba": "kotoba-whisper-v2.0)",
    "parakeet": "parakeet-tdt_ctc-0.6b-ja)",
}
GAIN_ROW = {-20: "-20 dB", -12: "-12 dB", 0: "unity", 6: "+6 dB"}


def _rows():
    """(doc, key, values, what): `key` and every value must share one table row."""
    for name, jp, en, speed, peak, _note, src in cd.PICKER:
        yield src, None, [v for v in (jp, en, speed) if v], name
        if peak:
            yield cd.PICKER_SOURCE["peak"], PEAK_ROW[name], [peak], f"{name} peak"
    for size, jp, en in cd.WHISPER_SIZES["rows"]:
        yield cd.WHISPER_SIZES["doc"], f"`{size}`", [jp, en], f"whisper {size}"
    for p in cd.WINDOWS:
        for x, jp, en in p["rows"]:
            yield p["doc"], f"{x}s", [v for v in (jp, en) if v], f"{p['name']} {x}s"
    for p in cd.PRECISION:
        for q, jp, peak in p["rows"]:
            yield p["doc"], q, [jp, peak], f"{p['name']} {q}"
    for p in cd.BATCH:
        for b, xrt in p["rows"]:
            yield p["doc"], f"| {b} |", [xrt], f"{p['name']} batch {b}"
    for x, jp, en in cd.DELAY["rows"]:
        yield cd.DELAY["doc"], f"{x}ms", [jp, en], f"delay {x}"
    for x, jp, en in cd.GAIN["rows"]:
        yield cd.GAIN["doc"], GAIN_ROW[x], [jp, en], f"gain {x}"


@pytest.mark.parametrize("doc,key,values,what", list(_rows()),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_each_chart_row_matches_one_table_row(doc, key, values, what):
    lines = [ln for ln in (ROOT / doc).read_text().splitlines() if ln.startswith("|")]
    hits = [ln for ln in lines
            if all(v in ln for v in values) and (key is None or key in ln)]
    assert hits, (f"{what}: no table row in {doc} holds {values}"
                  + (f" next to {key!r}" if key else ""))


def test_every_sweep_marks_a_value_it_actually_has():
    for panel in [cd.WHISPER_SIZES, cd.DELAY, cd.GAIN] + cd.WINDOWS + cd.PRECISION + cd.BATCH:
        xs = [r[0] for r in panel["rows"]]
        assert panel["chosen"] in xs, panel.get("name", panel["doc"])


@pytest.mark.parametrize("name", CHARTS)
def test_every_chart_is_rendered_in_both_modes_and_embedded(name):
    for mode in ("light", "dark"):
        assert (IMG / f"{name}-{mode}.svg").exists(), (
            f"{name}-{mode}.svg missing; run scripts/docs/gen_charts.py")
    docs = "".join(p.read_text() for p in ROOT.rglob("*.md") if ".venv" not in p.parts)
    assert f"img/{name}-light.svg" in docs, f"{name} is rendered but not embedded"


def test_committed_svgs_match_the_current_data(tmp_path):
    """Regenerate into a scratch dir and compare byte for byte.

    The generator is deterministic (no date metadata, fixed hash salt), so any
    difference means chart_data.py or the generator changed without the committed
    SVGs being regenerated. Needs matplotlib (the `eval` extra); text metrics come from
    the local fonts, so run it where the SVGs were generated.
    """
    pytest.importorskip("matplotlib")
    import gen_charts

    stale = [p.name for p in gen_charts.render_all(tmp_path)
             if p.read_bytes() != (IMG / p.name).read_bytes()]
    assert not stale, (f"stale SVGs, regenerate with scripts/docs/gen_charts.py: "
                       f"{', '.join(stale)}")
