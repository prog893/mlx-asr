"""The doc charts must agree with the doc tables they illustrate.

The figures in `scripts/docs/chart_data.py` are copied from published tables. A copy
drifts, and a chart that disagrees with its own table is worse than no chart, because
the picture is what a reader takes away. So every value is checked against the text of
the doc it is attributed to, and every chart the generator writes must exist and be
embedded somewhere.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "docs"))

import chart_data as cd  # noqa: E402

CHARTS = ["picker", "whisper-sizes", "windows", "precision", "batch", "delay", "gain"]


def _doc(rel: str) -> str:
    return (ROOT / rel).read_text()


def _values_by_doc():
    """(doc, value) for every figure a chart draws."""
    for name, jp, en, speed, peak, _note, src in cd.PICKER:
        for v in (jp, en, speed):
            if v:
                yield src, v, name
        if peak:
            yield cd.PICKER_SOURCE["peak"], peak, name
    for row in cd.WHISPER_SIZES["rows"]:
        for v in row[1:]:
            yield cd.WHISPER_SIZES["doc"], v, row[0]
    for panel in cd.WINDOWS + cd.PRECISION + cd.BATCH:
        for row in panel["rows"]:
            for v in row[1:]:
                if v:
                    yield panel["doc"], v, f"{panel['name']} {row[0]}"
    for sweep in (cd.DELAY, cd.GAIN):
        for row in sweep["rows"]:
            for v in row[1:]:
                yield sweep["doc"], v, str(row[0])


@pytest.mark.parametrize("doc,value,what", list(_values_by_doc()))
def test_every_charted_value_is_in_its_source_doc(doc, value, what):
    assert value in _doc(doc), f"{what}: {value} not found in {doc}"


def test_every_sweep_marks_a_value_it_actually_has():
    for panel in [cd.WHISPER_SIZES, cd.DELAY, cd.GAIN] + cd.WINDOWS + cd.PRECISION + cd.BATCH:
        xs = [r[0] for r in panel["rows"]]
        assert panel["chosen"] in xs, panel.get("name", panel["doc"])


@pytest.mark.parametrize("name", CHARTS)
def test_every_chart_is_rendered_in_both_modes_and_embedded(name):
    img = ROOT / "docs" / "benchmarks" / "img"
    for mode in ("light", "dark"):
        assert (img / f"{name}-{mode}.svg").exists(), f"{name}-{mode}.svg missing; run scripts/docs/gen_charts.py"
    docs = "".join(p.read_text() for p in (ROOT / "docs").rglob("*.md"))
    assert f"img/{name}-light.svg" in docs, f"{name} is rendered but not embedded in any doc"
