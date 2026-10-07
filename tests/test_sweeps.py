"""Every generic sweep chart (scripts/docs/sweeps/*.py) must match its doc table.

Same contract as test_charts.py: each row's key and values sit together on one table
row of the doc the chart is embedded in, the ringed default is one of the chart's own
points, and the chart is embedded in that doc.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "docs"))

import chart_data as cd  # noqa: E402

ROWS = [(name, spec, row) for name, spec in cd.SWEEPS.items() for row in spec["rows"]]


@pytest.mark.parametrize("name,spec,row", ROWS,
                         ids=[f"{n}-{r[0]}" for n, _, r in ROWS])
def test_sweep_row_matches_one_table_row(name, spec, row):
    if spec.get("layout") == "columns":
        pytest.skip("column-laid table; checked per series below")
    x, key, values = row[:3]
    vals = [v for v in values if v is not None]
    if len(row) > 3 and row[3]:   # the CI strings must be printed on the same row
        vals += [c for c in row[3] if c]
    lines = [ln for ln in (ROOT / spec["doc"]).read_text().splitlines()
             if ln.startswith("|")]
    assert any(key in ln and all(v in ln for v in vals) for ln in lines), (
        f"{name} x={x}: no table row in {spec['doc']} holds {key!r} with {vals}")


COLUMN_SERIES = [(name, spec, i) for name, spec in cd.SWEEPS.items()
                 if spec.get("layout") == "columns" for i in range(len(spec["series"]))]


@pytest.mark.parametrize("name,spec,i", COLUMN_SERIES,
                         ids=[f"{n}-{s['series'][i]}" for n, s, i in COLUMN_SERIES])
def test_column_laid_series_matches_one_table_row(name, spec, i):
    """For a table whose rows are the series and whose columns are the x values
    (e.g. one config per row, one threshold per column): every value of series `i`
    sits on the one table line that carries that series' key."""
    key = spec["series_keys"][i]
    vals = [r[2][i] for r in spec["rows"] if r[2][i] is not None]
    lines = [ln for ln in (ROOT / spec["doc"]).read_text().splitlines()
             if ln.startswith("|")]
    assert any(key in ln and all(v in ln for v in vals) for ln in lines), (
        f"{name} series {spec['series'][i]!r}: no row in {spec['doc']} holds "
        f"{key!r} with {vals}")


@pytest.mark.parametrize("name", list(cd.SWEEPS))
def test_sweep_default_is_a_point_and_chart_is_embedded(name):
    spec = cd.SWEEPS[name]
    if spec.get("chosen") is not None:
        assert spec["chosen"] in [r[0] for r in spec["rows"]], name
    for i, label in enumerate(spec["series"]):
        assert all(len(r[2]) == len(spec["series"]) for r in spec["rows"]), name
    doc = (ROOT / spec["doc"]).read_text()
    assert f"img/{name}-light.svg" in doc, f"{name} not embedded in {spec['doc']}"
