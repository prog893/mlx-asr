"""Known-answer tests for eval_timing.

Timing metrics are easy to write and easy to get silently wrong, so each test
here fixes a case where the correct answer is known by construction: a file
against itself must read zero, a uniformly shifted file must read that shift as a
constant offset with no slope, and a progressively stretched file must read as
slope rather than offset. That last distinction is the whole point of the metric,
so it is the one worth pinning.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from metrics.eval_timing import break_stats, drift_stats, parse_cues, score


def ts(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round(s % 1 * 1000)):03d}"


def write_srt(path, cues):
    lines = []
    for i, (st, en, text) in enumerate(cues, 1):
        lines.append(f"{i}\n{ts(st)} --> {ts(en)}\n{text}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


# Japanese text, since that is the corpus this metric is for, and long enough
# that each cue clears the default 8-character anchor threshold.
#
# The fixture spans minutes, not seconds, on purpose: drift slope is measured in
# ms/min, so a 20-second fixture cannot exercise it and an early version of these
# tests failed for that reason rather than because the metric was wrong.
_LINES = [
    "皆さんようこそミュージックバーナイトフライへ",
    "本日から日本国内のクラフトビールについて教えていきたいと思います",
    "ぜひ押さえてほしい銘柄について話していきます",
    "第一回はこちらの宇宙ブリューイングでございます",
    "二千十八年にできた比較的新しい醸造所になります",
    "できてすぐ皆さんの心をつかみまくった醸造所でもあります",
    "一体どういうものを作っているのかというところについて",
    "定番の土星というビールから話していければなと思います",
    "手に入らない激レアビールとしての評判は今も続いています",
    "オンラインショップでは十九時発売で数十秒で売り切れます",
]
CUES = []
_t = 5.0
for _i in range(4):                      # ~4 minutes of material
    for _line in _LINES:
        CUES.append((_t, _t + 4.0, _line))
        _t += 6.0


def test_identical_file_has_no_timing_error(tmp_path):
    p = write_srt(tmp_path / "a.srt", CUES)
    d = score(p, p)["drift"]
    assert d["anchors"] > 0
    assert d["anchor_coverage"] == pytest.approx(1.0)
    assert d["median_abs_ms"] == pytest.approx(0.0, abs=1e-6)
    assert d["p95_abs_ms"] == pytest.approx(0.0, abs=1e-6)
    assert abs(d["drift_slope_ms_per_min"]) < 1e-6


def test_uniform_shift_reads_as_constant_offset_not_slope(tmp_path):
    ref = write_srt(tmp_path / "ref.srt", CUES)
    shift = 0.4
    hyp = write_srt(tmp_path / "hyp.srt",
                    [(st + shift, en + shift, t) for st, en, t in CUES])
    d = score(ref, hyp)["drift"]
    assert d["constant_offset_ms"] == pytest.approx(400, abs=20)
    assert d["median_abs_ms"] == pytest.approx(400, abs=20)
    # a pure shift must NOT be reported as drift
    assert abs(d["drift_slope_ms_per_min"]) < 30


def test_progressive_stretch_reads_as_slope(tmp_path):
    """The failure that matters: error accumulating through the file."""
    ref = write_srt(tmp_path / "ref.srt", CUES)
    # 3% stretch: late cues drift far later, early ones barely move
    hyp = write_srt(tmp_path / "hyp.srt",
                    [(st * 1.03, en * 1.03, t) for st, en, t in CUES])
    d = score(ref, hyp)["drift"]
    assert d["drift_slope_ms_per_min"] is not None
    # 3% of one minute is 1800ms, so slope must be clearly positive and large
    assert d["drift_slope_ms_per_min"] > 500
    assert d["max_abs_ms"] > d["median_abs_ms"]


def test_negative_shift_is_signed(tmp_path):
    ref = write_srt(tmp_path / "ref.srt", CUES)
    hyp = write_srt(tmp_path / "hyp.srt",
                    [(max(0.0, st - 0.5), max(0.1, en - 0.5), t)
                     for st, en, t in CUES])
    d = score(ref, hyp)["drift"]
    assert d["constant_offset_ms"] < -100      # hypothesis is early
    assert d["median_abs_ms"] > 100            # magnitude is unsigned


def test_anchor_coverage_drops_when_text_diverges(tmp_path):
    """Omitted-from-reference material must lower coverage, not inflate drift."""
    ref = write_srt(tmp_path / "ref.srt", CUES)
    extra = CUES + [(21.0, 30.0, "これは参照には含まれない余計な雑談です")]
    hyp = write_srt(tmp_path / "hyp.srt", extra)
    d = score(ref, hyp)["drift"]
    # the shared prefix still anchors and still reads as on-time
    assert d["median_abs_ms"] == pytest.approx(0.0, abs=50)
    assert d["anchor_coverage"] > 0.8


def test_identical_breaks_score_perfect(tmp_path):
    p = write_srt(tmp_path / "a.srt", CUES)
    b = score(p, p)["breaks"]
    assert b["break_f1"] == pytest.approx(1.0)
    assert b["mid_phrase_rate"] == pytest.approx(0.0)


def test_splitting_every_cue_in_two_is_penalised(tmp_path):
    """Same words and timings, twice the cues: breaks must score worse."""
    ref = write_srt(tmp_path / "ref.srt", CUES)
    split = []
    for st, en, t in CUES:
        mid = (st + en) / 2
        h = len(t) // 2
        split.append((st, mid, t[:h]))
        split.append((mid, en, t[h:]))
    hyp = write_srt(tmp_path / "hyp.srt", split)
    b = score(ref, hyp)["breaks"]
    assert b["hyp_cues"] == 2 * b["ref_cues"]
    assert b["break_f1"] < 0.8                 # precision halves
    assert b["mid_phrase_rate"] > 0.2          # the added cuts are mid-phrase


def test_vtt_and_srt_parse_equivalently(tmp_path):
    srt = write_srt(tmp_path / "a.srt", CUES)
    vtt = tmp_path / "a.vtt"
    body = ["WEBVTT", ""]
    for st, en, t in CUES:
        body.append(f"{ts(st).replace(',', '.')} --> {ts(en).replace(',', '.')}")
        body.append(t)
        body.append("")
    vtt.write_text("\n".join(body), encoding="utf-8")
    assert len(parse_cues(srt)) == len(parse_cues(str(vtt)))
    a, b = parse_cues(srt), parse_cues(str(vtt))
    for (s1, e1, _), (s2, e2, _) in zip(a, b):
        assert s1 == pytest.approx(s2, abs=0.002)
        assert e1 == pytest.approx(e2, abs=0.002)


def test_cli_runs(tmp_path):
    """Invoked as a module, not a path.

    These files import each other (`from metrics.eval_cer import normalize`), so
    running one by path puts its own directory on sys.path rather than the package
    parent and the sibling import fails. `-m` from scripts/ is the documented way.
    """
    p = write_srt(tmp_path / "a.srt", CUES)
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-m", "metrics.eval_timing", p, p],
                       capture_output=True, text=True, cwd=str(root / "scripts"))
    assert r.returncode == 0, r.stderr
    assert "drift" in r.stdout and "cue breaks" in r.stdout
