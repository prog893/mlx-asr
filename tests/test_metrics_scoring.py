"""Known-answer tests for the scoring functions.

These were absent, which is how a reference-loading bug worth 16.8 WER points on one
file survived every sweep in the project. `test_metrics_timing.py` covers `eval_timing`
this way already; the character and word scorers had only been exercised indirectly,
through drivers whose output nobody could check by hand.

Every case here has an answer derivable on paper, so a failure localises to one
function rather than to "the numbers moved".
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("rapidfuzz", reason="scorers need the eval extra")

from metrics.eval_cer import normalize, srt_to_text  # noqa: E402
from metrics.eval_coverage import (  # noqa: E402
    is_space_delimited,
    load_reference,
    score_pair,
    tokenize,
)

JA_SRT = """1
00:00:01,000 --> 00:00:03,000
これはテスト

2
00:00:03,000 --> 00:00:05,000
です
"""

# Two cues that break MID-SENTENCE, which is what a real subtitle track does and what
# the "" join corrupted: "through" + "some" became "throughsome".
EN_SRT_MIDSENTENCE = """1
00:00:01,000 --> 00:00:03,000
we are going to take you through

2
00:00:03,000 --> 00:00:05,000
some real shenanigans today
"""


def test_identity_scores_zero_on_both_units():
    """A hypothesis equal to its reference must score 0, whatever the unit."""
    for ref in ("これは完全に正しい文字起こしです", "this is a perfect transcript of the audio"):
        unit, s = score_pair(ref, ref, 30, 6)
        assert s["coverage_cer"] == 0.0, (unit, s)
        assert s["coverage"] == 1.0
        assert abs(s["extra_ratio"] - 1.0) < 1e-9


def test_one_substitution_is_charged_exactly():
    """Hand-computable: 5 reference chars, 1 substituted, so 1/5 = 20%."""
    unit, s = score_pair("あいうえお", "あいXえお", 30, 6)
    assert unit == "char"
    assert s["ref_chars"] == 5
    assert s["subs"] == 1
    assert abs(s["coverage_cer"] - 0.2) < 1e-9


def test_subtitle_cues_are_not_fused_into_one_word():
    """The bug this file exists for.

    Joining cue lines with "" made "through" + "some" into "throughsome", costing a
    deletion plus two insertions per break. A subtitle reference breaks mid-sentence
    constantly, so on the one English `.srt` in the corpus this was worth 16.8 WER
    points, and 3.7 points of the published English aggregate.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False,
                                     encoding="utf-8") as f:
        f.write(EN_SRT_MIDSENTENCE)
        path = f.name
    text = srt_to_text(path)
    assert "throughsome" not in text, f"cue lines were fused: {text!r}"
    assert "through" in tokenize(text) and "some" in tokenize(text)
    # A perfect hypothesis of that reference must now score zero.
    perfect = "we are going to take you through some real shenanigans today"
    unit, s = score_pair(text, perfect, 30, 6)
    assert unit == "word"
    assert s["coverage_cer"] == 0.0, s


def test_plain_transcript_lines_are_not_fused_either():
    """The `.txt` path had the same defect; it just did not bite on this corpus,
    because those references happen to end every line with punctuation."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write("we are going to take you through\nsome real shenanigans today\n")
        path = f.name
    text = load_reference(path)
    assert "throughsome" not in text, f"lines were fused: {text!r}"
    unit, s = score_pair(text, "we are going to take you through some real "
                               "shenanigans today", 30, 6)
    assert s["coverage_cer"] == 0.0, s


def test_the_join_fix_does_not_move_japanese():
    """Japanese must be byte-identical either way: it has no word spaces, and
    `normalize` strips whitespace before scoring. This is why the bug hid."""
    assert normalize("これはテスト\nです") == normalize("これはテストです")
    a = score_pair("これはテストです", "これはテストです", 30, 6)[1]["coverage_cer"]
    b = score_pair("これはテスト\nです", "これはテストです", 30, 6)[1]["coverage_cer"]
    assert a == b == 0.0


def test_word_tokenizer_splits_on_punctuation():
    """Why only mid-sentence breaks matter: a fusion across '.' still tokenizes into
    two words, so a reference whose lines end in punctuation is unharmed."""
    assert tokenize("on.I still have") == tokenize("on. I still have")
    assert tokenize("throughsome real") != tokenize("through some real")


def test_unit_is_chosen_from_the_reference_script():
    """Choosing the wrong unit puts a file in the wrong aggregate entirely."""
    assert is_space_delimited("this is english text with spaces")
    assert not is_space_delimited("これは日本語のテキストです")


def test_long_insertion_runs_are_excused_and_short_ones_are_not():
    """The excusal rule, stated as a test rather than as a docstring claim.

    The references here deliberately omit audio, so a long inserted run is treated as
    correctly-transcribed cut material. A short one is charged as a real error. This
    pins the behaviour that `min_cut` controls; it is not an endorsement of the rule.
    """
    ref = "あ" * 100
    short = "あ" * 50 + "X" * 5 + "あ" * 50          # 5 inserted: charged
    long_run = "あ" * 50 + "X" * 60 + "あ" * 50      # 60 inserted: excused
    _, s_short = score_pair(ref, short, 30, 6)
    _, s_long = score_pair(ref, long_run, 30, 6)
    assert s_short["coverage_cer"] > 0.0
    assert s_long["insertions_excused"] > 0
    assert s_long["coverage_cer"] < s_short["coverage_cer"]


def test_extra_ratio_flags_an_editorial_reference():
    """`extra_ratio` is the guardrail that catches a hypothesis much longer than its
    reference, which is the shape excusal is meant for and also the shape an
    over-producing config has. It must be reported, not just computed."""
    ref = "あ" * 100
    _, s = score_pair(ref, "あ" * 100 + "い" * 100, 30, 6)
    assert s["extra_ratio"] > 1.5, s
