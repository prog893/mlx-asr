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


# --- coverage-aware kana and lenient CER ----------------------------------
#
# Needed for Qwen3-ASR, whose Japanese finetune does inverse text normalisation: it
# prefers "2018年" where a reference typist may have written the reading out. Plain
# coverage CER charges every such difference as a misrecognition, so a model that
# formats numbers differently from the reference scores worse than one that misheard
# them. These exist so the size of that confound is measurable rather than argued
# about.

pytest.importorskip("pykakasi", reason="kana metrics need the eval extra")

from metrics.eval_cer_lenient import read_number, to_kana  # noqa: E402
from metrics.eval_coverage_kana import coverage_kana, coverage_lenient  # noqa: E402


@pytest.mark.parametrize("digits,reading", [
    ("0", "れい"),
    ("7", "なな"),
    ("10", "じゅう"),          # 1 is dropped before じゅう
    ("18", "じゅうはち"),
    ("100", "ひゃく"),         # and before ひゃく
    ("300", "さんびゃく"),      # irregular: not さんひゃく
    ("600", "ろっぴゃく"),
    ("800", "はっぴゃく"),
    ("1000", "せん"),
    ("3000", "さんぜん"),       # irregular: not さんせん
    ("8000", "はっせん"),
    ("2018", "にせんじゅうはち"),
    ("10000", "いちまん"),      # 万 upward keeps いち, unlike じゅう/ひゃく/せん
    ("2026", "にせんにじゅうろく"),
    ("1,500", "せんごひゃく"),   # thousands separator
    ("１９", "じゅうきゅう"),     # full-width
    ("3.5", "さんてんご"),       # decimals are read digit by digit after てん
])
def test_number_readings_are_correct(digits, reading):
    """pykakasi does not read numerals, so this table is the metric's own arithmetic.

    Hand-checkable, and the irregular forms are the point: a naive ones-plus-unit
    generator gets 300, 600, 800, 3000 and 8000 wrong, and each error would show up as
    a permanent residual charge on any file mentioning a price or a year.
    """
    assert read_number(digits) == reading


@pytest.mark.parametrize("digits", [
    "0" * 40,
    "1234567890123456789012345",        # 25 digits, past 京
    "9" * 64,
    "007",                              # leading zeros: an identifier, not seven
    "00",
])
def test_absurd_digit_runs_do_not_crash(digits):
    """A real hypothesis crashed the sweep here, so this is a regression test.

    A transcript contains digit strings that are not quantities: identifiers, and
    run-together timestamps such as `19時0分23秒`. Anything past 京 has no everyday group
    name, and indexing the table for one raised IndexError mid-run, after four files had
    already decoded. These read digit by digit, which is what a speaker does anyway.
    """
    out = read_number(digits)
    assert out and not any(c.isdigit() for c in out), out


def test_no_digit_string_can_crash_the_reader():
    """Exhaustive over lengths and shapes, because this runs inside a scorer.

    A crash here does not produce a wrong number, it destroys a benchmark arm that has
    already spent minutes of GPU time. Cheap to rule out completely.
    """
    import random

    rng = random.Random(0)
    cases = ["", "0", "9"]
    for length in range(1, 40):
        cases.append("".join(rng.choice("0123456789") for _ in range(length)))
        cases.append("1" + "0" * (length - 1))
    cases += ["1.5", "0.000", "1,234,567.89", "１２，３４５", "12.", ".5", "1..2"]
    for case in cases:
        read_number(case)                     # must not raise
        to_kana(f"これは{case}年です", read_digits=True)


def test_pykakasi_alone_does_not_read_digits():
    """States the gap this fills, so the opt-in is not mistaken for redundancy.

    Without it, kana CER charges a digit/spelled-out difference in full, which on an
    inverse-text-normalising model is the largest orthographic difference there is.
    """
    assert to_kana("2018年") == "2018ねん"
    assert to_kana("2018年", read_digits=True) == "にせんじゅうはちねん"


def test_a_digit_spelling_difference_is_charged_plainly_and_forgiven_on_readings():
    """The whole reason these two metrics exist, on a hand-checkable pair.

    Same sentence, the year written as digits in one and spelled out in the other.
    Nothing was misheard, so coverage CER should be clearly nonzero (it charges every
    differing character) while the reading-based figures should be much lower.
    """
    ref = "答えは1500でした"
    hyp = "こたえはせんごひゃくでした"
    _, plain = score_pair(ref, hyp, 30, 6)
    kana = coverage_kana(ref, hyp, 30)
    lenient = coverage_lenient(ref, hyp, 30)
    assert plain["coverage_cer"] > 0.3, plain
    assert kana["coverage_cer"] < plain["coverage_cer"], (kana, plain)
    assert lenient["lenient_cer"] < plain["coverage_cer"], (lenient, plain)


def test_identical_text_scores_zero_on_every_variant():
    ref = "これは完全に正しい文字起こしです"
    assert coverage_kana(ref, ref, 30)["coverage_cer"] == 0.0
    assert coverage_lenient(ref, ref, 30)["lenient_cer"] == 0.0


def test_a_real_misrecognition_is_not_forgiven():
    """Leniency must not swallow errors. Different word, different reading: charged
    by all three metrics, or the kana figure would be meaningless as a quality
    number."""
    ref = "ビールを飲みました"
    hyp = "ワインを買いました"
    _, plain = score_pair(ref, hyp, 30, 6)
    assert plain["coverage_cer"] > 0.0
    assert coverage_kana(ref, hyp, 30)["coverage_cer"] > 0.0
    assert coverage_lenient(ref, hyp, 30)["lenient_cer"] > 0.0


def test_kana_metrics_refuse_space_delimited_text():
    """pykakasi passes English through unchanged, so running it there would report the
    plain figure under a kana label. Returning None makes the caller skip the file
    instead of aggregating a mislabelled number."""
    assert coverage_kana("this is english text", "this is english text") is None
    assert coverage_lenient("this is english text", "this is english text") is None


def test_cut_material_is_still_excused_on_the_kana_path():
    """The references omit audio on purpose, so the coverage rule has to survive the
    reading conversion. Without the min_cut rescaling, a run excused on characters
    could be charged on kana purely because kana expand."""
    ref = "本日は晴天なり" * 12
    hyp = ref[:40] + "これは参照から省かれた音声の書き起こしです" * 4 + ref[40:]
    kana = coverage_kana(ref, hyp, 30)
    assert kana["insertions_excused"] > 0, kana
    assert kana["coverage_cer"] < 0.25, kana


def test_lenient_excludes_cut_runs_before_comparing_readings():
    """The composition order matters. Forgiving first would run reading comparison
    over cut segments thousands of characters long: slow, and it would compare
    unrelated text. Coverage first, then forgive what is left charged."""
    ref = "本日は晴天なり" * 12
    hyp = ref + "まったく別の話題についての長い挿入がここに入ります" * 4
    lenient = coverage_lenient(ref, hyp, 30)
    assert lenient["lenient_cer"] < 0.05, lenient
