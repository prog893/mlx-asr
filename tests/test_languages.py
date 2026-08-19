"""Language-tag normalisation: one spelling per language, whatever the user types.

Exists because the three engines disagree on what a language argument is, and each of
them accepts a wrong one silently. mlx-whisper wants an ISO 639-1 code and treats
anything else as a decoder prompt hint; Qwen3-ASR wants an English name and interpolates
an unrecognised string straight into its prompt, so `--language ja` would produce the
prefix `language ja<asr_text>` and a token sequence the model never saw in training. Both
failures look like an unexplained accuracy loss rather than a rejected argument, which is
the exact class of bug this CLI refuses everywhere else.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlx_asr.languages import (  # noqa: E402
    UnknownLanguage,
    base_code,
    to_english_name,
    to_iso,
)

# Every spelling of Japanese a user might reasonably type.
JAPANESE_FORMS = ["ja", "ja_JP", "ja-JP", "JA", "Ja", "jpn", "ja-Hira-JP",
                  "Japanese", "japanese", "JAPANESE", " ja ", "ja-jp"]


@pytest.mark.parametrize("form", JAPANESE_FORMS)
def test_every_form_of_japanese_maps_to_one_iso_code(form):
    assert to_iso(form) == "ja", form


@pytest.mark.parametrize("form", JAPANESE_FORMS)
def test_every_form_of_japanese_maps_to_one_english_name(form):
    """What Qwen3-ASR needs. `ja` must become `Japanese`, not pass through."""
    assert to_english_name(form) == "Japanese", form


def test_regional_and_script_subtags_are_dropped():
    """An engine selects a language, not a locale. en-US and en-GB are both `en`."""
    for form in ("en-US", "en_GB", "en-Latn-US", "pt-BR", "zh-Hans-CN"):
        assert base_code(form) == form.replace("_", "-").split("-")[0].lower()


def test_syntactically_valid_nonsense_is_rejected():
    """`xx` and `zz` parse as BCP-47 and mean nothing.

    Accepting them would hand an engine a language token that does not exist, which is
    the same silent misfire the name mapping prevents.
    """
    for form in ("xx", "zz", "qq"):
        assert base_code(form) is None, form
        with pytest.raises(UnknownLanguage):
            to_iso(form)


def test_unrecognised_input_names_the_accepted_set():
    """"Unknown language" without the vocabulary leaves the user guessing."""
    with pytest.raises(UnknownLanguage) as e:
        to_iso("Klingon", alias="whisper-turbo")
    msg = str(e.value)
    assert "whisper-turbo" in msg
    assert "ja" in msg          # the vocabulary is shown
    assert "jpn" in msg         # and so is the hint that codes work


def test_an_engine_only_accepts_what_it_claims_to_support():
    """The engine's own vocabulary is authoritative, not this module's world view.

    A checkpoint that supports 8 languages must refuse the 9th even though this module
    knows how to spell it, so dropping a language upstream needs no change here.
    """
    eight = ["Chinese", "English", "Japanese", "Korean", "German",
             "French", "Spanish", "Arabic"]
    assert to_english_name("ja", eight) == "Japanese"
    assert to_english_name("korean", eight) == "Korean"
    for unsupported in ("sv", "Swedish", "th"):
        with pytest.raises(UnknownLanguage):
            to_english_name(unsupported, eight)


def test_empty_and_whitespace_are_not_languages():
    for form in ("", "   ", "\t"):
        assert base_code(form) is None
        with pytest.raises(UnknownLanguage):
            to_iso(form)


def test_the_full_qwen_vocabulary_round_trips():
    """Every language Qwen3-ASR claims must survive code -> name -> code."""
    from mlx_asr.languages import _CODE_TO_ENGLISH_NAME

    for code, name in _CODE_TO_ENGLISH_NAME.items():
        assert to_english_name(code) == name, code
        assert to_iso(name) == code, name
