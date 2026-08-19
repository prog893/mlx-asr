"""One place that turns whatever the user typed into whatever an engine wants.

Three engines, three conventions, and the failure mode is silence rather than an error:

    whisper-*   ISO 639-1 codes: "ja". Passing "Japanese" makes it transcribe the word
                "Japanese" as a prompt hint rather than selecting a language.
    kotoba      forced to "ja" internally; a flag here is meaningless.
    qwen3-asr   English language NAMES: "Japanese". Its prompt builder looks the string
                up case-insensitively and, on a miss, interpolates it verbatim, so
                `--language ja` yields the prefix `language ja<asr_text>` and the model
                gets a token sequence it never saw in training. No exception, no warning.

That last one is exactly the class of bug this CLI refuses elsewhere: a flag that looks
honoured and does nothing. So the mapping lives here, is applied before the engine sees
it, and an unmappable value is a hard error rather than a guess.

`langcodes` does the part that is genuinely hard, which is parsing the tag zoo: `ja`,
`ja_JP`, `ja-JP`, `JA`, `jpn`, `ja-Hira-JP` all reduce to the base subtag `ja`. It is
178KB, pure Python, and has zero required dependencies.

Deliberately NOT using `langcodes[data]`. English display names would need
`language_data` (5.6MB) plus `marisa-trie`, a native extension that would need an arm64
wheel pinned in the Homebrew formula, all to resolve about thirty fixed strings that each
engine already publishes in its own config. Instead an engine's own vocabulary is matched
case-insensitively first, and `langcodes` handles everything that is a code. A name the
engine does not itself list is not a name worth accepting.
"""

# Base subtag -> the English name Qwen3-ASR's prompt builder expects. Only the languages
# that model claims; anything else must fail rather than be invented, because a plausible
# guess is what produces a silently wrong prompt.
#
# Kept as data rather than derived from CLDR so the accepted set is auditable and matches
# the model card exactly. Cantonese ("yue") is separate from Chinese ("zh") here because
# the model treats them as different languages.
_CODE_TO_ENGLISH_NAME = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian",
    "mk": "Macedonian", "ro": "Romanian",
}


class UnknownLanguage(ValueError):
    """The value could not be resolved to something the engine accepts.

    Carries the accepted set, because "unknown language" without the vocabulary leaves
    the user guessing at which of several plausible spellings this engine wants.
    """

    def __init__(self, value, accepted, alias=""):
        self.value, self.accepted, self.alias = value, accepted, alias
        shown = ", ".join(sorted(accepted)[:8])
        more = f", and {len(accepted) - 8} more" if len(accepted) > 8 else ""
        target = f" by --model {alias}" if alias else ""
        super().__init__(
            f"--language {value!r} is not recognised{target}. Accepted: {shown}{more}. "
            f"ISO codes work too, in any form: ja, ja-JP, jpn all mean Japanese."
        )


def base_code(value: str) -> str | None:
    """Reduce any language tag to its base subtag. "ja_JP" -> "ja", "jpn" -> "ja".

    Returns None when the value is not a language tag at all, so callers can fall back to
    matching an engine's own vocabulary before giving up.
    """
    if not value or not value.strip():
        return None
    try:
        import langcodes
    except ImportError:      # pragma: no cover - langcodes is a hard dependency
        # Degrade to the obvious case rather than crash: a bare two or three letter code
        # is already its own base subtag.
        token = value.strip().replace("_", "-").split("-")[0].lower()
        return token if token.isalpha() and 2 <= len(token) <= 3 else None
    try:
        parsed = langcodes.Language.get(value.strip().replace("_", "-"))
    except Exception:
        return None
    # `is_valid` separates a real subtag from one that merely parses: "xx" and "zz" are
    # well-formed BCP-47 and mean nothing, so accepting them would send an engine a
    # language token that does not exist and produce exactly the silent misfire this
    # module prevents for names.
    return parsed.language if parsed.is_valid() else None


def to_iso(value: str, alias: str = "") -> str:
    """Normalise to the ISO 639-1 code the whisper-* engines take.

    Accepts codes in any form, and English names for the languages this module knows.
    """
    code = base_code(value)
    if code:
        return code
    # Not a tag; try an English name, since a user who typed "Japanese" for Qwen may type
    # it for Whisper too and the remedy is the same either way.
    for iso, name in _CODE_TO_ENGLISH_NAME.items():
        if name.lower() == value.strip().lower():
            return iso
    raise UnknownLanguage(value, set(_CODE_TO_ENGLISH_NAME), alias)


def to_english_name(value: str, accepted=None, alias: str = "") -> str:
    """Normalise to the English language name Qwen3-ASR expects.

    ``accepted`` is the engine's own published vocabulary. Passing it means the answer is
    checked against what that specific checkpoint claims to support, rather than against
    this module's idea of the world, so a model that drops a language stops accepting it
    without any change here.
    """
    accepted = set(accepted) if accepted else set(_CODE_TO_ENGLISH_NAME.values())
    raw = value.strip()

    # The engine's own vocabulary first, case-insensitively. This is what makes
    # "japanese", "Japanese" and "JAPANESE" all work with no name database.
    by_lower = {name.lower(): name for name in accepted}
    if raw.lower() in by_lower:
        return by_lower[raw.lower()]

    code = base_code(raw)
    name = _CODE_TO_ENGLISH_NAME.get(code) if code else None
    if name and name in accepted:
        return name
    raise UnknownLanguage(value, accepted, alias)
