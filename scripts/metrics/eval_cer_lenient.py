"""Lenient CER for Japanese: don't penalize legitimate spelling variants.

Japanese has no fixed orthography. The same word can be written in kanji,
hiragana or katakana, and numbers can be digits or spelled out, all pronounced
identically. Plain CER charges a full edit for each such difference, so it
conflates "the model misheard" with "the model spelled it differently than the
reference typist did".

Two metrics here, in increasing leniency:

  kana CER    convert both sides to hiragana readings, then compare. Cheap and
              already a good proxy: it collapses kanji/kana/digit variants that
              share a reading. (This is what eval_cer_kana.py does.)

  lenient CER after the alignment, forgive any substitution whose reference and
              hypothesis segments share a reading. Implemented as a lattice-lite:
              instead of building all plausible respellings up front (as Karita
              et al., ACL CAWL 2023 do with lexical resources and an MT model),
              align once on characters, then re-score each edit region by
              comparing readings. This catches variants that kana CER misses
              because the *segmentation* differs, e.g. a kanji compound vs its
              hiragana spelling, where the reading matches but the character
              count does not.

Reported together so the gap between them is visible; the honest headline number
for model quality is the kana or lenient figure, while plain CER is what matches
the reference verbatim.

    uv run python eval_cer_lenient.py ref.srt hyp.srt [--verbose]
"""

import argparse
import re

import pykakasi
from rapidfuzz.distance import Levenshtein

from metrics.eval_cer import load_text, normalize

_kks = pykakasi.kakasi()

# --- Japanese number readings ---------------------------------------------
#
# pykakasi does NOT read digits: it converts `2018年` to `2018ねん`, leaving the
# numeral alone. So kana CER as originally written collapses kanji/kana variants but
# NOT digits-versus-spelled-out, which is the single largest orthographic difference
# on this material and the one an inverse-text-normalising model produces most.
# Measured on the pair `答えは1500でした` / `こたえはせんごひゃくでした`: kana CER
# reads 55% for two spellings of the same sentence, all of it from the numeral,
# against 0% once the digits are read.
#
# `read_digits=True` fills that in. It is OPT-IN and off by default so that every
# figure already published from `eval_cer_kana.py` and `eval_cer_lenient.py` keeps
# its meaning; only the coverage-aware wrappers in eval_coverage_kana.py ask for it.
#
# Limits, stated rather than discovered later: counter-specific readings are not
# modelled (4年 is read よんねん here, not よねん; 19時 becomes じゅうきゅうじ, not じゅうくじ),
# and a digit string a speaker would read one digit at a time is still read as a
# number. Each leaves a small residual charge, so this narrows the confound without
# claiming to erase it.
_ONES = {1: "いち", 2: "に", 3: "さん", 4: "よん", 5: "ご",
         6: "ろく", 7: "なな", 8: "はち", 9: "きゅう"}
# Irregular in exactly the places Japanese phonology requires: 300/600/800 and
# 3000/8000 are not the regular ones-plus-unit forms, and 1 is dropped before
# じゅう/ひゃく/せん.
_TENS = {1: "じゅう", **{d: _ONES[d] + "じゅう" for d in range(2, 10)}}
_HUNDREDS = {1: "ひゃく", 2: "にひゃく", 3: "さんびゃく", 4: "よんひゃく", 5: "ごひゃく",
             6: "ろっぴゃく", 7: "ななひゃく", 8: "はっぴゃく", 9: "きゅうひゃく"}
_THOUSANDS = {1: "せん", 2: "にせん", 3: "さんぜん", 4: "よんせん", 5: "ごせん",
              6: "ろくせん", 7: "ななせん", 8: "はっせん", 9: "きゅうせん"}
# 万 upward take the full ones reading, so 1万 is いちまん and never まん.
_GROUPS = ["", "まん", "おく", "ちょう", "けい"]

# Above 京 (10^20) Japanese has no everyday group name, and a digit string that long is
# not a quantity anyway: it is an ID, an account number or a run-together timestamp,
# which a speaker reads digit by digit. Both cases are handled by reading each digit
# separately rather than by extending the table with names nobody says.
_MAX_GROUPED_DIGITS = 4 * len(_GROUPS)

_DIGIT_RUN = re.compile(r"[0-9０-９][0-9０-９,，]*(?:[.．][0-9０-９]+)?")


def _read_group(n: int) -> str:
    """Read 1..9999, the unit Japanese numbers are grouped in."""
    out = ""
    if n >= 1000:
        out += _THOUSANDS[n // 1000]
        n %= 1000
    if n >= 100:
        out += _HUNDREDS[n // 100]
        n %= 100
    if n >= 10:
        out += _TENS[n // 10]
        n %= 10
    if n:
        out += _ONES[n]
    return out


def read_number(token: str) -> str:
    """Hiragana reading of a digit string. `2018` -> `にせんじゅうはち`.

    Handles thousands separators and a decimal point (read as てん, then digit by
    digit, which is how a decimal is actually spoken). Returns the input unchanged if
    it is not a number, so a caller never has to pre-validate.
    """
    token = token.translate(str.maketrans("０１２３４５６７８９，．", "0123456789,."))
    token = token.replace(",", "")
    whole, _, frac = token.partition(".")
    if not whole.isdigit():
        return token
    # Beyond 京, and for leading-zero strings, read digit by digit. A 24-digit run in a
    # transcript is an identifier or a run-together timestamp, not a quantity, and that
    # is how a speaker says it. Also keeps the group table from being indexed past its
    # end, which a real hypothesis did hit.
    if len(whole) > _MAX_GROUPED_DIGITS or (len(whole) > 1 and whole[0] == "0"):
        return "".join("れい" if c == "0" else _ONES[int(c)] for c in whole)
    n = int(whole)
    if n == 0:
        head = "れい"
    else:
        head, groups = "", []
        # Split into 4-digit groups from the right, which is how 万/億/兆 work.
        while n:
            groups.append(n % 10000)
            n //= 10000
        for i in reversed(range(len(groups))):
            if not groups[i]:
                continue
            head += _read_group(groups[i]) + _GROUPS[i]
    if frac.isdigit():
        head += "てん" + "".join(_ONES[int(c)] if c != "0" else "れい" for c in frac)
    return head


def to_kana(text: str, read_digits: bool = False) -> str:
    """Hiragana reading of a string.

    ``read_digits`` also reads numerals aloud, which pykakasi does not do. Off by
    default: turning it on changes every kana figure this project has published, so
    it is requested explicitly by the coverage-aware wrappers rather than assumed.
    """
    if read_digits:
        text = _DIGIT_RUN.sub(lambda m: read_number(m.group(0)), text)
    return "".join(item["hira"] for item in _kks.convert(text))


def _script(ch: str) -> str:
    o = ord(ch)
    if 0x3040 <= o <= 0x309F:
        return "hira"
    if 0x30A0 <= o <= 0x30FF:
        return "kata"
    if 0x4E00 <= o <= 0x9FFF:
        return "kanji"
    if ch.isdigit():
        return "digit"
    return "other"


def _same_reading(a: str, b: str, read_digits: bool = False) -> bool:
    """True when the two strings differ only in how the SAME word is written.

    Two guards, both needed:

    1. Identical reading, compared exactly. No collapsing of long vowels or
       gemination: a looser version forgave pairs differing by a chouon (a
       kanji word vs a katakana rendering with an extra long vowel), which is a
       misrecognition rather than a spelling choice.
    2. One side is a pure-kana rendering of the other. This is what separates an
       orthographic variant from a homophone confusion. A kanji word rewritten
       in hiragana is the same word respelled, so it is forgiven; two different
       kanji words that merely share a reading are NOT, because a reader sees
       the wrong word.

    ``read_digits`` extends guard 1 to numerals, so `2018年` and `にせんじゅうはちねん`
    compare equal. Off by default, since it changes every figure already published
    from this module; see ``to_kana``.
    """
    if a == b:
        return True
    ka, kb = to_kana(a, read_digits), to_kana(b, read_digits)
    if not ka or ka != kb:
        return False
    # Require that the difference is a script change, not a kanji-to-different
    # -kanji swap: at least one side must contain no kanji at all. A digit run
    # counts as "not kanji" for this purpose: a numeral is a spelling of a number,
    # not a word choice, so `2018年` vs `にせんじゅうはちねん` must pass here.
    a_kanji = any(_script(c) == "kanji" for c in a)
    b_kanji = any(_script(c) == "kanji" for c in b)
    if a_kanji and b_kanji:
        return False
    # Reading equality via pykakasi is not proof of sameness: it assigns a
    # single most-likely reading per kanji run, so an unrelated kana string can
    # collide with it. Only forgive when the kana side is a plausible
    # respelling, i.e. it does not itself mix hiragana and katakana (which
    # signals a garbled transcription rather than a deliberate spelling).
    kana_side = a if b_kanji else b
    scripts = {_script(c) for c in kana_side}
    return not ({"hira", "kata"} <= scripts)


def lenient_ops(ref: str, hyp: str, context: int = 4, read_digits: bool = False):
    """Align once, then forgive edits that are only spelling differences.

    Returns (n_errors, forgiven, detail) where detail lists the forgiven pairs.
    Adjacent edit operations are grouped into regions before comparison, because
    a single orthographic variant usually shows up as several character edits.

    ``read_digits`` also forgives digits-versus-spelled-out numbers; see ``to_kana``
    for why that is opt-in.
    """
    ops = Levenshtein.editops(ref, hyp)
    if not ops:
        return 0, 0, []

    # Group ops into contiguous regions (in both coordinate systems).
    regions = []
    cur = [ops[0]]
    for o in ops[1:]:
        prev = cur[-1]
        if o.src_pos - prev.src_pos <= 1 and o.dest_pos - prev.dest_pos <= 1:
            cur.append(o)
        else:
            regions.append(cur)
            cur = [o]
    regions.append(cur)

    n_err = 0
    forgiven = 0
    detail = []
    for reg in regions:
        cost = len(reg)
        rs = min(o.src_pos for o in reg)
        re_ = max(o.src_pos for o in reg) + 1
        hs = min(o.dest_pos for o in reg)
        he = max(o.dest_pos for o in reg) + 1
        # Widen by a little context so a variant that shifts segmentation still
        # lines up (e.g. 一つ vs ひとつ, where the character counts differ), then
        # strip the shared prefix/suffix back off. Without the strip, context
        # kanji leak into the comparison and defeat the script-change guard.
        r_seg = ref[max(rs - context, 0) : min(re_ + context, len(ref))]
        h_seg = hyp[max(hs - context, 0) : min(he + context, len(hyp))]
        i = 0
        while i < min(len(r_seg), len(h_seg)) and r_seg[i] == h_seg[i]:
            i += 1
        j = 0
        while (j < min(len(r_seg), len(h_seg)) - i
               and r_seg[len(r_seg) - 1 - j] == h_seg[len(h_seg) - 1 - j]):
            j += 1
        r_core = r_seg[i : len(r_seg) - j]
        h_core = h_seg[i : len(h_seg) - j]
        if _same_reading(r_core, h_core, read_digits):
            forgiven += cost
            detail.append((r_core, h_core, cost))
        else:
            n_err += cost
    return n_err, forgiven, detail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("hypothesis")
    p.add_argument("--verbose", action="store_true",
                   help="list the forgiven spelling variants")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--read-digits", action="store_true",
                   help="also read numerals aloud, so 2018年 and にせんじゅうはちねん "
                        "compare equal. pykakasi does not do this, so kana CER "
                        "without it charges every digit/spelled-out difference as a "
                        "misrecognition. Off by default because it changes every "
                        "figure previously published from this script")
    a = p.parse_args()

    ref = normalize(load_text(a.reference))
    hyp = normalize(load_text(a.hypothesis))

    strict = Levenshtein.distance(ref, hyp)
    kana_ref = to_kana(ref, a.read_digits)
    kana_hyp = to_kana(hyp, a.read_digits)
    kana_d = Levenshtein.distance(kana_ref, kana_hyp)
    n_err, forgiven, detail = lenient_ops(ref, hyp, read_digits=a.read_digits)

    print(f"ref_chars={len(ref)} hyp_chars={len(hyp)}")
    print(f"CER        = {strict / max(len(ref), 1):.4f}  ({strict} edits)")
    print(f"kanaCER    = {kana_d / max(len(kana_ref), 1):.4f}  ({kana_d} edits "
          f"over {len(kana_ref)} kana)")
    print(f"lenientCER = {n_err / max(len(ref), 1):.4f}  ({n_err} edits, "
          f"{forgiven} forgiven as spelling variants)")
    if strict:
        print(f"orthographic share of plain CER: {100 * forgiven / strict:.0f}%")

    if a.verbose and detail:
        print(f"\nforgiven variants (showing {min(a.limit, len(detail))} of "
              f"{len(detail)}), ref -> hyp:")
        for r, h, c in sorted(detail, key=lambda x: -x[2])[: a.limit]:
            print(f"  [{c}] {r!r} -> {h!r}")


if __name__ == "__main__":
    main()
