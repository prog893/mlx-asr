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

import pykakasi
from rapidfuzz.distance import Levenshtein

from evals.eval_cer import load_text, normalize

_kks = pykakasi.kakasi()


def to_kana(text: str) -> str:
    """Hiragana reading of a string."""
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


def _same_reading(a: str, b: str) -> bool:
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
    """
    if a == b:
        return True
    ka, kb = to_kana(a), to_kana(b)
    if not ka or ka != kb:
        return False
    # Require that the difference is a script change, not a kanji-to-different
    # -kanji swap: at least one side must contain no kanji at all.
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


def lenient_ops(ref: str, hyp: str, context: int = 4):
    """Align once, then forgive edits that are only spelling differences.

    Returns (n_errors, forgiven, detail) where detail lists the forgiven pairs.
    Adjacent edit operations are grouped into regions before comparison, because
    a single orthographic variant usually shows up as several character edits.
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
        if _same_reading(r_core, h_core):
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
    a = p.parse_args()

    ref = normalize(load_text(a.reference))
    hyp = normalize(load_text(a.hypothesis))

    strict = Levenshtein.distance(ref, hyp)
    kana_ref, kana_hyp = to_kana(ref), to_kana(hyp)
    kana_d = Levenshtein.distance(kana_ref, kana_hyp)
    n_err, forgiven, detail = lenient_ops(ref, hyp)

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
