"""Coverage CER on hiragana readings, and on readings-forgiven alignments.

Both metrics already exist here, but only as *plain* rates
(`eval_cer_kana.py`, `eval_cer_lenient.py`), and plain rates are meaningless on this
corpus: the references deliberately omit audio, so they read 130-152% on correct
output (see eval_coverage.py). This module composes the two ideas, so a kana or
lenient figure can be quoted next to the coverage CER it belongs beside.

Why it is needed now rather than earlier: Qwen3-ASR's Japanese finetune advertises
inverse text normalisation, i.e. it writes numbers as digits where a reference typist
may have spelled the reading out. Plain coverage CER charges every one of those as an
error, so a model that formats numbers differently from the reference would look
worse than one that misheard them. The gap between coverage CER and coverage kana CER
is exactly the size of that confound, which is why the pair is reported rather than
either alone.

  coverage kana CER    convert both sides to readings first, then run the same
                       coverage accounting. Collapses kanji/kana/digit variants that
                       share a reading. `min_cut` is scaled by the kana expansion
                       ratio, because one kanji becomes two or three kana and a cut
                       run measured in characters is longer measured in readings.

  coverage lenient CER align on characters, forgive edit regions whose two sides are
                       the same word respelled, and excuse long insertion runs the
                       same way. Catches variants kana CER misses because the
                       segmentation differs, not just the script.

Japanese only. Both return None on space-delimited text: a reading has no meaning
there, and pykakasi would pass English through unchanged, which would silently report
the plain figure under a kana label.

    uv run python eval_coverage_kana.py REFERENCE HYPOTHESIS
"""

import argparse

from rapidfuzz.distance import Levenshtein

from metrics.eval_cer import load_text, normalize
from metrics.eval_cer_lenient import to_kana
from metrics.eval_coverage import coverage_score, is_space_delimited, load_reference


def coverage_kana(ref_text: str, hyp_text: str, min_cut: int = 30,
                  read_digits: bool = True):
    """Coverage CER computed on hiragana readings. None for non-CJK text.

    ``read_digits`` defaults to True here, unlike in `eval_cer_lenient.to_kana`.
    pykakasi leaves numerals alone (`2018年` becomes `2018ねん`), so without it the
    digits confound this metric exists to remove survives untouched: measured on two
    spellings of one sentence, kana CER read 55% with the numeral left as digits and
    0% with it read. Published figures from the older plain scripts are unaffected,
    since the default there stays off.
    """
    if is_space_delimited(ref_text):
        return None
    ref = to_kana(normalize(ref_text), read_digits)
    hyp = to_kana(normalize(hyp_text), read_digits)
    # A cut segment is a fixed amount of *audio*, so its length in kana is longer
    # than in characters. Scaling by the measured expansion keeps `min_cut` meaning
    # the same thing on both scales; without it, short kana runs would be excused
    # that the character path counts, and the two figures would differ partly
    # because of the threshold rather than because of the readings.
    chars = len(normalize(ref_text))
    scale = (len(ref) / chars) if chars else 1.0
    return coverage_score(ref, hyp, max(int(round(min_cut * scale)), min_cut))


def coverage_lenient(ref_text: str, hyp_text: str, min_cut: int = 30,
                     read_digits: bool = True):
    """Coverage CER with orthographic variants forgiven. None for non-CJK text.

    Structured as "coverage first, then forgive": the coverage pass decides which
    insertion runs are omitted-from-reference audio, and the lenient pass then asks
    of the *charged* edits how many are only spelling. Doing it the other way round
    would run the reading comparison over cut segments tens of thousands of
    characters long, which is both slow and meaningless.
    """
    if is_space_delimited(ref_text):
        return None
    ref, hyp = normalize(ref_text), normalize(hyp_text)
    base = coverage_score(ref, hyp, min_cut)

    # Rebuild the hypothesis without the excused runs, so the alignment the lenient
    # pass sees is the one the coverage number was charged on.
    keep, cursor = [], 0
    for pos, length in sorted(base["excused_runs"]):
        keep.append(hyp[cursor:pos])
        cursor = pos + length
    keep.append(hyp[cursor:])
    trimmed = "".join(keep)

    from metrics.eval_cer_lenient import lenient_ops

    n_err, forgiven, _ = lenient_ops(ref, trimmed, read_digits=read_digits)
    n = max(len(ref), 1)
    return {**base,
            "lenient_cer": n_err / n,
            "forgiven": forgiven,
            "charged_edits": Levenshtein.distance(ref, trimmed)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("hypothesis")
    p.add_argument("--min-cut", type=int, default=30)
    a = p.parse_args()

    ref_text = load_reference(a.reference)
    hyp_text = load_text(a.hypothesis)
    if is_space_delimited(ref_text):
        print("reference is space-delimited; kana and lenient CER do not apply")
        return 0

    plain = coverage_score(normalize(ref_text), normalize(hyp_text), a.min_cut)
    kana = coverage_kana(ref_text, hyp_text, a.min_cut)
    len_ = coverage_lenient(ref_text, hyp_text, a.min_cut)
    print(f"coverageCER        = {plain['coverage_cer']:.4f}  "
          f"({plain['ref_chars']} ref chars)")
    print(f"coverage kanaCER   = {kana['coverage_cer']:.4f}  "
          f"({kana['ref_chars']} kana)")
    print(f"coverage lenientCER= {len_['lenient_cer']:.4f}  "
          f"({len_['forgiven']} edits forgiven as spelling variants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
