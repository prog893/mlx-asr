"""CER for references that deliberately omit parts of the audio.

Some reference transcripts are edited for a purpose other than ASR evaluation:
off-topic chatter, side conversations in another language, and music during
breaks get cut because they are irrelevant to the story being told. The audio
still contains them, so a correct transcription legitimately includes text the
reference does not.

Plain CER is meaningless on such a pair. Measured on one such file: the reference
covers 2313 characters, the hypothesis 4993, and plain CER reads 140% with 2838
"insertions" -- while 82% of the reference characters are in fact present and
correct. Scoring that as near-total failure would be wrong, and worse, it would
rank configs by how much omitted audio they skip rather than by accuracy.

The metric here treats the reference as a *subsequence* to be located, not as the
whole truth:

  coverage CER   align reference to hypothesis, then charge substitutions and
                 deletions (the model got reference content wrong or missed it)
                 but NOT insertions that fall outside the aligned span of
                 reference material. Insertions *inside* a matched region still
                 count, since those are real intrusions into transcribed speech.

  coverage       fraction of reference characters matched exactly. A high value
                 with a long hypothesis means "transcribed everything, including
                 material the reference omits".

  extra ratio    hypothesis length / reference length. Well above 1 flags an
                 edited reference; near 1 means the pair is comparable and plain
                 CER can be trusted.

Insertions are not ignored blindly: a run of inserted characters is only excused
when it is longer than ``--min-cut`` (default 30), i.e. plausibly a cut segment
rather than a hallucinated word. Short insertions inside otherwise-matching text
are counted as errors, which is what catches genuine repetition loops.

    uv run python eval_coverage.py REFERENCE HYPOTHESIS [--verbose]
"""

import argparse
import re
import unicodedata

from rapidfuzz.distance import Levenshtein

from metrics.eval_cer import load_text, normalize

SPEAKER = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_ .'-]{0,30}:\s*$")
# Keep spaces: the word-level path needs them. eval_cer.normalize() strips all
# whitespace, which is right for Japanese and destroys English tokenization.
PUNCT_KEEP_SPACE = re.compile(
    r"[、。，．,.!?！？…・「」『』（）()\[\]〈〉《》：:;；\-~〜?\"]+"
)
WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def is_space_delimited(text: str, threshold: float = 0.15) -> bool:
    """True for languages written with spaces between words.

    Japanese and Chinese have no word boundaries, so CER is the standard metric
    there; for English and similar, WER is. Decide from the text itself rather
    than a flag, since a corpus can mix both.
    """
    sample = text[:4000]
    cjk = sum(
        1 for c in sample
        if 0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF
    )
    return len(sample) > 0 and cjk / len(sample) < threshold


def normalize_words(text: str) -> str:
    """NFKC + strip punctuation but preserve single spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = PUNCT_KEEP_SPACE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str):
    return WORD.findall(normalize_words(text))


def load_reference(path: str) -> str:
    """Read a reference transcript, dropping speaker-label lines.

    Speaker turns ("Name:" on its own line) are diarization metadata, not speech,
    so they must not be scored against a model that emits no speaker labels.

    Lines are rejoined with a NEWLINE, not with "". Joining with "" fuses the last
    word of each line to the first word of the next, which is invisible on Japanese
    (no word spaces, and `normalize` strips whitespace anyway) and corrupts the
    word-level path: the fused pair costs one deletion plus two insertions. It only
    bites where a line ends mid-sentence, so a `.txt` transcript whose lines all end
    in punctuation is unaffected while a subtitle track is not. Measured on this
    corpus: 0 fused words in the two plain transcripts, 131 in the one `.srt`, worth
    **16.8 WER points** on that file alone and 3.7 points of the English aggregate.
    """
    raw = open(path, encoding="utf-8", errors="replace").read()
    if path.endswith(".srt"):
        return load_text(path)
    keep = [ln for ln in raw.split("\n") if ln.strip() and not SPEAKER.match(ln)]
    return "\n".join(keep)


def coverage_score(ref, hyp, min_cut: int = 30):
    """Return a dict of coverage-aware statistics.

    Uses Levenshtein opcodes so each edit is attributed to a span, letting long
    insertion runs (cut material) be separated from short ones (hallucination).

    ``ref``/``hyp`` may be strings (character units) or lists of tokens (word
    units); rapidfuzz handles both, so the same accounting serves CER and WER.
    Scale ``min_cut`` accordingly: a cut segment is tens of characters but only a
    handful of words.
    """
    ops = Levenshtein.opcodes(ref, hyp)
    matched = charged = 0
    excused_ins = counted_ins = subs = dels = 0
    excused_runs = []
    for o in ops:
        rlen = o.src_end - o.src_start
        hlen = o.dest_end - o.dest_start
        if o.tag == "equal":
            matched += rlen
        elif o.tag == "replace":
            subs += max(rlen, hlen)
            charged += max(rlen, hlen)
        elif o.tag == "delete":
            dels += rlen
            charged += rlen
        elif o.tag == "insert":
            if hlen >= min_cut:
                excused_ins += hlen
                excused_runs.append((o.dest_start, hlen))
            else:
                counted_ins += hlen
                charged += hlen
    n = max(len(ref), 1)
    return {
        "ref_chars": len(ref),
        "hyp_chars": len(hyp),
        "extra_ratio": len(hyp) / n,
        "coverage": matched / n,
        "cer": Levenshtein.distance(ref, hyp) / n,
        "coverage_cer": charged / n,
        "subs": subs,
        "dels": dels,
        "insertions_counted": counted_ins,
        "insertions_excused": excused_ins,
        "excused_runs": excused_runs,
    }


def score_pair(ref_text: str, hyp_text: str, min_cut_chars: int = 30,
               min_cut_words: int = 6, unit: str = "auto"):
    """Score a pair in the unit appropriate to its script.

    Returns (unit, stats). ``unit`` is "word" for space-delimited text and
    "char" for CJK, chosen from the reference unless overridden.
    """
    if unit == "auto":
        unit = "word" if is_space_delimited(ref_text) else "char"
    if unit == "word":
        return unit, coverage_score(tokenize(ref_text), tokenize(hyp_text),
                                    min_cut_words)
    return unit, coverage_score(normalize(ref_text), normalize(hyp_text),
                                min_cut_chars)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("hypothesis")
    p.add_argument("--min-cut", type=int, default=30,
                   help="character-unit insertion runs at least this long are "
                        "treated as omitted-from-reference audio, not errors")
    p.add_argument("--min-cut-words", type=int, default=6,
                   help="same threshold in the word-unit path")
    p.add_argument("--unit", default="auto", choices=["auto", "char", "word"])
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    ref_text = load_reference(a.reference)
    hyp_text = load_text(a.hypothesis)
    unit, s = score_pair(ref_text, hyp_text, a.min_cut, a.min_cut_words, a.unit)
    u = "chars" if unit == "char" else "words"
    m = "CER" if unit == "char" else "WER"

    print(f"unit={unit}  ref_{u}={s['ref_chars']} hyp_{u}={s['hyp_chars']} "
          f"extra_ratio={s['extra_ratio']:.2f}")
    print(f"coverage       = {s['coverage']:.4f}   "
          f"(reference {u} found in the hypothesis)")
    print(f"{m} (plain)    = {s['cer']:.4f}   "
          f"(unusable if extra_ratio is well above 1)")
    print(f"coverage{m}    = {s['coverage_cer']:.4f}   "
          f"sub={s['subs']} del={s['dels']} ins_counted={s['insertions_counted']}")
    print(f"excused as omitted-from-reference audio: "
          f"{s['insertions_excused']} {u} in {len(s['excused_runs'])} runs")
    if s["extra_ratio"] > 1.15:
        print(f"\nNOTE: the hypothesis is much longer than the reference, so this "
              f"reference omits audio on purpose. Use coverage{m}, not {m}.")
    if a.verbose and s["excused_runs"]:
        seq = tokenize(hyp_text) if unit == "word" else normalize(hyp_text)
        print(f"\nlongest excused runs (position, length, text):")
        for pos, ln in sorted(s["excused_runs"], key=lambda x: -x[1])[:10]:
            frag = seq[pos:pos + (12 if unit == "word" else 60)]
            print(f"  @{pos:>6} [{ln:>5}] "
                  f"{(' '.join(frag) if unit == 'word' else frag)!r}")


if __name__ == "__main__":
    main()
