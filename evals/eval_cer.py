"""CER evaluation of a hypothesis transcript/SRT against a reference SRT.

Japanese has no word boundaries, so CER on normalized text (NFKC, punctuation
and whitespace stripped) is the standard metric. Levenshtein via rapidfuzz.
"""

import argparse
import re
import unicodedata

from rapidfuzz.distance import Levenshtein

SRT_TIME = re.compile(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}")
PUNCT = re.compile(r"[\s、。，．,.!?！？…・「」『』（）()\[\]〈〉《》：:;；\-~〜?]+")


def srt_to_text(path: str) -> str:
    lines = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.isdigit() or SRT_TIME.match(line):
            continue
        lines.append(line)
    return "".join(lines)


def load_text(path: str) -> str:
    if path.endswith(".srt"):
        return srt_to_text(path)
    return open(path, encoding="utf-8").read()


def normalize(text: str) -> str:
    # NFKC + strip punctuation/whitespace. The prolonged sound mark is
    # phonemic in Japanese, so it stays.
    text = unicodedata.normalize("NFKC", text)
    return PUNCT.sub("", text)


def cer(ref: str, hyp: str) -> float:
    return Levenshtein.distance(ref, hyp) / max(len(ref), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("hypothesis")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    ref = normalize(load_text(a.reference))
    hyp = normalize(load_text(a.hypothesis))
    d = Levenshtein.distance(ref, hyp)
    print(f"ref_chars={len(ref)} hyp_chars={len(hyp)} edit_distance={d}")
    print(f"CER={d / max(len(ref), 1):.4f}")
    if a.verbose:
        ops = Levenshtein.editops(ref, hyp)
        subs = sum(1 for o in ops if o.tag == "replace")
        ins = sum(1 for o in ops if o.tag == "insert")
        dele = sum(1 for o in ops if o.tag == "delete")
        print(f"sub={subs} ins={ins} del={dele}")


if __name__ == "__main__":
    main()
