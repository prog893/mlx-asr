"""Phonetic CER: convert both texts to hiragana readings with pykakasi first.

Separates true misrecognition from orthographic variance (kanji vs katakana,
digits vs spelled-out numbers pronounce identically but differ as characters).
"""

import argparse

import pykakasi

from evals.eval_cer import load_text, normalize
from rapidfuzz.distance import Levenshtein

_kks = pykakasi.kakasi()


def to_kana(text: str) -> str:
    return "".join(item["hira"] for item in _kks.convert(text))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("hypothesis")
    a = p.parse_args()

    ref = to_kana(normalize(load_text(a.reference)))
    hyp = to_kana(normalize(load_text(a.hypothesis)))
    d = Levenshtein.distance(ref, hyp)
    print(f"kana_ref={len(ref)} kana_hyp={len(hyp)} edit_distance={d}")
    print(f"kanaCER={d / max(len(ref), 1):.4f}")


if __name__ == "__main__":
    main()
