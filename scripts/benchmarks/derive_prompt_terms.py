"""Derive the topic and terms prompts for sweep_prompt_language.py from the references.

    .venv/bin/python scripts/benchmarks/derive_prompt_terms.py --corpus DIR \\
        --variant freq --out bench_out/prompt_terms_freq.json

The derivation the sweep's docstring recommends, made mechanical so nothing is hand-picked:
a token counts only if it appears in at least two reference files, Japanese references
give the Japanese texts and English references the English ones.

- `freq`: every candidate token (Japanese: katakana runs, Latin tokens, kanji compounds;
  English: content words), ranked by total frequency.
- `names`: proper-name-like tokens only (Japanese: katakana runs and Latin tokens;
  English: capitalized words not at a sentence start), ranked by how many files use them.

The terms list puts the highest-ranked token LAST, because only the final tokens survive
when a prompt is truncated to the decoder's prompt capacity. The topic is the top three
tokens, substituted into the sweep's own topic template.

The output names corpus content, so it is written to a local file (bench_out/ is
gitignored) and must never be committed; the sweep's result JSON repeats the texts, and
archive_results.py drops them.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.eval_coverage import load_reference  # noqa: E402

AUDIO = {".wav", ".m4a", ".mp3", ".flac", ".mp4", ".mov"}
KATAKANA = re.compile(r"[ァ-ヺー]{3,}")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9&.+-]{1,}")
KANJI = re.compile(r"[一-龯々]{2,}")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
STOP = set("""the and that this with for are was were you your have has had not but
what when where which who will would can could should there their they them then than
about from into just like know think really yeah okay going get got one all out some
very much more also been being because here how its it's i'm don't that's we're you're
our us his her she him he me my mine so if or an as at by on in of to do did does""".split())


def references(corpus: Path):
    for audio in sorted(corpus.iterdir()):
        if audio.suffix.lower() not in AUDIO:
            continue
        for cand in (f"{audio.stem}_transcript.txt", f"{audio.stem}.srt", f"{audio.stem}.txt"):
            if (corpus / cand).exists():
                yield load_reference(str(corpus / cand))
                break


def tokens(text: str, japanese: bool, variant: str) -> list[str]:
    if japanese:
        found = KATAKANA.findall(text) + LATIN.findall(text)
        if variant == "freq":
            found += KANJI.findall(text)
        return found
    if variant == "freq":
        return [w for w in WORD.findall(text) if w.lower() not in STOP]
    # a capitalized word not at a sentence start
    return [m.group(1) for m in re.finditer(r"(?<![.!?\n]\s)(?<!^)\b([A-Z][a-zA-Z]{2,})", text)
            if m.group(1).lower() not in STOP]


def ranked(texts: list[str], japanese: bool, variant: str, n: int) -> list[str]:
    total, files = Counter(), Counter()
    for t in texts:
        toks = tokens(t, japanese, variant)
        total.update(toks)
        files.update(set(toks))
    keep = [w for w in total if files[w] >= 2]
    key = (lambda w: (total[w], files[w])) if variant == "freq" else (lambda w: (files[w], total[w]))
    return sorted(keep, key=key, reverse=True)[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--variant", choices=["freq", "names"], required=True)
    p.add_argument("--n", type=int, default=30, help="terms per language")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    refs = list(references(Path(a.corpus)))
    ja = [t for t in refs if re.search(r"[ぁ-んァ-ヺ一-龯]", t)]
    en = [t for t in refs if t not in ja]
    out = {}
    for lang, texts, sep in (("ja", ja, "、"), ("en", en, ", ")):
        top = ranked(texts, lang == "ja", a.variant, a.n)
        out[f"terms_{lang}"] = sep.join(reversed(top))       # highest-ranked last
        out[f"topic_{lang}"] = sep.join(top[:3])
        out[f"n_{lang}"] = {"files": len(texts), "terms": len(top)}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{a.variant}: ja {out['n_ja']}, en {out['n_en']} -> {a.out}")


if __name__ == "__main__":
    main()
