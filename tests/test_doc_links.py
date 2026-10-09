"""Every relative markdown link in the docs must resolve, including its #anchor.

The benchmark pages link to each other's sections (corpus.md#the-20-file-corpus and so
on), and a heading rename breaks those links without any error. Anchors are computed the
way GitHub does: lowercase, punctuation other than hyphens and spaces dropped, spaces to
hyphens, and a numeric suffix for repeated headings.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted([*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md")])
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _slug(heading: str) -> str:
    text = re.sub(r"`|\*\*|\*", "", heading.strip()).lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set:
    seen, out, in_code = {}, set(), False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        m = None if in_code else re.match(r"#{1,6} (.+)", line)
        if m:
            base = _slug(m.group(1))
            n = seen.get(base, 0)
            out.add(base if n == 0 else f"{base}-{n}")
            seen[base] = n + 1
    return out


def _links():
    for doc in DOCS:
        in_code = False
        for line in doc.read_text().splitlines():
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            for target in LINK.findall(line):
                if re.match(r"[a-z]+://|mailto:", target):
                    continue
                yield doc, target


@pytest.mark.parametrize("doc,target", list(_links()),
                         ids=lambda v: v if isinstance(v, str) else v.name)
def test_link_resolves(doc, target):
    path, _, anchor = target.partition("#")
    dest = (doc.parent / path).resolve() if path else doc
    assert dest.exists(), f"{doc.relative_to(ROOT)}: {target} -> missing file"
    if anchor and dest.suffix == ".md":
        assert anchor in anchors(dest), (
            f"{doc.relative_to(ROOT)}: {target} -> no heading with anchor #{anchor} "
            f"in {dest.relative_to(ROOT)}")
