"""Every page in docs/benchmarks/ follows one template, checked rather than eyeballed.

Lever and engine pages (`# Lever: ...`, `# Engine: ...`):

    # Lever: <topic>
    <conclusion paragraphs>
    | setting | default | why |          one table, nothing else before the first ##
    **Setup:** <one paragraph>
    ## Experiment: <topic>                one or more, first
       **Basis:** <material, machine>     first line of every experiment
       <chart>                            optional; then its caption and table
       **Table:** <what it shows>         directly above every table
       <table, reading>                   no ### inside an experiment
    ## How it works                        optional, ### allowed
    ## Superseded                          optional, ### allowed (one per old result)
    ## Not settled                         optional
    ## Reproducing                         optional
    ## Related                             required, last

Reference pages (`# Reference: ...`): a lead paragraph before the first ##, free topic
sections, optional `## Reproducing`, `## Related` last. The index (README.md) is exempt.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted(p for p in (ROOT / "docs" / "benchmarks").rglob("*.md") if p.name != "README.md")
TAIL = ["How it works", "Superseded", "Not settled", "Reproducing", "Related"]


def _blocks(text):
    """Lines outside code fences, with their index."""
    out, fence = [], False
    for i, line in enumerate(text.splitlines()):
        if line.startswith("```"):
            fence = not fence
            continue
        if not fence:
            out.append((i, line))
    return out


def lint(path: Path) -> list[str]:
    lines = _blocks(path.read_text())
    errs = []
    title = lines[0][1] if lines else ""
    kind = re.match(r"# (Lever|Engine|Reference): \S", title)
    if not kind:
        return [f"title must start '# Lever: ', '# Engine: ' or '# Reference: ' ({title!r})"]
    kind = kind.group(1)

    h2 = [(i, l[3:].strip()) for i, l in lines if l.startswith("## ")]
    if not h2 or h2[-1][1] != "Related":
        errs.append("last ## section must be 'Related'")
    first_h2 = h2[0][0] if h2 else len(lines)
    head = [l for i, l in lines[1:] if i < first_h2]
    head_text = [l for l in head if l.strip()]
    if not head_text or head_text[0].startswith(("|", "**Setup", "<picture", "#")):
        errs.append("a conclusion/lead paragraph must come right after the title")

    # every table says what it is: a '**Table:** <what it shows>' line directly above it
    # (the setting/default/why table at the top is exempt); a chart is followed by that
    # caption and then its table
    for n, (i, l) in enumerate(lines):
        if l.strip() == "</picture>":
            nxt = next((m for _, m in lines[n + 1:] if m.strip()), "")
            if not nxt.startswith("**Table:**"):
                errs.append(f"line {i + 1}: a chart must be followed by its '**Table:**' caption and table")
        starts_table = (l.startswith("|") and n > 0 and not lines[n - 1][1].startswith("|"))
        if starts_table and not l.startswith("| setting | default | why |"):
            prev = next((m for _, m in reversed(lines[:n]) if m.strip()), "")
            if not prev.startswith("**Table:**"):
                errs.append(f"line {i + 1}: table without a '**Table:** <what it shows>' line above it")

    if kind == "Reference":
        if any(l.startswith("| setting | default |") for l in head):
            errs.append("reference pages have no defaults table")
        names = [n for _, n in h2]
        if "Reproducing" in names and names.index("Reproducing") != len(names) - 2:
            errs.append("'Reproducing' must be the section just before 'Related'")
        return errs

    # --- lever / engine pages ---
    tables = [l for l in head if l.startswith("| setting | default | why |")]
    if len(tables) != 1:
        errs.append("exactly one '| setting | default | why |' table before the first ##")
    # a table starts on a '|' line whose previous line is not one; only the defaults
    # table may start before the first ##
    starts = [l for n, l in enumerate(head)
              if l.startswith("|") and not (n and head[n - 1].startswith("|"))]
    if any(not l.startswith("| setting | default | why |") for l in starts):
        errs.append("no table other than the defaults table before the first ##")
    setup = [n for n, l in enumerate(head) if l.startswith("**Setup:**")]
    if len(setup) != 1:
        errs.append("exactly one '**Setup:**' paragraph before the first ##")
    if any(l.startswith(("<picture", "### ")) for l in head):
        errs.append("no charts or ### headings before the first ##")
    if tables and setup:
        t_idx = next(n for n, l in enumerate(head) if l.startswith("| setting | default | why |"))
        if not t_idx < setup[0]:
            errs.append("the defaults table comes before the Setup paragraph")
        # nothing but the Setup paragraph after the defaults table
        after = [l for l in head[t_idx:] if l.strip()]
        trailing = [l for l in after if not l.startswith("|")]
        if trailing and not trailing[0].startswith("**Setup:**"):
            errs.append("the Setup paragraph must directly follow the defaults table")
        setup_par_end = next((n for n in range(setup[0], len(head)) if not head[n].strip()),
                             len(head))
        if any(l.strip() for l in head[setup_par_end:]):
            errs.append("nothing may follow the Setup paragraph before the first ##")

    names = [n for _, n in h2]
    exps = [n for n in names if n.startswith("Experiment: ")]
    if not exps:
        errs.append("at least one '## Experiment: ...' section")
    allowed = set(TAIL)
    for n in names:
        if not n.startswith("Experiment: ") and n not in allowed:
            errs.append(f"'## {n}' is not a template section (Experiment: ..., {', '.join(TAIL)})")
    # order: experiments first, then the tail in template order
    order = [("Experiment" if n.startswith("Experiment: ") else n) for n in names]
    rank = {"Experiment": 0, **{t: k + 1 for k, t in enumerate(TAIL)}}
    ranks = [rank.get(o, 99) for o in order]
    if ranks != sorted(ranks):
        errs.append(f"sections out of template order: {order}")
    if len(set(o for o in order if o != "Experiment")) != len([o for o in order if o != "Experiment"]):
        errs.append("each tail section appears at most once")

    # per experiment: Basis first, no ###, plain topic title
    h2_idx = [i for i, _ in h2] + [10 ** 9]
    for k, (i, name) in enumerate(h2):
        body = [l for j, l in lines if i < j < h2_idx[k + 1]]
        if name.startswith("Experiment: "):
            topic = name[len("Experiment: "):]
            if re.search(r"\?|\bn=|\.$|\bon (one|a single) clip\b|\bon the corpus\b"
                         r"|\b\d+ files\b|\bsubset\b", topic):
                errs.append(f"'## {name}': title names the topic only (no '?', 'n=', "
                            f"trailing '.', or basis words; the basis goes in **Basis:**)")
            first = next((l for l in body if l.strip()), "")
            if not first.startswith("**Basis:**"):
                errs.append(f"'## {name}': first line must be '**Basis:** ...'")
            if any(l.startswith("### ") for l in body):
                errs.append(f"'## {name}': no ### inside an experiment; split it into experiments")
        elif name not in ("How it works", "Superseded"):
            if any(l.startswith("### ") for l in body):
                errs.append(f"'## {name}': ### only under 'How it works' and 'Superseded'")
    return errs


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_follows_the_template(page):
    errs = lint(page)
    assert not errs, f"{page.name}:\n  " + "\n  ".join(errs)
