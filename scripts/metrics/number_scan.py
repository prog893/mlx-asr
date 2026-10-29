"""Find measured numbers and source markers on the published surface.

Shared by tests/test_sources.py (every number must cite saved run data) and the local
number audit, so both agree on what counts as a measured number and what a marker is.

A measured number is a percentage, a realtime factor (`33.6x`), a size in GB, a duration or
time (`ms`, `s`, `min`, `h`), a point difference, or a confidence interval. Configuration
values (chunk lengths, delays, machine RAM) are not measurements and are skipped.

Markers (the convention in AGENTS.md):

- markdown table: a `Source:` line directly under the table;
- markdown prose: a footnote reference `[^id]` whose definition `[^id]: ...` is a marker;
- marker body: one or more links to a run-group JSON, `[name.json](relative/path.json)`,
  each optionally followed by its runs in backticks;
- chart spec (scripts/docs): a `"source"` field, `group.json#run,run;group2.json#run`;
- code comment or docstring (mlx_asr/): `source: group.json#run[,run]` on the line or within
  the comment block around it; profiles.json: a `"source"` field per profile.
"""

import re
from pathlib import Path

NUM = re.compile(r"(?<![\w.#/-])(\*\*)?(~?[+-]?\d+(?:\.\d+)?)\s?"
                 r"(%|x\b|GB\b|GiB\b|ms\b|s\b|min\b|h\b| points?\b| pts\b|kb/s)")
CI = re.compile(r"\[\s*[+-]?\d+(?:\.\d+)?%?,\s*[+-]?\d+(?:\.\d+)?%?\s*\]")
SKIP_LINE = re.compile(r"^\s*(```|<img|<source|!\[)")
# Spans that hold digits but no measurement: a URL, a version, a date. They are cut out of
# the line before scanning, rather than skipping the whole line, so a table row that also
# carries a link is still scanned.
NOT_NUMBERS = re.compile(r"https?://\S+|mlx[ -]?0\.\d+(\.\d+)?|macOS \d+(\.\d+)*|"
                         r"v\d+\.\d+(\.\d+)?|\d{4}-\d{2}-\d{2}")
LINK = re.compile(r"\[([^\]]+?\.json)\]\(([^)\s]+?\.json)\)")
TICK = re.compile(r"`([^`]+)`")
FOOTREF = re.compile(r"\[\^([\w-]+)\](?!:)")
FOOTDEF = re.compile(r"^\[\^([\w-]+)\]:\s*(.*)$")
SOURCE_LINE = re.compile(r"^\s*(\*\*)?Source:(\*\*)?\s")
SPEC_SOURCE = re.compile(r"([\w.-]+\.json)(?:#([^;]+))?")
CODE_SOURCE = re.compile(r"source:\s*([\w.-]+\.json)(?:#([\w.,#-]+))?")
# A number that is not a measurement (a design constant, an arithmetic example, a format
# string) is exempted explicitly, with its reason, so every exemption is reviewable:
# `<!-- not-measured: reason -->` in markdown, `# not-measured: reason` in code.
NOT_MEASURED = re.compile(r"not-measured:\s*\S")


def is_measure(tok: str, unit: str, line: str) -> bool:
    unit = unit.strip()
    if unit in ("GB", "GiB") and re.search(r"\b(16|18|24|32|36|48|64|96|128|192|256|512) ?GB\b",
                                          tok + unit) and \
            re.search(r"(Mac|RAM|nameplate|16GB Mac|128GB|M4 16GB|Ultra 128GB)", line):
        return False
    if unit in ("s", "min", "h") and re.search(r"chunk|window|--chunk|delay|overlap|timeout",
                                              line) and \
            not re.search(r"audio|wall|decode in|took|realtime", line):
        return False
    return True


def measured(line: str) -> list[str]:
    """The measured numbers printed on one line (empty for skipped lines)."""
    if SKIP_LINE.search(line):
        return []
    line = NOT_NUMBERS.sub(" ", line)
    out = [m.group(2) + m.group(3).strip() for m in NUM.finditer(line)
           if is_measure(m.group(2), m.group(3), line)]
    return out + [m.group(0) for m in CI.finditer(line)]


def marker_links(body: str):
    """[(json path text, [runs])] in a marker body, runs being the backticked tokens that
    follow each link up to the next link."""
    out, spans = [], list(LINK.finditer(body))
    for i, m in enumerate(spans):
        end = spans[i + 1].start() if i + 1 < len(spans) else len(body)
        out.append((m.group(2), TICK.findall(body[m.end():end])))
    return out


def spec_links(source: str):
    """[(group file name, [runs])] in a chart spec's "source" field."""
    out = []
    for part in source.split(";"):
        m = SPEC_SOURCE.search(part.strip())
        if m:
            out.append((m.group(1), [r for r in (m.group(2) or "").split(",") if r]))
    return out


def md_blocks(text: str):
    """Yield ("table", [(lineno, line)], next_nonblank_line) and ("prose", lineno, line),
    skipping fenced code."""
    lines = text.split("\n")
    i, fence = 0, ""             # the open fence's marker run, e.g. "```" or "~~~~"
    while i < len(lines):
        ln = lines[i]
        m = re.match(r" {0,3}(`{3,}|~{3,})(.*)$", ln)
        if m and not fence:
            fence = m.group(1)
            i += 1
            continue
        if fence:
            # CommonMark: closed by the same character, at least as long, nothing after it
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) \
                    and not m.group(2).strip():
                fence = ""
            i += 1
            continue
        if ln.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append((i + 1, lines[i]))
                i += 1
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            yield "table", block, (lines[j] if j < len(lines) else "")
            continue
        yield "prose", i + 1, ln
        i += 1


def group_runs(path: Path):
    """The set of run names (and run#arm) in a run-group JSON."""
    import json
    g = json.loads(path.read_text(encoding="utf-8"))
    names = set()
    for r in g.get("runs", []):
        names.add(r["run"])
        if r.get("arm"):
            names.add(f"{r['run']}#{r['arm']}")
    return names
