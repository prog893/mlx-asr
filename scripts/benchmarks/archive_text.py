"""Save outputs that are not runner JSON (probe stdout, CLI --stats-json) as a run-group.

    .venv/bin/python scripts/benchmarks/archive_text.py --corpus DIR --queue QUEUE.zsh \\
        --group NAME --question TEXT --doc PAGE --out benchmarks/results/<yyyy-mm>/NAME.json \\
        OUT.txt|OUT.stats.json|OUT.log [...]

Probes print their tables and the CLI writes `--stats-json`; neither has the per-file rows
archive_results.py reads. Each output becomes one run: its `command` is the queue line that
produced it (`run NAME ...` or `arm NAME ...`, loop variables filled in from the name), and
its body is the probe's stdout (progress bars dropped), the stats JSON, or an `arm` line's
log (`$D/NAME.log`: the runner's console output, e.g. `/usr/bin/time -p` totals). A stats
file whose name ends in a corpus stem also gets a per-file row. `--alias STEM=NAME` names a
derived input that is not a corpus file by its corpus.json `derived` label. Everything goes through the same
Scrubber as archive_results.py, and nothing is written if a private name or path survives.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from archive_results import Scrubber

PROGRESS = re.compile(r"(Fetching \d+ files|\d+%\|)")
VAR = re.compile(r"\$\{?(\w+)\}?")


def queue_commands(path: Path):
    """[(name regex, variable names, command template)] for every run/arm line."""
    out = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        m = re.search(r"\b(?:run|arm) (\S+) (.+?)(?:; *done.*)?$", line)
        if not m or line.lstrip().startswith("#"):
            continue
        tmpl, cmd = m.groups()
        # A command that opens a double quote it does not close (`$PY -c "` followed by a
        # script) continues on the next lines until the quote closes.
        j = i
        while _open_quote(cmd) and j + 1 < len(lines):
            j += 1
            cmd += "\n" + lines[j]
        names = VAR.findall(tmpl)
        pattern = "(.+?)".join(re.escape(piece) for piece in VAR.split(tmpl)[::2])
        out.append((re.compile(f"^{pattern}$"), names, cmd.strip()))
    return out


def _open_quote(cmd: str) -> bool:
    """True if `cmd` leaves a double quote open (backslash-escaped quotes do not count)."""
    return len(re.findall(r'(?<!\\)"', re.sub(r"'[^']*'", "", cmd))) % 2 == 1


def command_for(name: str, commands) -> str:
    for rx, names, cmd in commands:
        m = rx.match(name)
        if m:
            for v, val in zip(names, m.groups()):
                cmd = re.sub(r"\$\{?" + v + r"\}?(?!\w)", val, cmd)
            return cmd
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", action="append", required=True)
    p.add_argument("--source-ids")
    p.add_argument("--queue", required=True, help="the queue script whose lines made these")
    p.add_argument("--group", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--doc", action="append", default=[])
    p.add_argument("--note", default="")
    p.add_argument("--alias", action="append", default=[], metavar="STEM=NAME",
                   help="a derived input outside the corpus, saved as NAME (listed under "
                        "corpus.json derived)")
    p.add_argument("--out", required=True)
    p.add_argument("outputs", nargs="+")
    a = p.parse_args()
    stems = {f.stem for c in a.corpus for f in Path(c).iterdir()
             if f.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".mp4")}
    scrub = Scrubber(stems, json.load(open(a.source_ids)) if a.source_ids else None,
                     dict(x.split("=", 1) for x in a.alias))
    commands = queue_commands(Path(a.queue))
    runs = []
    for out in map(Path, a.outputs):
        name = out.name.removesuffix(".stats.json").removesuffix(".txt").removesuffix(".log")
        cmd = command_for(name, commands)
        if not cmd:   # stats written into a per-run directory: <dir>/<x> came from run <dir>_<x>
            cmd = command_for(f"{out.parent.name}_{name}", commands)
            name = f"{out.parent.name}_{name}" if cmd else name
        kind = ("stats-json" if out.name.endswith(".stats.json")
                else "log" if out.name.endswith(".log") else "stdout")
        run = {"run": name, "output": kind,
               "command": cmd, "params": {}, "machine": {}, "aggregate": None,
               "measures": {}, "files": [],
               "missing": ["machine state at start (not recorded by this output)"]
               + ([] if cmd else ["command (no queue line matched)"])}
        if out.name.endswith(".stats.json"):
            stats = json.loads(out.read_text())
            run["machine"] = stats.pop("machine", {}) or {}
            run["measures"] = stats
            stem = next((s for s in sorted(stems, key=len, reverse=True)
                         if name.endswith(s)), None)
            if stem:
                run["files"] = [{"file": stem, **{k: stats[k] for k in (
                    "duration_s", "x_realtime", "wall_s", "peak_memory_gb", "cues")
                    if k in stats}}]
        else:
            body = "log" if kind == "log" else "stdout"
            run[body] = "\n".join(l for l in out.read_text(errors="replace").splitlines()
                                   if not PROGRESS.search(l))
        prev = next((r for r in runs if r["run"] == name), None)
        if prev:                    # stdout and --stats-json of one run: one saved run
            if kind not in prev["output"].split("+"):
                prev["output"] += "+" + kind
            for k in ("machine", "measures", "files", "stdout", "log"):
                if run.get(k) and not prev.get(k):
                    prev[k] = run[k]
            continue
        runs.append(run)
    group = scrub({"group": a.group, "question": a.question, "docs": a.doc, "note": a.note,
                   "runs": runs})
    text = json.dumps(group, indent=1, ensure_ascii=False)
    if scrub.leaks(text):
        print(f"REFUSED {a.group}: {scrub.leaks(text)} private strings survived", file=sys.stderr)
        return 1
    Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(f"{a.group}: {len(runs)} runs -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
