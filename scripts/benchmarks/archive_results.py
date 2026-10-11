"""Archive runs as one JSON per experiment run-group, with corpus files hashed.

    uv run python scripts/benchmarks/archive_results.py --corpus DIR \\
        --group NAME --question TEXT --doc PAGE [--doc PAGE ...] \\
        --out benchmarks/results/2026-10/NAME.json RUN.json [RUN.json ...]

The corpus is private: its filenames and content may not be published. A file is named
here only by `sha256(source id)[:12]`, followed by any derived suffix (`.wav`,
`.16k.wav`); the source id is the filename stem for a recording and the video id for a
downloaded public video.
This hides names from casual reading, not from someone who guesses candidates; if the
stem-to-hash table is lost, `corpus.json`'s metadata is enough to match files again.

`benchmarks/results/corpus.json` maps each hash to the metadata that may be published:
language, duration, codec, sample rate, bit depth and bitrate. A run-group JSON holds one
entry per run: date, hardware and machine state, input parameters, the command rebuilt
from the recorded config, the aggregates, and every per-file measure, so the published
metrics can be recomputed from raw data. Anything a run did not record is listed under
`missing`, which makes it a rerun candidate. Transcripts are never archived.

The output is checked before it is written: if any original stem or absolute path
survives, nothing is written.
"""

import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

# Any absolute path with at least one directory, wherever it sits in a string. A "/" that
# follows a word character, ".", ":", "/", "~" or "-" is inside a relative path, a URL or a
# model id, not the start of an absolute path.
ABS_PATH = re.compile(r"(?<![\w.:/~-])/(?:[^\s\"'/]+/)+[^\s\"']*")
AUDIO_SUFFIX = re.compile(r"(\.16k)?\.(wav|flac|mp3|m4a|mp4)$")
PER_FILE_DROP = {"ref_chars", "kana_ref_chars", "lenient_ref_chars"}   # see --drop-ref-lengths
QWEN3_REPO = re.compile(r"Qwen3-ASR-(?P<size>[\d.]+B)-(?P<quant>\w+)$")
RUN_DROP = {"json", "keep_hyp"}            # local output paths, not inputs
# Directories whose names may say something about the machine or its owner. Used to cut a
# whole parameter value (which may contain spaces) and as a last-resort leak check.
LOCAL_ROOT = re.compile(r"/(?:Users|private|tmp|home|var/folders|Volumes|mnt)/")


LOCAL = "<local>"   # documented placeholder for a removed machine-local parent directory


def last_component(value):
    """A parameter value that is an absolute path becomes `<local>/<last component>`: the
    private parent directories go, the placeholder says the path must be resolved against
    the reader's own copy (see the archive README). The whole value is one path, so a
    directory name with spaces is cut with it. Relative paths replay from the repo root and
    are kept."""
    # any absolute path, root-level ones such as "/data" included; a bare "/" names nothing
    if isinstance(value, str) and value.startswith("/") and value.strip("/"):
        return f"{LOCAL}/" + value.rstrip("/").rsplit("/", 1)[-1]
    return value


def token_pattern(name: str) -> re.Pattern:
    """`name` as a whole token: not preceded or followed by a letter or digit."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])")


def file_id(stem: str) -> str:
    return hashlib.sha256(stem.encode()).hexdigest()[:12]


class Scrubber:
    def __init__(self, stems, source_ids=None):
        # The id hashes the file's source identifier: its stem for a recording, the bare
        # video id for a downloaded public video (source_ids maps the exceptions).
        src = source_ids or {}
        # longest first so one stem that prefixes another cannot be half-replaced
        self.map = {s: file_id(src.get(s, s)) for s in sorted(stems, key=len, reverse=True)}
        # two files sharing an id (same source id, or a truncated-hash collision) would make
        # archived rows ambiguous, so refuse rather than merge them
        if len(set(self.map.values())) != len(self.map):
            sys.exit("REFUSED: two corpus files map to the same id")
        # a stem counts only as a whole token (not inside a longer word), so a short stem
        # cannot rewrite schema keys or ordinary text
        self.pattern = {s: token_pattern(s) for s in self.map}

    def __call__(self, obj):
        if isinstance(obj, dict):
            return {self(k): self(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self(v) for v in obj]
        if isinstance(obj, str):
            s = obj
            # any absolute path, also inside a command line, keeps only its last part
            s = ABS_PATH.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1], s)
            for stem, h in self.map.items():
                s = self.pattern[stem].sub(h, s)
            return s
        return obj

    def leaks(self, text: str) -> int:
        spans = [m.span() for m in ABS_PATH.finditer(text)]
        # a local root the path pattern missed (e.g. a directory name with spaces) counts too
        extra = [m for m in LOCAL_ROOT.finditer(text)
                 if not any(a <= m.start() < b for a, b in spans)]
        return (sum(bool(p.search(text)) for p in self.pattern.values())
                + len(spans) + len(extra))


def runner_and_rows(d: dict):
    """(script, config, [(arm, arm_params, aggregate, per_file_rows, machine, extra, gaps)]).

    `arm` names a sweep arm ("" for a plain run); `arm_params` are the runner arguments that
    select it, so the rebuilt command reruns exactly that arm; `gaps` are things the runner
    did not record, which make the run a rerun candidate.
    """
    if isinstance(d.get("results"), list):
        script = {"mlx-whisper": "run_whisper.py", "mlx-qwen3": "run_qwen3.py"}.get(
            d.get("engine"), "run_corpus.py")
        extra = {k: d[k] for k in ("x_realtime", "peak_memory_gb", "files_scored",
                                   "files_expected", "complete", "truncated_files",
                                   "looped_files", "languages") if k in d}
        return script, d.get("config", {}), [("", {}, d.get("aggregate"), d["results"],
                                             d.get("machine"), extra, [])]
    if isinstance(d.get("results"), dict):                 # sweep_gain.py
        return "sweep_gain.py", d.get("config", {}), [
            (f"mode {k}", {"modes": k}, v.get("aggregate"), v.get("per_file", []), None, {}, [])
            for k, v in d["results"].items()]
    if "arms" in d:                                         # sweep_qwen3_batch.py
        # The sweep records only model, window and language. Corpus and the per-group token
        # budget come from the queue that ran it (bench_out/rerun_queue.md section 8).
        m = QWEN3_REPO.search(d.get("model", ""))
        cfg = {"corpus": "bench_out/corpus_all", "chunk_seconds": d.get("chunk_seconds"),
               "size": m["size"] if m else None, "quantization": m["quant"] if m else None,
               "language": d.get("language"), "group_budget": True}
        identity = {k: v for k, v in (d.get("machine") or {}).items()}
        return "sweep_qwen3_batch.py", cfg, [
            (f"batch {k}", {"batches": k}, v.get("aggregate"), v.get("per_file", []),
             {**identity, **(v.get("machine_at_arm_start") or {})},
             {kk: v[kk] for kk in ("x_realtime", "peak_gb", "wall_s") if kk in v},
             ["corpus and group budget not recorded by the runner (taken from the queue)"])
            for k, v in d["arms"].items()]
    raise ValueError("unrecognised run JSON")


def command(script: str, cfg: dict, out: str = "") -> str:
    """The runner invocation, ending with a relative `--json` destination so it can be
    replayed (sweep_qwen3_batch.py requires one; the others would otherwise save nothing)."""
    parts = [f"scripts/benchmarks/{script}"]
    for k, v in cfg.items():
        if k in RUN_DROP or v is None or v is False or v == "":
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            parts.append(flag)
            continue
        v = shlex.quote(str(v))
        # a value that starts with "-" (a negative gain) must be attached to its flag
        parts.append(f"{flag}={v}" if v.startswith("-") else f"{flag} {v}")
    if out:
        parts.append(f"--json {shlex.quote(out)}")
    return " ".join(parts)


NEEDED = {"machine state": lambda r: bool(r["machine"]),
          "x_realtime": lambda r: r["measures"].get("x_realtime") is not None,
          "peak memory": lambda r: any(r["measures"].get(k) is not None
                                       for k in ("peak_memory_gb", "peak_gb")),
          "per-file rows": lambda r: bool(r["files"]),
          "per-file duration": lambda r: bool(r["files"]) and all(
              f.get("duration_s") is not None for f in r["files"]),
          "per-file timing": lambda r: bool(r["files"]) and all(
              f.get("x_realtime") is not None or f.get("wall_s") is not None
              for f in r["files"])}


def mark_invalid(run: dict, revisions: dict):
    """Flag rows that used a superseded audio copy (see benchmarks/results/revisions.json).

    A row is matched to a revision by its recorded duration, so no run date is needed; a
    row of a revised file that records no duration is flagged too, being unverifiable.
    """
    bad = []
    for row in run["files"]:
        fid = row.get("file", "")
        for rev in revisions.get(fid, []):
            dur = row.get("duration_s")
            if dur is None or abs(dur - rev["previous"]["duration_s"]) < 1.0:
                row["invalid"] = f"superseded audio revision ({rev['superseded']}): {rev['reason']}"
                bad.append(fid)
    if bad:
        run["invalid_files"] = sorted(set(bad))
        run["note"] = ("aggregate and measures include the invalid files; recompute from the "
                       "rows without an 'invalid' key")


def build_run(path: Path, keep_ref: bool):
    d = json.loads(path.read_text())
    script, cfg, subs = runner_and_rows(d)
    out = []
    for arm, arm_params, agg, rows, machine, extra, gaps in subs:
        files = []
        for r in rows:
            r = {k: v for k, v in r.items() if keep_ref or k not in PER_FILE_DROP}
            if isinstance(r.get("file"), str):        # bare file id, no audio suffix
                r["file"] = AUDIO_SUFFIX.sub("", r["file"])
            files.append(r)
        params = {k: last_component(v) for k, v in
                  {**{k: v for k, v in cfg.items() if k not in RUN_DROP}, **arm_params}.items()}
        run = {"run": path.stem, **({"arm": arm} if arm else {}),
               "command": command(script, params, f"{path.stem}.json"),
               "params": params,
               "machine": machine or {},
               "aggregate": agg,
               "measures": extra,
               "files": files}
        run["missing"] = [name for name, ok in NEEDED.items() if not ok(run)] + gaps
        if run["machine"].get("busy"):
            # recorded, not hidden: this run's timing shared the GPU with other work
            run["caveats"] = ["other GPU work at start: " +
                              "; ".join(run["machine"].get("busy_reasons") or ["busy"])]
        out.append(run)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", action="append", required=True,
                   help="corpus directory; every audio stem in it is a private name")
    p.add_argument("--group", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--doc", action="append", default=[])
    p.add_argument("--note", default="")
    p.add_argument("--source-ids", help="local JSON {stem: source identifier} for files whose "
                   "id hashes something other than the stem (a video id)")
    p.add_argument("--revisions", help="committed revisions.json; flags rows that used "
                   "a superseded audio copy")
    p.add_argument("--out", required=True)
    p.add_argument("--drop-ref-lengths", action="store_true",
                   help="drop per-file reference lengths (then weighted metrics cannot be "
                        "recomputed from the rows)")
    p.add_argument("--keep-ref-lengths", action="store_true",
                   help="no-op, the default; kept so older invocations still work")
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    stems = {f.stem for c in a.corpus for f in Path(c).iterdir()
             if f.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".mp4")}
    scrub = Scrubber(stems, json.load(open(a.source_ids)) if a.source_ids else None)
    group = {"group": a.group, "question": a.question, "docs": a.doc, "note": a.note,
             "runs": [r for run in a.runs for r in build_run(Path(run), not a.drop_ref_lengths)]}
    group = scrub(group)
    if a.revisions:
        revisions = json.load(open(a.revisions))
        for run in group["runs"]:
            mark_invalid(run, revisions)
    text = json.dumps(group, indent=1, ensure_ascii=False)
    if scrub.leaks(text):
        print(f"REFUSED {a.group}: {scrub.leaks(text)} private strings survived", file=sys.stderr)
        return 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(f"{a.group}: {len(group['runs'])} runs -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
