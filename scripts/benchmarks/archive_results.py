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
import sys
from pathlib import Path

ABS_PATH = re.compile(r"(?<![\w.])/(?:Users|private|tmp|home|var/folders)/[^\s\"']*")
PER_FILE_DROP = {"ref_chars", "kana_ref_chars", "lenient_ref_chars"}   # see --keep-ref-lengths
RUN_DROP = {"json", "keep_hyp"}            # local output paths, not inputs


def file_id(stem: str) -> str:
    return hashlib.sha256(stem.encode()).hexdigest()[:12]


class Scrubber:
    def __init__(self, stems, source_ids=None):
        # The id hashes the file's source identifier: its stem for a recording, the bare
        # video id for a downloaded public video (source_ids maps the exceptions).
        src = source_ids or {}
        # longest first so one stem that prefixes another cannot be half-replaced
        self.map = {s: file_id(src.get(s, s)) for s in sorted(stems, key=len, reverse=True)}

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
                s = s.replace(stem, h)
            return s
        return obj

    def leaks(self, text: str) -> int:
        return sum(stem in text for stem in self.map) + len(ABS_PATH.findall(text))


def runner_and_rows(d: dict):
    """(script, config, [(sub_label, aggregate, per_file_rows, machine, extra)])."""
    if isinstance(d.get("results"), list):
        script = {"mlx-whisper": "run_whisper.py", "mlx-qwen3": "run_qwen3.py"}.get(
            d.get("engine"), "run_corpus.py")
        extra = {k: d[k] for k in ("x_realtime", "peak_memory_gb", "files_scored",
                                   "files_expected", "complete", "truncated_files",
                                   "looped_files", "languages") if k in d}
        return script, d.get("config", {}), [("", d.get("aggregate"), d["results"],
                                             d.get("machine"), extra)]
    if isinstance(d.get("results"), dict):                 # sweep_gain.py
        return "sweep_gain.py", d.get("config", {}), [
            (k, v.get("aggregate"), v.get("per_file", []), None, {})
            for k, v in d["results"].items()]
    if "arms" in d:                                         # sweep_qwen3_batch.py
        cfg = {k: d[k] for k in ("model", "chunk_seconds", "language") if k in d}
        return "sweep_qwen3_batch.py", cfg, [
            (f"batch {k}", v.get("aggregate"), v.get("per_file", []),
             v.get("machine_at_arm_start"),
             {kk: v[kk] for kk in ("x_realtime", "peak_gb", "wall_s") if kk in v})
            for k, v in d["arms"].items()]
    raise ValueError("unrecognised run JSON")


def command(script: str, cfg: dict) -> str:
    parts = [f"scripts/benchmarks/{script}"]
    for k, v in cfg.items():
        if k in RUN_DROP or v is None or v is False or v == "":
            continue
        flag = "--" + k.replace("_", "-")
        parts.append(flag if v is True else f"{flag} {v}")
    return " ".join(parts)


NEEDED = {"machine state": lambda r: bool(r["machine"]),
          "x_realtime": lambda r: r["measures"].get("x_realtime") is not None,
          "peak memory": lambda r: any(k in r["measures"] for k in ("peak_memory_gb", "peak_gb")),
          "per-file duration": lambda r: all("duration_s" in f for f in r["files"]),
          "per-file timing": lambda r: all("x_realtime" in f or "wall_s" in f for f in r["files"])}


def build_run(path: Path, keep_ref: bool):
    d = json.loads(path.read_text())
    script, cfg, subs = runner_and_rows(d)
    out = []
    for sub, agg, rows, machine, extra in subs:
        files = [{k: v for k, v in r.items() if keep_ref or k not in PER_FILE_DROP}
                 for r in rows]
        run = {"run": path.stem + (f" [{sub}]" if sub else ""),
               "command": command(script, cfg) + (f"  ({sub})" if sub else ""),
               "params": {k: v for k, v in cfg.items() if k not in RUN_DROP},
               "machine": machine or {},
               "aggregate": agg,
               "measures": extra,
               "files": files}
        run["missing"] = [name for name, ok in NEEDED.items() if not ok(run)]
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
    p.add_argument("--out", required=True)
    p.add_argument("--keep-ref-lengths", action="store_true",
                   help="keep per-file reference lengths (needed to recompute weighted metrics)")
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    stems = {f.stem for c in a.corpus for f in Path(c).iterdir()
             if f.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".mp4")}
    scrub = Scrubber(stems, json.load(open(a.source_ids)) if a.source_ids else None)
    group = {"group": a.group, "question": a.question, "docs": a.doc, "note": a.note,
             "runs": [r for run in a.runs for r in build_run(Path(run), a.keep_ref_lengths)]}
    text = json.dumps(scrub(group), indent=1, ensure_ascii=False)
    if scrub.leaks(text):
        print(f"REFUSED {a.group}: {scrub.leaks(text)} private strings survived", file=sys.stderr)
        return 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(f"{a.group}: {len(group['runs'])} runs -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
