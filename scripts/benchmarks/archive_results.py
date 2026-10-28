"""Save runs as one JSON per experiment run-group, with corpus files hashed.

    uv run python scripts/benchmarks/archive_results.py --corpus DIR \\
        --group NAME --question TEXT --doc PAGE [--doc PAGE ...] \\
        --out benchmarks/results/2026-10/NAME.json RUN.json [RUN.json ...]

The corpus is private: its filenames and content may not be published. A file is named
here only by `sha256(source id)[:12]`, followed by any derived suffix (`.wav`,
`.16k.wav`); the source id is a private identifier kept with the local corpus, usually
the filename stem.
This hides names from casual reading, not from someone who guesses candidates; if the
stem-to-hash table is lost, `corpus.json`'s metadata is enough to match files again.

`benchmarks/results/corpus.json` maps each hash to the metadata that may be published:
language, duration, codec, sample rate, bit depth and bitrate. A run-group JSON holds one
entry per run: date, hardware and machine state, input parameters, the command rebuilt
from the recorded config, the aggregates, and every per-file measure, so the published
metrics can be recomputed from raw data. Anything a run did not record is listed under
`missing`, which makes it a rerun candidate. Transcripts are never saved.

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
RUN_DROP = {"json", "keep_hyp"}
# sweep_prompt_language.py: only these content shapes have fixed, publishable texts
FIXED_PROMPTS = {"none", "instruction", "description"}
PROMPT_TEXT_KEYS = {"topic_ja", "topic_en", "terms_ja", "terms_en"}
PROMPT_PLACEHOLDER = "<private: scripts/benchmarks/derive_prompt_terms.py, variant in the label>"            # local output paths, not inputs
# Directories whose names may say something about the machine or its owner. Used to cut a
# whole parameter value (which may contain spaces) and as a last-resort leak check.
LOCAL_ROOT = re.compile(r"/(?:Users|private|tmp|home|var/folders|Volumes|mnt)/")


LOCAL = "<local>"   # documented placeholder for a removed machine-local parent directory


def last_component(value):
    """A parameter value that is an absolute path becomes `<local>/<last component>`: the
    private parent directories go, the placeholder says the path must be resolved against
    the reader's own copy (see the monthly README in benchmarks/results/). The whole value is one path, so a
    directory name with spaces is cut with it. Relative paths replay from the repo root and
    are kept."""
    # any absolute path, root-level ones such as "/data" included; a bare "/" names nothing
    if isinstance(value, list):                       # --pair A R, --files a b
        return [last_component(v) for v in value]
    if isinstance(value, str) and value.startswith("/") and value.strip("/"):
        return f"{LOCAL}/" + value.rstrip("/").rsplit("/", 1)[-1]
    return value


def token_pattern(name: str) -> re.Pattern:
    """`name` as a whole token: not preceded or followed by a letter or digit."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])")


def file_id(stem: str) -> str:
    return hashlib.sha256(stem.encode()).hexdigest()[:12]


class Scrubber:
    def __init__(self, stems, source_ids=None, aliases=None):
        # The id hashes the file's source identifier: usually its stem (source_ids maps
        # the exceptions).
        src = source_ids or {}
        # longest first so one stem that prefixes another cannot be half-replaced
        # aliases name a derived input that is not a corpus file (an excerpt cut from one) by
        # the generic label corpus.json lists it under
        named = {s: file_id(src.get(s, s)) for s in stems} | dict(aliases or {})
        self.map = {s: named[s] for s in sorted(named, key=len, reverse=True)}
        # two files sharing an id (same source id, or a truncated-hash collision) would make
        # saved rows ambiguous, so refuse rather than merge them
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
    if "overlaps" in d.get("config", {}) and "duration_s" in d:   # sweep_overlap.py (clip)
        cfg = {k: v for k, v in d["config"].items() if k != "overlaps"}
        stem = Path(cfg.get("audio", "")).stem
        return "sweep_overlap.py", cfg, [
            (f"overlap {r['overlap_s']:g}", {"overlaps": f"{r['overlap_s']:g}"},
             {"char": r["cer"]},
             [{"file": stem, "duration_s": round(d["duration_s"], 1), "unit": "char",
               **{k: r[k] for k in ("cer", "x_realtime", "wall_s", "steps", "extra_audio_s",
                                    "cues") if k in r}}],
             None, {}, ["plain CER on one clip; sweep_overlap.py records no machine state"])
            for r in d["results"]]
    if "variants" in d.get("config", {}) and "chunks" in d:     # ab_seam_context.py (clip)
        cfg = {k: v for k, v in d["config"].items() if k != "variants"}
        stem = Path(cfg.get("audio", "")).stem
        return "ab_seam_context.py", cfg, [
            (f"variant {r['variant']}", {"variants": r["variant"]}, {"char": r["cer"]},
             [{"file": stem, "unit": "char", "chunks": d["chunks"],
               **{k: r[k] for k in ("cer", "x_realtime", "wall_s", "note") if k in r}}],
             None, {}, ["plain CER on one clip; ab_seam_context.py records no machine state "
                        "or duration"])
            for r in d["results"]]
    if "min_anchor" in d and "cue_overrides" in d:              # run_timing_sweep.py
        cfg = {"corpus": d.get("corpus"), "min_anchor": d["min_anchor"],
               "break_tol": d["break_tol"],
               **{k: v for k, v in (d.get("cue_overrides") or {}).items()}}
        engines = sorted({r["engine"] for r in d["results"]})
        return "run_timing_sweep.py", cfg, [
            (f"engine {e}", {}, None,
             [{k: v for k, v in r.items() if k != "engine"} for r in d["results"]
              if r["engine"] == e],
             None, {}, ["run_timing_sweep.py records no machine state; whisper-python and "
                        "srt-dir came from the queue"])
            for e in engines]
    if "shipped_defaults" in d and "break_tol" in d:           # sweep_cues.py: grid rows
        return "sweep_cues.py", {"corpus": d.get("corpus"), "break_tol": d["break_tol"],
                                 "files": d.get("files")}, [
            ("", {}, None, [], None, {"rows": d["results"],
                                      "shipped_defaults": d["shipped_defaults"]},
             ["sweep_cues.py records no machine state"])]
    if set(d) == {"reference", "results"}:                     # compare_text_vs_srt.py
        return "compare_text_vs_srt.py", {"reference": d["reference"]}, [
            ("", {}, None, [], None, {"rows": d["results"]},
             ["compare_text_vs_srt.py scores CLI outputs made by the queue's run lines"])]
    if "steps_per_measurement" in d:                      # mlx-asr-bench: no corpus
        cfg = {"model": d.get("model"), "steps": d["steps_per_measurement"],
               "batches": ",".join(str(r["batch"]) for r in d["results"]),
               "kv_bits": d.get("kv_bits")}
        return "mlx-asr-bench", cfg, [("", {}, None, [], d.get("machine"),
                                      {"rows": d["results"]},
                                      ["machine load and GPU memory at start (not recorded "
                                       "by mlx-asr-bench)"])]
    if isinstance(d.get("results"), list):
        script = {"mlx-whisper": "run_whisper.py", "mlx-qwen3": "run_qwen3.py",
                  "mlx-parakeet": "run_parakeet.py", "sherpa-onnx": "run_reazon.py",
                  "mlx-voxtral-v1": "run_qwen3.py",
                  "mlx-chunked": "run_kotoba_mlx.py",
                  "mlx-qwen3-wholefile": "probes/probe_qwen3_budget.py",
                  "transformers-chunked": "run_kotoba_hf.py",
                  "whisper.cpp": "run_competitors.py",
                  "faster-whisper": "run_competitors.py"}.get(
            d.get("engine"), "run_corpus.py")
        extra = {k: d[k] for k in ("x_realtime", "peak_memory_gb", "peak_rss_gb", "files_scored",
                                   "files_expected", "complete", "truncated_files",
                                   "looped_files", "languages") if k in d}
        return script, d.get("config", {}), [("", {}, d.get("aggregate"), d["results"],
                                             d.get("machine"), extra, [])]
    if isinstance(d.get("results"), dict):                 # sweep_gain.py
        return "sweep_gain.py", d.get("config", {}), [
            (f"mode {k}", {"modes": k}, v.get("aggregate"), v.get("per_file", []), None, {}, [])
            for k, v in d["results"].items()]
    if "arms_expected" in d:                                # sweep_prompt_language.py
        # Topic and terms texts are derived from the private references
        # (derive_prompt_terms.py), so they are never saved: the config keeps a
        # placeholder and the arm keeps only the text's length.
        cfg = {k: (PROMPT_PLACEHOLDER if k in PROMPT_TEXT_KEYS and v else v)
               for k, v in d.get("config", {}).items()}
        return "sweep_prompt_language.py", cfg, [
            (f"{v['content']} {v['lang']}", {"only": v["content"]}, v.get("aggregate"),
             v.get("results", []), d.get("machine"),
             {**{k: v[k] for k in ("x_realtime", "prompt_tokens", "prompt_capacity",
                                   "truncated") if k in v},
              **({"text": v["text"]} if v["content"] in FIXED_PROMPTS
                 else {"text_chars": len(v.get("text", ""))})},
             ["--only selects a content shape, so the command reruns every language "
              "variant of it"] if v["lang"] != "-" else [])
            for v in d["arms"]]
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
    # an installed entry point (mlx-asr-bench) runs as itself, a script from the repo root
    parts = [script if script.startswith("mlx-asr") else f"scripts/benchmarks/{script}"]
    for k, v in cfg.items():
        if k in RUN_DROP or v is None or v is False or v == "":
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            parts.append(flag)
            continue
        if isinstance(v, list) and v and all(isinstance(x, list) for x in v):
            for group in v:                           # action="append", nargs=2: --pair A R
                parts.append(" ".join([flag, *map(shlex.quote, map(str, group))]))
            continue
        if isinstance(v, list):                       # nargs="+", e.g. --files
            parts.append(" ".join([flag, *map(shlex.quote, map(str, v))]))
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


def mark_invalid(run: dict, revisions: dict, current: bool = False):
    """Flag rows that used a superseded audio copy (see benchmarks/results/revisions.json).

    A row is matched to a revision by its recorded duration, so no run date is needed; a
    row of a revised file that records no duration is flagged too, being unverifiable,
    unless `current`: the group was run after every revision (--current-audio), so it
    used the replacement copy.
    """
    bad = []
    for row in run["files"]:
        fid = row.get("file", "")
        for rev in revisions.get(fid, []):
            dur = row.get("duration_s")
            if (dur is None and not current) or (
                    dur is not None and abs(dur - rev["previous"]["duration_s"]) < 1.0):
                row["invalid"] = f"superseded audio revision ({rev['superseded']}): {rev['reason']}"
                bad.append(fid)
    if bad:
        run["invalid_files"] = sorted(set(bad))
        run["note"] = ("aggregate and measures include the invalid files; recompute from the "
                       "rows without an 'invalid' key")


def clean_rows(rows, keep_ref: bool):
    files = []
    for r in rows:
        r = {k: v for k, v in r.items() if keep_ref or k not in PER_FILE_DROP}
        if isinstance(r.get("file"), str):        # bare file id, no audio suffix
            r["file"] = AUDIO_SUFFIX.sub("", r["file"])
        files.append(r)
    return files


def run_params(cfg: dict, arm_params: dict) -> dict:
    return {k: last_component(v) for k, v in
            {**{k: v for k, v in cfg.items() if k not in RUN_DROP}, **arm_params}.items()}


# What a deterministic engine must reproduce exactly on a canary file. Not timing or memory.
CANARY_KEYS = ("duration_s", "unit", "chunks", "ref_chars", "hyp_chars", "cer",
               "coverage_cer", "subs", "dels", "insertions_counted")
# Arguments a supplement may set differently from the run it extends. The corpus may sit
# elsewhere; the canary row and the file ids check that it holds the same audio.
SUPPLEMENT_FREE = {"label", "files", "limit", "corpus"}


def add_supplement(run: dict, path: Path, keep_ref: bool):
    """Append the rows of a supplement (the same config rerun on a few corpus files) to
    `run`. A supplement row for a file the run already holds at the same duration is a
    canary: it must match the run's row exactly, otherwise nothing is written. Any other
    row is appended with `supplement` pointing at its provenance in `run["supplements"]`
    (date, machine, command); the run's own `aggregate` and `measures` are left as the
    runner wrote them."""
    script, cfg, subs = runner_and_rows(json.loads(path.read_text()))
    if len(subs) != 1 or len(run["files"]) == 0:
        sys.exit(f"REFUSED {path.name}: supplements are supported for single-arm runs only")
    _, arm_params, _, rows, machine, extra, _ = subs[0]
    params = run_params(cfg, arm_params)
    differ = sorted(k for k in set(params) | set(run["params"])
                    if k not in SUPPLEMENT_FREE and params.get(k) != run["params"].get(k))
    if differ:
        sys.exit(f"REFUSED {path.name}: config differs from {run['run']} in {differ}")
    held = {r["file"]: r for r in run["files"]}
    index = len(run.setdefault("supplements", []))
    canaries, added, drift = [], [], {}
    for r in clean_rows(rows, keep_ref):
        if "error" in r:
            sys.exit(f"REFUSED {path.name}: a supplement row failed")
        old = held.get(r["file"])
        if old is not None and abs(old.get("duration_s", -1) - r["duration_s"]) < 0.05:
            bad = [k for k in CANARY_KEYS if old.get(k) != r.get(k)]
            # Whisper samples on temperature fallback, so its canary can differ between
            # sessions; the difference is recorded with the supplement instead of refused
            if bad and script == "run_whisper.py":
                drift[r["file"]] = {k: [old.get(k), r.get(k)] for k in bad}
            elif bad:
                sys.exit(f"REFUSED {path.name}: canary row differs from {run['run']} in {bad}")
            canaries.append(r["file"])
            continue
        added.append({**r, "supplement": index})
    if not canaries or not added:
        sys.exit(f"REFUSED {path.name}: a supplement needs a canary and a new row")
    run["files"].extend(added)
    run["supplements"].append({"run": path.stem,
                               "command": command(script, params, f"{path.stem}.json"),
                               "params": params, "machine": machine or {},
                               "measures": extra, "canary": canaries,
                               **({"canary_drift": drift} if drift else {}),
                               "added": [r["file"] for r in added]})


def run_name(path: Path, runs) -> str:
    """The run's name in its group: the file stem, or `<directory>__<stem>` when another run
    in the same group has the same stem, so every run name in a group is unique."""
    same = [r for r in runs if Path(r).stem == path.stem]
    return f"{path.parent.name}__{path.stem}" if len(same) > 1 else path.stem


def build_run(path: Path, keep_ref: bool, supplements=(), name: str = "", overrides=None):
    """`overrides`: parameters the runner did not record (or recorded as a fixed guess)
    whose real value the queue line gives; they replace the recorded ones."""
    d = json.loads(path.read_text())
    script, cfg, subs = runner_and_rows(d)
    out = []
    for arm, arm_params, agg, rows, machine, extra, gaps in subs:
        files = clean_rows(rows, keep_ref)
        params = run_params({**cfg, **(overrides or {})}, arm_params)
        run = {"run": name or path.stem, **({"arm": arm} if arm else {}),
               "command": command(script, params, f"{path.stem}.json"),
               "params": params,
               "machine": machine or {},
               "aggregate": agg,
               "measures": extra,
               "files": files}
        for s in supplements:
            add_supplement(run, s, keep_ref)
        if "rows" in run["measures"]:     # synthetic decode, cue grid, text-vs-srt: no corpus rows
            run["rows"] = run["measures"].pop("rows")
            run["missing"] = gaps
        else:
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
                   "id hashes something other than the stem")
    p.add_argument("--revisions", help="committed revisions.json; flags rows that used "
                   "a superseded audio copy")
    p.add_argument("--out", required=True)
    p.add_argument("--drop-ref-lengths", action="store_true",
                   help="drop per-file reference lengths (then weighted metrics cannot be "
                        "recomputed from the rows)")
    p.add_argument("--keep-ref-lengths", action="store_true",
                   help="no-op, the default; kept so older invocations still work")
    p.add_argument("--redact", action="append", default=[], metavar="RUN=KEY",
                   help="replace parameter KEY of RUN (its JSON stem) with a placeholder, in "
                        "params and command, e.g. a prompt derived from the private corpus")
    p.add_argument("--current-audio", action="store_true",
                   help="every run in the group was made after every audio revision, so rows "
                        "that record no duration used the current copy (recorded in the note)")
    p.add_argument("--command-prefix", action="append", default=[], metavar="RUN=TEXT",
                   help="text the run's command started with that its JSON does not record, "
                        "e.g. an environment variable; RUN is the run JSON's stem")
    p.add_argument("--alias", action="append", default=[], metavar="STEM=NAME",
                   help="a derived input outside the corpus, saved as NAME (listed under "
                        "corpus.json derived)")
    p.add_argument("--supplement", action="append", default=[], metavar="RUN.json=SUPP.json",
                   help="rows from SUPP.json (RUN's config rerun on a canary and new files) "
                        "are added to RUN; repeatable")
    p.add_argument("--set", action="append", default=[], metavar="RUN=KEY=VALUE",
                   help="a parameter RUN's JSON does not record correctly, from the queue line "
                        "that ran it (VALUE is JSON, else a string); RUN is the run JSON's stem")
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    overrides = {}
    for spec in a.set:
        run, key, value = spec.split("=", 2)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        overrides.setdefault(run, {})[key] = value
    supplements = {}
    for pair in a.supplement:
        run, _, supp = pair.partition("=")
        if Path(run).resolve() not in {Path(r).resolve() for r in a.runs}:
            sys.exit(f"--supplement: {run} is not one of the runs")
        supplements.setdefault(Path(run).resolve(), []).append(Path(supp))
    stems = {f.stem for c in a.corpus for f in Path(c).iterdir()
             if f.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".mp4")}
    scrub = Scrubber(stems, json.load(open(a.source_ids)) if a.source_ids else None,
                     dict(x.split("=", 1) for x in a.alias))
    group = {"group": a.group, "question": a.question, "docs": a.doc, "note": a.note,
             "runs": [r for run in a.runs for r in build_run(
                 Path(run), not a.drop_ref_lengths, supplements.get(Path(run).resolve(), ()),
                 run_name(Path(run), a.runs), overrides.get(Path(run).stem))]}
    # --redact names a run by its input JSON stem; a saved run may be renamed
    # `<dir>__<stem>` when stems collide, so both spellings resolve, and every request
    # must redact something: a silent miss would publish the private value.
    saved_names = {}
    for r in a.runs:
        saved_names.setdefault(Path(r).stem, set()).add(run_name(Path(r), a.runs))
        saved_names.setdefault(run_name(Path(r), a.runs), set()).add(run_name(Path(r), a.runs))
    redact = {}
    for x in a.redact:
        run, _, key = x.partition("=")
        if run not in saved_names:
            sys.exit(f"--redact {x}: no input run has stem or saved name {run!r}")
        for name in saved_names[run]:
            redact.setdefault(name, set()).add(key)
    done = set()

    def redact_record(rec, key):
        if key not in rec.get("params", {}):
            return False
        rec["params"][key] = PROMPT_PLACEHOLDER
        # keep the runner's path below scripts/benchmarks/ (e.g. probes/), which
        # command() prepends again
        script = rec["command"].split()[0]
        script = script.split("scripts/benchmarks/", 1)[-1] if "scripts/benchmarks/" in script \
            else Path(script).name
        rec["command"] = command(script, rec["params"],
                                 rec["command"].rsplit("--json ", 1)[-1])
        return True

    for run in group["runs"]:
        for key in redact.get(run["run"], ()):
            # supplements carry their own params and command, so they are redacted too
            hits = [redact_record(rec, key) for rec in [run, *run.get("supplements", [])]]
            if any(hits):
                done.add((run["run"], key))
    missed = [f"{n}={k}" for n, ks in redact.items() for k in ks if (n, k) not in done]
    if missed:
        sys.exit(f"--redact: no parameter to redact for {', '.join(missed)}")
    prefixes = dict(x.split("=", 1) for x in a.command_prefix)
    for run in group["runs"]:
        if run["run"] in prefixes:
            run["command"] = f"{prefixes[run['run']]} {run['command']}"
    group = scrub(group)
    if a.revisions:
        revisions = json.load(open(a.revisions))
        for run in group["runs"]:
            mark_invalid(run, revisions, a.current_audio)
        if a.current_audio:
            group["note"] = (group["note"] + " " if group["note"] else "") + (
                "Every run was made after every audio revision in revisions.json, so it used "
                "the current copies.")
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
