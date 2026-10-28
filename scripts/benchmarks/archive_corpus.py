"""Write benchmarks/results/corpus.json: the publishable metadata of every corpus file.

    uv run python scripts/benchmarks/archive_corpus.py --corpus DIR --languages RUN.json \\
        [--derived NAME=STEM:SECONDS ...] [--filtered NAME=STEM:PATH:FILTER ...] \\
        [--also PATH ...] --out benchmarks/results/corpus.json

Keyed by `sha256(stem)[:12]`, the same id archive_results.py writes into run-groups. Only
language, duration, codec, sample rate, bit depth and bitrate are recorded; nothing about
the content. Language comes from a run JSON's scoring unit (char = Japanese, word =
English). `--derived` records an input cut from a corpus file (for example the padded
worst-case tests) by its parent id and cut length, so it is traceable without a name;
`--filtered` records a whole parent file passed through an ffmpeg audio filter (an
attenuated or resampled copy) by its parent id and the filter, probing the copy at PATH.
"""

import argparse
import json
from pathlib import Path

import av

import re
import sys

from archive_results import ABS_PATH, file_id, token_pattern

DERIVED_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

UNIT_LANGUAGE = {"char": "Japanese", "word": "English"}

# What happened to the audio between the recording and the model, step by step. The metadata
# under "files" describes step 1's output, which is what every run read.
PROCESSING = {
    "1_corpus_copy": "The corpus copies every run read: the per-file metadata below "
                     "(16 kHz mono 16-bit PCM WAV). Sources that were 96 kHz mono WAV were "
                     "converted with `ffmpeg -ac 1 -ar 16000 -c:a pcm_s16le` (ffmpeg's "
                     "default resampler). Not recorded: those sources' original bit depth "
                     "(integer or float), and the original format of the other sources "
                     "before their 16 kHz copies.",
    "2_runner_cache": "Each runner (run_corpus.py, run_whisper.py, run_qwen3.py) converts "
                      "with `ffmpeg -ac 1 -ar 16000 -c:a pcm_s16le` into a temp cache. For "
                      "these 16 kHz mono 16-bit copies that is a re-encode with no change; "
                      "a 24-bit or float source would be rounded to 16-bit here.",
    "3_decode": "Decoded to 16 kHz mono float32 in [-1, 1) for the model.",
    "derived": "A derived input is the first cut_first_s seconds of its parent's step-1 "
               "copy, decoded to float32 (divided by 32768), then written back as 16-bit "
               "PCM (multiplied by 32767, rounded): a gain change of 32767/32768 "
               "(-0.0003 dB) plus rounding, not a bit-exact cut.",
    "filtered": "A filtered input is its parent's whole step-1 copy passed through "
                "`ffmpeg -af <ffmpeg_af> -c:a pcm_s16le` (16 kHz out); its own duration and "
                "format are probed from the written copy.",
}


def probe(path: Path) -> dict:
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        cc = s.codec_context
        dur = float(s.duration * s.time_base) if s.duration else c.duration / 1e6
        return {"duration_s": round(dur, 3), "codec": cc.name, "sample_rate": cc.sample_rate,
                "bit_depth": cc.format.bits, "bitrate": s.bit_rate or c.bit_rate}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--languages", action="append", required=True,
                   help="run JSON(s) that together cover every file; repeatable")
    p.add_argument("--also", action="append", default=[], metavar="PATH",
                   help="an audio file outside --corpus to catalog too (e.g. the single clip)")
    p.add_argument("--derived", action="append", default=[], help="NAME=STEM:SECONDS")
    p.add_argument("--filtered", action="append", default=[], metavar="NAME=STEM:PATH:FILTER",
                   help="the whole STEM passed through ffmpeg -af FILTER, written to PATH")
    p.add_argument("--source-ids", help="local JSON {stem: source identifier}")
    p.add_argument("--revisions", help="committed revisions.json: replaced audio, by id")
    p.add_argument("--probe-from", action="append", default=[],
                   help="STEM=PATH: probe this file for STEM (a staged replacement)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    src = json.load(open(a.source_ids)) if a.source_ids else {}
    revisions = json.load(open(a.revisions)) if a.revisions else {}
    probe_from = dict(spec.split("=", 1) for spec in a.probe_from)
    rows = {Path(r["file"]).stem: r for run in a.languages
            for r in json.load(open(run))["results"]}
    units = {s: r["unit"] for s, r in rows.items()}
    files = {}
    for f in sorted(Path(a.corpus).iterdir()) + [Path(x) for x in a.also]:
        if f.suffix.lower() != ".wav":
            continue
        fid = file_id(src.get(f.stem, f.stem))
        if fid in files:   # same source id twice, or a truncated-hash collision
            sys.exit("REFUSED: two corpus files map to the same id")
        files[fid] = {
            "language": UNIT_LANGUAGE[units[f.stem]], **probe(Path(probe_from.get(f.stem, f))),
            "reference_length": rows[f.stem]["ref_chars"],
            "reference_unit": "characters" if units[f.stem] == "char" else "words"}
        if fid in revisions:
            files[fid]["revisions"] = revisions[fid]
    private = set(rows) | set(src.values())          # every name that must not be published
    derived = {}

    def generic(name):
        # a derived input is published by NAME, so NAME must be a generic label, and one
        # label names one input: a repeat would silently replace the first entry
        if name in derived:
            sys.exit(f"REFUSED: derived name {name!r} is given twice")
        if not DERIVED_NAME.match(name) or any(token_pattern(p).search(name) or
                                               token_pattern(name).search(p) for p in private):
            sys.exit(f"REFUSED: derived name {name!r} is not a generic label or overlaps a "
                     f"private name; use something like 'worst32'")
        return name

    for spec in a.derived:
        name, rest = spec.split("=", 1)
        stem, secs = rest.rsplit(":", 1)
        derived[generic(name)] = {"derived_from": file_id(src.get(stem, stem)),
                                  "cut_first_s": float(secs),
                                  "language": UNIT_LANGUAGE[units[stem]], "codec": "pcm_s16le",
                                  "sample_rate": 16000, "bit_depth": 16}
    for spec in a.filtered:
        name, rest = spec.split("=", 1)
        stem, path, af = rest.split(":", 2)
        derived[generic(name)] = {"derived_from": file_id(src.get(stem, stem)), "ffmpeg_af": af,
                                  "language": UNIT_LANGUAGE[units[stem]], **probe(Path(path))}
    out = {"id": "sha256(source id)[:12]; the source id is a private identifier kept "
                 "with the local corpus (usually the filename stem)",
           "fields": "language, duration_s, codec, sample_rate, bit_depth, bitrate, and the "
              "reference transcript's length (reference_length, in reference_unit)",
           "processing": PROCESSING, "files": files, "derived": derived}
    # sorted by id, not by filename: the order of names would itself leak information
    out["files"] = dict(sorted(files.items()))
    text = json.dumps(out, indent=1)
    leaked = (sum(bool(token_pattern(p).search(text)) for p in private)
              + len(ABS_PATH.findall(text)))
    if leaked:
        sys.exit(f"REFUSED: {leaked} private names or absolute paths in the catalog")
    Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(f"{len(files)} files, {len(derived)} derived -> {a.out}")


if __name__ == "__main__":
    main()
