"""Write benchmarks/results/corpus.json: the publishable metadata of every corpus file.

    uv run python scripts/benchmarks/archive_corpus.py --corpus DIR --languages RUN.json \\
        [--derived NAME=STEM:SECONDS ...] --out benchmarks/results/corpus.json

Keyed by `sha256(stem)[:12]`, the same id archive_results.py writes into run-groups. Only
language, duration, codec, sample rate, bit depth and bitrate are recorded; nothing about
the content. Language comes from a run JSON's scoring unit (char = Japanese, word =
English). `--derived` records an input cut from a corpus file (for example the padded
worst-case tests) by its parent id and cut length, so it is traceable without a name.
"""

import argparse
import json
from pathlib import Path

import av

from archive_results import file_id

UNIT_LANGUAGE = {"char": "Japanese", "word": "English"}

# What happened to the audio between the recording and the model, step by step. The metadata
# under "files" describes step 1's output, which is what every run read.
PROCESSING = {
    "1_corpus_copy": "The corpus copies every run read: the per-file metadata below "
                     "(16 kHz mono 16-bit PCM WAV). The recordings were 96 kHz mono WAV, "
                     "converted with `ffmpeg -ac 1 -ar 16000 -c:a pcm_s16le` (ffmpeg's "
                     "default resampler). Not recorded: the recordings' original bit depth "
                     "(integer or float), and the original format of the downloaded public "
                     "videos before their 16 kHz copies.",
    "2_runner_cache": "Each runner (run_corpus.py, run_whisper.py, run_qwen3.py) converts "
                      "with `ffmpeg -ac 1 -ar 16000 -c:a pcm_s16le` into a temp cache. For "
                      "these 16 kHz mono 16-bit copies that is a re-encode with no change; "
                      "a 24-bit or float source would be rounded to 16-bit here.",
    "3_decode": "Decoded to 16 kHz mono float32 in [-1, 1) for the model.",
    "derived": "A derived input is the first cut_first_s seconds of its parent's step-1 "
               "copy, decoded to float32 (divided by 32768), then written back as 16-bit "
               "PCM (multiplied by 32767, rounded): a gain change of 32767/32768 "
               "(-0.0003 dB) plus rounding, not a bit-exact cut.",
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
    p.add_argument("--languages", required=True, help="a run JSON covering every file")
    p.add_argument("--derived", action="append", default=[], help="NAME=STEM:SECONDS")
    p.add_argument("--source-ids", help="local JSON {stem: source identifier}")
    p.add_argument("--revisions", help="committed revisions.json: replaced audio, by id")
    p.add_argument("--probe-from", action="append", default=[],
                   help="STEM=PATH: probe this file for STEM (a staged replacement)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    src = json.load(open(a.source_ids)) if a.source_ids else {}
    revisions = json.load(open(a.revisions)) if a.revisions else {}
    probe_from = dict(spec.split("=", 1) for spec in a.probe_from)
    rows = {Path(r["file"]).stem: r for r in json.load(open(a.languages))["results"]}
    units = {s: r["unit"] for s, r in rows.items()}
    files = {}
    for f in sorted(Path(a.corpus).iterdir()):
        if f.suffix.lower() != ".wav":
            continue
        fid = file_id(src.get(f.stem, f.stem))
        files[fid] = {
            "language": UNIT_LANGUAGE[units[f.stem]], **probe(Path(probe_from.get(f.stem, f))),
            "reference_length": rows[f.stem]["ref_chars"],
            "reference_unit": "characters" if units[f.stem] == "char" else "words"}
        if fid in revisions:
            files[fid]["revisions"] = revisions[fid]
    derived = {}
    for spec in a.derived:
        name, rest = spec.split("=", 1)
        stem, secs = rest.rsplit(":", 1)
        derived[name] = {"derived_from": file_id(src.get(stem, stem)), "cut_first_s": float(secs),
                         "language": UNIT_LANGUAGE[units[stem]], "codec": "pcm_s16le",
                         "sample_rate": 16000, "bit_depth": 16}
    out = {"id": "sha256(source id)[:12]; the source id is the filename stem for a recording "
                 "and the video id for a downloaded public video",
           "fields": "language, duration_s, codec, sample_rate, bit_depth, bitrate, and the "
              "reference transcript's length (reference_length, in reference_unit)",
           "processing": PROCESSING, "files": files, "derived": derived}
    # sorted by id, not by filename: the order of names would itself leak information
    out["files"] = dict(sorted(files.items()))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{len(files)} files, {len(derived)} derived -> {a.out}")


if __name__ == "__main__":
    main()
