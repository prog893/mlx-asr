"""Where in a chunk do the seam errors land: at its start or its end?

This decides the design of overlapping chunks. Both the encoder and the decoder
are causal, so extra audio appended AFTER a chunk cannot change the tokens
emitted before it. Extra audio PREPENDED before a chunk can, because it gives
the model left context it otherwise lacks at position 0.

So if the errors cluster at chunk starts, overlap must be a prefix (decode extra
leading audio, then discard its transcript). If they cluster at chunk ends, the
cause is something else (rows ending early / EOS) and overlap will not help.

Method: align hypothesis to reference once, then bucket the resulting edit
operations by their distance from the nearest chunk boundary, separating "just
after a boundary" from "just before" one.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from rapidfuzz.distance import Levenshtein

from evals.eval_cer import load_text, normalize
from mlx_asr.audio import SAMPLE_RATE, load_audio_16k, split_at_silences

CUE = re.compile(r"(\d\d):(\d\d):(\d\d),(\d+) --> ")


def cue_times(path):
    """[(start_s, text)] from an SRT."""
    out, cur = [], None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        m = CUE.match(line)
        if m:
            cur = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
        elif line and not line.isdigit() and cur is not None:
            out.append((cur, line))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--chunk-seconds", type=float, default=30.0)
    p.add_argument("--window", type=float, default=3.0,
                   help="seconds either side of a boundary to count as 'near'")
    a = p.parse_args()

    audio = load_audio_16k(a.audio)
    chunks = split_at_silences(audio, target_s=a.chunk_seconds)
    bounds = np.cumsum([len(c) for c in chunks[:-1]]) / SAMPLE_RATE
    print(f"{len(chunks)} chunks, {len(bounds)} internal boundaries")

    hyp_cues = cue_times(a.hypothesis)
    # Map each hypothesis character to a timestamp via its cue start.
    char_t = []
    for t, text in hyp_cues:
        char_t.extend([t] * len(text))
    hyp = "".join(text for _, text in hyp_cues)

    # Normalization drops punctuation, so track times through it.
    keep = [(c, t) for c, t in zip(hyp, char_t) if normalize(c)]
    hyp_n = "".join(c for c, _ in keep)
    times = [t for _, t in keep]
    ref_n = normalize(load_text(a.reference))

    ops = Levenshtein.editops(ref_n, hyp_n)
    print(f"ref={len(ref_n)} hyp={len(hyp_n)} ops={len(ops)}")

    # For each op, find its time via the hypothesis index (dest_pos).
    after = {"replace": 0, "insert": 0, "delete": 0}
    before = {"replace": 0, "insert": 0, "delete": 0}
    far = {"replace": 0, "insert": 0, "delete": 0}
    for o in ops:
        i = min(o.dest_pos, len(times) - 1)
        if i < 0:
            continue
        t = times[i]
        d = bounds - t                      # >0 means boundary is ahead
        nearest_ahead = d[d >= 0].min() if (d >= 0).any() else 1e9
        nearest_behind = (-d[d < 0]).min() if (d < 0).any() else 1e9
        if nearest_behind <= a.window:
            after[o.tag] += 1               # just after a boundary
        elif nearest_ahead <= a.window:
            before[o.tag] += 1              # just before a boundary
        else:
            far[o.tag] += 1

    # Fraction of audio that is "near" each way, for a fair baseline.
    dur = len(audio) / SAMPLE_RATE
    frac_after = min(len(bounds) * a.window / dur, 1.0)
    frac_before = min(len(bounds) * a.window / dur, 1.0)
    tot = sum(after.values()) + sum(before.values()) + sum(far.values())

    print(f"\nerror location vs chunk boundaries (window {a.window:.0f}s):")
    print(f"{'region':>28} {'sub':>5} {'ins':>5} {'del':>5} {'total':>6} "
          f"{'% of ops':>9} {'% of audio':>11} {'enrichment':>11}")
    for label, d, frac in (
        (f"first {a.window:.0f}s of a chunk", after, frac_after),
        (f"last {a.window:.0f}s of a chunk", before, frac_before),
        ("elsewhere", far, max(1 - frac_after - frac_before, 1e-9)),
    ):
        n = sum(d.values())
        pct = 100 * n / max(tot, 1)
        print(f"{label:>28} {d['replace']:>5} {d['insert']:>5} {d['delete']:>5} "
              f"{n:>6} {pct:>8.1f}% {100*frac:>10.1f}% {pct/(100*frac):>10.2f}x")
    print("\nenrichment > 1 means errors concentrate there more than chance.")


if __name__ == "__main__":
    main()
