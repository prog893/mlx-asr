"""Timing quality against a timed reference: drift and cue-break placement.

Every other metric in this project scores *text* and throws timestamps away.
That misses two failure modes that make a subtitle file unusable even when the
words are right:

  drift        the words are correct but land at the wrong time. The dangerous
               shape is *accumulating* drift, where a seam or a dropped segment
               shifts everything after it, so the end of a 90-minute file is
               seconds out even though every individual cue looks plausible.
  cue breaks   the timings are right but the text is chopped mid-phrase, so the
               reader gets fragments. Independent of drift: a model can be
               perfectly locked to the audio and still break badly.

They are reported separately and never combined, because a single "timing score"
would hide which one failed.

## How anchoring works

These references are edited (chatter cut, lines reflowed), so reference cue N and
hypothesis cue N are not the same material. Comparing them positionally would
report omitted audio as drift, which is the mistake plain CER makes on this
corpus (see eval_coverage.py).

Instead the two texts are aligned character-by-character with Levenshtein
opcodes, exactly as coverage CER does. Every run of >= `--min-anchor` matched
characters becomes an *anchor*: a piece of text both sides agree on. Each
anchor's reference time comes from the reference cue containing it, and its
hypothesis time from the hypothesis cue containing it, both by interpolating
linearly within the cue. The signed difference is the timing error at that
point.

`anchor_coverage` (fraction of reference characters that landed in an anchor) is
reported alongside, because a file with few anchors gives a weak timing estimate
and should be read with suspicion rather than averaged in silently.

## What the numbers mean

  median_abs_ms      typical offset. Under ~300ms is imperceptible for reading.
  p95_abs_ms         tail. This is what viewers actually complain about.
  drift_slope_ms_per_min
                     regression of signed error on time. Near zero means the
                     model stays locked; a large magnitude means error grows
                     through the file, which is the seam/skip failure and is far
                     worse than a constant offset because it cannot be fixed by
                     shifting the whole file.
  constant_offset_ms median signed error. A pure constant offset is benign and
                     trivially correctable, so it is separated from slope.

  break_f1           F1 of hypothesis cue boundaries against reference ones,
                     within `--break-tol` seconds, matched greedily nearest-first.
  mid_phrase_rate    fraction of hypothesis cue ends that fall inside a
                     reference cue rather than near its edge, i.e. splits the
                     author did not make.

    uv run python eval_timing.py REF.srt HYP.srt [--verbose]
"""

import argparse
import re
import sys
from bisect import bisect_right

from rapidfuzz.distance import Levenshtein

from metrics.eval_cer import normalize

SRT_BLOCK = re.compile(
    r"(\d+)\s*\n(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*\n(.*?)(?=\n\s*\n|\Z)",
    re.S,
)
VTT_BLOCK = re.compile(
    r"(?:(\d{2}):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*"
    r"(?:(\d{2}):)?(\d{2}):(\d{2})[.,](\d{3})[^\n]*\n(.*?)(?=\n\s*\n|\Z)",
    re.S,
)


def parse_cues(path):
    """Return [(start_s, end_s, raw_text)] from an SRT or VTT file."""
    raw = open(path, encoding="utf-8", errors="replace").read()
    cues = []
    if ".vtt" in path.lower() or raw.lstrip().startswith("WEBVTT"):
        for m in VTT_BLOCK.finditer(raw):
            h1, m1, s1, ms1, h2, m2, s2, ms2, text = m.groups()
            st = int(h1 or 0) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000
            en = int(h2 or 0) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000
            cues.append((st, en, text))
    else:
        for m in SRT_BLOCK.finditer(raw):
            _, h1, m1, s1, ms1, h2, m2, s2, ms2, text = m.groups()
            st = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000
            en = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000
            cues.append((st, en, text))
    return cues


def build_timeline(cues):
    """Flatten cues into normalized text plus a per-character time estimate.

    Time within a cue is interpolated linearly across its characters. That is an
    approximation (speech is not uniform), but the error it introduces is bounded
    by one cue's duration and is the same for both sides, so it does not bias the
    comparison.
    """
    text_parts, times = [], []
    for st, en, raw in cues:
        t = normalize(raw)
        if not t:
            continue
        n = len(t)
        span = max(en - st, 1e-6)
        for i, ch in enumerate(t):
            times.append(st + span * (i + 0.5) / n)
        text_parts.append(t)
    return "".join(text_parts), times


def anchors(ref_text, hyp_text, min_anchor, max_span=None):
    """Matched runs of >= min_anchor characters, as (ref_pos, hyp_pos, length).

    Long runs are subdivided into <= ``max_span`` pieces. Without this, an
    accurate transcript produces one enormous matched run, which yields a single
    timing sample for the whole file: the median would still be right but slope
    would be uncomputable, so accumulating drift (the failure that matters most)
    would be invisible exactly when the text is good. Subdividing costs nothing
    and keeps the sample count proportional to file length.
    """
    out = []
    for op in Levenshtein.opcodes(ref_text, hyp_text):
        if op.tag != "equal":
            continue
        ln = op.src_end - op.src_start
        if ln < min_anchor:
            continue
        if max_span and ln > max_span:
            n = ln // max_span
            step = ln // n
            for k in range(n):
                sub = step if k < n - 1 else ln - step * (n - 1)
                out.append((op.src_start + k * step, op.dest_start + k * step,
                            sub))
        else:
            out.append((op.src_start, op.dest_start, ln))
    return out


def _median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _percentile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q / 100 * (len(s) - 1)))))
    return s[i]


def _slope(xs, ys):
    """Least-squares slope of ys on xs, or None if degenerate."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-9:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def drift_stats(ref_cues, hyp_cues, min_anchor=8, max_span=40):
    ref_text, ref_times = build_timeline(ref_cues)
    hyp_text, hyp_times = build_timeline(hyp_cues)
    if not ref_text or not hyp_text:
        return None
    errs, at_min, matched = [], [], 0
    for rp, hp, ln in anchors(ref_text, hyp_text, min_anchor, max_span):
        matched += ln
        # sample the anchor midpoint: its interior is agreed text on both sides
        ri, hi = rp + ln // 2, hp + ln // 2
        if ri >= len(ref_times) or hi >= len(hyp_times):
            continue
        errs.append((hyp_times[hi] - ref_times[ri]) * 1000.0)
        at_min.append(ref_times[ri] / 60.0)
    if not errs:
        return {"anchors": 0, "anchor_coverage": 0.0}
    abs_errs = [abs(e) for e in errs]
    return {
        "anchors": len(errs),
        "anchor_coverage": matched / max(len(ref_text), 1),
        "median_abs_ms": _median(abs_errs),
        "p95_abs_ms": _percentile(abs_errs, 95),
        "max_abs_ms": max(abs_errs),
        "constant_offset_ms": _median(errs),
        "drift_slope_ms_per_min": _slope(at_min, errs),
        "span_min": max(at_min) - min(at_min) if at_min else 0.0,
    }


def break_stats(ref_cues, hyp_cues, tol=0.5):
    """Boundary agreement and mid-phrase splitting."""
    if not ref_cues or not hyp_cues:
        return None
    ref_ends = sorted(c[1] for c in ref_cues)
    hyp_ends = sorted(c[1] for c in hyp_cues)
    # greedy nearest-first match so one reference end cannot absorb many
    # hypothesis ends (which would flatter a model that breaks constantly)
    pairs = sorted((abs(h - r), h, r) for h in hyp_ends for r in ref_ends
                   if abs(h - r) <= tol)
    used_h, used_r, tp = set(), set(), 0
    for _, h, r in pairs:
        if h in used_h or r in used_r:
            continue
        used_h.add(h)
        used_r.add(r)
        tp += 1
    prec = tp / len(hyp_ends)
    rec = tp / len(ref_ends)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    # a hypothesis cue end strictly inside a reference cue (not near its edges)
    # is a split the author did not make
    starts = [c[0] for c in ref_cues]
    mid = 0
    for h in hyp_ends:
        i = bisect_right(starts, h) - 1
        if 0 <= i < len(ref_cues):
            st, en = ref_cues[i][0], ref_cues[i][1]
            if st + tol < h < en - tol:
                mid += 1
    def durs(cs):
        return [c[1] - c[0] for c in cs]
    def chars(cs):
        return [len(normalize(c[2])) for c in cs]
    return {
        "ref_cues": len(ref_cues), "hyp_cues": len(hyp_cues),
        "boundary_precision": prec, "boundary_recall": rec, "break_f1": f1,
        "mid_phrase_rate": mid / len(hyp_ends),
        "ref_median_dur_s": _median(durs(ref_cues)),
        "hyp_median_dur_s": _median(durs(hyp_cues)),
        "ref_median_chars": _median(chars(ref_cues)),
        "hyp_median_chars": _median(chars(hyp_cues)),
    }


def score(ref_path, hyp_path, min_anchor=8, tol=0.5, max_span=40):
    ref_cues, hyp_cues = parse_cues(ref_path), parse_cues(hyp_path)
    return {"drift": drift_stats(ref_cues, hyp_cues, min_anchor, max_span),
            "breaks": break_stats(ref_cues, hyp_cues, tol),
            "ref_cues": len(ref_cues), "hyp_cues": len(hyp_cues)}


def fmt(v, nd=0, suffix=""):
    return "-" if v is None else f"{v:.{nd}f}{suffix}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference", help="timed SRT/VTT reference")
    p.add_argument("hypothesis", help="timed SRT/VTT from the model")
    p.add_argument("--min-anchor", type=int, default=8,
                   help="matched-character run length that counts as an anchor")
    p.add_argument("--break-tol", type=float, default=0.5,
                   help="seconds within which two cue ends count as the same")
    p.add_argument("--max-anchor-span", type=int, default=40,
                   help="subdivide matched runs longer than this, so a clean "
                        "transcript still yields many timing samples")
    p.add_argument("--json")
    a = p.parse_args()

    s = score(a.reference, a.hypothesis, a.min_anchor, a.break_tol,
              a.max_anchor_span)
    d, b = s["drift"], s["breaks"]
    if not d or not d.get("anchors"):
        print("no anchors found: the two texts do not agree anywhere, so timing "
              "cannot be measured", file=sys.stderr)
        return 2

    print(f"cues: {s['ref_cues']} reference, {s['hyp_cues']} hypothesis")
    print(f"anchors: {d['anchors']} "
          f"({d['anchor_coverage']*100:.1f}% of reference characters)")
    if d["anchor_coverage"] < 0.5:
        print("  NOTE: fewer than half the reference characters anchored; "
              "treat these timing numbers as weak")
    print("\n--- drift (is the text locked to the audio?) ---")
    print(f"  median |error|        {fmt(d['median_abs_ms'], 0, ' ms')}")
    print(f"  p95 |error|           {fmt(d['p95_abs_ms'], 0, ' ms')}")
    print(f"  max |error|           {fmt(d['max_abs_ms'], 0, ' ms')}")
    print(f"  constant offset       {fmt(d['constant_offset_ms'], 0, ' ms')}"
          f"   (benign, shift-correctable)")
    print(f"  drift slope           "
          f"{fmt(d['drift_slope_ms_per_min'], 1, ' ms/min')}"
          f"   over {fmt(d['span_min'], 1, ' min')}")
    if d["drift_slope_ms_per_min"] is not None:
        total = d["drift_slope_ms_per_min"] * d["span_min"]
        print(f"  => implied end-to-end accumulation {total:+.0f} ms")

    if b:
        print("\n--- cue breaks (are the cuts in sensible places?) ---")
        print(f"  boundary F1           {b['break_f1']*100:.1f}%  "
              f"(P {b['boundary_precision']*100:.1f} / "
              f"R {b['boundary_recall']*100:.1f}, +/-{a.break_tol}s)")
        print(f"  mid-phrase splits     {b['mid_phrase_rate']*100:.1f}% "
              f"of hypothesis cue ends")
        print(f"  median cue duration   "
              f"{fmt(b['hyp_median_dur_s'], 2, 's')} vs "
              f"{fmt(b['ref_median_dur_s'], 2, 's')} reference")
        print(f"  median cue length     "
              f"{fmt(b['hyp_median_chars'], 0, ' ch')} vs "
              f"{fmt(b['ref_median_chars'], 0, ' ch')} reference")

    if a.json:
        import json
        with open(a.json, "w") as f:
            json.dump({"reference": a.reference, "hypothesis": a.hypothesis,
                       "min_anchor": a.min_anchor, "break_tol": a.break_tol,
                       **s}, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
