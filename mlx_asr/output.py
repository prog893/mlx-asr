"""Cue grouping and output formats (srt, vtt, txt, json)."""

import json
from pathlib import Path

from .audio import TOKEN_S


def _prepare(path):
    """Create the parent directory so `-o subdir/name` works as documented."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def _complete(buf: bytes) -> bool:
    """True if buf ends on a UTF-8 character boundary (so it decodes cleanly)."""
    try:
        buf.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def build_cues(model, timed_chunks, chunk_offsets, gap_s=0.7, max_chars=56,
               max_dur_s=9.0, time_map=None):
    """Group timed tokens into subtitle cues.

    A cue breaks on: a silence gap > gap_s, text length > max_chars, duration >
    max_dur_s, or sentence-ending punctuation once the cue has some body.
    Returns a list of (start_s, end_s, text).

    The defaults are the grid point closest to the hand-made timed references: the
    highest break F1 among points whose cue count is within 10% of the references'
    (not-measured: the selection rule). Over 168 points of gap_s x max_chars x max_dur_s
    on the 7 timed files, (0.7, 56, 9.0) scores 42.65% break F1, 54.1% mid-phrase ends
    and 1.07 cues per reference cue, against 37.43%, 58.1% and 1.04 for the previous
    (1.2, 28, 7.0). The best break F1 overall, (0.4, 56, 5.0) at 45.35%, emits 1.97
    cues per reference cue, so it is further from the references in shape.
    (source: m2-ultra-cue-layout.json#cue_sweep_union; the sweep regrouped tokens
    from one 4bit decode at 30s / B32, see docs/benchmarks/cue-layout.md.)
    """
    from .audio import map_time

    tok = model._tokenizer
    n_special = tok.n_special

    events = []
    for timed, offset in zip(timed_chunks, chunk_offsets):
        for t, t_s in timed:
            if t < n_special or t in tok.special_ids:
                continue
            events.append((tok.token_bytes(t), offset + t_s))

    cues = []
    cur = bytearray()
    cur_start = cur_end = None

    def flush():
        # errors="replace" only bites on a truly truncated tail (the stream ended
        # mid-character); every interior break is guarded by _complete().
        nonlocal cur, cur_start, cur_end
        text = cur.decode("utf-8", errors="replace").strip()
        if text:
            cues.append((cur_start, cur_end, text))
        cur = bytearray()
        cur_start = cur_end = None

    for b, t_s in events:
        # Only consider breaking where the buffer ends on a complete character.
        # Tekken is a byte-level BPE, so one character can span several tokens;
        # flushing mid-character splits it and both halves decode to U+FFFD.
        # Deferring the break to the next character boundary costs at most a
        # couple of characters of cue length and never corrupts text.
        if cur_start is not None and _complete(cur):
            text_len = len(cur.decode("utf-8"))
            if t_s - cur_end > gap_s or text_len >= max_chars or t_s - cur_start > max_dur_s:
                flush()
        if cur_start is None:
            cur_start = max(t_s - TOKEN_S, 0.0)
        cur += b
        cur_end = t_s
        if _complete(cur):
            tail = cur.decode("utf-8")
            if len(tail) >= max_chars // 2 and tail and tail[-1] in "。！？?!":
                flush()
    if cur:
        flush()

    if time_map is not None:
        cues = [(map_time(s, time_map), map_time(e, time_map), t) for s, e, t in cues]
    return cues


def fmt_ts(s: float, sep: str = ",") -> str:
    # not-measured: millisecond arithmetic in a format string
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


def wrap_cjk(text: str, width: int = 16) -> str:
    return "\n".join(text[i : i + width] for i in range(0, len(text), width))


def write_srt(cues, path, line_width=16, min_dur=0.5):
    with open(_prepare(path), "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(cues, 1):
            end = max(end, start + min_dur)
            f.write(f"{i}\n{fmt_ts(start)} --> {fmt_ts(end)}\n")
            f.write(wrap_cjk(text, line_width) + "\n\n")


def write_vtt(cues, path, line_width=16, min_dur=0.5):
    with open(_prepare(path), "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, (start, end, text) in enumerate(cues, 1):
            end = max(end, start + min_dur)
            f.write(f"{i}\n{fmt_ts(start, '.')} --> {fmt_ts(end, '.')}\n")
            f.write(wrap_cjk(text, line_width) + "\n\n")


def write_txt(cues, path):
    with open(_prepare(path), "w", encoding="utf-8") as f:
        f.write("\n".join(c[2] for c in cues) + "\n")


def write_json(cues, path, meta=None, full_text=None):
    doc = {
        # full_text comes from the token stream when available (see text.py);
        # joining cue text is a fallback that inherits cue layout artifacts.
        "text": full_text if full_text is not None else "".join(c[2] for c in cues),
        "segments": [
            {"id": i, "start": round(s, 3), "end": round(e, 3), "text": t}
            for i, (s, e, t) in enumerate(cues)
        ],
    }
    if meta:
        doc["meta"] = meta
    with open(_prepare(path), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


WRITERS = {"srt": write_srt, "vtt": write_vtt, "txt": write_txt, "json": write_json}
