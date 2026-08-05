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


def build_cues(model, timed_chunks, chunk_offsets, gap_s=1.2, max_chars=28,
               max_dur_s=7.0, time_map=None):
    """Group timed tokens into subtitle cues.

    A cue breaks on: a silence gap > gap_s, text length > max_chars, duration >
    max_dur_s, or sentence-ending punctuation once the cue has some body.
    Returns a list of (start_s, end_s, text).

    **These values are deliberately NOT tuned to the project's references, because
    every timed reference available here was authored by one editor and therefore
    encodes one subtitle house style.** Two sweeps were run and both are recorded
    in docs/benchmarks/cue-layout.md rather than applied:

    * n=1 (one file): suggested (1.2, 28), F1 35.4% -> 43.6%.
    * n=7 (seven files, `scripts/benchmarks/sweep_cues.py`): suggested (0.7, 32) instead,
      and showed the n=1 result only wins on the single file it was fitted to.

    These shipped values score F1 37.4% end to end across the 7 timed references
    against 42.8% for (0.7, 32); see docs/benchmarks/cue-layout.md.
    Giving up those 5.4 points is the deliberate trade described above; what is not
    negotiable is reporting the number the default actually produces, which an
    earlier version of this docstring did not.

    The n=7 sweep is better evidence than the n=1 one, but it is not evidence of
    *good segmentation*; it is evidence of agreement with one subtitle convention.
    All seven references share line-wrap width and pause conventions, so "optimal
    gap_s" there may be a fact about that convention rather than about Japanese
    subtitles. Fitting a shipped default to it would export that style to every
    user.

    What the sweeps DO establish, and what a caller can act on:

    * `gap_s` is the only knob that matters. Mean F1 moves 44.5 / 42.3 / 40.2 /
      36.2 / 35.9% across gap 0.6 / 0.7 / 0.8 / 1.0 / 1.2 on these references, so a
      user with their own reference SRTs should sweep this one parameter and
      nothing else. `mlx-asr --gap-seconds` exists for exactly that; it was advice
      with no flag behind it until the flags were added.
    * `max_chars` is worth under a point anywhere in 28-72.
    * `max_dur_s` is inert: no cue reached it at any value from 4 to 9 seconds,
      because gap_s or max_chars always fires first. It stays as a safety valve
      for long unbroken speech.
    * Lower gap_s buys F1 partly by emitting more cues (cues/ref 0.99 at gap 1.0,
      1.25 at 0.7, 1.36 at 0.6), and that component is recall the reader does not
      benefit from.

    Cue placement is the weaker half of this project's timing story either way
    (Whisper reaches 56.0% F1 on the same files). Drift is the strong half and is
    unaffected by these knobs: regrouping cues cannot move a token's timestamp.
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
