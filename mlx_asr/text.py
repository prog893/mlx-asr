"""Plain-text transcript assembled directly from tokens, bypassing cue layout.

Why this exists separately from build_cues(): cue text is built by appending raw
token *bytes* into a buffer and decoding it at every flush. Tekken is a byte-level
BPE, so one character can span several tokens, and a cue boundary that lands in
the middle of a multi-byte character splits it. Each half then decodes to U+FFFD
via errors="replace". Across this project's saved SRTs that produced 217 stray
replacement characters, every one of which counts as a CER error while having
nothing to do with what the model actually recognized.

Cue segmentation is also a genuinely underdetermined problem: many groupings of
the same tokens are equally correct, and the choice shifts characters between
lines. Scoring the concatenated cue text therefore measures the cue heuristics as
much as the transcription.

So: decode the whole token stream at once, per chunk, and concatenate. That yields
the transcript the model actually produced, with no layout decisions and no split
characters. Use this to compare model/decode configurations; use the SRT path when
the timing and line breaks are what matter.
"""


def chunk_text(model, timed) -> str:
    """Decode one chunk's timed tokens to text.

    Decodes the whole id list in a single call so multi-byte characters can never
    be split, and lets the tokenizer strip specials rather than filtering by id.
    """
    ids = [t for t, _ in timed]
    if not ids:
        return ""
    tok = model._tokenizer
    n_special = tok.n_special
    ids = [t for t in ids if t >= n_special and t not in tok.special_ids]
    if not ids:
        return ""
    return tok.decode(ids)


def transcript_text(model, timed_chunks, sep: str = "") -> str:
    """Full transcript from all chunks.

    ``sep`` defaults to empty because Japanese has no inter-word spacing and the
    chunk boundary is an artifact of this tool, not of the speech. Pass "\\n" to
    keep chunk structure visible for debugging.
    """
    parts = [chunk_text(model, t) for t in timed_chunks]
    return sep.join(p for p in parts if p)


def write_text(model, timed_chunks, path, sep: str = "") -> str:
    from pathlib import Path

    text = transcript_text(model, timed_chunks, sep)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return text
