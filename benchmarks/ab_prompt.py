"""A/B the vocabulary-bias prompt: writing style, ordering, separators, length.

The bias prompt is injected into the decoder's left-pad region, so only the LAST
(n_left_pad_tokens - 1) tokens survive; anything earlier is dropped silently.
That makes several things worth measuring rather than guessing:

  style       a bare keyword list vs a natural sentence vs a topic description.
              A sentence may condition the model better because it looks like
              text the model would actually emit, but costs tokens on function
              words.
  separator   punctuation choice changes how many keywords fit at all. CJK
              full-width marks, ASCII commas, and bare spaces tokenize
              differently.
  ordering    keywords must go LAST to survive truncation.
  length      more terms vs fewer, higher-value ones.

Supply your own vocabulary; nothing here is domain-specific. Provide either
--terms (comma-separated, most important LAST) or --prompt-file with one prompt
variant per line as "label<TAB>text". Without either, only the no-prompt
baseline runs, which is still a useful reference.

    uv run python benchmarks/ab_prompt.py --audio AUDIO --reference REF.srt \\
        --terms "term1,term2,important-term" --topic "what the recording is about"

The model is loaded once and reused across variants.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from mlx_audio.stt.utils import get_model_path
from mlx_audio.stt.utils import load as load_model

from evals.eval_cer import cer, load_text, normalize
from mlx_asr.audio import SAMPLE_RATE, load_audio_16k, split_with_overlap
from mlx_asr.decode import prompt_capacity, transcribe_batch
from mlx_asr.text import transcript_text

DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def build_variants(terms, topic, language_hint=None):
    """Construct prompt variants from a caller-supplied term list.

    Terms arrive in ascending importance so that truncation to the last N tokens
    keeps the most valuable ones. Each variant is one hypothesis about what the
    model responds to: raw vocabulary, prose, or a topic statement.
    """
    variants = {"none": ""}
    if not terms:
        return variants

    fwd = list(terms)
    rev = list(reversed(terms))

    variants["list_cjk_sep"] = "、".join(fwd) + "。"
    variants["list_ascii_sep"] = ", ".join(fwd)
    variants["list_ascii_tight"] = ",".join(fwd)
    variants["list_space_sep"] = " ".join(fwd)
    variants["list_newline_sep"] = "\n".join(fwd)
    # Same content, importance-first: tests whether truncation is really what
    # decides the outcome (this variant loses the important terms).
    variants["list_reversed"] = "、".join(rev) + "。"
    # Fewer, higher-value terms: does a shorter prompt that fully fits beat a
    # longer one that gets cut?
    variants["list_short"] = "、".join(fwd[-3:]) + "。"
    if topic:
        variants["topic_only"] = topic
        variants["topic_plus_list"] = f"{topic} " + "、".join(fwd) + "。"
    if language_hint:
        variants["lang_plus_list"] = f"{language_hint} " + "、".join(fwd) + "。"
    return variants


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--chunk-seconds", type=float, default=60.0)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--delay-ms", type=int, default=2400)
    p.add_argument("--kv-bits", type=int, default=8)
    p.add_argument("--terms", default="",
                   help="comma-separated vocabulary, MOST IMPORTANT LAST")
    p.add_argument("--topic", default="",
                   help="one-sentence description of the recording")
    p.add_argument("--language-hint", default="",
                   help="optional leading sentence naming the language/register")
    p.add_argument("--prompt-file",
                   help="label<TAB>text per line, used instead of the generated set")
    p.add_argument("--track-terms", default="",
                   help="comma-separated terms to count hits for (default: --terms)")
    p.add_argument("--json")
    a = p.parse_args()

    terms = [t.strip() for t in a.terms.split(",") if t.strip()]
    if a.prompt_file:
        variants = {"none": ""}
        for line in Path(a.prompt_file).read_text(encoding="utf-8").splitlines():
            if not line.strip() or "\t" not in line:
                continue
            label, text = line.split("\t", 1)
            variants[label.strip()] = text.strip()
    else:
        variants = build_variants(terms, a.topic, a.language_hint)

    tracked = [t.strip() for t in (a.track_terms or a.terms).split(",") if t.strip()]

    audio = load_audio_16k(a.audio)
    chunks, offsets, warmup = split_with_overlap(
        audio, target_s=a.chunk_seconds, overlap_s=0.0
    )
    ref = normalize(load_text(a.reference))

    model = load_model(a.model)
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer

    tek = Tekkenizer.from_file(str(get_model_path(a.model) / "tekken.json"))
    cap = prompt_capacity(model.config)
    ref_counts = {t: ref.count(t) for t in tracked}

    print(f"prompt window: {cap} tokens (anything earlier is dropped)")
    if ref_counts:
        print(f"tracked terms in reference: {ref_counts}")
    print(f"{len(chunks)} chunks of ~{a.chunk_seconds:.0f}s, batch {a.max_batch}\n")
    print(f"{'variant':>20} {'tok':>4} {'kept':>5} {'CER':>8} {'delta':>7}"
          + ("  term hits" if tracked else ""))

    rows, base = [], None
    for name, text in variants.items():
        ids = tek.encode(text, bos=False, eos=False) if text else None
        t0 = time.perf_counter()
        timed_all = []
        for i in range(0, len(chunks), a.max_batch):
            sl = slice(i, i + a.max_batch)
            timed, _ = transcribe_batch(
                model, chunks[sl], a.delay_ms, ids, a.kv_bits,
                log=lambda *x: None, warmup_tokens=warmup[sl],
            )
            timed_all.extend(timed)
        # Score the token stream, not cue text: cue layout is irrelevant here.
        hyp = normalize(transcript_text(model, timed_all))
        c = cer(ref, hyp)
        if name == "none":
            base = c
        hits = {t: hyp.count(t) for t in tracked}
        n_tok = len(ids) if ids else 0
        rows.append({"variant": name, "prompt": text, "tokens": n_tok,
                     "kept": min(n_tok, cap), "cer": round(c, 4),
                     "delta_vs_none": round(c - base, 4) if base is not None else None,
                     "term_hits": hits, "wall_s": round(time.perf_counter() - t0, 1)})
        d = f"{(c - base)*100:+.2f}" if base is not None and name != "none" else ""
        line = (f"{name:>20} {n_tok:>4} {min(n_tok, cap):>5} {c*100:>7.2f}% {d:>7}")
        if tracked:
            line += "  " + " ".join(f"{k}={v}/{ref_counts[k]}" for k, v in hits.items())
        print(line)

    best = min(rows, key=lambda r: r["cer"])
    print(f"\nbest: {best['variant']} at CER {best['cer']*100:.2f}%")
    if best["variant"] != "none":
        print(f"  ({best['delta_vs_none']*100:+.2f} points vs no prompt; compare "
              f"against the ~1 point noise floor on a single clip)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"model": a.model, "audio": a.audio, "prompt_capacity": cap,
                       "reference_counts": ref_counts, "results": rows}, f,
                      indent=2, ensure_ascii=False)
        print(f"[saved] {a.json}")


if __name__ == "__main__":
    main()
