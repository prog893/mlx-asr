"""Can chunk-seam accuracy loss be recovered by carrying context across seams?

Chunking is this tool's own choice, not the model's requirement: the paper notes
the decoder reuses KV state as audio is appended, so a chunk boundary is a place
where we throw context away. Measured cost of that: 60s chunks score 7.2% CER
while 30s chunks score 9.1% on identical audio, purely from having twice as many
seams. Deletions dominate the difference.

Whisper's long-form algorithm fixes the analogous problem by conditioning each
window on the previous window's text. Voxtral has no <|startofprev|> token, but
it has something equivalent by construction: the decoder input at every position
is audio_embed + embed(previous_token), and the 32-token left pad is where a
prompt can be injected. So the tail of chunk N-1's transcript can be written
into chunk N's left pad.

Variants:
  none        independent chunks (current behaviour)
  static      the same caller-supplied keyword prompt in every chunk
              (skipped unless --terms is given)
  carry       last N tokens of the previous chunk's output (sequential, so
              chunks cannot be decoded in one batch: measures the ACCURACY
              ceiling of context carry, not a shippable fast path)
  carry_pair  two-pass: decode all chunks batched, then re-decode only with
              carried context, still batched (pass 2 uses pass 1's tails)

carry_pair is the interesting one: it keeps full batching (so it stays fast) and
costs exactly 2x decode.
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
from mlx_asr.audio import SAMPLE_RATE, load_audio_16k, split_at_silences
from mlx_asr.decode import prompt_capacity, transcribe_batch
from mlx_asr.output import build_cues



DEFAULT_MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def chunk_texts(model, timed):
    """Per-chunk text from timed token lists."""
    tok = model._tokenizer
    out = []
    for t in timed:
        b = bytearray()
        for tid, _ in t:
            if tid < tok.n_special or tid in tok.special_ids:
                continue
            b += tok.token_bytes(tid)
        out.append(b.decode("utf-8", errors="replace"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--chunk-seconds", type=float, default=30.0)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--delay-ms", type=int, default=2400)
    p.add_argument("--kv-bits", type=int, default=8)
    p.add_argument("--terms", default="",
                   help="comma-separated vocabulary for the 'static' variant, "
                        "most important LAST; omit to skip that variant")
    p.add_argument("--variants", default="none,carry_pair,carry")
    p.add_argument("--json")
    a = p.parse_args()

    audio = load_audio_16k(a.audio)
    duration = len(audio) / SAMPLE_RATE
    chunks = split_at_silences(audio, target_s=a.chunk_seconds)
    offsets = list(np.cumsum([0] + [len(c) for c in chunks[:-1]]) / SAMPLE_RATE)
    ref = normalize(load_text(a.reference))

    model = load_model(a.model)
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer

    tek = Tekkenizer.from_file(str(get_model_path(a.model) / "tekken.json"))
    cap = prompt_capacity(model.config)
    terms = [t.strip() for t in a.terms.split(",") if t.strip()]
    kw_ids = tek.encode("、".join(terms) + "。", bos=False, eos=False) if terms else None
    quiet = lambda *x: None

    def score(timed):
        cues = build_cues(model, timed, offsets)
        return cer(ref, normalize("".join(c[2] for c in cues))), cues

    def run(extra, per_chunk=False):
        """extra is one shared id list, or (per_chunk=True) one entry per chunk."""
        out, st_tot = [], {"encode_s": 0.0, "decode_s": 0.0, "steps": 0}
        for i in range(0, len(chunks), a.max_batch):
            sl = slice(i, i + a.max_batch)
            e = extra[sl] if per_chunk else extra
            timed, st = transcribe_batch(model, chunks[sl], a.delay_ms, e, a.kv_bits, quiet)
            out.extend(timed)
            for k in st_tot:
                st_tot[k] += st[k]
        return out, st_tot

    print(f"{a.audio}: {duration:.0f}s -> {len(chunks)} chunks of ~{a.chunk_seconds:.0f}s, "
          f"batch {a.max_batch}, prompt window {cap} tokens\n")
    print(f"{'variant':>12} {'CER':>8} {'delta':>8} {'wall s':>8} {'x rt':>7}  note")

    rows, base = [], None
    want = a.variants.split(",")

    for name in want:
        t0 = time.perf_counter()
        note = ""
        if name == "none":
            timed, _ = run(None)
        elif name == "static":
            if kw_ids is None:
                print(f"{name:>12}  skipped (no --terms given)")
                continue
            timed, _ = run(kw_ids)
        elif name == "carry_pair":
            # pass 1: plain batched. pass 2: each chunk gets the previous
            # chunk's pass-1 tail as its prompt. Both passes stay batched.
            t1, _ = run(None)
            texts = chunk_texts(model, t1)
            extra = [None]
            for prev in texts[:-1]:
                ids = tek.encode(prev[-40:], bos=False, eos=False) if prev.strip() else None
                extra.append(ids)
            timed, _ = run(extra, per_chunk=True)
            note = "2 passes"
        elif name == "carry":
            # strictly sequential: chunk i waits for chunk i-1's text.
            timed, prev_text = [], ""
            for c in chunks:
                ids = tek.encode(prev_text[-40:], bos=False, eos=False) if prev_text.strip() else None
                t1, _ = transcribe_batch(model, [c], a.delay_ms, ids, a.kv_bits, quiet)
                timed.extend(t1)
                prev_text = chunk_texts(model, t1)[0]
            note = "sequential, batch=1"
        else:
            continue
        wall = time.perf_counter() - t0
        c, _ = score(timed)
        if base is None:
            base = c
        rows.append({"variant": name, "cer": round(c, 4),
                     "delta": round(c - base, 4), "wall_s": round(wall, 1),
                     "x_realtime": round(duration / wall, 1), "note": note})
        print(f"{name:>12} {c*100:>7.2f}% {(c-base)*100:>+7.2f} {wall:>8.1f} "
              f"{duration/wall:>7.1f}  {note}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"config": vars(a), "chunks": len(chunks), "results": rows},
                      f, indent=2, ensure_ascii=False)
        print(f"[saved] {a.json}")


if __name__ == "__main__":
    main()
