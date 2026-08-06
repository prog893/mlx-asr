"""Cross prompt CONTENT against prompt LANGUAGE, over the whole corpus.

The prompt field is read by the decoder as text it has already emitted (see
docs/benchmarks/prompt.md), which predicts something the earlier experiments could not
test: what matters is not only *what* the prompt says but *which language it is written
in*, because register carries over into the output. The one instruction ever measured was
English, on a corpus that is mostly Japanese, so "an instruction costs 6 CER points"
and "an English prompt on Japanese audio costs 6 CER points" were indistinguishable.

This runs each content shape in both languages against every file in the corpus, so each
cell is one prompt applied to 17 Japanese files and 3 English ones. The scoring already
splits by unit, so a single run yields both an audio-Japanese and an audio-English row:
that crossing (prompt language x audio language) is the point, and it is free.

    instruction   the known trap. "Transcribe the audio accurately."
    description   what the recording IS, not what to do with it. Untested shape.
    topic         a short sentence naming the subject matter. Won at n=1 by 0.19.
    terms         domain vocabulary: trademarks, product names, abbreviations.

Only `none`, `instruction` and `description` run without arguments. `topic` and `terms`
appear only when their text is supplied, since both would otherwise have to name the
material.

Terms and the topic come from `--terms-ja`/`--terms-en` and `--topic-ja`/`--topic-en`, so no
domain vocabulary and no description of the private corpus is committed here;
derive them from your own references. The recommended derivation is mechanical: tokens
appearing in at least two files, ranked by frequency, which avoids hand-picking terms
that flatter the result.

Every arm reuses one loaded model and one 16kHz cache, and arms run sequentially, so the
comparison is clean but the whole sweep is (variants x corpus) long. Budget for it.

    uv run python scripts/benchmarks/sweep_prompt_language.py --corpus DIR \\
        --terms-ja "..." --terms-en "..." --topic-ja "..." --topic-en "..." \\
        --json out.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx
from mlx_audio.stt.utils import get_model_path
from mlx_audio.stt.utils import load as load_model

from benchmarks.machine_state import machine_state, warn_if_busy
from benchmarks.run_corpus import DEFAULT_MODEL, find_pairs, to_16k_mono
from metrics.eval_coverage import load_reference, score_pair
from mlx_asr.audio import SAMPLE_RATE, load_audio_16k, split_with_overlap
from mlx_asr.decode import prompt_capacity, transcribe_batch
from mlx_asr.text import transcript_text

# The instruction and description texts are fixed here rather than passed in, because
# they are the experiment: an imperative and a scene-setting statement, each written
# natively in both languages rather than machine-translated word-for-word, so each reads
# as something a speaker of that language would actually write.
FIXED = {
    "instruction": {
        "en": "Transcribe the audio accurately.",
        "ja": "音声を正確に文字起こししてください。",
    },
    "description": {
        "en": "A raw uncut recording of a person recording a video.",
        "ja": "動画を収録している人の未編集の録音です。",
    },
}

# The topic arm names the corpus subject matter, which is private here, so it is passed
# in rather than baked in. `{topic}` is substituted with --topic-en / --topic-ja.
TOPIC_TEMPLATE = {
    "en": "A conversation about {topic}.",
    "ja": "{topic}についての会話。",
}


def build_variants(terms_ja: str, terms_en: str, topic_ja: str = "",
                   topic_en: str = "") -> list[dict]:
    """One row per (content shape, prompt language), plus the no-prompt baseline.

    The baseline is not a language, so it appears once. Everything else appears twice.
    """
    out = [{"content": "none", "lang": "-", "text": ""}]
    for content in ("instruction", "description"):
        for lang in ("en", "ja"):
            out.append({"content": content, "lang": lang,
                        "text": FIXED[content][lang]})
    for lang, topic in (("en", topic_en), ("ja", topic_ja)):
        if topic:
            out.append({"content": "topic", "lang": lang,
                        "text": TOPIC_TEMPLATE[lang].format(topic=topic)})
    # Terms last within each language, since only the final tokens survive truncation.
    for lang, terms in (("en", terms_en), ("ja", terms_ja)):
        if terms:
            out.append({"content": "terms", "lang": lang, "text": terms})
    return out


def score_arm(model, prepared, prompt_ids, args, log=print, hypdir=None,
              arm_label=""):
    """Decode and score the whole corpus once, under one prompt.

    ``hypdir`` keeps the transcripts. Worth doing for at least the term arms: a CER
    aggregate cannot say *why* a prompt helped, and the specific question these arms
    raise is whether a term list makes the model emit those terms or merely shifts its
    register. That needs the text, and re-decoding 8h of audio to recover it is far
    more expensive than writing it down the first time.
    """
    rows = []
    agg_ref = {"char": 0, "word": 0}
    agg_charged = {"char": 0.0, "word": 0.0}
    tot_wall = tot_dur = 0.0

    for stem, wav, refpath, dur in prepared:
        audio = load_audio_16k(str(wav))
        # Keyword args, and overlap 0: --prompt and overlap over-condition the decoder
        # together (18.64% against 9.04% on one clip), so an overlap arm here would be
        # measuring that interaction rather than the prompt. Same call shape as
        # run_corpus.py so the two are comparable.
        chunks, _offsets, warmup = split_with_overlap(
            audio, target_s=args.chunk_seconds, overlap_s=0.0
        )
        t0 = time.perf_counter()
        timed_all = []
        try:
            for i in range(0, len(chunks), args.max_batch):
                sl = slice(i, i + args.max_batch)
                timed, _ = transcribe_batch(
                    model, chunks[sl], args.delay_ms, prompt_ids, args.kv_bits,
                    log=lambda *a: None, warmup_tokens=warmup[sl],
                )
                timed_all.extend(timed)
        except Exception as e:
            rows.append({"file": stem, "error": f"{type(e).__name__}: {e}"[:200]})
            mx.clear_cache()
            continue
        wall = time.perf_counter() - t0
        hyp = transcript_text(model, timed_all)
        if hypdir:
            (hypdir / f"{stem}.{arm_label}.txt").write_text(
                hyp + "\n", encoding="utf-8")
        unit, s = score_pair(load_reference(str(refpath)), hyp,
                             args.min_cut, args.min_cut_words)
        agg_ref[unit] += s["ref_chars"]
        agg_charged[unit] += s["coverage_cer"] * s["ref_chars"]
        tot_wall += wall
        tot_dur += dur
        rows.append({"file": stem, "duration_s": round(dur, 1), "unit": unit,
                     "x_realtime": round(dur / wall, 1),
                     **{k: v for k, v in s.items() if k != "excused_runs"}})
        mx.clear_cache()

    agg = {u: round(agg_charged[u] / agg_ref[u], 5)
           for u in ("char", "word") if agg_ref[u]}
    return {"aggregate": agg, "ref_units": agg_ref, "results": rows,
            "x_realtime": round(tot_dur / max(tot_wall, 1e-9), 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--chunk-seconds", type=float, default=30.0)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--delay-ms", type=int, default=2400)
    p.add_argument("--kv-bits", type=int, default=8)
    p.add_argument("--min-cut", type=int, default=30)
    p.add_argument("--min-cut-words", type=int, default=6)
    p.add_argument("--limit", type=int, help="only the N shortest files")
    p.add_argument("--topic-ja", default="", help="subject matter, e.g. 'town planning'; "
                                                 "wrapped in a sentence template")
    p.add_argument("--topic-en", default="")
    p.add_argument("--terms-ja", default="", help="comma-separated, important LAST")
    p.add_argument("--terms-en", default="", help="comma-separated, important LAST")
    p.add_argument("--only", help="comma-separated content shapes to run")
    p.add_argument("--keep-hyp", help="directory to write per-arm transcripts into, "
                                      "so term recall can be checked without redecoding")
    p.add_argument("--json")
    a = p.parse_args()

    corpus = Path(a.corpus).expanduser()
    pairs = find_pairs(corpus)
    if not pairs:
        print(f"no audio/transcript pairs found in {corpus}", file=sys.stderr)
        return 2

    tmpdir = Path(__import__("tempfile").gettempdir()) / "voxtral_corpus_16k"
    tmpdir.mkdir(exist_ok=True)
    prepared = []
    for audio, ref in pairs:
        wav = to_16k_mono(audio, tmpdir)
        prepared.append((audio.stem, wav, ref,
                         wav.stat().st_size / (SAMPLE_RATE * 2)))
    prepared.sort(key=lambda x: x[3])
    if a.limit:
        prepared = prepared[: a.limit]

    hypdir = Path(a.keep_hyp).expanduser() if a.keep_hyp else None
    if hypdir:
        hypdir.mkdir(parents=True, exist_ok=True)

    variants = build_variants(a.terms_ja, a.terms_en, a.topic_ja, a.topic_en)
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        variants = [v for v in variants if v["content"] in want]

    state = machine_state()
    print(f"machine: {state['label']}, load {state['load_1min']}, "
          f"{state['gpu_in_use_gb']}GB GPU already in use")
    warn_if_busy(state)
    total = sum(d for _, _, _, d in prepared)
    print(f"corpus: {len(prepared)} files, {total/3600:.2f}h audio")
    print(f"arms:   {len(variants)} "
          f"({total*len(variants)/3600:.1f}h of audio to decode in total)\n")

    model = load_model(a.model)
    cap = prompt_capacity(model.config)
    tek = None
    if any(v["text"] for v in variants):
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
        tek = Tekkenizer.from_file(str(get_model_path(a.model) / "tekken.json"))

    print(f"{'content':<12} {'lang':<5} {'tok':>4} {'trunc':>6} "
          f"{'JP covCER':>10} {'EN covWER':>10} {'x rt':>7}")
    arms = []
    for v in variants:
        ids = None
        n_tok = 0
        if v["text"]:
            ids = tek.encode(v["text"], bos=False, eos=False)
            n_tok = len(ids)
        arm = score_arm(model, prepared, ids, a, hypdir=hypdir,
                        arm_label=f"{v['content']}_{v['lang']}")
        jp = arm["aggregate"].get("char")
        en = arm["aggregate"].get("word")
        trunc = "yes" if n_tok > cap else "no"
        print(f"{v['content']:<12} {v['lang']:<5} {n_tok:>4} {trunc:>6} "
              f"{(jp*100 if jp else float('nan')):>9.2f}% "
              f"{(en*100 if en else float('nan')):>9.2f}% "
              f"{arm['x_realtime']:>6.1f}x")
        arms.append({**v, "prompt_tokens": n_tok, "prompt_capacity": cap,
                     "truncated": n_tok > cap, **arm})
        if a.json:      # written after every arm; a long sweep can be interrupted
            with open(a.json, "w") as f:
                json.dump({"config": vars(a), "machine": state,
                           "complete": len(arms) == len(variants),
                           "arms_done": len(arms), "arms_expected": len(variants),
                           "arms": arms}, f, indent=2, ensure_ascii=False)

    if a.json:
        print(f"\n[saved] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
