"""mlx-asr: feed-and-forget batch transcription on Apple Silicon.

    mlx-asr audio.wav --output-format srt
    mlx-asr audio.wav --model whisper-turbo
    mlx-asr --list-models

One CLI over three engines, because on measured evidence no single one wins
everything (docs/benchmarks/engines.md):

    voxtral        fastest, deterministic, best timestamp stability, no language
                   token needed. The default.
    whisper-*      most accurate on our corpus, but samples (so not
                   reproducible) and needs to be told the language.
    kotoba         Japanese-specialised. Runs on MLX through our own chunked
                   driver (chunked.py), since Whisper's sequential algorithm
                   costs a 2-layer distil decoder 68 points.

Defaults per model come from `models.py`; defaults per machine come from
`hardware.py`, which uses measured profiles where a machine has been benchmarked
and a formula elsewhere. Batch size is not a fixed constant because throughput is
not monotonic in it: on a 16GB M4, B=2..8 is *slower* per step than B=1 (see
docs/benchmarks/decode-throughput.md), so "bigger batch is better" lands in the worst regime.

The flags are mostly NOT portable across the three engines, because the engines do
not share a long-form algorithm. **An unsupported flag is a hard error** (exit 2),
never a warning and never silently dropped: a flag that looks accepted and then
does nothing yields output the user reads as having been produced with it, which is
exactly how a published figure in this project came to describe a config the CLI
never applied. `_run_other_backend` rejects the Voxtral-only flags; the Voxtral path
rejects `--language`, the one flag that applies everywhere except here.
`--chunk-seconds` is the single flag that crosses over, and it means a different
thing on each side (chunk length for Voxtral, independent-window length for the
chunked drivers, and unsupported on sequential Whisper, whose 30s window is fixed by
the model).
"""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from .audio import (
    SAMPLE_RATE,
    AudioError,
    apply_gain,
    dbfs,
    compact_silence,
    load_audio_16k,
    split_with_overlap,
)
from .hardware import machine_info, resolve_profile
from .models import DEFAULT_ALIAS, REGISTRY, describe_registry, resolve as resolve_model
from .output import WRITERS, build_cues
from .text import transcript_text, write_text

PROG = "mlx-asr"


class UnsupportedFlags(Exception):
    """A flag was passed that the selected engine cannot honour.

    An error rather than a warning on purpose. The three engines do not share a
    long-form algorithm, so most flags belong to exactly one of them; a flag that
    is accepted and then quietly does nothing yields output the user will read as
    having been produced with it. That is how a published break-F1 figure in this
    project came to describe a cue config the CLI never applied.
    """

    def __init__(self, flags, alias):
        self.flags, self.alias = flags, alias
        # --language is the one flag that goes the other way: every engine but
        # Voxtral takes it, so the remedy differs.
        if flags == ["--language"]:
            hint = (f"{alias} detects the language itself and takes no language "
                    f"token. Drop the flag, or pass it to a whisper-* or kotoba "
                    f"model.")
        else:
            hint = ("These are Voxtral-only, because the engines do not share a "
                    "long-form algorithm. Drop the flag, or use the default "
                    "--model voxtral.")
        super().__init__(f"{', '.join(flags)}: not supported by --model {alias}. "
                         + hint)


def _resolved_cue_config(overrides: dict) -> dict:
    """The cue-grouping values actually used, defaults filled in.

    Read from ``build_cues``' signature rather than duplicated here, so the
    recorded config cannot drift from the shipped default the way the published
    break-F1 number once did.
    """
    sig = inspect.signature(build_cues).parameters
    return {name: overrides.get(name, sig[name].default)
            for name in ("gap_s", "max_chars", "max_dur_s")}


def _write_outputs(a, cues, full_text, meta, log):
    """Shared output path: every backend ends up here with the same cue list."""
    stem = (Path(a.output).with_suffix("") if a.output
            else Path(a.audio).with_suffix(""))
    formats = list(WRITERS) if a.output_format == "all" else [a.output_format]
    for fmt in formats:
        path = (a.output if (a.output and a.output_format != "all")
                else f"{stem}.{fmt}")
        if fmt == "txt":
            with open(path, "w", encoding="utf-8") as f:
                f.write(full_text.strip() + "\n")
        elif fmt == "json":
            WRITERS[fmt](cues, path, meta=meta, full_text=full_text)
        else:
            WRITERS[fmt](cues, path)
        log(f"[saved] {path}")


def _run_other_backend(a, spec, log, t_start):
    """Whisper / transformers path. Kept separate from the Voxtral path because
    almost none of the Voxtral machinery (batching, chunk seams, KV quantization,
    prompt bias) applies, and pretending otherwise would mean silently ignoring
    half the flags the user passed."""
    from .backends import run

    # --chunk-seconds is the one chunking flag that carries over, and only to the
    # chunked drivers: there it picks the window length, which is the largest
    # lever those engines have (23 points across 10-30s on kotoba). The
    # sequential Whisper driver has no such knob, its 30s window is fixed.
    overrides = {}
    if a.chunk_seconds and spec.chunked_long_form:
        overrides["chunk_length_s"] = a.chunk_seconds
        log(f"[{spec.backend}] window length {a.chunk_seconds:g}s "
            f"(default {spec.opts.get('chunk_length_s', 10.0):g}s; "
            f"material-dependent, so sweep it on your own audio)")

    # Every flag that cannot be honoured here is a hard error, not a warning.
    # A flag that appears to be accepted and then does nothing produces a result
    # the user will read as measuring what they asked for, which is worse than
    # refusing: it is the same class of mistake as publishing a number from a
    # config that was never applied. Warnings scroll past; a nonzero exit does not.
    #
    # --delay-ms, --gain, --peak-dbfs and --rms-dbfs have non-None defaults, so
    # they only count as "passed" when set away from it: the default is not
    # something the user asked for.
    unsupported = [name for name, val in (
        ("--max-batch", a.max_batch),
        ("--chunk-seconds", None if spec.chunked_long_form else a.chunk_seconds),
        ("--overlap-seconds", a.overlap_seconds), ("--kv-bits", a.kv_bits),
        ("--prompt", a.prompt), ("--vad", a.vad),
        ("--compact-silence", a.compact_silence), ("--fast", a.fast),
        ("--no-kv-quant", a.no_kv_quant),
        ("--delay-ms", a.delay_ms if a.delay_ms != 2400 else None),
        ("--gain", a.gain if a.gain != "auto" else None),
        # The two --gain targets travel with it: gain is applied on the Voxtral
        # path only (apply_gain below), so a level target passed here would look
        # honoured and change nothing.
        ("--peak-dbfs", a.peak_dbfs if a.peak_dbfs != -1.0 else None),
        ("--rms-dbfs", a.rms_dbfs if a.rms_dbfs != -23.0 else None),
        # Cue grouping is Voxtral's own heuristic over per-token timestamps. The
        # other engines emit segments and their layout is theirs, so these cannot
        # be honoured there.
        ("--gap-seconds", a.gap_seconds), ("--max-chars", a.max_chars),
        ("--max-dur-seconds", a.max_dur_seconds),
    ) if val]
    if unsupported:
        raise UnsupportedFlags(unsupported, spec.alias)
    if not spec.deterministic:
        log(f"[{spec.backend}] this engine samples, so repeat runs differ "
            f"(~0.5 CER points on our corpus); it is not reproducible like "
            f"voxtral")
    if spec.languages != "multilingual" and not a.language:
        log(f"[{spec.backend}] {spec.alias} is {spec.languages}-only; "
            f"forcing language={spec.languages}")

    duration = len(load_audio_16k(a.audio)) / SAMPLE_RATE
    log(f"[model] {spec.label} ({spec.repo})")
    cues, full_text, meta = run(spec, a.audio, language=a.language, log=log,
                                **overrides)
    total = time.perf_counter() - t_start

    _write_outputs(a, cues, full_text, {
        "model": spec.repo, "backend": spec.backend,
        "duration_s": round(duration, 2), "wall_s": round(total, 2),
        "x_realtime": round(duration / total, 2),
        **meta,
    }, log)
    log(f"[total] {duration:.1f}s audio in {total:.1f}s "
        f"({duration / total:.1f}x realtime, incl. model load)")
    if a.stats_json:
        with open(a.stats_json, "w") as f:
            json.dump({"duration_s": duration, "wall_s": total,
                       "x_realtime": duration / total, "model": spec.repo,
                       "backend": spec.backend, "cues": len(cues),
                       "machine": machine_info(), **meta}, f, indent=2)
    return 0


def build_parser():
    """The argument parser, built separately so tests can enumerate the flags.

    Split out of ``main`` because the flag list is a thing to assert about: a
    Voxtral-only flag that is added here and nowhere else reaches the other engines
    and is silently ignored, which is the failure this project keeps guarding
    against. Enumerating the parser catches that; a hand-copied list in a test
    cannot, since the same omission would be made twice.
    """
    p = argparse.ArgumentParser(
        prog=PROG,
        description="Fast multi-model batch ASR on Apple Silicon "
                    "(Voxtral, Whisper, kotoba-whisper)",
    )
    p.add_argument("audio", nargs="?",
                   help="input audio/video file (anything ffmpeg reads)")
    p.add_argument("--model", default=DEFAULT_ALIAS,
                   help=f"registry alias or any HF repo id "
                        f"(default: {DEFAULT_ALIAS}). "
                        f"Aliases: {', '.join(REGISTRY)}")
    p.add_argument("--list-models", action="store_true",
                   help="show the model registry with per-model caveats and exit")
    p.add_argument("--language",
                   help="language code for the Whisper/kotoba backends. Voxtral "
                        "takes none. Whisper autodetects from the first 30s if "
                        "omitted, which misfires on mixed-language audio")
    p.add_argument("-f", "--output-format", default="srt",
                   choices=[*WRITERS.keys(), "all"])
    p.add_argument("-o", "--output", help="output path (default: input stem + ext)")
    p.add_argument("--prompt",
                   help="domain keywords or a topic sentence, in the same language as "
                        "the audio. A weak lever, and easy to make things worse with: it "
                        "is NOT an instruction field (the decoder reads it as text it "
                        "already emitted), it does not recall vocabulary, and on "
                        "space-delimited audio such as English it wrecks word spacing, "
                        "so leave it empty there. Only the last ~31 tokens are used. "
                        "Ignored when --overlap-seconds is active. "
                        "See docs/benchmarks/prompt.md")
    p.add_argument("--chunk-seconds", type=float, default=None,
                   help="chunk/window length. Voxtral: default from the hardware "
                        "profile, and a throughput knob rather than an accuracy one. "
                        "kotoba: the window length, and its biggest lever by far; it is "
                        "material-dependent, so sweep it on your own audio. Ignored by "
                        "the whisper-* models, whose 30s window is fixed. "
                        "See docs/benchmarks/chunking.md")
    p.add_argument("--max-batch", type=int, default=None,
                   help="default: from the hardware profile (see mlx-asr-bench)")
    p.add_argument("--delay-ms", type=int, default=2400,
                   help="transcription delay. The largest accuracy lever here and it "
                        "costs no speed; 2400 is both the default and the maximum the "
                        "model supports. See docs/benchmarks/delay.md")
    p.add_argument("--kv-bits", type=int, default=None, choices=[4, 8],
                   help="quantize the KV cache; 8 is the profile default (faster, "
                        "no measured accuracy cost)")
    p.add_argument("--no-kv-quant", action="store_true",
                   help="disable the profile's KV quantization")
    p.add_argument("--fast", action="store_true",
                   help="halve the chunk length, double the batch, and add "
                        "warm-up overlap to pay back the extra chunk seams. "
                        "Declines automatically when it would not help")
    p.add_argument("--overlap-seconds", type=float, default=None,
                   help="prepend this much preceding audio to each chunk as "
                        "warm-up context and discard its transcript, to recover "
                        "accuracy lost at chunk seams. Helped on one clip and not on a "
                        "corpus, so it is off unless --fast asks for short chunks "
                        "(default: from profile). See docs/benchmarks/chunking.md")
    p.add_argument("--gain", default="auto",
                   help="input level. 'auto' (default) boosts only audio quieter "
                        "than -6 dBFS peak, up to -1 dBFS, and leaves anything "
                        "louder untouched: the mel front end clamps at an absolute "
                        "floor, so quiet input silently loses detail, while "
                        "amplifying healthy audio is at best neutral. Also accepts "
                        "a number of dB, 'peak', 'rms', or 'none'")
    p.add_argument("--peak-dbfs", type=float, default=-1.0,
                   help="target for --gain peak")
    p.add_argument("--rms-dbfs", type=float, default=-23.0,
                   help="target speech-frame RMS for --gain rms")
    p.add_argument("--vad", action="store_true",
                   help="place chunk boundaries with Silero VAD instead of energy "
                        "minima (needs the [vad] extra; never removes audio)")
    p.add_argument("--compact-silence", action="store_true",
                   help="drop long pauses before decode (fewer steps; timestamps remapped)")
    # Cue layout. Defaults are None so build_cues() stays the single source of
    # truth for what ships; passing nothing must behave exactly as before.
    # gap_s is the only one of the three that measurably moves anything, and it
    # is exposed because the right value depends on the subtitle conventions you
    # are matching, which this project cannot know (see mlx_asr/output.py).
    p.add_argument("--gap-seconds", type=float, default=None, metavar="SEC",
                   help="split a cue when speech pauses longer than this "
                        "(default 1.2). The one cue knob worth sweeping: lower "
                        "means more, shorter cues")
    p.add_argument("--max-chars", type=int, default=None,
                   help="split a cue once it reaches this many characters "
                        "(default 28); barely affects accuracy anywhere in its useful "
                        "range")
    p.add_argument("--max-dur-seconds", type=float, default=None, metavar="SEC",
                   help="hard cap on cue duration (default 7.0); a safety valve "
                        "that the other two rules normally pre-empt")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--stats-json", help="write timing stats here")
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)

    if a.list_models:
        print(describe_registry())
        return 0
    if not a.audio:
        p.error("the audio argument is required (or use --list-models)")

    log = (lambda *x: None) if a.quiet else print
    t_start = time.perf_counter()
    spec = resolve_model(a.model)

    # Non-Voxtral engines do their own long-form segmentation, so they skip the
    # chunking/batching path entirely and only share the output writers.
    if spec.backend != "voxtral":
        return _run_other_backend(a, spec, log, t_start)

    # The mirror of the ignored-flag report in _run_other_backend: --language is
    # the one flag that applies to every engine except this one, so passing it
    # here is a plausible mistake and worth naming rather than dropping.
    if a.language:
        # Same rule in the other direction: --language applies to every engine but
        # this one, so passing it here is a plausible mistake and must not look
        # honoured.
        raise UnsupportedFlags(["--language"], spec.alias)

    # Imported here rather than at module scope so that `--model whisper-turbo`
    # and `--model kotoba` work without mlx-audio installed. Only the Voxtral
    # path needs it, and it is the heaviest dependency in the tree.
    from mlx_audio.stt.utils import load as load_model

    from .decode import prompt_capacity, transcribe_batch

    audio = load_audio_16k(a.audio)
    duration = len(audio) / SAMPLE_RATE
    if a.gain not in ("none", ""):
        import numpy as _np

        before = dbfs(audio)
        audio = apply_gain(audio, a.gain, a.peak_dbfs, a.rms_dbfs)
        after = dbfs(audio)
        if abs(after - before) < 0.01:
            log(f"[gain] {a.gain}: peak {before:+.1f} dBFS already healthy, "
                f"left unchanged")
        else:
            clipped = 100 * float(_np.mean(abs(audio) >= 0.999))
            log(f"[gain] {a.gain}: peak {before:+.1f} -> {after:+.1f} dBFS"
                + (f", {clipped:.2f}% of samples clipped" if clipped > 0.01 else ""))
    time_map = None
    if a.compact_silence:
        before = len(audio)
        audio, time_map = compact_silence(audio)
        log(f"[silence] {before / SAMPLE_RATE:.1f}s -> {len(audio) / SAMPLE_RATE:.1f}s "
            f"({100 * (1 - len(audio) / before):.0f}% removed)")

    info = machine_info()
    prof = resolve_profile(info, weights_gb=spec.weights_gb,
                           chunk_seconds=a.chunk_seconds)
    batch = a.max_batch or prof["batch"]
    chunk_s = a.chunk_seconds or prof["chunk_seconds"]
    if a.fast:
        # Decode wall clock is set by the LONGEST row in each batch, so halving
        # the chunk length halves decode only if the batch can still hold every
        # chunk in one pass. Encoder cost, by contrast, grows with chunk count
        # (it is compute-bound, see docs/benchmarks/decode-throughput.md), so this trade is only
        # worth taking when the shorter chunks still fit.
        new_chunk = max(15.0, chunk_s / 2) if not a.chunk_seconds else chunk_s
        new_batch = batch * 2 if not a.max_batch else batch
        passes_now = -(-duration // (batch * chunk_s))
        passes_new = -(-duration // (new_batch * new_chunk))
        if passes_new <= passes_now and new_chunk < chunk_s:
            chunk_s, batch = new_chunk, new_batch
        else:
            log("[fast] no benefit at this duration; keeping the profile config")
    kv_bits = None if a.no_kv_quant else (a.kv_bits or prof.get("kv_bits"))
    overlap_s = a.overlap_seconds
    if overlap_s is None:
        overlap_s = prof.get("overlap_seconds", 0.0)
        # Overlap pays only where seams are dense. At the profile's long chunks a
        # paired test finds no benefit (60s: -0.69 points, CI [-1.47, +0.07]),
        # while at 30s it won +1.80 points, CI [+0.62, +3.20], on a single clip.
        # That win did NOT reproduce on the 20-file corpus (-1.47, CI [-4.33,
        # +2.36], sign reversed), which is why overlap is tied to --fast rather
        # than defaulted on: --fast is an explicit request for the short-chunk
        # regime the single-clip result describes. See docs/benchmarks/chunking.md.
        if a.fast and chunk_s < prof["chunk_seconds"]:
            overlap_s = max(overlap_s, 8.0)
    if a.vad:
        try:
            from .vad import split_at_vad_with_overlap

            chunks, offsets, warmup = split_at_vad_with_overlap(
                audio, target_s=chunk_s, overlap_s=overlap_s
            )
        except ImportError:
            from .backends import _install_hint

            log(f"[vad] onnxruntime not installed ({_install_hint('vad')}); "
                f"falling back to energy-based chunking")
            chunks, offsets, warmup = split_with_overlap(
                audio, target_s=chunk_s, overlap_s=overlap_s
            )
    else:
        chunks, offsets, warmup = split_with_overlap(
            audio, target_s=chunk_s, overlap_s=overlap_s
        )
    log(f"[machine] {info['chip']} {info['ram_gb']}GB, "
        f"{info['gpu_working_set_gb']}GB GPU -> batch {batch}, chunk {chunk_s:.0f}s"
        + (f", kv {kv_bits}-bit" if kv_bits else "")
        + f" ({prof['matched']})")
    if prof["matched"] != "profile" and not a.max_batch:
        log("[machine] no measured profile for this hardware; run "
            "`mlx-asr-bench` to find the best batch size and share it")
    log(f"[audio] {duration:.1f}s -> {len(chunks)} chunks of ~{chunk_s:.0f}s"
        + (f" (+{overlap_s:.0f}s warm-up overlap)" if overlap_s else ""))

    t0 = time.perf_counter()
    model = load_model(spec.repo)
    t_model = time.perf_counter() - t0
    log(f"[model] loaded in {t_model:.1f}s")

    prompt_ids = None
    if a.prompt and overlap_s > 0:
        # Measured conflict: --prompt with --overlap-seconds sends the decoder
        # into degenerate repetition loops (a short phrase repeated many times),
        # costing ~9 CER points
        # (30s chunks, 4s overlap: 9.04% with prompt alone, 7.16% with overlap
        # alone, 18.64% with both). Both knobs work by handing the model text it
        # is meant to treat as already-emitted, and stacking them apparently
        # over-conditions it. Overlap is the stronger effect, so it wins.
        log("[prompt] ignored: --prompt and --overlap-seconds conflict and "
            "together degrade output badly; overlap is kept (see docs/benchmarks/prompt.md). "
            "Use --overlap-seconds 0 if you need the prompt.")
    elif a.prompt:
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
        from mlx_audio.stt.utils import get_model_path

        tek = Tekkenizer.from_file(str(get_model_path(spec.repo) / "tekken.json"))
        prompt_ids = tek.encode(a.prompt, bos=False, eos=False)
        cap = prompt_capacity(model.config)
        if len(prompt_ids) > cap:
            log(f"[prompt] {len(prompt_ids)} tokens exceeds the {cap}-token prompt "
                f"window; keeping the LAST {cap} (put key terms last)")
        else:
            log(f"[prompt] {len(prompt_ids)} bias tokens")

    timed_all, stats = [], {"encode_s": 0.0, "decode_s": 0.0, "steps": 0}
    for i in range(0, len(chunks), batch):
        timed, st = transcribe_batch(
            model, chunks[i : i + batch], a.delay_ms, prompt_ids, kv_bits, log,
            warmup_tokens=warmup[i : i + batch],
        )
        timed_all.extend(timed)
        for k in stats:
            stats[k] += st[k]

    # Only pass the cue knobs the user actually set, so build_cues' own defaults
    # remain the single definition of what ships.
    cue_opts = {k: v for k, v in (("gap_s", a.gap_seconds),
                                  ("max_chars", a.max_chars),
                                  ("max_dur_s", a.max_dur_seconds)) if v is not None}
    cues = build_cues(model, timed_all, list(offsets), time_map=time_map, **cue_opts)
    total = time.perf_counter() - t_start

    stem = Path(a.output).with_suffix("") if a.output else Path(a.audio).with_suffix("")
    formats = list(WRITERS) if a.output_format == "all" else [a.output_format]
    for fmt in formats:
        path = a.output if (a.output and a.output_format != "all") else f"{stem}.{fmt}"
        if fmt == "txt":
            # Decode the token stream directly rather than joining cue text: cue
            # boundaries are a layout choice and used to be able to split
            # multi-byte characters. Plain text should be the model's output, not
            # the subtitle layout flattened.
            write_text(model, timed_all, path)
        elif fmt == "json":
            WRITERS[fmt](cues, path, meta={
                "model": spec.repo, "backend": "voxtral",
                "duration_s": round(duration, 2),
                "wall_s": round(total, 2), "x_realtime": round(duration / total, 2),
                "chunks": len(chunks), "batch": batch, "delay_ms": a.delay_ms,
            }, full_text=transcript_text(model, timed_all))
        else:
            WRITERS[fmt](cues, path)
        log(f"[saved] {path}")

    log(f"[total] {duration:.1f}s audio in {total:.1f}s "
        f"({duration / total:.1f}x realtime, incl. {t_model:.1f}s model load)")

    if a.stats_json:
        with open(a.stats_json, "w") as f:
            json.dump({
                "duration_s": duration, "wall_s": total,
                "x_realtime": duration / total, "model": spec.repo,
                "chunks": len(chunks), "chunk_seconds": chunk_s, "batch": batch,
                "overlap_seconds": overlap_s,
                "delay_ms": a.delay_ms, "kv_bits": kv_bits,
                "compact_silence": a.compact_silence, "gain": a.gain,
                "cues": len(cues),
                # Resolved, not as-passed: a cue count is only interpretable
                # alongside the grouping that produced it, and a published break-F1
                # figure once drifted from the shipped default for want of this.
                "cue_config": _resolved_cue_config(cue_opts),
                "machine": info, "profile_source": prof["matched"],
                "model_load_s": t_model, **stats,
            }, f, indent=2)
    return 0


def cli():
    """Console-script wrapper: turn expected failures into one-line messages.

    Kept separate from `main` so tests and the benchmark scripts can call `main`
    and still see the exception. A missing file or a non-audio input is ordinary
    user error, and a traceback for it buries the one line that matters.
    """
    try:
        return main()
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except UnsupportedFlags as e:
        # 2, matching argparse's usage-error convention: this is a bad invocation,
        # not a failure to process the input.
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(cli())
