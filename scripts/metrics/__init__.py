"""Scoring functions: a reference plus a hypothesis in, a number out.

Deliberately separate from `scripts/benchmarks/`, which is the other half of the same job.
The two were previously named similarly enough to read as synonyms; the split is by
direction of dependency:

    scripts/metrics/     pure scoring. No model, no GPU, no audio decode. Given two texts
                 (or two subtitle files) it returns a number.
    scripts/benchmarks/  experiments. Drive an engine over material, then call these.

So `scripts/benchmarks/` imports `scripts/metrics/` and never the reverse. Nothing here ships with
the CLI, and it needs the `eval` extra (pykakasi, rapidfuzz).

Which metric is right depends on the reference, not on preference:

    eval_cer          plain CER. Correct only when the reference is verbatim.
    eval_coverage     for references that deliberately omit audio, where plain CER
                      reads past 100% on correct output. The right choice for this
                      project's spontaneous material.
    eval_cer_lenient  additionally forgives kana/kanji respelling of the same word.
                      The fair Japanese number.
    eval_cer_kana     collapses everything to readings. Too generous: it also
                      excuses homophone substitutions, which are wrong words rather
                      than spelling choices.
    eval_timing       timestamp drift and subtitle cue placement. Needs a timed
                      reference (.srt/.vtt); plain text cannot support it.

See docs/benchmarks/ for what each measures, when it is the wrong choice, and the
results they produced.
"""
