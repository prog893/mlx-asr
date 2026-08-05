# Lever: the `--prompt` vocabulary-bias field (Voxtral only)

**Applies to `--model voxtral` only.** The field is a Voxtral architectural feature, not a
general one: it writes into that model's decoder left-pad region. The `whisper-*` and
`kotoba` engines report it as ignored rather than silently dropping it. Whisper has its own
`initial_prompt` concept with different semantics, which this project has not measured, so
nothing here transfers to it.

**Conclusion first.** `--prompt` is a weak and unreliable lever, worth a few tenths of a
CER point at best, and it can be negative. The one thing that matters is a prohibition:
**it is not an instruction field.** Putting an ASR-style imperative there costs about 6
CER points on Japanese and 13.8 WER points on English, which makes it the most expensive
single mistake available in the CLI. It also conflicts badly with
`--overlap-seconds`, and the CLI refuses to use both.

## What the field actually is

Voxtral Realtime's decoder input is `audio_embed + embed(prev_token)`, and the architecture
puts a 32-token left-pad region before the audio. Prompt text is injected there, so **the
model reads it as text it has already emitted** during the leading silence, not as a
directive. That is why the field behaves the way it does, and why none of this generalises
to an engine with a different long-form design.

Two consequences follow directly and both were confirmed by measurement:

- An imperative makes the model *continue* in the register of the imperative. "Transcribe
  the audio accurately." is English, so it pulls output toward English.
- Only the last 31 tokens survive (`n_left_pad_tokens - 1`). Earlier tokens are dropped.
  The CLI now warns when it truncates; it used to do so silently.

## Corpus

Two clips and one corpus, because the effect is small enough that the material changes
the sign. Single-clip work: the 935s Japanese prepared-narration recording (4205 scored
characters) and a 180s excerpt (943 characters), both with verbatim references, so plain
CER applies. Corpus work: 7 spontaneous recordings, 5 Japanese and 2 English, coverage
metric. `scripts/benchmarks/ab_prompt.py`, `scripts/benchmarks/run_corpus.py`.

## Experiment: style, ordering and separators

M2 Ultra 128GB, 60s chunks, batch 16, kv8, 935s clip, 31-token window. Prompt contents are
redacted here because they were real domain vocabulary; the shape of each variant is
what matters.

| variant | tokens | kept | CER | vs none |
|---|---|---|---|---|
| ASCII-comma term list | 38 | 31 | **7.09%** | -0.19 |
| term list, importance-first (so truncation drops the important ones) | 33 | 31 | **7.09%** | -0.19 |
| topic sentence only | 17 | 17 | **7.09%** | -0.19 |
| topic sentence + terms | 33 | 31 | 7.11% | -0.17 |
| natural sentence using the terms | 23 | 23 | 7.11% | -0.17 |
| space-separated terms | 32 | 31 | 7.16% | -0.12 |
| CJK-comma term list | 33 | 31 | 7.21% | -0.07 |
| short list, fits fully | 16 | 16 | 7.21% | -0.07 |
| **no prompt** | 0 | 0 | 7.28% | - |
| language hint + terms | 23 | 23 | 7.49% | +0.21 |
| single term only | 4 | 4 | 7.59% | +0.31 |
| **generic instruction** ("Transcribe the Japanese audio accurately.") | 7 | 7 | **13.17%** | **+5.90** |

Excluding the outlier, eleven variants span 7.09-7.59%: a 0.50-point spread against a
noise floor of roughly 1 point on this clip. Paired testing puts every one of them inside
the noise:

| comparison | diff | 95% CI | verdict |
|---|---|---|---|
| generic instruction vs none | 5.90 | [+1.40, +11.64] | **significant** |
| best variant vs none | 0.19 | [-0.26, +0.60] | not resolvable |
| topic sentence vs none | 0.19 | [-0.17, +0.57] | not resolvable |
| term order forward vs reversed | 0.12 | [-0.26, +0.59] | not resolvable |

**The ordering result is the informative one.** Reversing the term list so that
truncation discards the terms you care about scored *identically* to keeping them. If the
small gains were really about specific vocabulary reaching the model, that could not
happen. Whatever the prompt does here, it is not vocabulary recall.

Confirming that from the other direction: none of the variants recovered the rare proper
nouns they targeted. The rarest term appears once in the reference and was hit 0 times in
every variant. One mid-frequency term went from 1 hit to 4 against 2 in the reference,
i.e. the prompt caused *over*-production rather than recall.

## Experiment: separators across two machines

The 180s clip, 943 reference characters, one row per variant, run on both machines:

| variant | tokens | CER, M4 16GB (nvfp4) | CER, M2 Ultra 128GB (4-bit affine) |
|---|---|---|---|
| no prompt | 0 | 10.05% | **9.03%** |
| CJK punctuation | 45 | **9.59%** | 9.71% |
| CJK, reordered | 40 | 9.71% | 10.05% |
| natural sentence | 26 | 9.71% | 9.71% |
| minimal keyword list | 27 | 10.05% | 9.71% |
| Latin `, ` | 44 | 10.05% | 9.71% |
| Latin `,` no space | 40 | 10.16% | 10.05% |
| bare spaces | 36 | 10.05% | 9.82% |

**The two machines disagree in sign.** Every variant helped on the M4 16GB and every one hurt
on the Ultra, where no-prompt was the best cell in the table. The models differ (nvfp4
versus 4-bit affine) so this is not a clean isolation of the prompt, but the spread
within each column (~0.5 points) is the same size as the disagreement between them. That
is the clearest statement of how weak this lever is.

## Experiment: the instruction trap on a real corpus

The single clip could not show that this effect is language-dependent. The corpus can.
Putting "Transcribe the audio accurately." in the prompt field:

| files | diff | 95% CI |
|---|---|---|
| English (2 files) | **+13.77 WER points** | [+10.34, +15.25] |
| Japanese (5 files) | +1.41 CER points | CI spans zero |
| all 7, pooled | +3.53 | [+0.41, +9.82] |

That fits the mechanism exactly. An English instruction is English text the decoder
believes it just emitted, so it pulls English output badly off-register while barely
perturbing Japanese. **The warning stands; the magnitude depends on whether the prompt
language matches the audio.**

## Experiment: prompt plus overlap is catastrophic

Found while re-running the config matrix. M2 Ultra 128GB, 30s chunks, batch 32, kv8:

| config | CER |
|---|---|
| no prompt, no overlap | 9.04% |
| prompt only | 9.04% |
| overlap 4s only | 7.16% |
| **prompt + overlap 4s** | **18.64%** |

The output degenerates into repetition loops, with a short phrase repeated a dozen times
and 148 characters of duplicated text. Both knobs work by handing the decoder text it
should treat as already-emitted, and stacking them over-conditions it. The positions do
not collide (the prompt occupies decoder positions 0-62, the warm-up region follows), so
this is a behavioural interaction rather than an indexing bug.

The CLI ignores `--prompt` when overlap is active and says so, keeping overlap because it
is by far the stronger effect. Benchmarks pass no prompt, so the overlap rows in
[chunking.md](chunking.md) are clean.

## Practical guidance

- **Never put an instruction there.** Use domain vocabulary or a short topic sentence
  describing the recording.
- Prefer prose: a 17-token topic sentence matched a 38-token term list.
- Put terms you care about **last**, since only the final 31 tokens survive. Though note
  the ordering experiment above suggests this matters less than it should.
- If you use it at all, verify on your own audio. The effect is small enough to flip with
  the model or the clip.
- Japanese punctuation is a marginally better separator than Latin commas.

## What ships

No prompt by default, and Voxtral only. The CLI help says "domain keywords" explicitly,
warns on truncation, and reports when it is ignoring the prompt, whether because overlap is
active or because the selected engine has no such field.

## Related

[chunking.md](chunking.md) for the overlap side of the conflict.
