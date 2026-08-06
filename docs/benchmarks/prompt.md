# Lever: the `--prompt` vocabulary-bias field (Voxtral only)

**Applies to `--model voxtral` only.** The field is a Voxtral architectural feature, not a
general one: it writes into that model's decoder left-pad region. The `whisper-*` and
`kotoba` engines report it as ignored rather than silently dropping it. Whisper has its own
`initial_prompt` concept with different semantics, which this project has not measured, so
nothing here transfers to it.

**Conclusion first.** `--prompt` is a weak and unreliable lever, worth a few tenths of a
CER point at best, and it can be negative. Two prohibitions matter more than any gain.

**It is not an instruction field.** Putting an ASR-style imperative there costs about 6 CER
points on Japanese on a single clip, and 14 WER points on English at n=20.

**On English audio, do not use it at all.** Measured across eight prompt variants on 20
files, *every* variant cost English 14 to 72 WER points, whatever it contained and whatever
language it was written in. The mechanism is specific and worth knowing: **the prompt
suppresses word spacing**, so the transcript comes back in the shape of
`Thisisroughlywhatitlookslike.` with the content largely intact but unreadable and
unscoreable at word level. Japanese is untouched by this because it has no word spaces,
which is most of why its numbers barely move. So the deciding factor is the language of the
*audio*, not whether the prompt language matches it, which is the opposite of what this
project concluded from the 7-file corpus.

If you use it on Japanese audio, write it in Japanese. All four Japanese-language variants
landed within 0.26 points of no-prompt; all four English-language ones cost 1.4 to 3.3.

**Do not expect it to recall vocabulary.** Counting the prompted terms in the transcripts,
a term prompt moved their emission count by under 7% while the model was already producing
them at 3.8x the reference rate without any prompt. The field conditions register, not word
choice.

It also conflicts badly with `--overlap-seconds`, and the CLI refuses to use both.

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

Two clips and two corpora, because the effect is small enough that the material changes
the sign. Single-clip work: the 935s Japanese prepared-narration recording (4205 scored
characters) and a 180s excerpt (943 characters), both with verbatim references, so plain
CER applies. Corpus work: the early sweeps used 7 spontaneous recordings (5 Japanese, 2
English); the crossed language experiment uses all 20 (17 Japanese, 3 English). Coverage
metric throughout. `scripts/benchmarks/ab_prompt.py`,
`scripts/benchmarks/sweep_prompt_language.py`, `scripts/benchmarks/run_corpus.py`.

The single-clip prompt contents are redacted below because they were real domain
vocabulary. The 20-file experiment does not need that redaction: its term lists are derived
from the references by a rule (tokens appearing in at least two files, ranked by frequency),
so the arm is reproducible from the corpus without publishing anyone's word list, and the
selection cannot be tuned to the outcome.

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
i.e. the prompt caused *over*-production rather than recall. The 20-file experiment below
reproduces this by counting term emissions directly, and finds the same over-production at
corpus scale.

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
perturbing Japanese.

The reading at the time was "the magnitude depends on whether the prompt language matches
the audio". The crossed experiment below tests that directly and **it is wrong**: what
predicts the damage is the *audio* language, not the match between prompt and audio.

## Experiment: prompt content crossed with prompt language, 20 files

The earlier corpus test had one prompt, written in English, on a corpus that is mostly
Japanese, so "an instruction is harmful" and "an English prompt on Japanese audio is
harmful" were the same measurement. This crosses four content shapes against both
languages, all 20 files, one prompt per arm applied to every file, so each arm yields an
audio-Japanese and an audio-English row. `scripts/benchmarks/sweep_prompt_language.py`,
30s chunks, batch 32, kv8, delay 2400, M2 Ultra 128GB. The no-prompt row reproduced the
headline exactly (16.22% / 25.24%), which is the check that this script decodes the same
way `run_corpus.py` does.

| prompt content | lang | tokens | JP coverage CER (17 files) | vs none | EN coverage WER (3 files) | vs none |
|---|---|---|---|---|---|---|
| none | - | 0 | 16.22% | - | 25.24% | - |
| instruction | en | 6 | 17.62% | +1.40 | 39.54% | +14.30 |
| instruction | ja | 12 | 16.48% | +0.26 | 82.05% | +56.81 |
| description | en | 12 | 18.22% | +2.00 | 95.33% | +70.09 |
| description | ja | 13 | 16.39% | +0.17 | 71.79% | +46.55 |
| topic | en | 10 | 19.53% | +3.31 | 96.48% | +71.25 |
| topic | ja | 16 | **16.20%** | **-0.02** | 78.99% | +53.76 |
| terms | en | 16 | 18.50% | +2.28 | 97.60% | +72.37 |
| terms | ja | 24 | **15.97%** | **-0.25** | 82.47% | +57.23 |

Nothing truncated: the longest arm is 24 tokens against a 31-token window, so none of this
is a truncation artifact. Prompt texts were the four shapes written natively in each
language; the term lists were derived mechanically from the references (tokens appearing in
at least two files, ranked by frequency) rather than hand-picked, so they could not be
chosen to flatter the result.

**Three results, in order of how much they change the guidance.**

**1. The English collapse is a spacing artifact, not lost transcription.** Every one of the
eight prompts cost English between 14 and 72 WER points, which reads as near-total failure.
Inspecting the transcripts shows it is not. **A prompt suppresses space emission on English
audio**, and word error rate charges every word of a space-free transcript as wrong.

| arm | spaces per character | word count | character count |
|---|---|---|---|
| none | 0.19 | 4016 | 20923 |
| description ja | 0.07 | 1167 | 16633 |
| terms ja | 0.05 | 826 | 15909 |
| description en | 0.01 | 186 | 15495 |
| terms en | 0.0001 | 3 | 15517 |

One 4112-word recording: without a prompt the model emits 4016 spaces, with the English
term list it emits **two**, producing a 15517-character string in the shape of
`Thisisroughlywhatitlookslike.` The character count barely moves, so the words are all
there.
The ratio holds on all three English files (0.19 unprompted, 0.00 to 0.11 prompted).

Rescoring the English files at character level with whitespace stripped from both sides
collapses the difference:

| file | covWER span across arms | covCER span, no whitespace |
|---|---|---|
| A | 25.92% to 97.58% | 71.17% to 74.12% |
| B | 27.99% to 97.02% | 84.05% to 88.04% |
| C | 20.09% to 98.34% | 102.01% to 104.02% |

So the honest statement is narrower than the aggregate suggested: **a prompt degrades
English word segmentation badly, and leaves the transcribed content roughly intact.** That
still makes the field unusable on English if you want readable output or a word-level
score, which is why the guidance stands, but the earlier reading of these numbers as
"catastrophic transcription failure" was wrong and is corrected here.

Japanese is unaffected by this mechanism because Japanese is not space-delimited, which is
also why its scores move so little. That is the real asymmetry: not that Japanese audio
resists prompting, but that the damage a prompt does is invisible to a character metric on
a language without word spaces. Note n=3 for English, so treat magnitudes as directional.

**2. On Japanese, a prompt in Japanese is at worst free and possibly a small gain.** All
four Japanese-language arms land within 0.26 points of baseline, two of them below it,
and per file they improve 8 to 12 of the 17 Japanese files. All four English-language arms
cost 1.40 to 3.31 points and hurt 11 to 13 of 17. That asymmetry is the one thing here a
user can act on, and it is consistent with the mechanism: the model treats the prompt as
its own prior output, so text in the wrong script is a register the model then has to
escape.

Having found that CER missed the English spacing collapse entirely, the Japanese arms were
checked for the equivalent blind spot, and there is none. Across the 17 Japanese files,
punctuation density stays at 7.6 to 10.6 marks per 1000 characters and the Latin fraction at
1.4 to 1.7% in every arm, against 30.7 and 5.0% in the references. Total character counts
vary by about 10%. So no arm is quietly degrading readability the way the English ones do;
the small CER differences are the whole story on Japanese.

That check does surface something unrelated to prompting: **the model emits Japanese
punctuation at roughly a third of the reference rate in every condition, prompted or not.**
The prompt-driven spread is 3.0 marks per 1000 against a 20.1 gap to the reference, so this
is a property of the model rather than of this lever. Coverage CER barely sees it because
`。` and `、` are single characters in a 4300-character transcript. It matters for anyone
reading the output rather than scoring it, and it is not something a prompt fixes.

**3. The best Japanese arm is a term list, but not because of vocabulary recall.**
`terms ja` at -0.25 is the best cell in the table and it beat the topic sentence, which
nominally reverses the single-clip ranking (there a 17-token topic sentence matched a
38-token term list). The effect is 0.25 points against a resolution floor of about 1.6, so
**this is not a resolved win** either way.

What the term arms do settle is the *mechanism*, because the transcripts were kept
(`--keep-hyp`) and the prompted terms can be counted directly. Across the Japanese files,
the five prompted terms appear **19 times in the references, 72 times without any prompt,
and 77 times with the term prompt**: adding the prompt moved emissions by 5 occurrences,
under 7%, while the model was already producing these terms at 3.8x the reference rate
unprompted. One term went from 47 emissions to 51 against 7 in the reference; one prompted
term never appeared at all, in any condition.

So the field is not performing vocabulary recall on this material, and the earlier
conclusion from the single clip stands, now for a stronger reason. It also means a term
prompt cannot fix a rare-word miss here: the words are not being missed, they are being
over-produced, which is a different failure that a bias prompt makes marginally worse.
Whatever the -0.25 points came from, it was not the terms.

**Your content shape did not matter the way the shapes suggest it should.** An imperative,
a scene-setting description of the recording, a topic sentence and a bare term list all
behave alike within a language: same sign, same rough magnitude, ordered differently on
Japanese than on English. Whatever the field does, it is not reading the prompt as an
instruction, a description or a vocabulary list. It is conditioning register.

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

- **On English audio, leave it empty.** Every variant tested at n=20 cost 14 to 72 WER
  points. This is the strongest guidance on this page, and it does not depend on what the
  prompt says.
- **Never put an instruction there**, in any language. Use domain vocabulary or a short
  topic sentence describing the recording.
- **Write the prompt in the language of the audio.** On Japanese audio the four
  Japanese-language variants were free to marginally positive; the English-language ones all
  cost 1.4 to 3.3 points.
- Content shape matters less than language. Term list, topic sentence, description and
  imperative all behave similarly within a language, which is why none of them is
  recommended over the others on the evidence here.
- **Do not reach for it to fix a missed proper noun.** Term counts moved under 7% when the
  terms were prompted, and on this material the model over-produces them unprompted
  anyway. If a name comes out wrong, this field is not the fix.
- Put terms you care about **last**, since only the final 31 tokens survive. Though note
  the ordering experiment and the emission counts both suggest this matters less than it
  should.
- If you use it at all, verify on your own audio. The effect is small enough to flip with
  the model or the clip.
- Japanese punctuation is a marginally better separator than Latin commas.

## What ships

No prompt by default, and Voxtral only. The CLI help says "domain keywords" explicitly,
warns on truncation, and reports when it is ignoring the prompt, whether because overlap is
active or because the selected engine has no such field.

## Related

[chunking.md](chunking.md) for the overlap side of the conflict.
