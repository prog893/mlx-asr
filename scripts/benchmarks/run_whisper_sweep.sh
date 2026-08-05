#!/bin/bash
# Whisper baseline sweep: every model size over the 7-file corpus and the single
# narration clip, scored by the same coverage metric as the Voxtral runs.
#
# Ordered cheapest-first from scripts/benchmarks/probe_whisper_speed.py, so a failure
# late in the sweep still leaves the cheap rows on disk.
#
# One process per model is load-bearing, not stylistic: mlx_whisper lru_caches
# loaded weights, and timing several models in one process left every previous
# model resident (large-v3 read 1.85x instead of 10.7x). Each line below is a
# fresh interpreter.
#
# The corpus runs pass NO --language, which makes run_whisper.py take each file's
# language from its own reference script. The two rejected alternatives, both
# measured: forcing ja scores the 2 English files at 100% WER, and Whisper's own
# 30s autodetect returned *Russian* for two Japanese files (102-106% CER,
# extra_ratio 8.5). The narration clip is Japanese-only, so it is forced there.
#
#   ./benchmarks/run_whisper_sweep.sh CORPUS_DIR NARRATION_WAV NARRATION_SRT
set -u

CORPUS="${1:-corpus}"
NARR_WAV="${2:-narration.wav}"
NARR_SRT="${3:-narration.srt}"
PY=.venv-whisper/bin/python
OUT=benchmarks
HYP=bench_out/whisper_hyp
export HF_HUB_DISABLE_PROGRESS_BARS=1

MODELS=(
  "mlx-community/whisper-tiny-mlx           tiny"
  "mlx-community/whisper-base-mlx           base"
  "mlx-community/whisper-small-mlx          small"
  "mlx-community/whisper-large-v3-turbo     large-v3-turbo"
  "mlx-community/whisper-medium-mlx         medium"
  "mlx-community/whisper-large-v3-mlx       large-v3"
  "mlx-community/whisper-large-v2-mlx       large-v2"
)

mkdir -p "$HYP"

for entry in "${MODELS[@]}"; do
  set -- $entry
  repo="$1"; label="$2"
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] CORPUS  $label"
  echo "=============================================================="
  $PY -u "$OUT/run_whisper.py" --corpus "$CORPUS" --model "$repo" \
      --label "$label" --keep-hyp "$HYP" \
      --json "$OUT/whisper_corpus_${label}.json" 2>&1 \
    | tr '\r' '\n' | grep --line-buffered -viE "it/s|frames/s|Warning: You are sending"

  echo "[$(date +%H:%M:%S)] NARRATION  $label"
  $PY -u "$OUT/run_whisper.py" --pair "$NARR_WAV" "$NARR_SRT" --model "$repo" \
      --label "$label" --language ja --keep-hyp "$HYP" \
      --json "$OUT/whisper_narration_${label}.json" 2>&1 \
    | tr '\r' '\n' | grep --line-buffered -viE "it/s|frames/s|Warning: You are sending"
done

# kotoba-whisper is deliberately NOT run here: it needs transformers' chunked
# long-form algorithm, not the sequential one mlx-whisper implements, and scoring
# it through the wrong runner cost 68 points. See scripts/benchmarks/run_kotoba_hf.py.

# `condition_on_previous_text=False` on the strongest general models, so the
# comparison is not "Whisper's best flag set versus Voxtral's defaults" in either
# direction. This is the single biggest Whisper lever on long audio.
for repo_label in "mlx-community/whisper-large-v3-mlx large-v3-nocond" \
                  "mlx-community/whisper-large-v3-turbo large-v3-turbo-nocond"; do
  set -- $repo_label
  echo "[$(date +%H:%M:%S)] CORPUS  $2"
  $PY -u "$OUT/run_whisper.py" --corpus "$CORPUS" --model "$1" --no-condition \
      --label "$2" --keep-hyp "$HYP" \
      --json "$OUT/whisper_corpus_$2.json" 2>&1 \
    | tr '\r' '\n' | grep --line-buffered -viE "it/s|frames/s|Warning: You are sending"
done

# Determinism check: Whisper's default temperature ladder can fall back to
# sampling, so unlike Voxtral's greedy decode a rerun may differ. Quantify it.
echo "[$(date +%H:%M:%S)] CORPUS  large-v3-turbo greedy (determinism check)"
$PY -u "$OUT/run_whisper.py" --corpus "$CORPUS" \
    --model mlx-community/whisper-large-v3-turbo --greedy \
    --label large-v3-turbo-greedy --keep-hyp "$HYP" \
    --json "$OUT/whisper_corpus_large-v3-turbo-greedy.json" 2>&1 \
  | tr '\r' '\n' | grep --line-buffered -viE "it/s|frames/s|Warning: You are sending"

# What a zero-config user actually gets: Whisper picking the language itself.
# Kept as a row rather than a footnote, since "you must tell it the language"
# is a real usability difference from Voxtral, which takes no language token.
echo "[$(date +%H:%M:%S)] CORPUS  large-v3-turbo autodetect"
$PY -u "$OUT/run_whisper.py" --corpus "$CORPUS" \
    --model mlx-community/whisper-large-v3-turbo --autodetect \
    --label large-v3-turbo-autodetect --keep-hyp "$HYP" \
    --json "$OUT/whisper_corpus_large-v3-turbo-autodetect.json" 2>&1 \
  | tr '\r' '\n' | grep --line-buffered -viE "it/s|frames/s|Warning: You are sending"

echo "[$(date +%H:%M:%S)] sweep complete"
