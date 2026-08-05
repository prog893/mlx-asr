#!/bin/sh
# End-to-end matrix on the current machine. Same audio, no prompt, varying
# chunk length, batch, overlap and the KV/silence knobs.
#
#   sh benchmarks/run_matrix.sh <audio> <reference.srt> [model]
#
# Writes every output format per config, so text-only CER (content) and
# SRT-joined CER (content + cue layout) can both be scored from one run; see
# benchmarks/compare_text_vs_srt.py. Emits one block per config so results paste
# straight into RESULTS.md.
#
# No --prompt: it is a weak lever (a few tenths of a point, sign varies by
# machine) and it CONFLICTS with --overlap-seconds, which together drive the
# decoder into repetition loops. Keeping it out makes the overlap rows honest.
#
# No `set -e`: one config failing (usually out of memory at a high batch) must
# not abort the remaining configs.
AUDIO=$1
REF=$2
MODEL=${3:-mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit}
PY=${PY:-.venv/bin/python}
OUT=${OUT:-bench_out}
if [ -z "$AUDIO" ] || [ -z "$REF" ]; then
  echo "usage: $0 <audio> <reference.srt> [model]" >&2
  exit 2
fi
mkdir -p "$OUT"

run() {
  label=$1; shift
  echo "=== $label ==="
  $PY -m mlx_asr.cli "$AUDIO" --model "$MODEL" --delay-ms 2400 \
      -f all -o "$OUT/$label" \
      --stats-json "$OUT/$label.stats.json" "$@" 2>&1 \
    | grep -E '^\[(machine|encode|decode|total|silence)'
  echo "  -- text-only (content) --"
  $PY eval_cer.py "$REF" "$OUT/$label.txt" --verbose | tail -2
  $PY eval_cer_lenient.py "$REF" "$OUT/$label.txt" | tail -3
  echo "  -- srt-joined (content + cue layout) --"
  $PY eval_cer.py "$REF" "$OUT/$label.srt" | tail -1
}

run c60b16         --chunk-seconds 60 --max-batch 16 --no-kv-quant --overlap-seconds 0
run c60b16_kv8     --chunk-seconds 60 --max-batch 16 --kv-bits 8   --overlap-seconds 0
run c30b32_kv8     --chunk-seconds 30 --max-batch 32 --kv-bits 8   --overlap-seconds 0
run c30b32_kv8_ov4 --chunk-seconds 30 --max-batch 32 --kv-bits 8   --overlap-seconds 4
run c30b32_kv8_ov8 --chunk-seconds 30 --max-batch 32 --kv-bits 8   --overlap-seconds 8
run c60b16_compact --chunk-seconds 60 --max-batch 16 --kv-bits 8   --overlap-seconds 0 --compact-silence
echo "=== done: $OUT ==="
