#!/usr/bin/env bash
# End-to-end pipeline for ONE model, over the reported shot conditions (0 and 4 by default).
#
#   scripts/run_pipeline.sh <alias> [shot ...]
#
# <alias> is a row in models.csv (or a raw Hugging Face repo id). Requires a GPU for the
# extraction step. After running this for every model, produce the scores and stats tables:
#   python scoring/regen_scores.py
#   python stats/data.py            # builds master_table.csv
#   python stats/test6_regression_perspan.py   # etc.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
ALIAS="${1:?usage: run_pipeline.sh <alias> [shot ...]}"; shift || true
SHOTS=("$@"); [ ${#SHOTS[@]} -eq 0 ] && SHOTS=(0 4)

for SH in "${SHOTS[@]}"; do
  echo "=== extract $ALIAS shot $SH (fwd + rev) ==="
  "$PY" extraction/extract.py --model "$ALIAS" \
    --template "data/prompts/repr_conc_polar_${SH}shot.txt" \
    --stimuli data/stimuli/items_648.xlsx \
    --counterbalance I PU --brackets all --bracket-pooling last \
    --save-span-tokens --layers all \
    --set-name "repr_conc_polar_${SH}sh_spantokE"

  echo "=== curvature + surrogate null $ALIAS shot $SH ==="
  "$PY" curvature/r1_curvature.py "$ALIAS" "$SH" fwd spantokE
  "$PY" curvature/batch_surrogate.py ONE "$ALIAS" "$SH" fwd

  echo "=== within-bracket CKA matrices $ALIAS shot $SH ==="
  "$PY" cka/cka_matrices.py "$ALIAS" "$SH"
done

echo "done: $ALIAS. Once every model is processed, run scoring/ then stats/."
