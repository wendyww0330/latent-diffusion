#!/usr/bin/env bash
set -euo pipefail

INFER_SCRIPT=${1:-/workspace/scripts/infer_numeric_batch.py}
CONF_GLOB=${2:-/workspace/configs/prediction_*.yaml}
LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

mapfile -t CFGS < <(ls -1v $CONF_GLOB 2>/dev/null || true)

if [[ ${#CFGS[@]} -eq 0 ]]; then
  echo "[ERR] No prediction configs match: $CONF_GLOB"
  exit 1
fi

for CFG in "${CFGS[@]}"; do
  NAME=$(basename "$CFG" .yaml)
  TS=$(date +%Y%m%d_%H%M%S)
  LOG="$LOG_DIR/${NAME}_${TS}.infer.log"

  echo "==> INFER  $NAME"
  echo "    config : $CFG"
  echo "    log    : $LOG"
  echo "----------------------------------------"

  #  out_dir 
  python "$INFER_SCRIPT" --config "$CFG" 2>&1 | tee "$LOG"

  echo "==> [$NAME] done."
  echo
done
