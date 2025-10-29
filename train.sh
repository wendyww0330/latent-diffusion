#!/usr/bin/env bash
set -euo pipefail

TRAIN_SCRIPT=${1:-/workspace/scripts/train_numeric_encoder_lora.py}
CONF_GLOB=${2:-/workspace/configs/training_*.yaml}
LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

# order（…para1, para2, para3…）
mapfile -t CFGS < <(ls -1v $CONF_GLOB 2>/dev/null || true)

if [[ ${#CFGS[@]} -eq 0 ]]; then
  echo "[ERR] No training configs match: $CONF_GLOB"
  exit 1
fi

for CFG in "${CFGS[@]}"; do
  NAME=$(basename "$CFG" .yaml)
  TS=$(date +%Y%m%d_%H%M%S)
  LOG="$LOG_DIR/${NAME}_${TS}.train.log"

  echo "==> TRAIN  $NAME"
  echo "    config : $CFG"
  echo "    log    : $LOG"
  echo "----------------------------------------"

  # outdir 
  python "$TRAIN_SCRIPT" --config "$CFG" 2>&1 | tee "$LOG"

  echo "==> [$NAME] done."
  echo
done

