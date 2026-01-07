#!/bin/bash
set -e

ROOT=/workspace/latent-v2
CFG_DIR="$ROOT/configs"
EXP_LATENT="$ROOT/experiments/train_unet_with_vae"
EXP_DDPM="$ROOT/experiments/train_ddpm"

# 原始的 infer 配置（作为模板）
LATENT_CFG_TEMPLATE="$CFG_DIR/infer_latent_without_cfg.yaml"
DDPM_CFG_TEMPLATE="$CFG_DIR/infer_ddpm_without_cfg.yaml"

# 要跑的 step 列表
STEPS="10000 20000 30000 40000 50000 60000 70000 80000 90000 100000"

echo "==== LATENT LDM inference for multiple checkpoints ===="
for S in $STEPS; do
  echo "---- latent step $S ----"

  TMP_CFG="$CFG_DIR/tmp_infer_latent_step${S}.yaml"

  # 用 sed 基于模板生成一个新的 yaml：
  # 1) 替换 unet_ckpt
  # 2) 替换 numeric_ckpt
  # 3) 替换 out_dir（避免不同 step 覆盖）
  sed -e "s#^unet_ckpt:.*#unet_ckpt: \"/workspace/latent-v2/experiments/train_unet_with_vae/unet_step${S}.pt\"#;" \
      -e "s#^numeric_ckpt:.*#numeric_ckpt: \"/workspace/latent-v2/experiments/train_unet_with_vae/numeric_enc_step${S}.pt\"#;" \
      -e "s#^out_dir:.*#out_dir: \"/workspace/latent-v2/runs/latent_without_cfg/step${S}\"#;" \
      "$LATENT_CFG_TEMPLATE" > "$TMP_CFG"

  python -u "$ROOT/scripts/infer_latent_without_cfg.py" \
    --config "$TMP_CFG"
done

echo "==== DDPM pixel-space inference for multiple checkpoints ===="
for S in $STEPS; do
  echo "---- ddpm step $S ----"

  TMP_CFG="$CFG_DIR/tmp_infer_ddpm_step${S}.yaml"

  sed -e "s#^unet_ckpt:.*#unet_ckpt: \"/workspace/latent-v2/experiments/train_ddpm/unet_step${S}.pt\"#;" \
      -e "s#^numeric_ckpt:.*#numeric_ckpt: \"/workspace/latent-v2/experiments/train_ddpm/numeric_enc_step${S}.pt\"#;" \
      -e "s#^out_dir:.*#out_dir: \"/workspace/latent-v2/runs/ddpm_without_cfg/step${S}\"#;" \
      "$DDPM_CFG_TEMPLATE" > "$TMP_CFG"

  python -u "$ROOT/scripts/infer_ddpm_without_cfg.py" \
    --config "$TMP_CFG"
done

echo "==== ALL INFERENCE DONE ===="
