#!/bin/bash
set -e  # 任何一步失败，立即退出整个脚本

echo "========== [1/7] Train VAE =========="
python -u /workspace/scripts/train_vae.py \
  --config /workspace/configs/train_VAE.yaml

echo "========== [2/7] Train UNet (WITH dropout) =========="
python -u /workspace/scripts/train_unet_with_vae.py \
  --config /workspace/configs/train_unet_with_vae_with_dropout.yaml

echo "========== [3/7] Train UNet (WITHOUT dropout) =========="
python -u /workspace/scripts/train_unet_with_vae.py \
  --config /workspace/configs/train_unet_with_vae_without_dropout.yaml

echo "========== [4/7] Infer latent (CFG, WITH dropout) =========="
python -u /workspace/scripts/infer_latent.py \
  --config /workspace/configs/infer_latent_with_cfg_with_dropout.yaml

echo "========== [5/7] Infer latent (CFG, WITHOUT dropout) =========="
python -u /workspace/scripts/infer_latent.py \
  --config /workspace/configs/infer_latent_with_cfg_without_dropout.yaml

echo "========== [6/7] Infer latent (NO CFG, WITH dropout) =========="
python -u /workspace/scripts/infer_latent.py \
  --config /workspace/configs/infer_latent_without_cfg_with_dropout.yaml

echo "========== [7/7] Infer latent (NO CFG, WITHOUT dropout) =========="
python -u /workspace/scripts/infer_latent.py \
  --config /workspace/configs/infer_latent_without_cfg_without_dropout.yaml

echo "========== ALL STEPS FINISHED SUCCESSFULLY =========="
