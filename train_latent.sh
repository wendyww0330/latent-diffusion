#!/bin/bash
set -e          # 一旦任何命令出错，立即停止脚本
set -o pipefail # 管道中任意命令失败，整个管道失败

echo "========================================="
echo "   [1/2] Training VAE (train_from_scratch_latent)"
echo "========================================="
python /workspace/scripts/train_vae.py \
  --config /workspace/configs/train_VAE.yaml

echo "========================================="
echo "   [2/2] Training LDM (train_ldm_gray_numeric)"
echo "========================================="
python /workspace/scripts/train_unet_with_vae.py \
  --config /workspace/configs/train_unet_with_vae.yaml

echo "========================================="
echo "   [DONE] All training finished successfully!"
echo "========================================="
