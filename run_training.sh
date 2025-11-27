#!/bin/bash
set -e  # 出错时立即退出

echo "===== 1. Training VAE ====="
python /workspace/scripts/train_vae.py \
  --config /workspace/configs/train_VAE.yaml \
  | tee /workspace/log_vae.txt

echo "===== VAE finished ====="

echo "===== 2. Training UNet with VAE ====="
python -u /workspace/scripts/train_unet_with_vae.py \
  --config /workspace/configs/train_unet_with_vae.yaml \
  | tee /workspace/log_unet_vae.txt

echo "===== UNet finished ====="

echo "===== 3. Training pixel DDPM ====="
python -u /workspace/scripts/train_ddpm.py \
  --config /workspace/configs/train_ddpm.yaml \
  | tee /workspace/log_ddpm.txt

echo "===== ALL TRAINING COMPLETED ====="
