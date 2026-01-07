#!/bin/bash

# ================================
# 20 个模型目录 + Ground Truth
# ================================

IMAGE_DIRS=(
    "../dataset_split/test"

    "../runs/ddpm_without_cfg/step10000"
    "../runs/ddpm_without_cfg/step20000"
    "../runs/ddpm_without_cfg/step30000"
    "../runs/ddpm_without_cfg/step40000"
    "../runs/ddpm_without_cfg/step50000"
    "../runs/ddpm_without_cfg/step60000"
    "../runs/ddpm_without_cfg/step70000"
    "../runs/ddpm_without_cfg/step80000"
    "../runs/ddpm_without_cfg/step90000"
    "../runs/ddpm_without_cfg/step100000"

    "../runs/latent_without_cfg/step10000"
    "../runs/latent_without_cfg/step20000"
    "../runs/latent_without_cfg/step30000"
    "../runs/latent_without_cfg/step40000"
    "../runs/latent_without_cfg/step50000"
    "../runs/latent_without_cfg/step60000"
    "../runs/latent_without_cfg/step70000"
    "../runs/latent_without_cfg/step80000"
    "../runs/latent_without_cfg/step90000"
    "../runs/latent_without_cfg/step100000"
)

# 输出目录
OUT_DIR="../plots/hist_all_models"
mkdir -p "$OUT_DIR"

# ================================
# 逐个调用 plot_histogram.py
# ================================

echo "== Running histogram plots for 21 folders =="

for DIR in "${IMAGE_DIRS[@]}"
do
    # 把路径替换成可用作文件名的形式  (去掉 / )
    SAFE_NAME=$(echo "$DIR" | sed 's/\//_/g')

    OUT_FILE="${OUT_DIR}/hist_${SAFE_NAME}.png"

    echo "[RUN] python plot_histogram.py --image_dir ${DIR} --out ${OUT_FILE}"

    python plot_histogram.py \
        --image_dir "${DIR}" \
        --out "${OUT_FILE}"
done

echo "== DONE!  All histograms saved into: ${OUT_DIR} =="
