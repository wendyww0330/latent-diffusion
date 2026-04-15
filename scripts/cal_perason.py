import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr


# ===============================
# 1️⃣  你的 pixel ratio 函数
# ===============================
def compute_white_black_ratio(img_path, threshold=128):
    im = Image.open(img_path).convert("L")
    arr = np.array(im)
    white = (arr >= threshold).sum()
    black = (arr < threshold).sum()
    if black == 0:
        return np.nan
    return white / float(black)


# ===============================
# 2️⃣  key 处理函数
# ===============================
def key_from_csv_name(name: str) -> str:
    base = os.path.basename(name)
    base = base.replace(".csv.tif", "")
    if "_facet_size_" in base:
        base = base.split("_facet_size_")[0]
    return base


def key_from_img_name(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    if stem.startswith("binary_"):
        stem = stem[len("binary_"):]
    return stem


# ===============================
# 3️⃣  主流程
# ===============================
def main(csv_path, image_dir):

    df = pd.read_csv(csv_path)

    # --- 构建 image key -> path 映射 ---
    image_map = {}
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(".png"):
            key = key_from_img_name(fname)
            image_map[key] = os.path.join(image_dir, fname)

    print(f"Found {len(image_map)} PNG images")

    # --- 计算 pixel ratio ---
    ratios = []
    missing = []

    for name in df["image_name"]:
        key = key_from_csv_name(name)

        if key not in image_map:
            ratios.append(np.nan)
            missing.append(name)
            continue

        ratio = compute_white_black_ratio(image_map[key])
        ratios.append(ratio)

    df["pixel_ratio"] = ratios

    if missing:
        print(f"Missing {len(missing)} matches")
        print("Example:", missing[:3])

    # --- Pearson correlation across images ---
    def pearson_safe(x, y):
        tmp = pd.concat([x, y], axis=1).dropna()
        if len(tmp) < 2:
            return np.nan
        r, _ = pearsonr(tmp.iloc[:, 0], tmp.iloc[:, 1])
        return r

    r_b50 = pearson_safe(df["pixel_ratio"], df["signal_b50"])
    r_b800 = pearson_safe(df["pixel_ratio"], df["signal_b800"])
    r_b2000 = pearson_safe(df["pixel_ratio"], df["signal_b2000"])

    print("\n===== Final Pearson Correlations =====")
    print(f"b50   : {r_b50:.6f}")
    print(f"b800  : {r_b800:.6f}")
    print(f"b2000 : {r_b2000:.6f}")

    df.to_csv("../csv/15k_baseline_merged_with_pixel_ratio.csv", index=False)
    print("\nSaved merged_with_pixel_ratio.csv")
 

if __name__ == "__main__":
    main(
        csv_path="../inverse_problem_all_signals_summary.csv",
        image_dir="../runs/15000_latent_train_unet_without_dropout_without_cfg"
    )