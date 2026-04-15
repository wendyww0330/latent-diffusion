import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr


# ===============================
# 1) pixel ratio（沿用你的定义：white/black）
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
# 2) key 提取（CSV）
#  CSV 里一般是 ../binary_PNG/binary_labels_mask_..._threshold_0_7.png
#  也可能是 labels_mask_... .csv.tif（你上一版）
# ===============================
def key_from_csv_name(name: str) -> str:
    base = os.path.basename(str(name))

    # 去掉常见扩展
    base = base.replace(".csv.tif", "")
    base = re.sub(r"\.png$", "", base, flags=re.IGNORECASE)

    # 如果有 facet_size 字段，截断
    if "_facet_size_" in base:
        base = base.split("_facet_size_")[0]

    # 去掉 binary_ 前缀
    if base.startswith("binary_"):
        base = base[len("binary_"):]

    return base


# ===============================
# 3) key 提取（图片文件名）— 重点：处理随机后缀
#  例子：
#   binary_labels_mask_..._threshold_0_5-7bf23b6a_gray.png
#  目标 key：
#   labels_mask_..._threshold_0_5
# ===============================
def key_from_img_name(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]

    # 去掉 binary_ 前缀
    if stem.startswith("binary_"):
        stem = stem[len("binary_"):]

    # 去掉末尾常见标记：_gray / _mask / _pred 等（按需扩展）
    stem = re.sub(r"_(gray|mask|pred|gen)$", "", stem, flags=re.IGNORECASE)

    # 去掉随机hash后缀：-7bf23b6a 这种（假设是连字符后跟 6~16 位十六进制）
    stem = re.sub(r"-[0-9a-fA-F]{6,16}$", "", stem)

    # 如果还有其它“连字符随机串”也可更宽松：
    # stem = re.sub(r"-[A-Za-z0-9]{4,}$", "", stem)

    return stem


# ===============================
# 4) Pearson（安全版）
# ===============================
def pearson_safe(x: pd.Series, y: pd.Series) -> float:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 2:
        return np.nan
    r, _ = pearsonr(tmp.iloc[:, 0].to_numpy(), tmp.iloc[:, 1].to_numpy())
    return float(r)


# ===============================
# 5) 主函数
# ===============================
def main(
    csv_path: str,
    image_dir: str,
    threshold: int = 128,
    output_csv: str = "merged_with_pixel_ratio_b50_b800.csv",
):
    # 自动识别分隔符（你的截图像 ;）
    df = pd.read_csv(csv_path, sep=None, engine="python")

    # 基础列检查（只需要 b50+b800）
    required = ["image_name", "signal_b50", "signal_b800"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}\nGot columns: {list(df.columns)}")

    # 建图像 key -> path
    image_map = {}
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(".png"):
            k = key_from_img_name(fname)
            image_map[k] = os.path.join(image_dir, fname)

    print(f"Found {len(image_map)} PNG files in {image_dir}")

    # 匹配并计算 pixel_ratio
    ratios = []
    missing_match = []

    for n in df["image_name"].astype(str):
        k = key_from_csv_name(n)

        # 直接命中
        if k in image_map:
            ratios.append(compute_white_black_ratio(image_map[k], threshold=threshold))
            continue

        # 兜底：如果 CSV key 是“更短/更长”的版本，做一次模糊匹配（前缀/包含）
        # 这步会稍慢，但 200 张图无所谓
        candidates = [kk for kk in image_map.keys() if kk == k or kk.startswith(k) or k.startswith(kk)]
        if len(candidates) == 1:
            ratios.append(compute_white_black_ratio(image_map[candidates[0]], threshold=threshold))
        else:
            ratios.append(np.nan)
            missing_match.append((n, k, candidates[:5]))

    df["pixel_ratio"] = ratios

    if missing_match:
        print(f"[WARN] Unmatched rows: {len(missing_match)}")
        print("Example (csv_name, csv_key, candidate_keys[:5]):")
        for ex in missing_match[:3]:
            print(ex)

    # 计算相关
    r_b50 = pearson_safe(df["pixel_ratio"], df["signal_b50"])
    r_b800 = pearson_safe(df["pixel_ratio"], df["signal_b800"])

    print("\n===== Final Pearson Correlations (across images) =====")
    print(f"b50   : {r_b50:.6f}")
    print(f"b800  : {r_b800:.6f}")

    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main(
        csv_path="../modified_signals_summary.csv",     # 改成你的csv
        image_dir="../runs/latent_train_unet_new_arch",                   # 改成你的png目录
        threshold=128,                               # 你之前的阈值
        output_csv="../csv/baseline_new_arch_merged_with_pixel_ratio_b50_b800.csv",
    )
