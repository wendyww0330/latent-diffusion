import os
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="inverse_problem_all_signals_summary.csv",
        help="summary CSV 文件路径",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="binary_PNG",
        help="PNG 图片所在文件夹",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="dataset_split",
        help="输出数据集根目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（保证可复现划分）",
    )
    return parser.parse_args()


def csv_name_to_img_name(csv_name: str) -> str:
    """
    把表格里的名字转换成 PNG 图片名。

    例子：
    labels_mask_case_id_TCGA-BC-A69I_0_0_cell_expansion_3_0_threshold_0_7_facet_size_3.0.csv.tif
    -> binary_labels_mask_case_id_TCGA-BC-A69I_0_0_cell_expansion_3_0_threshold_0_7.png
    """

    # 先去掉 .csv.tif 后缀
    if csv_name.endswith(".csv.tif"):
        base = csv_name[:-len(".csv.tif")]
    else:
        base = Path(csv_name).stem  # 兜底

    # 去掉后面的 _facet_size_... 部分（facet_size 可能不是 3.0，所以用 split）
    base = base.split("_facet_size")[0]

    # 最后加上 binary_ 前缀和 .png 后缀
    img_name = "binary_" + base + ".png"
    return img_name


def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    out_train = out_dir / "train"
    out_test = out_dir / "test"
    for d in [out_train, out_test]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. 读取 CSV
    df = pd.read_csv(csv_path)

    n = len(df)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n)

    # 2. 9:1 划分 train / test
    n_train = int(0.9 * n)
    idx_train = indices[:n_train]
    idx_test = indices[n_train:]

    splits = {
        "train": idx_train,
        "test": idx_test,
    }

    print(f"总样本数: {n}")
    print(f"train: {len(idx_train)}, test: {len(idx_test)}")

    missing_images = 0

    for split_name, idxs in splits.items():
        split_dir = out_dir / split_name

        for i in idxs:
            row = df.iloc[i]
            csv_name = str(row["image_name"])

            # ⭐ 用新的规则生成图片名
            img_name = csv_name_to_img_name(csv_name)

            src_img = images_dir / img_name

            if not src_img.exists():
                print(f"[警告] 找不到图片: {src_img}")
                missing_images += 1
                continue

            # 复制图片到 train/test 子文件夹
            dst_img = split_dir / img_name
            shutil.copy2(src_img, dst_img)

            # sidecar 文本：与图片同名 .txt
            sidecar_path = split_dir / (Path(img_name).stem + ".txt")

            b50 = float(row["signal_b50"])
            b800 = float(row["signal_b800"])
            b2000 = float(row["signal_b2000"])

            # 形如：24090.05, 12296.88, 6487.89
            text_line = f"{b50:.2f}, {b800:.2f}, {b2000:.2f}\n"

            with open(sidecar_path, "w", encoding="utf-8") as f:
                f.write(text_line)

    print("完成划分。")
    if missing_images > 0:
        print(f"有 {missing_images} 张图片在 {images_dir} 中未找到，请检查命名是否一致。")


if __name__ == "__main__":
    main()
