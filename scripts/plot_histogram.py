import os
import glob
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compute_white_black_ratio(img_path, threshold=128):
    """
    计算白/黑像素比例: white_pixels / black_pixels
    如果 black=0，则返回 np.nan
    """
    im = Image.open(img_path).convert("L")
    arr = np.array(im)

    white = (arr >= threshold).sum()
    black = (arr < threshold).sum()

    if black == 0:
        return np.nan  # 或者你想设成 white/1 ?
    return white / float(black)

def compute_white_ratio(img_path, threshold=128):
    """
    计算单张二值图中白色像素占比（0~1）
    """
    im = Image.open(img_path).convert("L")   # 灰度
    arr = np.array(im)

    total = arr.size
    white = (arr >= threshold).sum()        # >=128 视作白
    ratio = white / float(total)
    return ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True,
                    help="含有 test ground truth 图像的文件夹")
    ap.add_argument("--out", default="bw_ratio_hist.png",
                    help="直方图保存路径")
    args = ap.parse_args()

    img_dir = Path(args.image_dir)

    # 支持常见几种图片后缀
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    img_paths = []
    for e in exts:
        img_paths.extend(glob.glob(str(img_dir / e)))
    img_paths = sorted(img_paths)

    if not img_paths:
        raise RuntimeError(f"在 {img_dir} 下没有找到图片")

    ratios = []
    for p in img_paths:
        r = compute_white_black_ratio(p)
        ratios.append(r)

    ratios = np.array(ratios)
    print(f"共 {len(ratios)} 张图片")
    print(f"ratio 最小值: {ratios.min():.4f}, 最大值: {ratios.max():.4f}, 均值: {ratios.mean():.4f}")

    # 0~1，每 0.1 一个 bin
    bins = np.linspace(0.0, 1.0, 11)  # 0,0.1,...,1.0
    counts, edges = np.histogram(ratios, bins=bins)

    # 打印各区间统计
    print("区间\t\t数量")
    for i in range(len(counts)):
        left, right = edges[i], edges[i+1]
        print(f"[{left:.1f}, {right:.1f})\t{counts[i]}")

    # 画柱状图
    centers = 0.5 * (edges[:-1] + edges[1:])  # bin 中心点
    width = edges[1] - edges[0]

    plt.figure(figsize=(8, 4))
    plt.bar(centers, counts, align="center", width=width * 0.9)
    plt.xlabel("white pixel ratio (white / total)")
    plt.ylabel("number of images")
    plt.title("Histogram of white pixel ratio (ground truth)")
    plt.xticks(np.linspace(0.0, 1.0, 11))
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"[SAVE] 直方图保存到 {args.out}")


if __name__ == "__main__":
    main()
