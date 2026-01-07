import os
import glob
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


# ======== 1. 路径设置（根据需要改成你自己的） ========

CSV_PATH = "/workspace/latent-v2/inverse_problem_all_signals_summary.csv"

# 21 个目录（自动生成 label）
IMAGE_DIRS = [
    ("/workspace/latent-v2/dataset_split/test", "gt_test"),

    # DDPM
    ("/workspace/latent-v2/runs/ddpm_without_cfg/step10000",  "ddpm_step10000"),
    ("/workspace/latent-v2/runs/ddpm_without_cfg/step20000",  "ddpm_step20000"),
    ("/workspace/latent-v2/runs/ddpm_without_cfg/step30000",  "ddpm_step30000"),
    ("/workspace/latent-v2/runs/ddpm_without_cfg/step40000",  "ddpm_step40000"),
    ("/workspace/latent-v2/runs/ddpm_without_cfg/step50000",  "ddpm_step50000"),
    ("/workspace/latent-v2/runs/ddpm_without_cfg/step60000",  "ddpm_step60000"),
    ("/workspace/latent-v2/runs/ddpm_without_cfg/step70000",  "ddpm_step70000"),
    ("/workspace/latent-v2/runs/ddpm_without_cfg/step80000",  "ddpm_step80000"),
    ("/workspace/latent-v2/runs/ddpm_without_cfg/step90000",  "ddpm_step90000"),
    ("/workspace/latent-v2/runs/ddpm_without_cfg/step100000", "ddpm_step100000"),

    # LATENT LDM
    ("/workspace/latent-v2/runs/latent_without_cfg/step10000",  "latent_step10000"),
    ("/workspace/latent-v2/runs/latent_without_cfg/step20000",  "latent_step20000"),
    ("/workspace/latent-v2/runs/latent_without_cfg/step30000",  "latent_step30000"),
    ("/workspace/latent-v2/runs/latent_without_cfg/step40000",  "latent_step40000"),
    ("/workspace/latent-v2/runs/latent_without_cfg/step50000",  "latent_step50000"),
    ("/workspace/latent-v2/runs/latent_without_cfg/step60000",  "latent_step60000"),
    ("/workspace/latent-v2/runs/latent_without_cfg/step70000",  "latent_step70000"),
    ("/workspace/latent-v2/runs/latent_without_cfg/step80000",  "latent_step80000"),
    ("/workspace/latent-v2/runs/latent_without_cfg/step90000",  "latent_step90000"),
    ("/workspace/latent-v2/runs/latent_without_cfg/step100000", "latent_step100000"),
]


OUT_DIR = "/workspace/latent-v2/plots/ratio_3d_figs"  # 所有图存这里


# ======== 2. 名字匹配：CSV ↔ 图像 ========

def key_from_csv_name(name: str) -> str:
    """
    CSV 里：labels_mask_case_id_XXX_..._threshold_0_7_facet_size_3.0.csv.tif
    -> 变成 key: labels_mask_case_id_XXX_..._threshold_0_7
    """
    base = os.path.basename(name)
    base = base.replace(".csv.tif", "")
    if "_facet_size_" in base:
        base = base.split("_facet_size_")[0]
    return base


def key_from_img_name(name: str) -> str:
    """
    图片：binary_labels_mask_case_id_XXX_..._threshold_0_7.png
        或  labels_mask_case_id_XXX_..._threshold_0_7.png
    -> 去掉 binary_ 前缀、去掉扩展名
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    if stem.startswith("binary_"):
        stem = stem[len("binary_"):]
    return stem


# ======== 3. 计算 white/black ========

def compute_white_black_ratio(img_path, threshold=128):
    im = Image.open(img_path).convert("L")
    arr = np.array(im)
    white = (arr >= threshold).sum()
    black = (arr < threshold).sum()
    if black == 0:
        return np.nan
    return white / float(black)


def main():
    # --- 从 CSV 读取信号 ---
    key_to_signals = {}
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["image_name"]
            k = key_from_csv_name(name)
            try:
                b50 = float(row["signal_b50"])
                b800 = float(row["signal_b800"])
                b2000 = float(row["signal_b2000"])
            except ValueError:
                continue
            key_to_signals[k] = (b50, b800, b2000)
    print(f"[INFO] loaded {len(key_to_signals)} entries from CSV")

    # --- 先扫一遍所有目录，收集每个目录的点，顺便统计全局 color 范围 ---
    dir_points = []  # 每个元素: (label, xs, ys, zs, cs)
    global_min_c, global_max_c = np.inf, -np.inf

    total_imgs = 0
    total_matched = 0
    total_missing = 0

    for img_dir, label in IMAGE_DIRS:
        img_dir = Path(img_dir)
        if not img_dir.exists():
            print(f"[WARN] dir not found: {img_dir}")
            dir_points.append((label, [], [], [], []))
            continue

        pngs = sorted(glob.glob(str(img_dir / "*.png")))
        print(f"[INFO] dir {label} ({img_dir}): {len(pngs)} images")

        xs, ys, zs, cs = [], [], [], []
        for p in pngs:
            total_imgs += 1
            k = key_from_img_name(p)
            if k not in key_to_signals:
                total_missing += 1
                continue

            ratio = compute_white_black_ratio(p)
            if not np.isfinite(ratio):
                continue

            b50, b800, b2000 = key_to_signals[k]
            val = min(ratio * 255.0, 255.0)  # clamp到 0~255

            xs.append(b50)
            ys.append(b800)
            zs.append(b2000)
            cs.append(val)

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        cs = np.array(cs)
        total_matched += len(xs)

        if len(cs) > 0:
            global_min_c = min(global_min_c, cs.min())
            global_max_c = max(global_max_c, cs.max())

        dir_points.append((label, xs, ys, zs, cs))

    print(f"[INFO] total images scanned: {total_imgs}")
    print(f"[INFO] matched points: {total_matched}, missing: {total_missing}")
    print(f"[INFO] global color range: {global_min_c:.2f} ~ {global_max_c:.2f}")

    if not np.isfinite(global_min_c) or not np.isfinite(global_max_c):
        print("[ERROR] no valid points to plot.")
        return

    # --- 为每个目录单独画一张 3D 图 ---
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    for label, xs, ys, zs, cs in dir_points:
        if len(xs) == 0:
            print(f"[INFO] skip {label}: no points")
            continue

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            xs, ys, zs,
            c=cs,
            cmap="viridis",
            s=8,
            vmin=global_min_c,
            vmax=global_max_c,
        )

        ax.set_xlabel("signal_b50")
        ax.set_ylabel("signal_b800")
        ax.set_zlabel("signal_b2000")
        ax.set_title(f"3D scatter ({label})\ncolor = white/black * 255")

        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cb.set_label("white/black * 255")

        plt.tight_layout()
        out_path = out_root / f"ratio_3d_{label}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[SAVE] {out_path}")

    print(f"[DONE] all figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
