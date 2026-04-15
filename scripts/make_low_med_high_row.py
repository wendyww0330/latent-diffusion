from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# =========================
# 配置：改这里就行
# =========================
ROOT = Path("../dataset_split")  # 现在是 dataset_split 的上层
SEARCH_DIRS = ["train", "test"]  # 在这两个文件夹里找
OUT_STEM = "../plots/dataset_distribution/0305_b_signal_low_med_high"

FILES = {
    "LOW": "binary_labels_mask_case_id_TCGA-BC-A69I_0_0_cell_expansion_9_0_threshold_0_8.txt",
    "MED": "binary_labels_mask_case_id_TCGA-DD-AAED_1_0_cell_expansion_13_0_threshold_0_6.txt",
    "HIGH": "binary_labels_mask_case_id_TCGA-G3-A5SK_3_4_cell_expansion_5_0_threshold_0_4.txt",
}

COLS = ["b50", "b800", "b2000"]  # sidecar 顺序


# =========================
# 工具函数
# =========================
def read_signals(txt_path: Path):
    s = txt_path.read_text(encoding="utf-8").strip()
    parts = [x.strip() for x in re.split(r"[,\s]+", s) if x.strip()]
    vals = [float(x) for x in parts]
    if len(vals) != 3:
        raise ValueError(f"{txt_path}: expected 3 values but got {len(vals)} -> {vals}")
    return dict(zip(COLS, vals))


def find_png_for_txt(txt_path: Path) -> Path:
    png = txt_path.with_suffix(".png")
    if png.exists():
        return png
    raise FileNotFoundError(f"Cannot find png for: {txt_path} (expected {png})")


def find_file_in_splits(root: Path, fname: str, splits=("train", "test")) -> Path:
    """在 dataset_split/train 和 dataset_split/test 里查找目标文件名（精确匹配）。"""
    hits = []
    for sp in splits:
        candidate = root / sp / fname
        if candidate.exists():
            hits.append(candidate)

    if len(hits) == 0:
        raise FileNotFoundError(
            f"Cannot find '{fname}' under: "
            + ", ".join(str(root / sp) for sp in splits)
        )
    if len(hits) > 1:
        raise RuntimeError(
            f"Ambiguous: '{fname}' exists in multiple splits:\n  "
            + "\n  ".join(str(h) for h in hits)
            + "\nPlease disambiguate (e.g., pick train/test explicitly)."
        )
    return hits[0]


# =========================
# 主逻辑
# =========================
items = []
for tag, fname in FILES.items():
    txt_path = find_file_in_splits(ROOT, fname, splits=SEARCH_DIRS)

    png_path = find_png_for_txt(txt_path)
    signals = read_signals(txt_path)

    # 读取图并强制灰度显示，避免 colormap 上色
    img = Image.open(png_path).convert("L")
    arr = np.array(img)

    title = (
        f"{tag}\n"
        f"b50={signals['b50']:.2f},  b800={signals['b800']:.2f},  b2000={signals['b2000']:.2f}"
        f"\n({txt_path.parent.name})"  # 顺手标记来自 train/test，避免论文图可追溯性问题
    )
    items.append((arr, title))

# 画 1×3
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
})

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (arr, title) in zip(axes, items):
    ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, pad=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig(f"{OUT_STEM}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")
print("Saved:", OUT_STEM + ".png")
print("Saved:", OUT_STEM + ".pdf")