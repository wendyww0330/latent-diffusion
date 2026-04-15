import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

ROOT = Path("../binary_PNG")   # 改成 test 也行
ROWS, COLS = 4, 5
SEED = 42
OUT_NAME = "../plots/dataset_distribution"

random.seed(SEED)

image_paths = sorted(ROOT.glob("*.png"))
assert len(image_paths) >= ROWS * COLS, f"Not enough images in {ROOT}"

selected = random.sample(image_paths, ROWS * COLS)

fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3, ROWS * 3))

for ax, img_path in zip(axes.flatten(), selected):
    img = Image.open(img_path)

    # 强制转成灰度（L），保证是单通道
    img = img.convert("L")
    arr = np.array(img)

    # 显示为灰度，不用默认 colormap
    ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")

plt.tight_layout()
plt.subplots_adjust(wspace=0.02, hspace=0.02)

plt.savefig(f"{OUT_NAME}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT_NAME}.pdf", bbox_inches="tight")
print("Saved:", OUT_NAME + ".png/.pdf")
