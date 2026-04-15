import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ==============================
# 1. 配置
# ==============================
ROOT = Path("../dataset_split")
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"

COL_NAMES = ["b = 50 s/mm²", "b = 800 s/mm²", "b = 2000 s/mm²"]
BINS = 50
USE_LOG1P = True

OUT_NAME = "../plots/0305_signal_distribution"


# ==============================
# 2. 读取 sidecar txt
# ==============================
def read_folder(folder):
    rows = []
    for p in sorted(folder.glob("*.txt")):
        s = p.read_text().strip()
        parts = [x.strip() for x in re.split(r"[,\s]+", s) if x.strip()]
        values = [float(x) for x in parts]
        rows.append(values)
    return np.array(rows)


train_data = read_folder(TRAIN_DIR)
test_data = read_folder(TEST_DIR)

print("Train samples:", train_data.shape[0])
print("Test samples:", test_data.shape[0])


# ==============================
# 3. 统一 log transform
# ==============================
all_train = train_data.copy()
all_test = test_data.copy()

if USE_LOG1P:
    minv = min(all_train.min(), all_test.min())
    shift = -minv if minv < 0 else 0.0
    all_train = np.log1p(all_train + shift)
    all_test = np.log1p(all_test + shift)


# ==============================
# 4. 计算全局 axis limits
# ==============================
global_min = min(all_train.min(), all_test.min())
global_max = max(all_train.max(), all_test.max())

# 为了计算统一 y 轴，需要先算 histogram
global_ymax = 0
for i in range(3):
    hist_train, _ = np.histogram(all_train[:, i], bins=BINS, range=(global_min, global_max), density=True)
    hist_test, _ = np.histogram(all_test[:, i], bins=BINS, range=(global_min, global_max), density=True)

    global_ymax = max(global_ymax, hist_train.max(), hist_test.max())


# ==============================
# 5. 论文风格设置
# ==============================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.figsize": (12, 4),
    "axes.grid": False
})


# ==============================
# 6. 绘图
# ==============================
fig, axes = plt.subplots(1, 3)

for i, ax in enumerate(axes):

    train_vals = all_train[:, i]
    test_vals = all_test[:, i]

    ax.hist(train_vals, bins=BINS, range=(global_min, global_max),
            alpha=0.5, density=True, label="Train")

    ax.hist(test_vals, bins=BINS, range=(global_min, global_max),
            alpha=0.5, density=True, label="Test")

    ax.set_title(COL_NAMES[i])
    ax.set_xlabel("log1p value" if USE_LOG1P else "value")
    ax.set_ylabel("Density")

    # 统一 axis
    ax.set_xlim(global_min, global_max)
    ax.set_ylim(0, global_ymax * 1.05)

axes[0].legend()

plt.tight_layout()

plt.savefig(f"{OUT_NAME}.png", dpi=300)
plt.savefig(f"{OUT_NAME}.pdf")

print("Saved:", OUT_NAME + ".png")
print("Saved:", OUT_NAME + ".pdf")