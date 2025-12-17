import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===== 参数 =====
INPUT_FILE = "../experiments/new2_train_unet_with_vae/ldm_train.tsv"
OUTPUT_FILE = "../plots/new2_vae_loss_curve_log.png"

# ===== 读取数据 =====
df = pd.read_csv(INPUT_FILE, sep="\t")

# 确保列存在
assert "step" in df.columns and "loss" in df.columns, "TSV must contain 'step' and 'loss' columns."

# ===== 取 log(loss) =====
df["log_loss"] = np.log(df["loss"].clip(lower=1e-12))  
# clip 防止 loss=0 导致 log(0) 问题

# ===== 绘图 =====
plt.figure(figsize=(8, 5))
plt.plot(df["step"], df["log_loss"], linewidth=1.8)

plt.xlabel("Step")
plt.ylabel("log(Loss)")
plt.title("Training Loss Curve (log scale)")
plt.grid(True)

# 保存图像
plt.savefig(OUTPUT_FILE, dpi=200)
plt.close()

print(f"Saved plot to {OUTPUT_FILE}")
