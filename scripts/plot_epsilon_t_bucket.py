import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root = Path("/workspace/latent-v2")

df = pd.read_csv(
    root / "experiments/loss_correction_epsilon_latent_train_unet_without_dropout/ldm_train.tsv",
    sep="\t"
)

out_dir = root / "plots/loss_correction_epsilon_latent_train_unet_without_dropout"
out_dir.mkdir(parents=True, exist_ok=True)

# 安全检查
if "t_bucket" not in df.columns:
    raise ValueError("t_bucket column not found in TSV.")

# log10 loss
df["log_loss"] = np.log10(df["loss"])

# ===============================
# 1️⃣ 画分桶平滑曲线
# ===============================
plt.figure(figsize=(8,6))

bucket_stats = []

for b in sorted(df["t_bucket"].unique()):
    sub = df[df["t_bucket"] == b].sort_values("step")

    # rolling window（你每50 step记录一次 → 40 ≈ 2000 training steps）
    smooth = sub["log_loss"].rolling(40, min_periods=1).mean()

    plt.plot(sub["step"], smooth, label=f"t_bucket={b}")

    # 保存统计信息
    bucket_stats.append({
        "t_bucket": b,
        "mean_log_loss": sub["log_loss"].mean(),
        "std_log_loss": sub["log_loss"].std(),
        "num_points": len(sub)
    })

plt.xlabel("step")
plt.ylabel("log10(loss)")
plt.title("Training loss by timestep bucket (epsilon-pred)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(out_dir / "loss_by_t_bucket.png", dpi=200)
plt.close()

# ===============================
# 2️⃣ 保存分桶统计结果
# ===============================
stats_df = pd.DataFrame(bucket_stats)
stats_df.to_csv(out_dir / "t_bucket_statistics.csv", index=False)

print("Saved:")
print(" - loss_by_t_bucket.png")
print(" - t_bucket_statistics.csv")
