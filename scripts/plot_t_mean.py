import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------
# Paths
# ------------------------
root = Path("/workspace/latent-v2")
tsv_path = root / "experiments/loss_correction_epsilon_latent_train_unet_without_dropout/ldm_train.tsv"
out_dir = root / "plots/loss_correction_epsilon_latent_loss_curve"
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------
# Load
# ------------------------
df = pd.read_csv(tsv_path, sep="\t")

required = {"loss", "t_bucket"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in TSV: {missing}. Found: {list(df.columns)}")

# Drop invalid rows (just in case)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["loss", "t_bucket"])

# ------------------------
# Compute log10(loss)
# ------------------------
df = df[df["loss"] > 0].copy()
df["log_loss"] = np.log10(df["loss"])

# ------------------------
# Method A: mean log loss per bucket
# ------------------------
bucket_mean = df.groupby("t_bucket")["log_loss"].mean().sort_index()
bucket_std  = df.groupby("t_bucket")["log_loss"].std().sort_index()
bucket_n    = df.groupby("t_bucket")["log_loss"].size().sort_index()

summary = pd.DataFrame({
    "t_bucket": bucket_mean.index.astype(int),
    "mean_log10_loss": bucket_mean.values,
    "std_log10_loss": bucket_std.values,
    "num_points": bucket_n.values
})

# Save the numeric results
summary_path = out_dir / "t_bucket_mean_log10_loss.csv"
summary.to_csv(summary_path, index=False)

# ------------------------
# Plot
# ------------------------
plt.figure(figsize=(7, 5))
plt.plot(summary["t_bucket"], summary["mean_log10_loss"], marker="o")
plt.xlabel("t_bucket")
plt.ylabel("mean log10(loss)")
plt.title("Average training loss by timestep bucket (epsilon-pred)")
plt.grid(True)
plt.tight_layout()

fig_path = out_dir / "t_bucket_mean_log10_loss.png"
plt.savefig(fig_path, dpi=200)
plt.close()

print("Saved:")
print(f" - {fig_path}")
print(f" - {summary_path}")
