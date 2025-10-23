import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# route setting
tsv_path = "./experiments/numeric_enc_lora_20251022_164834/train_loss1022.tsv"     
out_dir = Path("./plots")
out_dir.mkdir(exist_ok=True)
out_file = out_dir / "train_loss_curve_1022_log.png"


# load tsv
df = pd.read_csv(tsv_path, sep='\t')

# check column
print(df.head())

# draw
plt.figure(figsize=(8, 5))
plt.plot(df["step"], df["loss"], color="blue", linewidth=2)

plt.title("Training Loss Curve (log scale)")
plt.xlabel("Step")
plt.ylabel("Loss (log scale)")

#log y
plt.yscale("log")   

plt.grid(True, which="both", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(out_file, dpi=300)
print(f"plot is saved to: {out_file.resolve()}")