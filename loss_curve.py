# plot_five_losses.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# pair
pairs = [
    ("./experiments/pixel_unet_numeric_more_steps/train_loss.tsv", "./plots/scratch_more_steps.png"),
]

Path("./plots").mkdir(parents=True, exist_ok=True)

for tsv_path, out_png in pairs:
    # input
    df = pd.read_csv(tsv_path, sep="\t")

    # column
    cols = {c.lower(): c for c in df.columns}
    step_col = cols.get("step")
    loss_col = cols.get("loss")
    if step_col is None or loss_col is None:
        raise ValueError(f"{tsv_path} 缺少 step/loss 列")

    # value
    df = df[[step_col, loss_col]].dropna()
    df[step_col] = pd.to_numeric(df[step_col], errors="coerce")
    df[loss_col] = pd.to_numeric(df[loss_col], errors="coerce")
    df = df.dropna()

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(df[step_col], df[loss_col], linewidth=2)
    plt.title(f"{Path(tsv_path).parts[-2]} — Training Loss (log scale)")
    plt.xlabel("Step")
    plt.ylabel("Loss (log)")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"saved: {out_png}")
