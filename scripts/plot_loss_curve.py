import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==== 路径（按需要改） ====
root = Path("/workspace/latent-v2")

ddpm_log   = root / "experiments/train_ddpm/train_loss.tsv"
latent_log = root / "experiments/train_unet_with_vae/train_loss.tsv"
vae_log    = root / "experiments/train_vae/vae_train.tsv"

out_dir = root / "plots/loss_log"
out_dir.mkdir(parents=True, exist_ok=True)


# ================================
# Helper function to plot log10(loss)
# ================================
def plot_log_curve(df, x_col, y_col, title, out_file):
    # 防止 log(0)
    y = df[y_col].clip(lower=1e-12)

    y_log = np.log10(y)

    plt.figure(figsize=(7, 5))
    plt.plot(df[x_col], y_log)
    plt.title(title + " (log10(loss))")
    plt.xlabel(x_col)
    plt.ylabel("log10(loss)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"[OK] saved log plot to {out_file}")



# ===== 1) VAE loss =====
vae_df = pd.read_csv(vae_log, sep="\t")
plot_log_curve(
    df=vae_df,
    x_col="step",
    y_col="loss",
    title="VAE Training Loss",
    out_file=out_dir / "loss_vae_log.png"
)

# ===== 2) Latent UNet+VAE loss =====
latent_df = pd.read_csv(latent_log, sep="\t")
plot_log_curve(
    df=latent_df,
    x_col="step",
    y_col="loss",
    title="Latent UNet+VAE Training Loss",
    out_file=out_dir / "loss_latent_unet_log.png"
)

# ===== 3) DDPM pixel UNet loss =====
ddpm_df = pd.read_csv(ddpm_log, sep="\t")
plot_log_curve(
    df=ddpm_df,
    x_col="step",
    y_col="loss",
    title="Pixel DDPM Training Loss",
    out_file=out_dir / "loss_ddpm_log.png"
)
