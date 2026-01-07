import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# para
pixel_csv = "./latent_train_unet_with_vae_withoutcfg_withoutdropout.csv"      # out path
signal_csv = "modified_signals_summary.csv"    #  b50/b800  file
out_dir = Path("./plots")           # output folder
out_dir.mkdir(exist_ok=True)        # create a new one
out_file = out_dir / "latent_train_unet_with_vae_withoutcfg_withoutdropout.png"

# read csv
df_pixel = pd.read_csv(pixel_csv)
df_signal = pd.read_csv(signal_csv, sep=';')  # seperate ;

# get file name
df_signal['image_name'] = df_signal['image_name'].apply(lambda x: x.split('/')[-1])

# merge datasets
df = pd.merge(
    df_signal,
    df_pixel,
    how='inner',
    left_on='image_name',
    right_on='filename'
)

# calculate the ratio value (white/black * 255)
df['color_value'] = (df['white_pixels'] / df['black_pixels']).fillna(0) * 255
df['color_value'] = df['color_value'].clip(0, 255)

# plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(
    df['signal_b50'],
    df['signal_b800'],
    c=df['color_value'],
    cmap='viridis',
    edgecolor='none',
    alpha=0.9
)
plt.colorbar(sc, label='(white / black) × 255')
plt.xlabel('signal_b50')
plt.ylabel('signal_b800')
plt.title('Pixel ratio visualized by color intensity')

plt.tight_layout()

# save
plt.savefig(out_file, dpi=300)
print(f"result is saved to: {out_file.resolve()}")
