import os
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

# input folder
IMG_DIR = r"./runs/eval_pixel_dual_more_steps"   
# result path
OUT_XLSX = r"./eval_pixel_dual_more_steps.xlsx"

# method 1 threshold
USE_THRESHOLD = True
THRESH = 127   # 0~255 <thresh is black ,>thresh is white

# method 2 exact count
USE_EXACT_0_255 = False

# only pic file
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}

def count_black_white(img_path):
    # gray img
    im = Image.open(img_path).convert("L")
    arr = np.array(im)

    if USE_THRESHOLD:
        black = int(np.sum(arr <= THRESH))
        white = int(np.sum(arr > THRESH))
    elif USE_EXACT_0_255:
        black = int(np.sum(arr == 0))
        white = int(np.sum(arr == 255))
    else:
        # default thres
        black = int(np.sum(arr <= 127))
        white = int(np.sum(arr > 127))

    return black, white

def main():
    img_dir = Path(IMG_DIR)
    if not img_dir.is_dir():
        raise SystemExit(f"can't find folder: {img_dir}")

    rows = []
    for p in sorted(img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in EXTS:
            name = p.name
            if name.lower().endswith(".png") and len(name) > 9:
                name = name[:-9 - 4] + name[-4:] 
            try:
                black, white = count_black_white(p)
                
                rows.append({"filename": name, "black_pixels": black, "white_pixels": white})
            except Exception as e:
                rows.append({"filename": name, "black_pixels": None, "white_pixels": None, "error": str(e)})

    if not rows:
        raise SystemExit("so such file in the folder")

    df = pd.DataFrame(rows)
    # sum
    total_black = df["black_pixels"].fillna(0).sum()
    total_white = df["white_pixels"].fillna(0).sum()
    df.loc[len(df)] = {"filename": "TOTAL", "black_pixels": int(total_black), "white_pixels": int(total_white)}

    # in Excel
    out_path = Path(OUT_XLSX)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(OUT_XLSX).with_suffix(".csv")
    df.to_csv(out_path, index=False)
    print(f"result is saved as CSV: {out_path}")


    print(f"done！in sum {len(df)-1} pics")
    print(f"sum of black pixels: {total_black:,}  sum of white pixels: {total_white:,}")
    print(f"is saved as: {out_path}")

if __name__ == "__main__":
    main()
