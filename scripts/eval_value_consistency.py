import os, glob, argparse, warnings, numpy as np, pandas as pd
from PIL import Image
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

def load_pairs(real_dir, fake_dir, val_csv, img_col="image_name", value_col="signal_b50"):
    df = pd.read_csv(val_csv).dropna(subset=[img_col, value_col])
    df["stem"] = df[img_col].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    # map file stems -> paths
    def map_dir(d):
        if d is None: return {}
        exts = {".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"}
        M={}
        for p in glob.glob(os.path.join(d,"*")):
            if os.path.splitext(p)[1].lower() in exts:
                M[os.path.splitext(os.path.basename(p))[0]] = p
        return M
    R = map_dir(real_dir)
    F = map_dir(fake_dir) if fake_dir else {}
    # keep rows that have real (and optionally fake)
    if fake_dir:
        df = df[df["stem"].isin(R.keys() & F.keys())].copy()
    else:
        df = df[df["stem"].isin(R.keys())].copy()
    df = df.reset_index(drop=True)
    return df, R, F

def pil_to_rgb(p):
    return Image.open(p).convert("RGB")

# --- Feature extractors ---
def try_clip():
    try:
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        vis  = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        vis.eval().to("cuda" if torch.cuda.is_available() else "cpu")
        return proc, vis
    except Exception as e:
        return None, None

def feats_clip(imgs, proc, vis, device):
    with torch.no_grad():
        inputs = proc(images=imgs, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        out = vis(**inputs).image_embeds   # [B, D]
        return out.cpu().numpy()

def feats_fallback(imgs):
    # very simple RGB+hist features as fallback
    feats=[]
    for im in imgs:
        arr = np.array(im.resize((128,128))).astype("float32")/255.0  # [H,W,3]
        rgb_mean = arr.mean(axis=(0,1))
        rgb_std  = arr.std(axis=(0,1))
        # per-channel 16-bin hist
        h=[]
        for c in range(3):
            h.append(np.histogram(arr[...,c], bins=16, range=(0,1), density=True)[0])
        feats.append(np.concatenate([rgb_mean, rgb_std, *h], axis=0))
    return np.stack(feats,0)

def compute_features(paths, batch=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, vis = try_clip()
    X=[]
    if proc is not None:
        # CLIP path
        for i in range(0, len(paths), batch):
            imgs = [pil_to_rgb(p) for p in paths[i:i+batch]]
            X.append(feats_clip(imgs, proc, vis, device))
    else:
        # fallback
        for i in range(0, len(paths), batch):
            imgs = [pil_to_rgb(p) for p in paths[i:i+batch]]
            X.append(feats_fallback(imgs))
    return np.concatenate(X,0)

def evaluate_consistency(real_dir, fake_dir, val_csv, value_col):
    df, R, F = load_pairs(real_dir, fake_dir, val_csv, value_col=value_col)
    if df.empty:
        raise RuntimeError("No matched pairs found. Check filenames and directories.")
    # build arrays
    real_paths = [R[s] for s in df["stem"].tolist()]
    y = df[value_col].to_numpy().astype("float32")
    # split 80/20 on real for sanity check
    n = len(y); idx = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    split = int(0.8*n)
    tr, te = idx[:split], idx[split:]

    # features for real
    X_real = compute_features(real_paths)
    # train ridge on real-train
    reg = Ridge(alpha=1.0)
    reg.fit(X_real[tr], y[tr])
    # sanity on real-test
    y_pred_real = reg.predict(X_real[te])
    r_real = pearsonr(y[te], y_pred_real)[0] if len(te) > 2 else np.nan
    rmse_real = mean_squared_error(y[te], y_pred_real, squared=False)

    results = {
        "n_real": int(n),
        "real_test_pearson_r": float(0.0 if np.isnan(r_real) else r_real),
        "real_test_rmse": float(rmse_real),
        "feat_backend": "CLIP-B/32" if torch.cuda.is_available() and try_clip()[0] is not None else "fallback_hist"
    }

    if fake_dir:
        fake_paths = [F[s] for s in df["stem"].tolist()]
        X_fake = compute_features(fake_paths)
        y_pred_fake = reg.predict(X_fake)
        r_fake_p = pearsonr(y, y_pred_fake)[0] if len(y) > 2 else np.nan
        r_fake_s = spearmanr(y, y_pred_fake)[0] if len(y) > 2 else np.nan
        rmse_fake = mean_squared_error(y, y_pred_fake, squared=False)
        results.update({
            "n_fake": int(len(y)),
            "fake_pearson_r": float(0.0 if np.isnan(r_fake_p) else r_fake_p),
            "fake_spearman_r": float(0.0 if np.isnan(r_fake_s) else r_fake_s),
            "fake_rmse": float(rmse_fake),
        })

    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", default="/workspace/splits/val.csv")
    ap.add_argument("--real_dir", default="/workspace/val_images_rgb")
    ap.add_argument("--fake_dir", default="/workspace/eval_images_rgb")
    ap.add_argument("--value_col", default="signal_b50")
    args = ap.parse_args()

    res = evaluate_consistency(args.real_dir, args.fake_dir, args.val_csv, args.value_col)
    print("\n=== VALUE CONSISTENCY ===")
    for k,v in res.items():
        print(f"{k}: {v}")
