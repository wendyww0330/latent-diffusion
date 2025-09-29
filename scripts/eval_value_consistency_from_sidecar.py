import os, glob, re, warnings, numpy as np
from PIL import Image
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

VAL_DIR = "/workspace/val_data"      # 里头有 .txt + 同名图
FAKE_DIR= "/workspace/eval_val"      # 生成的同名 .png

def read_pairs(val_dir, fake_dir):
    # 读取 val_dir 下的 txt，抽取数值 & 找到真实图/生成图
    txts = sorted(glob.glob(os.path.join(val_dir, "*.txt")))
    pairs = []
    for t in txts:
        stem = os.path.splitext(os.path.basename(t))[0]
        with open(t,"r",encoding="utf-8") as f:
            s = f.read().strip()
        # 支持两种模板："signal_b50=12345" 或 "..., signal_b50_bin=3"
        m = re.search(r"signal_b50\s*=\s*([0-9]+(?:\.[0-9]+)?)", s)
        if m:
            value = float(m.group(1))
        else:
            m2 = re.search(r"signal_b50_bin\s*=\s*([0-9]+)", s)
            if m2: value = float(m2.group(1))
            else: continue

        # 找真实图（同 stem，不限扩展名）
        real_img = None
        for ext in [".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"]:
            p = os.path.join(val_dir, stem+ext)
            if os.path.isfile(p): real_img = p; break
        if real_img is None: continue

        fake_img = os.path.join(fake_dir, stem+".png")
        if not os.path.isfile(fake_img): continue

        pairs.append((real_img, fake_img, value))
    return pairs

def feats_clip(imgs):
    try:
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        vis  = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        X=[]; bs=64
        with torch.no_grad():
            for i in range(0, len(imgs), bs):
                batch = imgs[i:i+bs]
                inputs = proc(images=batch, return_tensors="pt").to(device)
                emb = vis(**inputs).image_embeds  # [B,D]
                X.append(emb.detach().cpu().numpy())
        return np.concatenate(X,0), "CLIP-B/32"
    except Exception:
        # 回退特征（RGB统计+直方图）
        X=[]
        for im in imgs:
            arr = np.array(im.resize((128,128)).convert("RGB")).astype("float32")/255.0
            rgb_mean = arr.mean(axis=(0,1))
            rgb_std  = arr.std(axis=(0,1))
            h=[]
            for c in range(3):
                h.append(np.histogram(arr[...,c], bins=16, range=(0,1), density=True)[0])
            X.append(np.concatenate([rgb_mean, rgb_std, *h], axis=0))
        return np.stack(X,0), "fallback_hist"

def main():
    pairs = read_pairs(VAL_DIR, FAKE_DIR)
    if not pairs:
        raise RuntimeError("No matched pairs found between val_data and eval_val by stem. Check directories.")

    real_paths = [p[0] for p in pairs]
    fake_paths = [p[1] for p in pairs]
    y = np.array([p[2] for p in pairs], dtype=np.float32)

    # 提取特征
    imgs_real = [Image.open(p).convert("RGB") for p in real_paths]
    X_real, backend = feats_clip(imgs_real)

    # 8:2 划分，在真实图上训练/验证回归器（sanity）
    n=len(y); idx=np.arange(n); rng=np.random.RandomState(42); rng.shuffle(idx)
    split=int(0.8*n); tr,te=idx[:split], idx[split:]
    reg = Ridge(alpha=1.0).fit(X_real[tr], y[tr])
    y_pred_real = reg.predict(X_real[te])
    r_real = pearsonr(y[te], y_pred_real)[0] if len(te)>2 else np.nan
    rmse_real = mean_squared_error(y[te], y_pred_real, squared=False)

    # 在生成图上评估一致性
    imgs_fake = [Image.open(p).convert("RGB") for p in fake_paths]
    X_fake,_ = feats_clip(imgs_fake)
    y_pred_fake = reg.predict(X_fake)
    r_fake_p = pearsonr(y, y_pred_fake)[0] if len(y)>2 else np.nan
    r_fake_s = spearmanr(y, y_pred_fake)[0] if len(y)>2 else np.nan
    rmse_fake = mean_squared_error(y, y_pred_fake, squared=False)

    print("\n=== VALUE CONSISTENCY (no CSV) ===")
    print("pairs_used:", n)
    print("feat_backend:", backend)
    print("real_test_pearson_r:", 0.0 if np.isnan(r_real) else float(r_real))
    print("real_test_rmse:", float(rmse_real))
    print("fake_pearson_r:", 0.0 if np.isnan(r_fake_p) else float(r_fake_p))
    print("fake_spearman_r:", 0.0 if np.isnan(r_fake_s) else float(r_fake_s))
    print("fake_rmse:", float(rmse_fake))

if __name__ == "__main__":
    main()
