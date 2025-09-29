import os, glob, argparse, numpy as np
from PIL import Image
import torch, lpips
from skimage.metrics import structural_similarity as ssim
from math import log10

def to_t(img):
    arr = np.array(img).astype("float32")/255.0
    if arr.ndim == 2:  # 灰度转RGB
        arr = np.stack([arr]*3, axis=-1)
    t = torch.from_numpy(arr).permute(2,0,1)  # [3,H,W]
    return t*2-1  # LPIPS 期望 [-1,1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True, help="真实图所在目录（与你的 val_data/test_data 一样）")
    ap.add_argument("--fake_dir", required=True, help="模型生成图目录（文件名需与真实图同stem）")
    args = ap.parse_args()

    # 真实图按 stem（不含后缀）收集
    real_map = {}
    for p in glob.glob(os.path.join(args.real_dir, "*")):
        stem, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() in [".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"]:
            real_map[stem] = p

    fake_list = sorted(glob.glob(os.path.join(args.fake_dir, "*.png")))
    assert fake_list, f"No generated PNGs in {args.fake_dir}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net='alex').to(device).eval()

    lpips_list, ssim_list, psnr_list, used = [], [], [], 0
    for f in fake_list:
        stem = os.path.splitext(os.path.basename(f))[0]
        r = real_map.get(stem)
        if not r: 
            continue
        imR = Image.open(r).convert("RGB")
        imF = Image.open(f).convert("RGB")
        if imR.size != imF.size:
            imF = imF.resize(imR.size, resample=Image.BICUBIC)

        # LPIPS
        R = to_t(imR).unsqueeze(0).to(device)
        F = to_t(imF).unsqueeze(0).to(device)
        lp = float(loss_fn(R, F).item()); lpips_list.append(lp)

        # SSIM/PSNR 在 [0,255] 上
        R8 = np.array(imR).astype(np.float32)
        F8 = np.array(imF).astype(np.float32)
        s = ssim(R8, F8, channel_axis=2, data_range=255.0); ssim_list.append(float(s))
        mse = np.mean((R8 - F8)**2); ps = 10*log10((255.0**2)/(mse+1e-8)); psnr_list.append(float(ps))
        used += 1

    print("\n=== RESULTS ===")
    print("Pairs matched:", used)
    if used == 0:
        print("No matching stems between real_dir and fake_dir. Check filenames.")
        return
    print(f"LPIPS (lower better): {np.mean(lpips_list):.4f}")
    print(f"SSIM  (higher better): {np.mean(ssim_list):.4f}")
    print(f"PSNR dB (higher better): {np.mean(psnr_list):.2f}")

if __name__ == "__main__":
    main()

