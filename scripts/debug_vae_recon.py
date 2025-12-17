import os
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from diffusers import AutoencoderKL


def load_gray_as_tensor(path: str, image_size: int, device: str):
    tf = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.Lambda(lambda im: im.convert("L")),
        transforms.ToTensor(),                 # [1,H,W] in [0,1]
        transforms.Normalize([0.5], [0.5]),    # -> [-1,1]
    ])
    im = Image.open(path)
    x = tf(im).unsqueeze(0).to(device)        # [1,1,H,W]
    return x


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_repo", required=True, help="Path to your trained VAE (diffusers save_pretrained dir)")
    ap.add_argument("--img", required=True, help="Path to a real mask image (png/jpg/...)")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--n_samples", type=int, default=4, help="How many recon samples to draw from posterior")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load VAE
    vae = AutoencoderKL.from_pretrained(args.vae_repo).to(device)
    vae.eval()

    # 2) Load image -> tensor in [-1,1]
    x = load_gray_as_tensor(args.img, args.image_size, device=device)

    # Save input (to [0,1])
    x_img = (x.clamp(-1, 1) + 1) * 0.5
    save_image(x_img, out_dir / "x_input.png")

    # 3) Encode
    enc_out = vae.encode(x)
    dist = enc_out.latent_dist
    z_mean = dist.mean
    z_logvar = dist.logvar

    # Debug stats
    print("[x] mean/std/min/max:",
          x.mean().item(), x.std().item(), x.min().item(), x.max().item())
    print("[z_mean] mean/std/min/max:",
          z_mean.mean().item(), z_mean.std().item(), z_mean.min().item(), z_mean.max().item())
    print("[z_logvar] mean/std/min/max:",
          z_logvar.mean().item(), z_logvar.std().item(), z_logvar.min().item(), z_logvar.max().item())

    # 4) Decode: use mean (deterministic) + several stochastic samples
    # 4.1 deterministic recon (use mean latent)
    x_rec_det = vae.decode(z_mean).sample
    x_rec_det_img = (x_rec_det.clamp(-1, 1) + 1) * 0.5
    save_image(x_rec_det_img, out_dir / "x_recon_det.png")

    # 4.2 stochastic recons (sample from posterior)
    for k in range(args.n_samples):
        z = dist.sample()
        x_rec = vae.decode(z).sample
        x_rec_img = (x_rec.clamp(-1, 1) + 1) * 0.5
        save_image(x_rec_img, out_dir / f"x_recon_sample{k}.png")

    # 5) Diff map (deterministic)
    diff = (x_img - x_rec_det_img).abs()
    save_image(diff, out_dir / "x_absdiff_det.png")

    print(f"[Saved] {out_dir}/x_input.png")
    print(f"[Saved] {out_dir}/x_recon_det.png")
    print(f"[Saved] {out_dir}/x_recon_sample0..{args.n_samples-1}.png")
    print(f"[Saved] {out_dir}/x_absdiff_det.png")


if __name__ == "__main__":
    main()
