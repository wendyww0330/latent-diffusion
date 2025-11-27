import os, glob, time, yaml, csv, math
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils

from diffusers import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup

class GrayDataset(Dataset):
    def __init__(self, root, image_size=256):
        self.imgs = []
        for ext in ["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp"]:
            self.imgs += glob.glob(os.path.join(root, ext))
        self.imgs = sorted(self.imgs)
        self.tf = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Lambda(lambda im: im.convert("L")),
            transforms.ToTensor(),                             # [1,H,W] in [0,1]
            transforms.Normalize([0.5],[0.5])                 # → [-1,1]
        ])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        x = Image.open(self.imgs[i])
        return self.tf(x)

def kld_loss(mean, logvar):
    # 0.5 * sum( exp(logvar)+mean^2 -1 -logvar )
    return 0.5 * torch.sum(torch.exp(logvar) + mean**2 - 1. - logvar, dim=(1,))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data
    ds = GrayDataset(cfg["data"]["train_dir"], image_size=int(cfg["data"]["image_size"]))
    dl = DataLoader(ds, batch_size=int(cfg["data"]["batch_size"]),
                    shuffle=True, num_workers=int(cfg["data"]["num_workers"]),
                    pin_memory=True, drop_last=True)

    # --- vae (diffusers AutoencoderKL: in/out 设置靠 .config；实际 forward 接 [B,1,H,W])
    # 我们先实例化一个与目标结构兼容的模型，然后随机初始化（因为要自训）。
    # 通过 from_config + 手动改 config 达成 1ch。
    vae = AutoencoderKL(
        in_channels=1, out_channels=1, latent_channels=cfg["model"]["params"]["ddconfig"]["z_channels"],
        down_block_types=("DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"),
        block_out_channels=(128,256,512,512),
        layers_per_block=2
    ).to(device)

    # 记录/输出目录
    out_dir = Path(cfg["optim"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "vae_train.tsv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(["step","recon","kl","loss","lr","ts"])
    
    # 追加写人类可读的日志
    txt_log_path = out_dir / "train_stdout.log"
    def log_line(msg: str):
        """
        同时打印到终端并写入 train_stdout.log（逐行追加）
        格式示例：2025-11-11 19:05:02 | step 100/20000 | rec 0.1070 | kl 0.0017 | loss 0.1070
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} | {msg}"
        print(line, flush=True)
        with open(txt_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


    # 优化器/调度器
    max_steps = int(cfg["optim"]["max_steps"])
    warmup    = int(cfg["optim"]["warmup"])
    opt = torch.optim.AdamW(vae.parameters(), lr=float(cfg["optim"]["lr"]), weight_decay=float(cfg["optim"]["weight_decay"]))
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=max_steps)

    kl_w = float(cfg["model"]["params"]["kl_weight"])
    save_every = int(cfg["optim"]["save_every"])
    log_every  = int(cfg["optim"]["log_every"])

    step = 0
    vae.train()
    while step < max_steps:
        for x in dl:
            x = x.to(device)                    # [B,1,H,W], in [-1,1]
            x = x.to(device)  # [B,1,H,W], in [-1,1]

            # 编码 → 得到分布（AutoencoderKLOutput.latent_dist）
            out  = vae.encode(x)                 # AutoencoderKLOutput
            dist = out.latent_dist               # DiagonalGaussianDistribution
            mean, logvar = dist.mean, dist.logvar  # [B,C,H/8,W/8]
            z = dist.sample()                    # reparameterize sample

            # 解码重建
            x_rec = vae.decode(z).sample         # [-1,1]

            # 重建损失 + KL
            rec = torch.mean((x_rec - x)**2, dim=(1,2,3))  # MSE per sample
            kld = kld_loss(mean, logvar) / (mean.shape[1] * mean.shape[2] * mean.shape[3])  # 归一化
            loss = rec.mean() + kl_w * kld.mean()

            # 反传与优化
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step(); sch.step(); step += 1


            if step % log_every == 0:
                with open(log_path, "a", newline="") as f:
                    csv.writer(f, delimiter="\t").writerow([
                        step, float(rec.mean().item()), float(kld.mean().item()),
                        float(loss.item()), float(sch.get_last_lr()[0]), int(time.time())
                    ])
                log_line(f"step {step}/{max_steps} | rec {rec.mean():.4f} | kl {kld.mean():.4f} | loss {loss.item():.4f}")

            if step % save_every == 0:
                vae.save_pretrained(str(out_dir / f"ckpt_step{step}"))   # diffusers 格式
                # 预览重建
                with torch.no_grad():
                    grid = vutils.make_grid((x_rec.clamp(-1,1)+1)*0.5, nrow=4)
                    vutils.save_image(grid, out_dir / f"recon_step{step}.png")

            if step >= max_steps: break

    vae.save_pretrained(str(out_dir / "final"))
    print(f"[OK] VAE saved to {out_dir}/final")

if __name__ == "__main__":
    main()
