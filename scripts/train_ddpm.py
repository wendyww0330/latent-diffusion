import os, re, glob, csv, time, yaml, random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from numeric_encoder import NumericEncoder

class SidecarDataset(Dataset):
    def __init__(self, root, image_size=256):
        self.txts = sorted(glob.glob(os.path.join(root, "*.txt")))
        self.re_b50  = re.compile(r"signal_b50\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")
        self.re_b800 = re.compile(r"signal_b800\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")
        self.tf = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Lambda(lambda im: im.convert("L")),             # binary
            transforms.ToTensor(),                                     # [1,H,W], 0..1
            transforms.Normalize([0.5], [0.5]),                        # -> [-1,1]
        ])

    def __len__(self): return len(self.txts)

    def __getitem__(self, i):
        t = self.txts[i]
        with open(t, "r", encoding="utf-8") as f:
            s = f.read().strip()
        def extr(regex):
            m = regex.search(s)
            if not m: raise ValueError(f"parse fail: {t}")
            return float(m.group(1).rstrip(",;").replace(",", ""))
        b50, b800 = extr(self.re_b50), extr(self.re_b800)
        stem = os.path.splitext(t)[0]
        img_path = None
        for ext in [".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"]:
            p = stem + ext
            if os.path.isfile(p): img_path = p; break
        if img_path is None:
            raise FileNotFoundError(f"image not found for {t}")
        im = Image.open(img_path)
        im = self.tf(im)                                               # [1,H,W]
        cond = torch.tensor([b50, b800], dtype=torch.float32)          # [2]
        return {"pixel_values": im, "cond": cond}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 123))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_loss.tsv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(["step","loss","lr","ts"])

    #  UNet pixel one channel FP32 
    arch = cfg["unet"]
    assert arch["sample_size"] == cfg["image_size"], "unet.sample_size = image_size"
    unet = UNet2DConditionModel(
        sample_size=arch["sample_size"],
        in_channels=arch["in_channels"],       # 1
        out_channels=arch["out_channels"],     # 1
        down_block_types=tuple(arch["down_block_types"]),
        up_block_types=tuple(arch["up_block_types"]),
        block_out_channels=tuple(arch["block_out_channels"]),
        layers_per_block=arch["layers_per_block"],
        cross_attention_dim=arch["cross_attention_dim"],
        attention_head_dim=arch["attention_head_dim"],
    ).to(device, dtype=torch.float32)
    unet.train()

    # NumericEncoder
    cond_dim = int(cfg.get("cond_dim", 2))
    hidden   = int(cfg.get("hidden_size", 768))
    seq_len  = int(cfg.get("seq_len", 16))
    enc = NumericEncoder(seq_len=seq_len, hidden_size=hidden, cond_dim=cond_dim).to(device).float()
    enc.train()

    #Scheduler DDPM
    sch = cfg["scheduler"]
    scheduler = DDPMScheduler(
        num_train_timesteps=sch["num_train_timesteps"],
        beta_start=sch["beta_start"], beta_end=sch["beta_end"],
        beta_schedule=sch["beta_schedule"]
    )

    # data
    ds = SidecarDataset(cfg["train_dir"], image_size=int(cfg.get("image_size", 256)))
    # cond 
    allc = np.stack([ds[i]["cond"].numpy() for i in range(len(ds))], 0)
    mean = allc.mean(0); std = np.clip(allc.std(0), 1e-6, None)
    enc.set_norm(mean, std)

    dl = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=True, num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True, drop_last=True
    )

    # optimization
    lr = float(cfg.get("lr", 5e-5))
    max_steps = int(cfg.get("max_steps", 20000))
    warmup    = int(cfg.get("warmup", 500))
    save_every= int(cfg.get("save_every", 1000))
    log_every = int(cfg.get("log_every", 50))

    params = list(unet.parameters()) + list(enc.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sch_lr = get_cosine_schedule_with_warmup(opt, warmup, max_steps)

    step = 0
    while step < max_steps:
        for batch in dl:
            # add noise from [-1,1] 
            imgs = batch["pixel_values"].to(device, dtype=torch.float32)     # [B,1,H,W]
            bsz  = imgs.size(0)

            timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(imgs, dtype=torch.float32)
            noisy = scheduler.add_noise(imgs, noise, timesteps)

            cond = batch["cond"].to(device).float()
            text_emb = enc(cond).to(dtype=torch.float32)                      # [B,L,768]

            noise_pred = unet(noisy, timesteps, encoder_hidden_states=text_emb).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            if not torch.isfinite(loss):
                print("[WARN] NaN/Inf loss — skip step")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sch_lr.step(); step += 1

            if step % log_every == 0:
                with open(log_path, "a", newline="") as f:
                    csv.writer(f, delimiter="\t").writerow([step, float(loss.item()), float(sch_lr.get_last_lr()[0]), int(time.time())])
                print(f"step {step}/{max_steps} | loss {loss.item():.4f}")

            if step % save_every == 0:
                torch.save(unet.state_dict(), out_dir / f"unet_step{step}.pt")
                torch.save(enc.state_dict_light(), out_dir / f"numeric_enc_step{step}.pt")

            if step >= max_steps: break

    torch.save(unet.state_dict(), out_dir / "unet_final.pt")
    torch.save(enc.state_dict_light(), out_dir / "numeric_enc_final.pt")
    print(f"[OK] saved to {out_dir}")

if __name__ == "__main__":
    main()
