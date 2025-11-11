import os, glob, re, random, time, csv, yaml
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from numeric_encoder import NumericEncoder

# --------- Dataset：解析 b50/b800 + 读图 ----------
class SidecarDataset(Dataset):
    def __init__(self, root, image_size=256, cond_dim=2):
        self.root = root
        self.txts = sorted(glob.glob(os.path.join(root, "*.txt")))
        self.cond_dim = cond_dim
        self.re_b50  = re.compile(r"signal_b50\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")
        self.re_b800 = re.compile(r"signal_b800\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")
        self.tf = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self): return len(self.txts)

    def __getitem__(self, i):
        t = self.txts[i]
        with open(t, "r", encoding="utf-8") as f:
            s = f.read().strip()
        m1 = self.re_b50.search(s); m2 = self.re_b800.search(s)
        if not (m1 and m2):
            raise ValueError(f"Parse fail: {t}")
        b50  = float(m1.group(1).rstrip(",;").replace(",", ""))
        b800 = float(m2.group(1).rstrip(",;").replace(",", ""))
        if self.cond_dim == 2:
            cond = torch.tensor([b50, b800], dtype=torch.float32)
        else:
            # 单变量：默认用 b50（改这里切换 b800）
            cond = torch.tensor([b50], dtype=torch.float32)

        stem = os.path.splitext(t)[0]
        img_path = None
        for ext in [".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"]:
            p = stem + ext
            if os.path.isfile(p): img_path = p; break
        if img_path is None:
            raise FileNotFoundError(f"image not found for {t}")
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        return {"pixel_values": img, "cond": cond}

# --------- 训练主函数 ----------
def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 123))
    random.seed(cfg.get("seed", 123))
    np.random.seed(cfg.get("seed", 123))

    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_loss.tsv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(["step","loss","lr","timestamp"])

    # 1) VAE：复用 SD-1.5 的 VAE（也可以替换成你自己训练的 VAE 路径）
    vae_base = cfg["vae_name"]
    vae = AutoencoderKL.from_pretrained(vae_base, subfolder="vae", torch_dtype=torch.float16).to(device)
    vae.eval()  # 通常固定 VAE（先不训练 VAE）

    # 2) UNet：随机初始化（从配置构建）
    unet_cfg = cfg["unet"]
    unet = UNet2DConditionModel(
        sample_size=unet_cfg["sample_size"],
        in_channels=unet_cfg["in_channels"],
        out_channels=unet_cfg["out_channels"],
        down_block_types=tuple(unet_cfg["down_block_types"]),
        up_block_types=tuple(unet_cfg["up_block_types"]),
        block_out_channels=tuple(unet_cfg["block_out_channels"]),
        layers_per_block=unet_cfg["layers_per_block"],
        cross_attention_dim=unet_cfg["cross_attention_dim"],
        attention_head_dim=unet_cfg["attention_head_dim"],
    ).to(device, dtype=torch.float16)

    # 3) 数值条件编码器（你自己的）
    cond_dim = int(cfg.get("cond_dim", 2))
    hidden_size = int(cfg.get("hidden_size", 768))
    seq_len = int(cfg.get("seq_len", 16))
    enc = NumericEncoder(seq_len=seq_len, hidden_size=hidden_size, cond_dim=cond_dim).to(device)
    enc.train()  # encoder 需要训练

    # 4) 调度器
    sch_cfg = cfg["scheduler"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=sch_cfg["num_train_timesteps"],
        beta_start=sch_cfg["beta_start"],
        beta_end=sch_cfg["beta_end"],
        beta_schedule=sch_cfg["beta_schedule"]
    )

    # 5) 数据
    ds = SidecarDataset(cfg["train_dir"], image_size=int(cfg.get("image_size", 256)), cond_dim=cond_dim)
    # cond 归一化统计（写入 encoder，和你之前一致）
    allc = np.stack([ds[i]["cond"].numpy() for i in range(len(ds))], 0)
    enc.set_norm(allc.mean(0), allc.std(0))

    dl = DataLoader(
        ds, batch_size=int(cfg.get("batch_size", 4)),
        shuffle=True, num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True, drop_last=True
    )

    # 6) 优化器与 LR 调度
    trainable = list(unet.parameters()) + list(enc.parameters())
    opt = torch.optim.AdamW(trainable, lr=float(cfg.get("lr", 1e-4)), weight_decay=1e-2)
    lr_sch = get_cosine_schedule_with_warmup(
        opt, int(cfg.get("warmup", 500)), int(cfg.get("max_steps", 20000))
    )

    # 7) 训练循环
    step = 0
    max_steps = int(cfg.get("max_steps", 20000))
    save_every = int(cfg.get("save_every", 1000))
    log_every  = int(cfg.get("log_every", 50))

    unet.train()
    for epoch in range(999999):
        for batch in dl:
            # a) 准备 latent（fp16）
            with torch.no_grad():
                imgs = batch["pixel_values"].to(device, dtype=torch.float16)
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            bsz = latents.size(0)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device, dtype=torch.long
            )
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # b) 条件 → 文本嵌入
            cond = batch["cond"].to(device)
            text_emb = enc(cond).to(noisy_latents.dtype)  # [B, L, 768] → half

            # c) UNet 预测噪声
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample

            # d) MSE 损失
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            # e) 反传 & 更新
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); lr_sch.step(); step += 1

            # 日志
            if step % log_every == 0:
                print(f"step {step}/{max_steps} | loss {loss.item():.4f}")
                with open(log_path, "a", newline="") as f:
                    csv.writer(f, delimiter="\t").writerow(
                        [step, float(loss.item()), float(lr_sch.get_last_lr()[0]), int(time.time())]
                    )

            # 保存
            if step % save_every == 0:
                torch.save(unet.state_dict(), out_dir / f"unet_step{step}.pt")
                torch.save(enc.state_dict_light(), out_dir / f"numeric_enc_step{step}.pt")

            if step >= max_steps:
                break
        if step >= max_steps:
            break

    # 最终保存
    torch.save(unet.state_dict(), out_dir / "unet_final.pt")
    torch.save(enc.state_dict_light(), out_dir / "numeric_enc_final.pt")
    print(f"[OK] saved UNet+Numeric to {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
