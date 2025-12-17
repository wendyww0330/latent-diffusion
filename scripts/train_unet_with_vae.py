import os
import re
import glob
import csv
import time
import yaml
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# 假设你的 NumericEncoder 定义在 numeric_encoder.py 中
from numeric_encoder import NumericEncoder


# ====================== Dataset ======================

class SidecarDataset(Dataset):
    """
    读取图像和对应的 .txt 文件 (包含 signal_b50, signal_b800)
    """
    def __init__(self, root, image_size=256):
        self.root = root
        self.txts = sorted(glob.glob(os.path.join(root, "*.txt")))

        # 图像预处理：必须与 VAE 训练保持一致
        self.tf = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Lambda(lambda im: im.convert("L")),  # 灰度
            transforms.ToTensor(),                         # [1,H,W] in [0,1]
            transforms.Normalize([0.5], [0.5])             # → [-1,1]
        ])

        _NUM_RE = r"([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)"
        self.re_b50  = re.compile(rf"signal_b50\s*=\s*{_NUM_RE}\s*[,;]?")
        self.re_b800 = re.compile(rf"signal_b800\s*=\s*{_NUM_RE}\s*[,;]?")

    def __len__(self):
        return len(self.txts)

    def _parse_pair(self, text: str):
        m1, m2 = self.re_b50.search(text), self.re_b800.search(text)
        if not (m1 and m2):
            return None, None

        def _to_float(m):
            return float(m.group(1).rstrip(",;").replace(",", ""))

        try:
            return _to_float(m1), _to_float(m2)
        except Exception:
            return None, None

    def __getitem__(self, i):
        tpath = self.txts[i]
        try:
            with open(tpath, "r", encoding="utf-8") as f:
                s = f.read().strip()
            b50, b800 = self._parse_pair(s)
            
            if b50 is None:
                print(f"[WARN] Parse fail: {tpath}")
                # 简单容错：如果解析失败，随机给个数据防止崩坏，或者抛出异常
                b50, b800 = 0.0, 0.0 

            # 找对应图像
            stem = os.path.splitext(tpath)[0]
            img_path = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
                p = stem + ext
                if os.path.isfile(p):
                    img_path = p
                    break
            
            if img_path is None:
                raise FileNotFoundError(f"Image not found for {tpath}")

            im = Image.open(img_path)
            im = self.tf(im)   # [1,H,W] in [-1,1]

            cond = torch.tensor([b50, b800], dtype=torch.float32)  # [2]
            return {"pixel_values": im, "cond": cond}
            
        except Exception as e:
            print(f"[Error loading {tpath}]: {e}")
            # 返回随机数据避免 DataLoader 崩溃 (可选)
            return {"pixel_values": torch.zeros(1, 256, 256), "cond": torch.tensor([0.0, 0.0])}


# ====================== 主训练逻辑 ======================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config for LDM")
    args = ap.parse_args()

    # ---------- 1. 读取配置 ----------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = int(cfg.get("seed", 123))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件
    log_path = out_dir / "ldm_train.tsv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(["step", "loss", "lr", "ts"])

    txt_log_path = out_dir / "train_stdout.log"
    def log_line(msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} | {msg}"
        print(line, flush=True)
        with open(txt_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    image_size = int(cfg.get("image_size", 256))

    


    # ---------- 2. 载入 VAE ----------
    vae_repo = cfg["vae_repo"]
    # 自动加载 VAE (只要你的 config.json 里写了 latent_channels=4，这里就会自动适配)
    vae = AutoencoderKL.from_pretrained(vae_repo).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    
    # ★关键修改1：设置缩放因子 (配合 z_channels=4 使用)
    SCALING_FACTOR = float(cfg.get("scaling_factor", 1.0))


    log_line(f"[VAE] Loaded from {vae_repo}. scaling_factor={SCALING_FACTOR}")

    log_line(f"[VAE] Loaded from {vae_repo}. using SCALING_FACTOR = {SCALING_FACTOR}")

    # ---------- 3. 构建 UNet ----------
    arch = cfg["unet"]
    unet = UNet2DConditionModel(
        sample_size=arch["sample_size"],           # 32
        in_channels=arch["in_channels"],           # ★ 请确保 YAML 里改成了 4
        out_channels=arch["out_channels"],         # ★ 请确保 YAML 里改成了 4
        down_block_types=tuple(arch["down_block_types"]),
        up_block_types=tuple(arch["up_block_types"]),
        block_out_channels=tuple(arch["block_out_channels"]),
        layers_per_block=arch["layers_per_block"],
        cross_attention_dim=arch["cross_attention_dim"],
        attention_head_dim=arch["attention_head_dim"],
    ).to(device, dtype=torch.float32)
    unet.train()
    
    log_line(f"[UNet] Configured: in={arch['in_channels']}, out={arch['out_channels']}")

    # ---------- 4. NumericEncoder ----------
    cond_dim  = int(cfg.get("cond_dim", 2))
    hidden    = int(cfg.get("hidden_size", 768))
    seq_len   = int(cfg.get("seq_len", 16))

    enc = NumericEncoder(
        seq_len=seq_len,
        hidden_size=hidden,
        cond_dim=cond_dim
    ).to(device).float()
    enc.train()

    # ---------- 5. Scheduler ----------
    sch_cfg = cfg["scheduler"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=sch_cfg["num_train_timesteps"],
        beta_start=sch_cfg["beta_start"],
        beta_end=sch_cfg["beta_end"],
        beta_schedule=sch_cfg["beta_schedule"]
    )
    print("prediction_type:", getattr(noise_scheduler.config, "prediction_type", None))
    # ---------- 6. 数据准备 ----------
    ds = SidecarDataset(cfg["train_dir"], image_size=image_size)
    dl = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True
    )
    log_line(f"[DATA] N={len(ds)} images")

    # 计算 Dataset 统计信息并注入 Encoder
    # 这样训练时模型看到的 cond 是归一化后的 (mean=0, std=1)
    allc = []
    # 为了速度，只采样一部分数据计算 mean/std，或者全部
    print("Calculating dataset stats...")
    # 这里简单起见，从 dataset 里取前 1000 个（或者全部）来算
    check_len = min(len(ds), 2000)
    for i in range(check_len):
        allc.append(ds[i]["cond"].numpy())
    allc = np.stack(allc, 0)
    
    cond_mean = allc.mean(0)
    cond_std  = np.clip(allc.std(0), 1e-6, None)
    
    if hasattr(enc, "set_norm"):
        enc.set_norm(cond_mean, cond_std)

        log_line(f"[ENC] Set Norm: mean={cond_mean}, std={cond_std}")
        np.save(out_dir / "cond_mean.npy", cond_mean)
        np.save(out_dir / "cond_std.npy",  cond_std)
        log_line(f"[ENC] Saved cond stats to {out_dir}/cond_mean.npy and cond_std.npy")

        # ★ 提醒：请把这两个打印出来的数组复制到 inference.py 中！

    # ---------- 7. 优化器配置 ----------
    lr         = float(cfg.get("lr", 1e-4))
    max_steps  = int(cfg.get("max_steps", 20000))
    warmup     = int(cfg.get("warmup", 500))
    save_every = int(cfg.get("save_every", 2000))
    log_every  = int(cfg.get("log_every", 50))
    
    # ★关键修改2：Dropout 概率 (推荐 0.1)
    cond_dropout_prob = 0.0
    log_line(f"[Config] CFG Dropout Prob = {cond_dropout_prob}")

    params = list(unet.parameters()) + list(enc.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sch = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=warmup, num_training_steps=max_steps
    )

    # ---------- 8. 训练循环 ----------
    step = 0
    log_line(f"[VAE] repo={vae_repo}")
    log_line(f"[VAE] scaling_factor={float(getattr(vae.config,'scaling_factor',1.0))}")

    while step < max_steps:

        if step == 50:
            lat_mean = latents.mean().item()   # latents 是 scaled 后的
            np.save(out_dir / "latent_mean.npy", np.array([lat_mean], dtype=np.float32))
            log_line(f"[LATENTS] Saved latent_mean={lat_mean} to {out_dir}/latent_mean.npy")

        for batch in dl:
            imgs = batch["pixel_values"].to(device, dtype=torch.float32)
            cond = batch["cond"].to(device, dtype=torch.float32)
            bsz = imgs.size(0)

            # --- A) VAE Encode ---
            with torch.no_grad():
                z_raw = vae.encode(imgs).latent_dist.sample()
                latents = z_raw * SCALING_FACTOR


            # --- B) Encoder & CFG Dropout ---
            # 1. 先通过 encoder 得到 embedding
            cond_emb = enc(cond).to(dtype=torch.float32) # [B, L, H]
            
            # 2. 应用 Dropout (Masking)
            # 如果随机数小于 0.1，就把整个 embedding 抹零
            # 这对应了推理时 uncond_emb = torch.zeros_like(...)
            cond_dropout_prob=0.0
            if cond_dropout_prob > 0:
                mask = (torch.rand(bsz, device=device) < cond_dropout_prob).float()  # [B]
                cond_emb = cond_emb * (1.0 - mask)[:, None, None]  # broadcast to [B,L,H]


            # --- C) 加噪 ---
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device, dtype=torch.long
            )
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # --- D) 预测 ---
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=cond_emb
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # --- E) 反传 ---
            if not torch.isfinite(loss):
                print("Loss is NaN, skipping...")
                opt.zero_grad()
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sch.step()
            step += 1


            if step % 200 == 0:
                log_line(f"[LATENTS] mean={latents.mean().item():.4f} std={latents.std().item():.4f} "
                        f"min={latents.min().item():.4f} max={latents.max().item():.4f}")

            # --- Log ---
            if step % log_every == 0:
                lr_now = sch.get_last_lr()[0]
                with open(log_path, "a", newline="") as f:
                    csv.writer(f, delimiter="\t").writerow([step, loss.item(), lr_now, int(time.time())])
                print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | LR: {lr_now:.2e}")

            # --- Save ---
            if step % save_every == 0:
                torch.save(unet.state_dict(), out_dir / f"unet_step{step}.pt")
                # 保存 Encoder (兼容 light 模式)
                if hasattr(enc, "state_dict_light"):
                    torch.save(enc.state_dict_light(), out_dir / f"numeric_enc_step{step}.pt")
                else:
                    torch.save(enc.state_dict(), out_dir / f"numeric_enc_step{step}.pt")
                log_line(f"Saved checkpoint at step {step}")

            if step >= max_steps:
                break

    # ---------- Final Save ----------
    torch.save(unet.state_dict(), out_dir / "unet_final.pt")
    if hasattr(enc, "state_dict_light"):
        torch.save(enc.state_dict_light(), out_dir / "numeric_enc_final.pt")
    else:
        torch.save(enc.state_dict(), out_dir / "numeric_enc_final.pt")
    log_line("Training Finished. Saved final models.")

if __name__ == "__main__":
    main()