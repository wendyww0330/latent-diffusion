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

from numeric_encoder import NumericEncoder


# ====================== Dataset ======================

class SidecarDataset(Dataset):
    """
    Read grayscale image + sidecar txt containing signal_b50 / signal_b800.

    Expected paired files:
      xxx.txt
      xxx.png (or jpg/jpeg/bmp/tif/tiff/webp)
    """
    def __init__(self, root, image_size=256):
        self.root = root
        self.txts = sorted(glob.glob(os.path.join(root, "*.txt")))

        self.tf = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Lambda(lambda im: im.convert("L")),
            transforms.ToTensor(),                         # [1,H,W] in [0,1]
            transforms.Normalize([0.5], [0.5]),            # -> [-1,1]
        ])

        _NUM_RE = r"([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)"
        self.re_b50  = re.compile(rf"signal_b50\s*=\s*{_NUM_RE}\s*[,;]?")
        self.re_b800 = re.compile(rf"signal_b800\s*=\s*{_NUM_RE}\s*[,;]?")

        self._img_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]

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
                # Fail-safe: keep training running, but log the file
                print(f"[WARN] Parse fail: {tpath}")
                b50, b800 = 0.0, 0.0

            stem = os.path.splitext(tpath)[0]
            img_path = None
            for ext in self._img_exts:
                p = stem + ext
                if os.path.isfile(p):
                    img_path = p
                    break
            if img_path is None:
                raise FileNotFoundError(f"Image not found for {tpath}")

            im = Image.open(img_path)
            im = self.tf(im)  # [1,H,W] in [-1,1]

            cond = torch.tensor([b50, b800], dtype=torch.float32)  # [2]
            return {"pixel_values": im, "cond": cond}

        except Exception as e:
            print(f"[Error loading {tpath}]: {e}")
            # Fail-safe sample to avoid DataLoader crash
            return {
                "pixel_values": torch.zeros(1, 256, 256, dtype=torch.float32),
                "cond": torch.tensor([0.0, 0.0], dtype=torch.float32),
            }


# ====================== Main ======================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config for latent diffusion training")
    args = ap.parse_args()

    # ---------- 1) Load config ----------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    seed = int(cfg.get("seed", 123))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.set_float32_matmul_precision("high")

    # Output dirs
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # ---------- 2) Read training hyperparams from YAML ----------
    image_size = int(cfg.get("image_size", 256))
    train_dir = cfg["train_dir"]

    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 4))

    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-2))

    max_steps = int(cfg.get("max_steps", 20000))
    save_every = int(cfg.get("save_every", 2000))
    log_every = int(cfg.get("log_every", 50))

    cond_dropout_prob = float(cfg.get("cond_dropout_prob", 0.0))

    # Latent scaling factor (must match inference)
    SCALING_FACTOR = float(cfg.get("scaling_factor", 1.0))

    # Prediction type: epsilon / sample / v_prediction
    pred_type = str(cfg.get("prediction_type", "epsilon"))

    log_line(f"[Config] image_size={image_size} batch_size={batch_size} num_workers={num_workers}")
    log_line(f"[Config] lr(fixed)={lr} weight_decay={weight_decay} max_steps={max_steps}")
    log_line(f"[Config] cond_dropout_prob={cond_dropout_prob}")
    log_line(f"[Config] scaling_factor(yaml)={SCALING_FACTOR}")
    log_line(f"[Config] prediction_type(yaml)={pred_type}")

    # ---------- 3) Load VAE (frozen) ----------
    vae_repo = cfg["vae_repo"]
    vae = AutoencoderKL.from_pretrained(vae_repo).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    log_line(f"[VAE] Loaded from {vae_repo} (frozen)")

    # ---------- 4) Build UNet ----------
    arch = cfg["unet"]
    unet = UNet2DConditionModel(
        sample_size=arch["sample_size"],
        in_channels=arch["in_channels"],
        out_channels=arch["out_channels"],
        down_block_types=tuple(arch["down_block_types"]),
        up_block_types=tuple(arch["up_block_types"]),
        block_out_channels=tuple(arch["block_out_channels"]),
        layers_per_block=arch["layers_per_block"],
        cross_attention_dim=arch["cross_attention_dim"],
        attention_head_dim=arch["attention_head_dim"],
    ).to(device, dtype=torch.float32)
    unet.train()
    log_line(f"[UNet] Configured: in={arch['in_channels']}, out={arch['out_channels']}")

    # ---------- 5) NumericEncoder ----------
    cond_dim  = int(cfg.get("cond_dim", 2))
    hidden    = int(cfg.get("hidden_size", 768))
    seq_len   = int(cfg.get("seq_len", 16))

    enc = NumericEncoder(seq_len=seq_len, hidden_size=hidden, cond_dim=cond_dim).to(device).float()
    enc.train()
    log_line(f"[ENC] NumericEncoder: cond_dim={cond_dim}, hidden={hidden}, seq_len={seq_len}")

    # ---------- 6) Noise scheduler ----------
    sch_cfg = cfg["scheduler"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=sch_cfg["num_train_timesteps"],
        beta_start=sch_cfg["beta_start"],
        beta_end=sch_cfg["beta_end"],
        beta_schedule=sch_cfg["beta_schedule"],
        prediction_type=pred_type,
    )
    log_line(f"[Scheduler] num_train_timesteps={noise_scheduler.config.num_train_timesteps} "
             f"prediction_type={getattr(noise_scheduler.config, 'prediction_type', None)}")

    # ---------- 7) Data ----------
    ds = SidecarDataset(train_dir, image_size=image_size)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    overfit_one_batch = bool(cfg.get("overfit_one_batch", False))
    fixed_batch = None

    log_line(f"[DATA] N={len(ds)} images from {train_dir}")

    # Compute cond stats and inject into encoder (save for inference)
    log_line("[DATA] Calculating cond mean/std (up to 2000 samples)...")
    check_len = min(len(ds), 2000)
    allc = []
    for i in range(check_len):
        allc.append(ds[i]["cond"].numpy())
    allc = np.stack(allc, 0)
    cond_mean = allc.mean(0)
    cond_std = np.clip(allc.std(0), 1e-6, None)

    if hasattr(enc, "set_norm"):
        enc.set_norm(cond_mean, cond_std)
        log_line(f"[ENC] Set Norm: mean={cond_mean}, std={cond_std}")
        np.save(out_dir / "cond_mean.npy", cond_mean.astype(np.float32))
        np.save(out_dir / "cond_std.npy",  cond_std.astype(np.float32))
        log_line(f"[ENC] Saved cond stats to {out_dir}/cond_mean.npy and {out_dir}/cond_std.npy")
    else:
        log_line("[WARN] Encoder has no set_norm(); continuing without normalization injection.")

    # ---------- 8) Optimizer (fixed LR) ----------
    params = list(unet.parameters()) + list(enc.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    log_line("[OPT] Using fixed learning rate (no scheduler).")

    # ---------- 9) Training loop ----------
    step = 0
    saved_latent_mean = False

    while step < max_steps:

        from collections import deque
        loss_win = deque(maxlen=200)


        for batch in dl:

            if overfit_one_batch:
                if fixed_batch is None:
                    fixed_batch = batch
                batch = fixed_batch


            imgs = batch["pixel_values"].to(device, dtype=torch.float32)  # [B,1,H,W] in [-1,1]
            cond = batch["cond"].to(device, dtype=torch.float32)          # [B,2]
            bsz = imgs.size(0)

            # A) VAE encode -> latents (scaled)
            with torch.no_grad():
                z_raw = vae.encode(imgs).latent_dist.sample()
                latents = z_raw * SCALING_FACTOR

            # Save a reference latent_mean once (optional, helps inference alignment/debug)
            if (not saved_latent_mean) and (step >= 50):
                lat_mean = float(latents.mean().item())
                np.save(out_dir / "latent_mean.npy", np.array([lat_mean], dtype=np.float32))
                log_line(f"[LATENTS] Saved latent_mean={lat_mean:.6f} to {out_dir}/latent_mean.npy")
                saved_latent_mean = True

            # B) Condition embedding + dropout
            cond_emb = enc(cond).to(dtype=torch.float32)  # [B,L,H]
            if cond_dropout_prob > 0:
                mask = (torch.rand(bsz, device=device) < cond_dropout_prob).float()
                cond_emb = cond_emb * (1.0 - mask)[:, None, None]

            # C) Add noise
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # D) Predict
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=cond_emb).sample

            # E) Target based on prediction_type
            if pred_type == "epsilon":
                target = noise
            elif pred_type == "sample":
                target = latents
            elif pred_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction_type={pred_type}")

            loss = torch.nn.functional.mse_loss(model_pred, target)

            # Backprop
            if not torch.isfinite(loss):
                log_line("[WARN] Loss is NaN/Inf, skipping step.")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1

            # Latent stats occasionally
            if step % 200 == 0:
                log_line(f"[LATENTS] mean={latents.mean().item():.4f} std={latents.std().item():.4f} "
                         f"min={latents.min().item():.4f} max={latents.max().item():.4f}")

            # Log
            loss_win.append(float(loss.item()))
            
            if step % log_every == 0:
                lr_now = opt.param_groups[0]["lr"]
                with open(log_path, "a", newline="") as f:
                    csv.writer(f, delimiter="\t").writerow([step, float(loss.item()), float(lr_now), int(time.time())])
                print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | "
                    f"Loss_ma200: {sum(loss_win)/len(loss_win):.4f} | LR: {lr_now:.2e}")

            # Save
            if step % save_every == 0:
                torch.save(unet.state_dict(), out_dir / f"unet_step{step}.pt")
                if hasattr(enc, "state_dict_light"):
                    torch.save(enc.state_dict_light(), out_dir / f"numeric_enc_step{step}.pt")
                else:
                    torch.save(enc.state_dict(), out_dir / f"numeric_enc_step{step}.pt")
                log_line(f"[CKPT] Saved checkpoint at step {step}")

            if step >= max_steps:
                break

    # Final save
    torch.save(unet.state_dict(), out_dir / "unet_final.pt")
    if hasattr(enc, "state_dict_light"):
        torch.save(enc.state_dict_light(), out_dir / "numeric_enc_final.pt")
    else:
        torch.save(enc.state_dict(), out_dir / "numeric_enc_final.pt")
    log_line("[DONE] Training finished. Saved final models.")


if __name__ == "__main__":
    main()
