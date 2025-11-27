import os, re, glob, csv, time, yaml, random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup

from numeric_encoder import NumericEncoder   # 你的数值编码器

# -----------------------------
# Dataset（灰度图 + 同名 sidecar）
# -----------------------------
class SidecarDataset(Dataset):
    def __init__(self, root, image_size=256):
        # 所有 sidecar txt
        self.txts = sorted(glob.glob(os.path.join(root, "*.txt")))
        self.tf = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Lambda(lambda im: im.convert("L")),   # 灰度 1 通道
            transforms.ToTensor(),                           # [1,H,W], 0..1
            transforms.Normalize([0.5], [0.5])               # -> [-1,1]
        ])

    def __len__(self):
        return len(self.txts)

    def __getitem__(self, i):
        t = self.txts[i]

        # ---- 读取 sidecar：三维数值 ----
        with open(t, "r", encoding="utf-8") as f:
            line = f.read().strip()

        if not line:
            raise ValueError(f"empty sidecar file: {t}")

        # 支持 "a, b, c" 或 "a b c" 两种写法
        # 先统一把逗号变成空格，再 split
        tokens = line.replace(",", " ").split()
        if len(tokens) < 3:
            raise ValueError(f"expect 3 values in sidecar, got {len(tokens)} in {t}: {tokens}")

        # 只取前三个（防止后面将来加别的东西）
        vals = [float(tok) for tok in tokens[:3]]
        b50, b800, b2000 = vals  # 只是可读性，变量名随意

        # ---- 找对应图片：和 txt 同名不同后缀 ----
        stem = os.path.splitext(t)[0]
        img_path = None
        for ext in [".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"]:
            p = stem + ext
            if os.path.isfile(p):
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError(f"image not found for sidecar: {t}")

        # ---- 读图 & 预处理 ----
        im = Image.open(img_path)
        im = self.tf(im)                                  # [1,H,W] in [-1,1]

        # ---- 数值条件（3 维）----
        cond = torch.tensor([b50, b800, b2000], dtype=torch.float32)  # [3]

        return {
            "pixel_values": im,
            "cond": cond,
        }

# -----------------------------
# 工具：快进 LR 调度器；读日志最后一步
# -----------------------------
def _fast_forward_scheduler(scheduler, steps:int):
    for _ in range(steps):
        scheduler.step()

def _read_last_logged_step(log_path: Path):
    last = 0
    try:
        if log_path.exists():
            with open(log_path, "r") as f:
                lines = f.read().strip().splitlines()
                if len(lines) >= 2:
                    last = int(lines[-1].split("\t")[0])
    except Exception:
        pass
    return last

def _infer_step_from_name(p: str):
    m = re.findall(r"step(\d+)", os.path.basename(p))
    return int(m[0]) if m else None

# -----------------------------
# 主程序：LDM（latent=1ch）训练
# -----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    # 续训（显式指定权重与步数）
    ap.add_argument("--resume_unet", type=str, default=None)
    ap.add_argument("--resume_enc",  type=str, default=None)
    ap.add_argument("--resume_step", type=int, default=None)
    # 可选：只预览，不训练（调试用）
    ap.add_argument("--preview_only", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 随机种子 & 设备
    seed = int(cfg.get("seed", 123))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 输出与日志
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_loss.tsv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(["step","loss","lr","ts"])
    last_logged_step = _read_last_logged_step(log_path)

    # -------------------------
    # VAE（灰度，latent=1）— 冻结
    # -------------------------
    vae_repo = cfg.get("vae_repo", "./experiments/vae_gray_f8/final")  # 你自训 VAE 的目录
    vae = AutoencoderKL.from_pretrained(vae_repo).to(device, dtype=torch.float32)
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()
    scaling = getattr(vae.config, "scaling_factor", 0.18215)
    print(f"[VAE] loaded from {vae_repo} | scaling_factor = {scaling}")

    # -------------------------
    # UNet（latent 空间，1→1），注意 sample_size = image_size//8
    # -------------------------
    arch = cfg["unet"]
    image_size = int(cfg["image_size"])
    latent_size = image_size // 8
    assert arch["sample_size"] == latent_size, "unet.sample_size must be image_size//8 in LDM"

    unet = UNet2DConditionModel(
        sample_size=arch["sample_size"],
        in_channels=1,                   # ★ latent 1 通道
        out_channels=1,                  # ★ 预测噪声 1 通道
        down_block_types=tuple(arch["down_block_types"]),
        up_block_types=tuple(arch["up_block_types"]),
        block_out_channels=tuple(arch["block_out_channels"]),
        layers_per_block=arch["layers_per_block"],
        cross_attention_dim=arch["cross_attention_dim"],    # = hidden_size
        attention_head_dim=arch["attention_head_dim"],
    ).to(device, dtype=torch.float32)
    unet.train()

    # 数值条件 Encoder
    cond_dim = int(cfg.get("cond_dim", 3))
    hidden   = int(cfg.get("hidden_size", 768))
    seq_len  = int(cfg.get("seq_len", 16))
    assert hidden == arch["cross_attention_dim"], "NumericEncoder.hidden_size must equal UNet.cross_attention_dim"
    enc = NumericEncoder(seq_len=seq_len, hidden_size=hidden, cond_dim=cond_dim).to(device).float()
    enc.train()

    # 调度器（DDPM epsilon 目标）
    sch_cfg = cfg["scheduler"]
    scheduler = DDPMScheduler(
        num_train_timesteps=sch_cfg["num_train_timesteps"],
        beta_start=sch_cfg["beta_start"], beta_end=sch_cfg["beta_end"],
        beta_schedule=sch_cfg["beta_schedule"]
    )
    assert scheduler.config.get("prediction_type", "epsilon") == "epsilon", \
        f"Scheduler prediction_type must be 'epsilon', got {scheduler.config.get('prediction_type')}"

    # 数据与 cond 归一化
    ds = SidecarDataset(cfg["train_dir"], image_size=image_size)
    allc = np.stack([ds[i]["cond"].numpy() for i in range(len(ds))], 0)
    mean = allc.mean(0); std = np.clip(allc.std(0), 1e-6, None)
    enc.set_norm(mean, std)

    dl = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if int(cfg.get("num_workers", 4)) > 0 else False,
        prefetch_factor=2 if int(cfg.get("num_workers", 4)) > 0 else None
    )

    # 优化
    lr = float(cfg.get("lr", 5e-5))
    max_steps = int(cfg.get("max_steps", 20000))
    warmup    = int(cfg.get("warmup", 500))
    save_every= int(cfg.get("save_every", 1000))
    log_every = int(cfg.get("log_every", 50))

    params = list(unet.parameters()) + list(enc.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sch_lr = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=max_steps)

    # -------------------------
    # 续训（显式指定）
    # -------------------------
    resume_unet = args.resume_unet
    resume_enc  = args.resume_enc
    resume_step = args.resume_step
    start_step = 0

    # 优先使用完整状态（training_state.pth）
    state_path = out_dir / "training_state.pth"
    used_state = False
    if state_path.exists() and not (resume_unet and resume_enc):
        try:
            st = torch.load(state_path, map_location="cpu")
            _step = int(st.get("step", 0))
            if _step > 0:
                unet.load_state_dict(torch.load(out_dir / f"unet_step{_step}.pt", map_location="cpu"), strict=True)
                enc_sd = torch.load(out_dir / f"numeric_enc_step{_step}.pt", map_location="cpu")
                try:
                    enc.load_state_dict(enc_sd, strict=True)
                except RuntimeError:
                    enc.load_state_dict(enc_sd, strict=False)
                opt.load_state_dict(st["opt"])
                sch_lr.load_state_dict(st["sch"])
                start_step = _step
                used_state = True
                print(f"[RESUME] from training_state.pth @ step={start_step}")
        except Exception as e:
            print(f"[RESUME] failed to load training_state.pth: {e}")

    # 如果显式给了 pt，则按给定权重+步数恢复；没有 state 就快进 LR
    if not used_state and (resume_unet and resume_enc):
        unet.load_state_dict(torch.load(resume_unet, map_location="cpu"), strict=True)
        enc_sd = torch.load(resume_enc, map_location="cpu")
        try:
            enc.load_state_dict(enc_sd, strict=True)
        except RuntimeError:
            enc.load_state_dict(enc_sd, strict=False)
        if resume_step is None:
            s1 = _infer_step_from_name(resume_unet)
            s2 = _infer_step_from_name(resume_enc)
            if s1 is not None and s2 is not None and s1 == s2:
                resume_step = s1
        if resume_step is None:
            raise ValueError("请用 --resume_step 指定起始步，或在文件名里包含 stepXXXX")
        start_step = int(resume_step)
        _fast_forward_scheduler(sch_lr, start_step)
        print(f"[RESUME] loaded weights @ step={start_step} (scheduler fast-forwarded)")

    # 仅预览：取一个 batch 编码/解码看是否正常
    if args.preview_only:
        batch = next(iter(dl))
        imgs = batch["pixel_values"].to(device, dtype=torch.float32)
        with torch.no_grad():
            lat = vae.encode(imgs).latent_dist.sample() * scaling
            rec = vae.decode(lat / scaling).sample
        rec = (rec.clamp(-1,1)+1)*0.5
        save_image(rec, out_dir / "preview_recon.png", nrow=4)
        print("[PREVIEW] saved preview_recon.png")
        return

    # -------------------------
    # ★ 小工具：从当前模型采样 4 张图，并拼成一张
    # -------------------------
    def sample_and_save(step, cond_batch):
        unet.eval()
        enc.eval()
        vae.eval()
        try:
            B = min(4, cond_batch.size(0))
            cond = cond_batch[:B].to(device).float()   # [B,3]
            with torch.no_grad():
                text_emb = enc(cond)                   # [B,L,hidden]

                num_infer_steps = int(cfg.get("sample_steps", 50))
                scheduler.set_timesteps(num_infer_steps, device=device)

                latents = torch.randn(
                    B, 1, latent_size, latent_size,
                    device=device, dtype=torch.float32
                ) * scheduler.init_noise_sigma

                for t in scheduler.timesteps:
                    noise_pred = unet(latents, t, encoder_hidden_states=text_emb).sample
                    latents = scheduler.step(noise_pred, t, latents).prev_sample

                imgs = vae.decode(latents / scaling).sample      # [-1,1]
                imgs = (imgs.clamp(-1, 1) + 1) * 0.5             # [0,1]

                save_path = out_dir / f"sample_step{step}.png"
                save_image(imgs, save_path, nrow=B)
                print(f"[SAMPLE] saved {B}-image grid at step {step} -> {save_path}")
        finally:
            unet.train()
            enc.train()
        # vae 一直是 eval 状态，无需改

    # -------------------------
    # 训练循环（latent 上加噪）
    # -------------------------
    step = start_step
    ema = None; ema_beta = 0.9

    while step < max_steps:
        for batch in dl:
            imgs = batch["pixel_values"].to(device, dtype=torch.float32)    # [B,1,H,W], [-1,1]

            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample() * scaling    # [B,1,H/8,W/8]

            bsz = latents.size(0)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(latents, dtype=torch.float32)
            noisy = scheduler.add_noise(latents, noise, timesteps)          # z_t

            cond = batch["cond"].to(device).float()
            text_emb = enc(cond).to(dtype=torch.float32)                     # [B,L,hidden]

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

            # 日志（追加写；避免重复）
            if step % log_every == 0 and step > last_logged_step:
                with open(log_path, "a", newline="") as f:
                    csv.writer(f, delimiter="\t").writerow(
                        [step, float(loss.item()), float(sch_lr.get_last_lr()[0]), int(time.time())]
                    )
                last_logged_step = step
                ema = float(loss.item()) if ema is None else ema_beta*ema + (1-ema_beta)*float(loss.item())
                print(f"step {step}/{max_steps} | loss {loss.item():.4f} | ema {ema:.4f}")

            # ★ 每 10000 步，从噪声采样 4 张图，拼成一张
            if step % 10000 == 0:
                sample_and_save(step, batch["cond"])

            # 定期保存（含状态，便于无缝续训）
            if step % save_every == 0:
                torch.save(unet.state_dict(), out_dir / f"unet_step{step}.pt")
                # 你的 NumericEncoder 若有 state_dict_light() 就用，没有就用 state_dict()
                if hasattr(enc, "state_dict_light"):
                    torch.save(enc.state_dict_light(), out_dir / f"numeric_enc_step{step}.pt")
                else:
                    torch.save(enc.state_dict(), out_dir / f"numeric_enc_step{step}.pt")

                state = {"step": step, "opt": opt.state_dict(), "sch": sch_lr.state_dict(), "seed": seed}
                torch.save(state, out_dir / "training_state.pth")

                # 可选：存一张 quick preview（把当前 batch 的 latents 解码）
                with torch.no_grad():
                    rec = vae.decode(latents[:4] / scaling).sample
                    rec = (rec.clamp(-1,1)+1)*0.5
                    save_image(rec, out_dir / f"preview_step{step}.png", nrow=2)

            if step >= max_steps: break

    torch.save(unet.state_dict(), out_dir / "unet_final.pt")
    if hasattr(enc, "state_dict_light"):
        torch.save(enc.state_dict_light(), out_dir / "numeric_enc_final.pt")
    else:
        torch.save(enc.state_dict(), out_dir / "numeric_enc_final.pt")
    print(f"[OK] saved to {out_dir}")

if __name__ == "__main__":
    main()
