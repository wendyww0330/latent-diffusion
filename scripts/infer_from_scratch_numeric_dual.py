import os, re, glob, math, yaml, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch import nn

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

from numeric_encoder import NumericEncoder   # 你训练时用的同一实现

# --------- 工具 ----------
_NUM_RE = r"([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)"
RE_B50  = re.compile(rf"signal_b50\s*=\s*{_NUM_RE}\s*[,;]?")
RE_B800 = re.compile(rf"signal_b800\s*=\s*{_NUM_RE}\s*[,;]?")

def parse_pair(text: str):
    m1 = RE_B50.search(text); m2 = RE_B800.search(text)
    if not (m1 and m2): return None, None
    def _to_f(m):
        s = m.group(1).rstrip(",;").replace(",", "")
        return float(s)
    try:
        return _to_f(m1), _to_f(m2)
    except ValueError:
        return None, None

@torch.no_grad()
def decode_latents_to_pil(vae: AutoencoderKL, latents: torch.Tensor):
    # 逆 SD 缩放因子
    latents = latents / 0.18215
    imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)            # [-1,1] -> [0,1]
    imgs = imgs.permute(0,2,3,1).detach().cpu().numpy()
    pil = [Image.fromarray((img*255).astype(np.uint8)) for img in imgs]
    return pil

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# --------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(int(cfg.get("seed", 123)))
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # 1) 加载 VAE（预训练，fp16）
    vae = AutoencoderKL.from_pretrained(
        cfg["vae_name"], subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae.eval()

    # 2) 构建 UNet（结构必须与训练一致），加载你训练的权重
    arch = cfg["unet_arch"]
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
    ).to(device, dtype=torch.float16)
    ckpt = torch.load(cfg["unet_ckpt"], map_location="cpu")
    unet.load_state_dict(ckpt, strict=True)
    unet.eval()

    # 3) NumericEncoder（双变量）
    cond_dim = int(cfg.get("cond_dim", 2))
    seq_len  = int(cfg.get("seq_len", 16))
    hidden   = int(cfg.get("hidden_size", 768))
    payload  = torch.load(cfg["numeric_ckpt"], map_location="cpu")  # 兼容 state_dict_light
    enc = NumericEncoder(seq_len=seq_len, hidden_size=hidden, cond_dim=cond_dim).to(device)
    enc.load_state_dict_light(payload)
    enc.eval()

    # 4) DDIM 调度器
    sch_cfg = cfg["scheduler"]
    scheduler = DDIMScheduler(
        num_train_timesteps=sch_cfg["num_train_timesteps"],
        beta_start=sch_cfg["beta_start"],
        beta_end=sch_cfg["beta_end"],
        beta_schedule=sch_cfg["beta_schedule"],
        clip_sample=False,
        set_alpha_to_one=False
    )
    steps = int(cfg.get("steps", 30))
    guidance = float(cfg.get("guidance", 7.5))
    eta = float(cfg.get("eta", 0.0))

    H, W = int(cfg.get("height", 512)), int(cfg.get("width", 512))
    # SD VAE 下采样 8 倍
    f = 8
    latent_h, latent_w = H // f, W // f

    # 5) 构建 .txt 列表
    txts = sorted(glob.glob(os.path.join(cfg["real_dir"], "*.txt")))
    print(f"[INFO] txt files: {len(txts)} | steps={steps} guidance={guidance}")

    bs = int(cfg.get("batch_size", 1))
    for i in range(0, len(txts), bs):
        batch_txts = txts[i: i+bs]
        stems, conds = [], []
        for t in batch_txts:
            with open(t, "r", encoding="utf-8") as f:
                s = f.read().strip()
            b50, b800 = parse_pair(s)
            if b50 is None:  # 跳过解析失败
                continue
            stems.append(os.path.splitext(os.path.basename(t))[0])
            conds.append([b50, b800])

        if not stems:
            continue

        cond = torch.tensor(conds, dtype=torch.float32, device=device)  # [B,2]

        # classifier-free guidance：构造正/负条件嵌入
        with torch.no_grad():
            pos = enc(cond).to(dtype=torch.float16)                       # [B, L, 768]
            neg = enc.negative(cond.size(0)).to(dtype=torch.float16)      # [B, L, 768]
            cond_embeds = torch.cat([neg, pos], dim=0)                    # [2B, L, 768]

        # 初始 latents（两份，用于 cfg）
        latents = torch.randn(
            (len(stems), 4, latent_h, latent_w),
            device=device, dtype=torch.float16
        )
        latents = latents * scheduler.init_noise_sigma
        latents = torch.cat([latents.clone(), latents.clone()], dim=0)    # [2B, 4, H/8, W/8]

        # DDIM 推理
        scheduler.set_timesteps(steps, device=device)
        for t in scheduler.timesteps:
            # UNet 预测噪声
            with torch.no_grad():
                noise_pred = unet(latents, t, encoder_hidden_states=cond_embeds).sample  # [2B, ...]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

            # 单步更新
            latents = scheduler.step(noise_pred, t, latents[:len(stems)], eta=eta).prev_sample
            # 为下一步再扩成 2B（uncond/cond 共享同一物理样本）
            latents = torch.cat([latents.clone(), latents.clone()], dim=0)

        # 只保留 cond 分支对应的那份
        latents = latents[:len(stems)]
        # 解码保存
        imgs = decode_latents_to_pil(vae, latents)
        for stem, im in zip(stems, imgs):
            im.save(os.path.join(cfg["out_dir"], f"{stem}.png"))

    print(f"[OK] saved images to: {cfg['out_dir']}")

if __name__ == "__main__":
    main()
