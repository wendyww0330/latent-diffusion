import os, glob, yaml
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image

from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from numeric_encoder import NumericEncoder


# ========== 解析三维数值的函数 ==========
def parse_triplet(text: str):
    """
    从 sidecar 文本中解析出三个数值，例如：
        "24090.05, 12296.88, 6487.89"
        "24090.05 12296.88 6487.89"
    返回 (b50, b800, b2000)；如果解析失败则返回 (None, None, None)
    """
    line = text.strip()
    if not line:
        return None, None, None

    # 统一把逗号变成空格，再 split
    tokens = line.replace(",", " ").split()
    if len(tokens) < 3:
        return None, None, None

    try:
        vals = [float(tok) for tok in tokens[:3]]
    except ValueError:
        return None, None, None

    b50, b800, b2000 = vals
    return b50, b800, b2000


@torch.no_grad()
def ddim_sample(unet, scheduler, zT, cond_emb):
    """
    zT: [B,C,H,W] latent 噪声
    cond_emb: [B,L,H]，NumericEncoder 输出
    """
    z = zT
    for t in scheduler.timesteps:
        eps = unet(z, t, encoder_hidden_states=cond_emb).sample
        out = scheduler.step(model_output=eps, timestep=t, sample=z)
        z = out.prev_sample
    return z


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    # ========== 1) 读取配置 ==========
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 123))

    # ========== 2) 加载 VAE（1 通道 latent 版本） ==========
    vae_repo = cfg["vae_repo"]
    vae = AutoencoderKL.from_pretrained(vae_repo).to(device)
    vae.eval()
    scaling = getattr(vae.config, "scaling_factor", 0.18215)
    print(f"[VAE] loaded from {vae_repo} | scaling_factor={scaling}")

    # ========== 3) 加载 UNet ==========
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
    ).to(device)
    unet_ckpt = cfg["unet_ckpt"]
    unet.load_state_dict(torch.load(unet_ckpt, map_location="cpu"))
    unet.eval()
    print(f"[UNet] loaded from {unet_ckpt}")

    # ========== 4) 加载 NumericEncoder ==========
    cond_dim   = int(cfg["cond_dim"])      # ⚠️ 这里 config 里请写 cond_dim: 3
    hidden     = int(cfg["hidden_size"])
    seq_len    = int(cfg["seq_len"])

    enc = NumericEncoder(seq_len=seq_len, hidden_size=hidden, cond_dim=cond_dim).to(device)
    numeric_ckpt = cfg["numeric_ckpt"]
    enc_sd = torch.load(numeric_ckpt, map_location="cpu")

    # 兼容几种保存方式
    if any(k.startswith("cond_mlp.") for k in enc_sd.keys()):
        enc.load_state_dict(enc_sd, strict=True)
        print(f"[Encoder] loaded FULL state_dict from {numeric_ckpt}")
    elif "cond_mlp" in enc_sd:
        sub_sd = enc_sd["cond_mlp"]
        enc.cond_mlp.load_state_dict(sub_sd, strict=True)
        print(f"[Encoder] loaded LIGHT state_dict (cond_mlp only) from {numeric_ckpt}")
    else:
        enc.load_state_dict(enc_sd, strict=False)
        print(f"[Encoder] loaded with strict=False from {numeric_ckpt}")

    enc.eval()

    # ========== 5) 构造 DDIM 调度器 ==========
    sch_cfg = cfg["scheduler"]
    scheduler = DDIMScheduler(
        num_train_timesteps=sch_cfg["num_train_timesteps"],
        beta_start=sch_cfg["beta_start"],
        beta_end=sch_cfg["beta_end"],
        beta_schedule=sch_cfg["beta_schedule"],
    )
    steps = int(cfg["steps"])
    scheduler.set_timesteps(steps)

    # latent 尺寸 / 通道数
    latent_size     = arch["sample_size"]
    latent_channels = arch["in_channels"]   # 1 对应你的灰度 latent

    # ========== 6) 准备数据（逐个 txt 读取三个值） ==========
    real_dir = cfg["real_dir"]
    out_dir  = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    txts = sorted(glob.glob(os.path.join(real_dir, "*.txt")))
    bs = int(cfg.get("batch_size", 1))

    print(f"[INFO] Found {len(txts)} txt files in {real_dir} | batch_size={bs} | steps={steps}")

    for i in range(0, len(txts), bs):
        batch_txts = txts[i:i+bs]
        stems, conds = [], []

        for tpath in batch_txts:
            with open(tpath, "r", encoding="utf-8") as f:
                s = f.read().strip()

            b50, b800, b2000 = parse_triplet(s)
            if b50 is None:
                print(f"[WARN] parse fail for {tpath}, skip")
                continue

            stem = os.path.splitext(os.path.basename(tpath))[0]
            stems.append(stem)
            conds.append([b50, b800, b2000])     # ★ 三维条件

        if not stems:
            continue

        cond = torch.tensor(conds, dtype=torch.float32, device=device)  # [B,3]
        with torch.no_grad():
            cond_emb = enc(cond).to(device=device, dtype=unet.dtype)    # [B,L,H]

        B = cond.shape[0]

        # ========== 7) 从高斯噪声开始 ==========
        zT = torch.randn(
            B, latent_channels, latent_size, latent_size,
            device=device, dtype=unet.dtype
        )

        # ========== 8) DDIM 去噪 ==========
        with torch.no_grad():
            z0 = ddim_sample(unet, scheduler, zT, cond_emb)  # [B,1,latent_size,latent_size]

            # ========== 9) 解码回图像 ==========
            x = vae.decode(z0 / scaling).sample             # [-1,1], [B,1,H,W]
            x = (x.clamp(-1, 1) + 1) * 0.5                  # [0,1]

        # 逐张保存
        for idx, stem in enumerate(stems):
            save_path = os.path.join(out_dir, f"{stem}.png")
            save_image(x[idx:idx+1], save_path, nrow=1)
            print(f"[SAVE] {save_path}")

    print(f"[DONE] all samples saved to {out_dir}")


if __name__ == "__main__":
    main()
