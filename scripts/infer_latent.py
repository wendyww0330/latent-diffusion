import os
import glob
import re
import yaml
import argparse

import torch
import numpy as np
from torchvision.utils import save_image

from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from numeric_encoder import NumericEncoder


# ========== 1) sidecar parsing ==========
_re_b50  = re.compile(r"signal_b50\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_re_b800 = re.compile(r"signal_b800\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")

def parse_pair(text: str):
    m1 = _re_b50.search(text)
    m2 = _re_b800.search(text)
    if not (m1 and m2):
        return None, None

    def to_float(m):
        return float(m.group(1).rstrip(",;").replace(",", ""))

    try:
        return to_float(m1), to_float(m2)
    except Exception:
        return None, None


# ========== 2) DDIM sampling (CFG supported) ==========
@torch.no_grad()
def ddim_sample(unet, scheduler, zT, cond_emb, guidance_scale=1.0, uncond_emb=None):
    z = zT

    for t in scheduler.timesteps:
        if guidance_scale > 1.0 and uncond_emb is not None:
            latent_model_input = torch.cat([z, z], dim=0)
            encoder_hidden_states = torch.cat([uncond_emb, cond_emb], dim=0)
        else:
            latent_model_input = z
            encoder_hidden_states = cond_emb

        model_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states
        ).sample

        if guidance_scale > 1.0 and uncond_emb is not None:
            pred_uncond, pred_cond = model_pred.chunk(2, dim=0)
            model_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        z = scheduler.step(model_output=model_pred, timestep=t, sample=z).prev_sample

    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config for inference")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = int(cfg.get("seed", 123))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ========== read key params ==========
    guidance_scale = float(cfg.get("guidance", 1.0))
    steps = int(cfg.get("steps", 50))

    pred_type = str(cfg.get("prediction_type", "epsilon"))  # must match training
    scaling_factor = float(cfg.get("scaling_factor", 1.0))

    do_binarize = bool(cfg.get("binarize", False))
    bin_thresh = float(cfg.get("binarize_threshold", 0.5))

    # ========== load VAE ==========
    vae_repo = cfg["vae_repo"]
    print(f"[VAE] Loading from {vae_repo} ...")
    vae = AutoencoderKL.from_pretrained(vae_repo).to(device)
    vae.eval()

    print(f"[Config] prediction_type={pred_type}")
    print(f"[Config] scaling_factor(yaml)={scaling_factor}")
    print(f"[Config] steps={steps} guidance={guidance_scale}")
    print(f"[Config] binarize={do_binarize} thresh={bin_thresh}")

    # ========== load UNet ==========
    arch = cfg["unet_arch"]
    unet_ckpt = cfg["unet_ckpt"]
    print(f"[UNet] Loading from {unet_ckpt} ...")

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

    unet.load_state_dict(torch.load(unet_ckpt, map_location="cpu"))
    unet.eval()

    # ========== load NumericEncoder ==========
    cond_dim   = int(cfg["cond_dim"])
    hidden     = int(cfg["hidden_size"])
    seq_len    = int(cfg["seq_len"])
    numeric_ckpt = cfg["numeric_ckpt"]

    enc = NumericEncoder(seq_len=seq_len, hidden_size=hidden, cond_dim=cond_dim).to(device)
    enc_sd = torch.load(numeric_ckpt, map_location="cpu")

    if isinstance(enc_sd, dict) and "cond_mlp" in enc_sd:
        enc.cond_mlp.load_state_dict(enc_sd["cond_mlp"], strict=True)
    elif isinstance(enc_sd, dict) and any(k.startswith("cond_mlp.") for k in enc_sd.keys()):
        enc.load_state_dict(enc_sd, strict=True)
    else:
        enc.load_state_dict(enc_sd, strict=False)
    enc.eval()

    # load cond stats saved by training (only for encoder norm; no per-image printing)
    mean_path = cfg["cond_mean_path"]
    std_path  = cfg["cond_std_path"]
    train_mean = np.load(mean_path).astype(np.float32)
    train_std  = np.load(std_path).astype(np.float32)

    if hasattr(enc, "set_norm"):
        enc.set_norm(train_mean, train_std)
        print(f"[Encoder] Norm loaded.")
    else:
        print("[WARN] Encoder has no set_norm(); continuing without norm injection.")

    # ========== scheduler ==========
    sch_cfg = cfg["scheduler"]
    scheduler = DDIMScheduler(
        num_train_timesteps=sch_cfg["num_train_timesteps"],
        beta_start=sch_cfg["beta_start"],
        beta_end=sch_cfg["beta_end"],
        beta_schedule=sch_cfg["beta_schedule"],
        prediction_type=pred_type,
        clip_sample=False,
    )
    scheduler.set_timesteps(steps)
    print("[Scheduler] prediction_type =", getattr(scheduler.config, "prediction_type", None))
    print("[Scheduler] clip_sample =", getattr(scheduler.config, "clip_sample", None))

    # ========== io ==========
    real_dir = cfg["real_dir"]
    out_dir  = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    txts = sorted(glob.glob(os.path.join(real_dir, "*.txt")))
    bs = int(cfg.get("batch_size", 1))
    print(f"[Start] Found {len(txts)} files. Batch={bs}")

    latent_size     = int(arch["sample_size"])
    latent_channels = int(arch["in_channels"])

    for i in range(0, len(txts), bs):
        batch_txts = txts[i:i+bs]
        stems, conds = [], []

        for tpath in batch_txts:
            with open(tpath, "r", encoding="utf-8") as f:
                s = f.read().strip()

            b50, b800 = parse_pair(s)
            if b50 is None:
                b50, b800 = 0.0, 0.0

            stem = os.path.splitext(os.path.basename(tpath))[0]
            stems.append(stem)
            conds.append([b50, b800])

        if not stems:
            continue

        cond = torch.tensor(conds, dtype=torch.float32, device=device)  # [B,2]
        B = cond.shape[0]

        with torch.no_grad():
            cond_emb = enc(cond).to(device=device, dtype=unet.dtype)  # [B,L,H]
            if guidance_scale > 1.0:
                uncond_emb = torch.zeros_like(cond_emb)
            else:
                uncond_emb = None

        zT = torch.randn(B, latent_channels, latent_size, latent_size, device=device, dtype=unet.dtype)

        z0 = ddim_sample(
            unet=unet,
            scheduler=scheduler,
            zT=zT,
            cond_emb=cond_emb,
            guidance_scale=guidance_scale,
            uncond_emb=uncond_emb,
        )

        with torch.no_grad():
            x = vae.decode(z0 / scaling_factor).sample  # [-1,1]
            x01 = (x.clamp(-1, 1) + 1) * 0.5           # [0,1]

            if do_binarize:
                out = (x01 > bin_thresh).float()
            else:
                out = x01

        for idx, stem in enumerate(stems):
            save_path = os.path.join(out_dir, f"{stem}.png")
            save_image(out[idx:idx+1], save_path, nrow=1)
            # 如不想每张都print，可以注释下一行
            print(f"[Saved] {save_path}")

    print(f"[Done] All saved to {out_dir}")


if __name__ == "__main__":
    main()
