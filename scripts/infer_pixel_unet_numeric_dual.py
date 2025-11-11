import os, re, glob, yaml, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from numeric_encoder import NumericEncoder

_NUM_RE = r"([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)"
RE_B50  = re.compile(rf"signal_b50\s*=\s*{_NUM_RE}\s*[,;]?")
RE_B800 = re.compile(rf"signal_b800\s*=\s*{_NUM_RE}\s*[,;]?")

def parse_pair(text: str):
    m1, m2 = RE_B50.search(text), RE_B800.search(text)
    if not (m1 and m2): return None, None
    def _f(m): return float(m.group(1).rstrip(",;").replace(",", ""))
    try: return _f(m1), _f(m2)
    except: return None, None

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(int(cfg.get("seed", 123)))

    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    H = W = int(cfg.get("image_size", 256))

    # UNet 
    arch = cfg["unet"]
    unet = UNet2DConditionModel(
        sample_size=arch["sample_size"],
        in_channels=arch["in_channels"],     # 1
        out_channels=arch["out_channels"],   # 1
        down_block_types=tuple(arch["down_block_types"]),
        up_block_types=tuple(arch["up_block_types"]),
        block_out_channels=tuple(arch["block_out_channels"]),
        layers_per_block=arch["layers_per_block"],
        cross_attention_dim=arch["cross_attention_dim"],
        attention_head_dim=arch["attention_head_dim"],
    ).to(device, dtype=torch.float32).eval()
    unet.load_state_dict(torch.load(cfg["unet_ckpt"], map_location="cpu"), strict=True)

    # NumericEncoder 
    enc = NumericEncoder(seq_len=int(cfg.get("seq_len", 16)),
                         hidden_size=int(cfg.get("hidden_size", 768)),
                         cond_dim=int(cfg.get("cond_dim", 2))).to(device).float().eval()
    enc.load_state_dict_light(torch.load(cfg["numeric_ckpt"], map_location="cpu"))

    # DDIM Scheduler
    sch = cfg["scheduler"]
    scheduler = DDIMScheduler(
        num_train_timesteps=sch["num_train_timesteps"],
        beta_start=sch["beta_start"], beta_end=sch["beta_end"],
        beta_schedule=sch["beta_schedule"],
        clip_sample=False,
        set_alpha_to_one=False
    )
    steps = int(cfg.get("steps", 30))
    guidance = float(cfg.get("guidance", 7.5))  # classifier-free guidance in pixel space
    eta = float(cfg.get("eta", 0.0))

    txts = sorted(glob.glob(os.path.join(cfg["real_dir"], "*.txt")))
    print(f"[INFO] txt files: {len(txts)} | steps={steps} guidance={guidance}")

    bs = int(cfg.get("batch_size", 1))
    for i in range(0, len(txts), bs):
        batch_txts = txts[i:i+bs]
        stems, conds = [], []
        for t in batch_txts:
            with open(t, "r", encoding="utf-8") as f:
                s = f.read().strip()
            b50, b800 = parse_pair(s)
            if b50 is None: continue
            stems.append(os.path.splitext(os.path.basename(t))[0])
            conds.append([b50, b800])

        if not stems: continue

        cond = torch.tensor(conds, dtype=torch.float32, device=device)  # [B,2]
        with torch.no_grad():
            pos = enc(cond).to(dtype=torch.float32)                      # [B,L,768]
            neg = enc.negative(len(stems)).to(dtype=torch.float32)       # [B,L,768]
            cond_embeds = torch.cat([neg, pos], dim=0)                   # [2B,L,768]

        # noise
        img = torch.randn((len(stems), 1, H, W), device=device, dtype=torch.float32)
        img = img * scheduler.init_noise_sigma
        img = torch.cat([img.clone(), img.clone()], dim=0)               # [2B,1,H,W]

        scheduler.set_timesteps(steps, device=device)
        for t in scheduler.timesteps:
            with torch.no_grad():
                eps = unet(img, t, encoder_hidden_states=cond_embeds).sample
                eps_u, eps_c = eps.chunk(2, dim=0)
                eps = eps_u + guidance * (eps_c - eps_u)
            img = scheduler.step(eps, t, img[:len(stems)], eta=eta).prev_sample
            img = torch.cat([img.clone(), img.clone()], dim=0)

        img = img[:len(stems)]
        # inverse [-1,1] -> [0,1]
        img = (img / 2 + 0.5).clamp(0, 1).cpu().numpy()                  # [B,1,H,W]
        for s, a in zip(stems, img):
            pil = Image.fromarray((a[0]*255).astype(np.uint8), mode="L")
            pil.save(out_dir / f"{s}.png")

    print(f"[OK] saved -> {out_dir}")

if __name__ == "__main__":
    main()
