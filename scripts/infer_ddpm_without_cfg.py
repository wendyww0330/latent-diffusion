import os, re, glob, yaml, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from numeric_encoder import NumericEncoder

# ========= 正则解析 b50 / b800 =========
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

    # ========= UNet =========
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
    ).to(device)
    unet.load_state_dict(torch.load(cfg["unet_ckpt"], map_location="cpu"))
    unet.eval()

    # ========= NumericEncoder =========
    enc = NumericEncoder(
        seq_len=int(cfg["seq_len"]),
        hidden_size=int(cfg["hidden_size"]),
        cond_dim=int(cfg["cond_dim"])
    ).to(device)
    enc.eval()

    enc_sd = torch.load(cfg["numeric_ckpt"], map_location="cpu")
    if hasattr(enc, "load_state_dict_light"):
        enc.load_state_dict_light(enc_sd)
    else:
        enc.load_state_dict(enc_sd, strict=False)

    # ========= Scheduler =========
    sch_cfg = cfg["scheduler"]
    scheduler = DDPMScheduler(
        num_train_timesteps=sch_cfg["num_train_timesteps"],
        beta_start=sch_cfg["beta_start"],
        beta_end=sch_cfg["beta_end"],
        beta_schedule=sch_cfg["beta_schedule"]
    )

    steps = int(cfg.get("steps", 30))
    scheduler.set_timesteps(steps, device=device)

    # ========= 扫描 TXT =========
    txts = sorted(glob.glob(os.path.join(cfg["real_dir"], "*.txt")))
    print(f"[INFO] txt files: {len(txts)}")

    bs = int(cfg.get("batch_size", 1))

    for i in range(0, len(txts), bs):
        batch_txts = txts[i:i+bs]
        stems, conds = [], []

        for t in batch_txts:
            with open(t, "r", encoding="utf-8") as f:
                s = f.read().strip()
            b50, b800 = parse_pair(s)
            if b50 is None:
                print(f"[WARN] parse fail: {t}")
                continue

            stem = os.path.splitext(os.path.basename(t))[0]
            stems.append(stem)
            conds.append([b50, b800])

        if not stems:
            continue

        cond = torch.tensor(conds, dtype=torch.float32, device=device)
        with torch.no_grad():
            cond_emb = enc(cond)

        B = len(stems)

        # ========= 初始噪声 =========
        img = torch.randn((B, 1, H, W), device=device)
        img = img * scheduler.init_noise_sigma

        # ========= DDPM 去噪 =========
        for t in scheduler.timesteps:
            with torch.no_grad():
                eps = unet(img, t, encoder_hidden_states=cond_emb).sample

            img = scheduler.step(eps, t, img).prev_sample

        # ========= 保存图像 =========
        img = (img / 2 + 0.5).clamp(0, 1).cpu().numpy()

        for stem, arr in zip(stems, img):
            pil = Image.fromarray((arr[0] * 255).astype(np.uint8), mode="L")
            pil.save(out_dir / f"{stem}.png")
            print(f"[SAVE] {out_dir / (stem + '.png')}")

    print(f"[OK] all saved -> {out_dir}")

if __name__ == "__main__":
    main()
