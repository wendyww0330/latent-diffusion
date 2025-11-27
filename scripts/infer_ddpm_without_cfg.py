import os, glob, yaml, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from numeric_encoder import NumericEncoder

# ========= 解析三维数值: b50, b800, b2000 =========
def parse_triplet(text: str):
    """
    从文本中解析出三个数值：
    例如:
        "24090.05, 12296.88, 6487.89"
        "24090.05 12296.88 6487.89"
    返回 [b50, b800, b2000] 或 None 表示失败
    """
    line = text.strip()
    if not line:
        return None

    # 统一用空格分隔，逗号替换为空格
    tokens = line.replace(",", " ").split()
    if len(tokens) < 3:
        return None

    try:
        vals = [float(tok) for tok in tokens[:3]]
    except ValueError:
        return None

    return vals  # [b50, b800, b2000]


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
        cond_dim=int(cfg["cond_dim"])   # 这里 config 里要写 cond_dim: 3
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

            vals = parse_triplet(s)
            if vals is None:
                print(f"[WARN] parse fail (need 3 numbers) in: {t}")
                continue

            b50, b800, b2000 = vals

            stem = os.path.splitext(os.path.basename(t))[0]
            stems.append(stem)
            conds.append([b50, b800, b2000])   # ★ 三维条件

        if not stems:
            continue

        cond = torch.tensor(conds, dtype=torch.float32, device=device)
        with torch.no_grad():
            cond_emb = enc(cond)   # [B, L, hidden]

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
            out_path = out_dir / f"{stem}.png"
            pil.save(out_path)
            print(f"[SAVE] {out_path}")

    print(f"[OK] all saved -> {out_dir}")


if __name__ == "__main__":
    main()
