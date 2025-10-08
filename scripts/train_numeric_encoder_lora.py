# /workspace/scripts/train_numeric_encoder_lora.py
import os, glob, re, random, numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_cosine_schedule_with_warmup

from numeric_encoder import NumericEncoder

# ---------- Dataset ----------
class SidecarDataset(Dataset):
    def __init__(self, root, image_size=512):
        self.root = root
        self.txts = sorted(glob.glob(os.path.join(root, "*.txt")))
        self.tf = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.re_b50  = re.compile(r"signal_b50\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")
        self.re_b800 = re.compile(r"signal_b800\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")


    def __len__(self): return len(self.txts)

    def __getitem__(self, i):
        t = self.txts[i]
        with open(t, "r", encoding="utf-8") as f:
            s = f.read().strip()
        m1 = self.re_b50.search(s); m2 = self.re_b800.search(s)
        if not (m1 and m2):
            raise ValueError(f"Parse fail: {t}")
        b50_str  = m1.group(1).rstrip(",;").replace(",", "")
        b800_str = m2.group(1).rstrip(",;").replace(",", "")
        b50  = float(b50_str)
        b800 = float(b800_str)
        cond = torch.tensor([b50, b800], dtype=torch.float32)
        stem = os.path.splitext(t)[0]
        img_path = None
        for ext in [".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"]:
            p = stem + ext
            if os.path.isfile(p): img_path = p; break
        if img_path is None: raise FileNotFoundError(f"image not found for {t}")
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        return {"pixel_values": img, "cond": cond}

# ---------- LoRA inject ----------
def add_lora_to_unet(unet: nn.Module, rank=16):
    attn_procs = {}
    for name, _ in unet.attn_processors.items():
        module = unet
        for attr in name.split(".")[:-1]:
            module = getattr(module, attr)
        hidden = module.to_q.in_features
        cross  = getattr(module, "cross_attention_dim", None)
        try:
            proc = LoRAAttnProcessor(hidden_size=hidden, cross_attention_dim=cross, rank=rank)
        except TypeError:
            proc = LoRAAttnProcessor(hidden, cross, rank)
        attn_procs[name] = proc
    unet.set_attn_processor(attn_procs)

def main():
    # seeds
    SEED=123; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    device="cuda"
    model_name=os.environ.get("MODEL_NAME","runwayml/stable-diffusion-v1-5")
    train_dir=os.environ.get("TRAIN_DIR","/workspace/train_data_cont")
    out_dir=os.environ.get("OUT_DIR","/workspace/experiments/numeric_enc_lora")
    os.makedirs(out_dir, exist_ok=True)

    # 1) base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    # 2) LoRA on UNet
    rank = int(os.environ.get("RANK","16"))
    add_lora_to_unet(pipe.unet, rank=rank)
    lora_params = list(nn.ModuleList(list(pipe.unet.attn_processors.values())).parameters())

    # 3) Numeric encoder
    ds = SidecarDataset(train_dir, image_size=512)
    # 计算 cond 的 mean/std 供归一化
    allc = np.stack([ds[i]["cond"].numpy() for i in range(len(ds))],0)
    cond_mean, cond_std = allc.mean(0), allc.std(0)

    enc = NumericEncoder(seq_len=16, hidden_size=pipe.text_encoder.config.hidden_size, cond_dim=2).to(device)
    enc.set_norm(cond_mean, cond_std)
    enc.train()

    # 4) dataloader / optim
    bs = int(os.environ.get("BATCH_SIZE","2"))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    lr = float(os.environ.get("LR","5e-5"))
    max_steps = int(os.environ.get("MAX_STEPS","10000"))
    warmup = int(os.environ.get("WARMUP","200"))
    SAVE_EVERY = int(os.environ.get("SAVE_EVERY","500"))

    trainable = list(enc.parameters()) + [p for p in lora_params if p.requires_grad]
    # 确保 LoRA 参数可训练
    for p in lora_params: p.requires_grad_(True)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-2)
    sch = get_cosine_schedule_with_warmup(opt, warmup, max_steps)

    pipe.unet.train()
    vae, unet = pipe.vae, pipe.unet
    noise_scheduler = pipe.scheduler

    step = 0
    while step < max_steps:
        for batch in dl:
            cond = batch["cond"].to(device)  # [B,2]

            # 先准备 latents（fp16），不参与梯度
            with torch.no_grad():
                imgs = batch["pixel_values"].to(device, dtype=torch.float16)
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            bsz = latents.size(0)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device, dtype=torch.long
            )
            noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # === 关键：先算 text_emb，再做 dtype 对齐 ===
            # enc 默认在 fp32 计算，避免精度损失
            text_emb = enc(cond)                            # [B, L, 768], float32
            text_emb = text_emb.to(noisy_latents.dtype)     # 对齐到 half（与 UNet/LoRA 一致）

            # UNet 前向与损失
            noise_pred = unet(
                noisy_latents, timesteps,
                encoder_hidden_states=text_emb              # dtype 已对齐
            ).sample
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sch.step()
            step += 1

            if step % 50 == 0:
                print(f"step {step}/{max_steps} | loss {loss.item():.4f} | "
                    f"latents.dtype={latents.dtype} text_emb.dtype={text_emb.dtype}")

            if step % SAVE_EVERY == 0:
                pipe.unet.save_attn_procs(out_dir)
                torch.save(enc.state_dict_light(), os.path.join(out_dir, f"numeric_enc_step{step}.pt"))

            if step >= max_steps:
                break

    # 结束保存
    pipe.unet.save_attn_procs(out_dir)
    torch.save(enc.state_dict_light(), os.path.join(out_dir, "numeric_enc_final.pt"))
    print(f"[OK] saved LoRA + numeric encoder to {out_dir}")


if __name__ == "__main__":
    main()
