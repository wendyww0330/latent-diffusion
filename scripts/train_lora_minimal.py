import os, glob
from PIL import Image
import torch, random, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# -------- Dataset: image + sidecar .txt (支持递归) --------
class SidecarDataset(Dataset):
    def __init__(self, root, image_size=512):
        self.root = root
        self.txts = sorted(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True))
        self.tf = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # to [-1,1]
        ])
    def __len__(self): return len(self.txts)
    def __getitem__(self, i):
        txt_path = self.txts[i]
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        stem = os.path.splitext(txt_path)[0]
        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]:
            p = stem + ext
            if os.path.isfile(p): img_path = p; break
        if img_path is None:
            raise FileNotFoundError(f"image not found for {txt_path}")
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        return {"pixel_values": img, "prompt": prompt}

# -------- LoRA helpers (diffusers 0.24.0 兼容) --------
def add_lora_to_unet(unet: nn.Module, rank=8):
    from diffusers.models.attention_processor import LoRAAttnProcessor
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
    # 固定随机种子（不使用 generator 参数）
    SEED = 123
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    device     = "cuda"
    model_name = os.environ.get("MODEL_NAME", "runwayml/stable-diffusion-v1-5")
    train_dir  = os.environ.get("TRAIN_DIR", "train_data")
    out_dir    = os.environ.get("OUT_DIR",  "lora_out_minimal")
    os.makedirs(out_dir, exist_ok=True)

    # 1) 基座
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)  # 用 DDPM 训练
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    # 2) 注入 LoRA，并收集可训练参数
    rank = int(os.environ.get("RANK", "8"))
    add_lora_to_unet(pipe.unet, rank=rank)
    lora_modules = torch.nn.ModuleList(list(pipe.unet.attn_processors.values()))
    for p in lora_modules.parameters(): p.requires_grad_(True)
    num_trainable = sum(p.numel() for p in lora_modules.parameters() if p.requires_grad)
    print("[INFO] trainable params in LoRA:", num_trainable)

    # 3) 数据
    ds = SidecarDataset(train_dir, image_size=512)
    bs = int(os.environ.get("BATCH_SIZE", "2"))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # 4) 优化器 & 学习率
    lr = float(os.environ.get("LR", "1e-4"))
    max_steps = int(os.environ.get("MAX_STEPS", "5000"))
    warmup = int(os.environ.get("WARMUP", "200"))
    opt = torch.optim.AdamW(lora_modules.parameters(), lr=lr, weight_decay=1e-2)
    sch = get_cosine_schedule_with_warmup(opt, warmup, max_steps)

    pipe.unet.train()
    vae, unet, te = pipe.vae, pipe.unet, pipe.text_encoder
    tok = pipe.tokenizer
    noise_scheduler = pipe.scheduler

    step = 0
    while step < max_steps:
        for batch in dl:
            with torch.no_grad():
                enc = tok(batch["prompt"], padding="max_length",
                          max_length=tok.model_max_length, truncation=True, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                text_emb = te(**enc)[0]

                imgs = batch["pixel_values"].to(device, dtype=torch.float16)
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            bsz = latents.size(0)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (bsz,), device=device, dtype=torch.long)
            noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_modules.parameters(), 1.0)  # ← 已修
            opt.step(); sch.step()
            step += 1

            if step % 50 == 0:
                print(f"step {step}/{max_steps} | loss {loss.item():.4f}")
            if step % 1000 == 0:
                pipe.unet.save_attn_procs(out_dir)
            if step >= max_steps:
                break

    pipe.unet.save_attn_procs(out_dir)
    print(f"[OK] LoRA saved to {out_dir}")

if __name__ == "__main__":
    main()
