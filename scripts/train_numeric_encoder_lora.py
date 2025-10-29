# /workspace/scripts/train_numeric_encoder_lora.py
import os, glob, re, random, numpy as np, argparse, yaml, csv, time
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_cosine_schedule_with_warmup

from numeric_encoder import NumericEncoder   # cond_dim=2

# settings
def load_cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to YAML config")

    ap.add_argument("--out_dir", type=str, help="override out_dir")
    ap.add_argument("--train_dir", type=str, help="override train_dir")
    ap.add_argument("--batch_size", type=int, help="override batch_size")
    ap.add_argument("--lr", type=float, help="override lr")
    ap.add_argument("--max_steps", type=int, help="override max_steps")
    ap.add_argument("--rank", type=int, help="override rank")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # overwrite
    for k in ["out_dir","train_dir","batch_size","lr","max_steps","rank"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg

# Dataset（b50+b800
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
        b50  = float(m1.group(1).rstrip(",;").replace(",", ""))
        b800 = float(m2.group(1).rstrip(",;").replace(",", ""))

        cond = torch.tensor([b50, b800], dtype=torch.float32)  # encoder
        stem = os.path.splitext(t)[0]
        img_path = None
        for ext in [".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"]:
            p = stem + ext
            if os.path.isfile(p): img_path = p; break
        if img_path is None: raise FileNotFoundError(f"image not found for {t}")
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        return {"pixel_values": img, "cond": cond}

# LoRA
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
    cfg = load_cfg()

    # seeds
    SEED = int(cfg.get("seed", 123))
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    device    = "cuda"
    model_name= cfg["model_name"]
    train_dir = cfg["train_dir"]
    out_dir   = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # log
    log_path = os.path.join(out_dir, "train_loss.tsv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(["step","loss","lr","timestamp"])

    # base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    # LoRA on UNet
    rank = int(cfg.get("rank", 16))
    add_lora_to_unet(pipe.unet, rank=rank)
    lora_params = list(nn.ModuleList(list(pipe.unet.attn_processors.values())).parameters())
    for p in lora_params: p.requires_grad_(True)

    # Numeric encoder
    image_size = int(cfg.get("image_size", 512))
    ds = SidecarDataset(train_dir, image_size=image_size)

    # cond
    allc = np.stack([ds[i]["cond"].numpy() for i in range(len(ds))], 0)  # [N,2]
    cond_mean, cond_std = allc.mean(0), allc.std(0)

    seq_len = int(cfg.get("seq_len", 16))
    enc = NumericEncoder(seq_len=seq_len,
                         hidden_size=pipe.text_encoder.config.hidden_size,
                         cond_dim=2).to(device)  # 2 dim
    enc.set_norm(cond_mean, cond_std)
    enc.train()

    # 4) dataloader / optim
    bs = int(cfg.get("batch_size", 2))
    num_workers = int(cfg.get("num_workers", 4))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=num_workers,
                    pin_memory=True, drop_last=True)

    lr         = float(cfg.get("lr", 5e-5))
    max_steps  = int(cfg.get("max_steps", 10000))
    warmup     = int(cfg.get("warmup", 200))
    SAVE_EVERY = int(cfg.get("save_every", 500))
    LOG_EVERY  = int(cfg.get("log_every", 50))

    trainable = list(enc.parameters()) + [p for p in lora_params if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-2)
    sch = get_cosine_schedule_with_warmup(opt, warmup, max_steps)

    pipe.unet.train()
    vae, unet = pipe.vae, pipe.unet
    noise_scheduler = pipe.scheduler

    step = 0
    while step < max_steps:
        for batch in dl:
            cond = batch["cond"].to(device)  # [B,2]

            # latents（fp16）
            with torch.no_grad():
                imgs = batch["pixel_values"].to(device, dtype=torch.float16)
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            bsz = latents.size(0)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (bsz,), device=device, dtype=torch.long)
            noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # encoder -> text_emb
            text_emb = enc(cond).to(noisy_latents.dtype)  # [B,L,768] half
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sch.step(); step += 1

            if step % LOG_EVERY == 0:
                print(f"step {step}/{max_steps} | loss {loss.item():.4f}")
                with open(log_path, "a", newline="") as f:
                    csv.writer(f, delimiter="\t").writerow(
                        [step, float(loss.item()), float(sch.get_last_lr()[0]), int(time.time())]
                    )

            if step % SAVE_EVERY == 0:
                pipe.unet.save_attn_procs(out_dir)
                torch.save(enc.state_dict_light(), os.path.join(out_dir, f"numeric_enc_step{step}.pt"))

            if step >= max_steps:
                break

    pipe.unet.save_attn_procs(out_dir)
    torch.save(enc.state_dict_light(), os.path.join(out_dir, "numeric_enc_final.pt"))
    print(f"[OK] saved LoRA + numeric encoder to {out_dir}")

if __name__ == "__main__":
    main()
