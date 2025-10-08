# /workspace/scripts/infer_numeric_batch.py
import os, glob, re, argparse
from PIL import Image
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from numeric_encoder import NumericEncoder

_NUM_RE = r"([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)"  # 允许千分位和科学计数

re_b50  = re.compile(r"signal_b50\s*=\s*([0-9.+-eE]+)")
re_b800 = re.compile(r"signal_b800\s*=\s*([0-9.+-eE]+)")

def _extract_num(text: str, key: str):
    # 匹配如：signal_b50 = 24,166.9,   或 1.23e4; 末尾可有 , ;
    m = re.search(rf"{key}\s*=\s*{_NUM_RE}\s*[,;]?", text)
    if not m:
        return None
    s = m.group(1).rstrip(",;").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None

def parse_cond(text: str):
    b50  = _extract_num(text, "signal_b50")
    b800 = _extract_num(text, "signal_b800")
    if b50 is None or b800 is None:
        return None
    import torch
    return torch.tensor([b50, b800], dtype=torch.float32)

def main(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(args.base, torch_dtype=torch.float16).to(device)
    pipe.load_lora_weights(args.lora_dir)

    # 恢复 numeric encoder
    payload=torch.load(os.path.join(args.lora_dir, args.numeric_ckpt), map_location=device)
    enc=NumericEncoder(seq_len=payload["seq_len"], hidden_size=pipe.text_encoder.config.hidden_size, cond_dim=payload["cond_dim"]).to(device)
    enc.load_state_dict_light(payload)
    enc.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    txts = sorted(glob.glob(os.path.join(args.real_dir, "*.txt")))
    print("txt files:", len(txts))

    for t in tqdm(txts):
        stem=os.path.splitext(os.path.basename(t))[0]
        # 读取 sidecar 文本并解析 (b50, b800)
        with open(t,"r",encoding="utf-8") as f: 
            s=f.read().strip()

        cond=parse_cond(s); 

        if cond is None: 
            # 读取 sidecar 文本并解析 (b50, b800)
            continue
         # 形状 [1, 2]，放到 GPU
        cond=cond.unsqueeze(0).to(device)  # [1,2]
        # 前向生成
        with torch.no_grad():
            pos = enc(cond)                               # [1, L, 768]
            neg = enc.negative(batch_size=1)              # [1, L, 768]

            # ★ 与 pipe 的 UNet dtype 对齐（通常是 float16）
            target_dtype = pipe.unet.dtype
            pos = pos.to(target_dtype)
            neg = neg.to(target_dtype)

            img = pipe(prompt_embeds=pos,
                       negative_prompt_embeds=neg,
                       num_inference_steps=args.steps,
                       guidance_scale=args.guidance).images[0]
        img.save(os.path.join(args.out_dir, stem + ".png"))

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--lora_dir", required=True)      # 训练输出目录（含 LoRA + numeric_enc_*.pt）
    ap.add_argument("--numeric_ckpt", default="numeric_enc_final.pt")
    ap.add_argument("--real_dir", required=True)      # val_data_cont
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    args=ap.parse_args(); main(args)
