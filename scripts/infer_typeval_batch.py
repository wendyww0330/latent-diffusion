import os, glob, re, argparse
from PIL import Image
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from numeric_encoder_typeval import TypeValEncoder

_NUM_RE = r"([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)"

def extract(text, key):
    import re
    m = re.search(rf"{key}\s*=\s*{_NUM_RE}\s*[,;]?", text)
    if not m: return None
    s = m.group(1).rstrip(",;").replace(",", "")
    try: return float(s)
    except: return None

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(args.base, torch_dtype=torch.float16).to(device)
    pipe.load_lora_weights(args.lora_dir)

    payload = torch.load(os.path.join(args.lora_dir, args.numeric_ckpt), map_location=device)
    enc = TypeValEncoder(seq_len=payload["seq_len"],
                         hidden_size=pipe.text_encoder.config.hidden_size,
                         num_types=payload["num_types"]).to(device)
    enc.load_state_dict_light(payload); enc.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    txts = sorted(glob.glob(os.path.join(args.real_dir, "*.txt")))
    print("txt files:", len(txts))

    gen=0
    for t in tqdm(txts):
        stem = os.path.splitext(os.path.basename(t))[0]
        with open(t,"r",encoding="utf-8") as f: s=f.read().strip()

        b50  = extract(s, "signal_b50")
        b800 = extract(s, "signal_b800")
        if b50 is not None:
            type_id = torch.tensor([0], device=device, dtype=torch.long)      # 0->50
            value   = torch.tensor([b50], device=device, dtype=torch.float32) # scalar
            with torch.no_grad():
                pos = enc(type_id, value).to(pipe.unet.dtype)
                neg = enc.negative(1).to(pipe.unet.dtype)
                img = pipe(prompt_embeds=pos, negative_prompt_embeds=neg,
                           num_inference_steps=args.steps, guidance_scale=args.guidance).images[0]
            img.save(os.path.join(args.out_dir, stem + "_b50.png")); gen+=1

        if b800 is not None:
            type_id = torch.tensor([1], device=device, dtype=torch.long)      # 1->800
            value   = torch.tensor([b800], device=device, dtype=torch.float32)
            with torch.no_grad():
                pos = enc(type_id, value).to(pipe.unet.dtype)
                neg = enc.negative(1).to(pipe.unet.dtype)
                img = pipe(prompt_embeds=pos, negative_prompt_embeds=neg,
                           num_inference_steps=args.steps, guidance_scale=args.guidance).images[0]
            img.save(os.path.join(args.out_dir, stem + "_b800.png")); gen+=1

    print(f"[OK] saved {gen} images to {args.out_dir}")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--numeric_ckpt", default="typeval_enc_final.pt")
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    args=ap.parse_args()
    main(args)
