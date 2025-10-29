# /workspace/scripts/infer_numeric_batch.py
import os, glob, re, argparse, yaml
from PIL import Image
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline
from numeric_encoder import NumericEncoder

_NUM_RE = r"([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)"  
RE_B50  = re.compile(rf"signal_b50\s*=\s*{_NUM_RE}\s*[,;]?")
RE_B800 = re.compile(rf"signal_b800\s*=\s*{_NUM_RE}\s*[,;]?")

def _extract(m):
    s = m.group(1).rstrip(",;").replace(",", "")
    return float(s)

def parse_pair(text):
    """return (b50, b800) or (None,None)"""
    m1, m2 = RE_B50.search(text), RE_B800.search(text)
    if not (m1 and m2): return None, None
    try:
        return _extract(m1), _extract(m2)
    except ValueError:
        return None, None

def load_cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    # overwrite
    ap.add_argument("--out_dir", type=str)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--guidance", type=float)
    ap.add_argument("--mode", choices=["dual","single"])
    ap.add_argument("--key", choices=["b50","b800"])
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    for k in ["out_dir","steps","guidance","mode","key"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg

def main():
    cfg = load_cfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = cfg["base_model"]
    lora_dir   = cfg["lora_dir"]
    numeric_ck = cfg.get("numeric_ckpt", "numeric_enc_final.pt")
    real_dir   = cfg["real_dir"]
    out_dir    = cfg["out_dir"]
    steps      = int(cfg.get("steps", 30))
    guidance   = float(cfg.get("guidance", 7.5))
    mode       = cfg.get("mode", "dual")
    key        = cfg.get("key", "b50")

    os.makedirs(out_dir, exist_ok=True)

    # pipe and LoRA
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to(device)
    pipe.load_lora_weights(lora_dir)

    # numeric encoder（ckpt-cond_dim / seq_len）
    payload = torch.load(os.path.join(lora_dir, numeric_ck), map_location=device)
    cond_dim = int(payload["cond_dim"])
    seq_len  = int(payload["seq_len"])

    enc = NumericEncoder(seq_len=seq_len,
                         hidden_size=pipe.text_encoder.config.hidden_size,
                         cond_dim=cond_dim).to(device)
    enc.load_state_dict_light(payload)
    enc.eval()

    # mode
    if mode == "dual" and cond_dim != 2:
        raise RuntimeError(f"Config asks for dual but ckpt cond_dim={cond_dim}.")
    if mode == "single" and cond_dim != 1:
        raise RuntimeError(f"Config asks for single but ckpt cond_dim={cond_dim}.")

    txts = sorted(glob.glob(os.path.join(real_dir, "*.txt")))
    print("txt files:", len(txts), "| mode:", mode, "| steps:", steps, "| guidance:", guidance)

    for t in tqdm(txts):
        stem = os.path.splitext(os.path.basename(t))[0]
        with open(t, "r", encoding="utf-8") as f:
            s = f.read().strip()

        #  cond
        if mode == "dual":
            b50, b800 = parse_pair(s)
            if b50 is None:  
                continue
            cond = torch.tensor([[b50, b800]], dtype=torch.float32, device=device)   # [1,2]
        else:  # single
            b50, b800 = parse_pair(s)
            if b50 is None:
                continue
            val = b50 if key == "b50" else b800
            cond = torch.tensor([[val]], dtype=torch.float32, device=device)        # [1,1]

        # gene
        with torch.no_grad():
            pos = enc(cond).to(pipe.unet.dtype)   # [1,L,768] -> half
            neg = enc.negative(1).to(pipe.unet.dtype)
            img = pipe(
                prompt_embeds=pos,
                negative_prompt_embeds=neg,
                num_inference_steps=steps,
                guidance_scale=guidance
            ).images[0]

        img.save(os.path.join(out_dir, stem + ".png"))

if __name__ == "__main__":
    main()
