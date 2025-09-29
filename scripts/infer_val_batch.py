import os, glob, argparse
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

ap=argparse.ArgumentParser()
ap.add_argument("--base", default="runwayml/stable-diffusion-v1-5")
ap.add_argument("--lora", default="/workspace/lora_out_minimal")
ap.add_argument("--real_dir", default="/workspace/val_data")
ap.add_argument("--out_dir", default="/workspace/eval_val")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--guidance", type=float, default=7.5)
args=ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
pipe=StableDiffusionPipeline.from_pretrained(args.base, torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights(args.lora)
# 不必 fuse_lora()，避免旧版警告；需要更快可加上

txts=sorted(glob.glob(os.path.join(args.real_dir,"*.txt")))
assert txts, f"No .txt in {args.real_dir}"
for i, t in enumerate(txts, 1):
    with open(t,"r",encoding="utf-8") as f: prompt=f.read().strip()
    img=pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance).images[0]
    stem=os.path.splitext(os.path.basename(t))[0]
    img.save(os.path.join(args.out_dir, stem+".png"))
    if i%50==0: print(f"[{i}/{len(txts)}]")
print("done ->", args.out_dir)
