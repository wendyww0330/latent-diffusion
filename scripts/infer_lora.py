# 在 /workspace/scripts 下创建推理脚本

from diffusers import StableDiffusionPipeline
import torch, os

BASE = os.environ.get("MODEL_NAME", "runwayml/stable-diffusion-v1-5")
LORA = os.environ.get("LORA_DIR", "/workspace/lora_out_minimal")
OUT  = os.environ.get("OUT_IMG", "/workspace/sample.png")
PROMPT = os.environ.get("PROMPT", "signal_b50=24000, histology patch, high detail")
STEPS = int(os.environ.get("STEPS", "30"))
GUIDE = float(os.environ.get("GUIDE", "7.5"))

pipe = StableDiffusionPipeline.from_pretrained(BASE, torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights(LORA)
pipe.fuse_lora()  # 可选

img = pipe(PROMPT, num_inference_steps=STEPS, guidance_scale=GUIDE).images[0]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
img.save(OUT)
print("saved ->", OUT)

