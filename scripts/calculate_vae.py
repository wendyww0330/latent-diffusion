import os
import glob
import yaml
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL

# ==========================================
# 1. 复制过来的 Dataset 定义 (保持完全一致)
# ==========================================
class GrayDataset(Dataset):
    def __init__(self, root, image_size=256):
        self.imgs = []
        # 你的代码里支持的后缀
        for ext in ["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp"]:
            self.imgs += glob.glob(os.path.join(root, ext))
        self.imgs = sorted(self.imgs)
        
        # 预处理必须和训练 VAE 时一模一样
        self.tf = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Lambda(lambda im: im.convert("L")),
            transforms.ToTensor(),              
            transforms.Normalize([0.5],[0.5])   
        ])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        x = Image.open(self.imgs[i])
        return self.tf(x)

# ==========================================
# 2. 计算 Scaling Factor 的主逻辑
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="VAE的配置文件路径 (例如 vae_config.yaml)")
    parser.add_argument("--vae_path", required=True, help="训练好的VAE权重路径 (例如 outputs/vae/final)")
    args = parser.parse_args()

    # 读取配置
    print(f"Loading config from {args.config} ...")
    cfg = yaml.safe_load(open(args.config, "r"))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 准备数据 ---
    data_dir = cfg["data"]["train_dir"]
    image_size = int(cfg["data"]["image_size"])
    batch_size = 16  # 推理不需要太大 batch，够快就行

    print(f"Data Dir: {data_dir}")
    print(f"Image Size: {image_size}")

    ds = GrayDataset(data_dir, image_size=image_size)
    if len(ds) == 0:
        print("错误：未找到图像文件，请检查路径。")
        return
        
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    # --- 加载 VAE ---
    print(f"Loading VAE from {args.vae_path} ...")
    try:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    except Exception as e:
        print(f"加载 VAE 失败: {e}")
        print("尝试加载 checkpoint 文件夹...")
        # 有时候可能指向的是 ckpt_stepXXXX 文件夹
        vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae").to(device)
        
    vae.eval()

    # --- 开始扫描 ---
    latents_list = []
    print("开始扫描数据集计算 Latent 统计信息...")
    
    with torch.no_grad():
        for x in tqdm(dl):
            x = x.to(device)
            # 编码
            dist = vae.encode(x).latent_dist
            # 获取均值 (Mean) 作为 Latent 的代表值
            z = dist.mean 
            latents_list.append(z.cpu().numpy())

    # 拼接结果 [Total_Images, Channels, H, W]
    all_latents = np.concatenate(latents_list, axis=0)
    
    # --- 统计计算 ---
    std = np.std(all_latents)
    mean = np.mean(all_latents)
    
    print("-" * 30)
    print(f"数据集样本数: {len(ds)}")
    print(f"Latent Mean (均值): {mean:.6f}")
    print(f"Latent Std  (标准差): {std:.6f}")
    print("-" * 30)
    
    if std < 1e-6:
        print("警告：标准差极小，可能是模型坍塌或全黑/全白输入。")
        calc_scale = 1.0
    else:
        calc_scale = 1.0 / std

    print(f"★ 建议使用的 Scaling Factor (1/std): {calc_scale:.6f}")
    print("-" * 30)

if __name__ == "__main__":
    main()