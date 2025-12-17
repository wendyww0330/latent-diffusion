import os
import glob
import re
import yaml
import argparse
from pathlib import Path

import torch
import numpy as np
from torchvision.utils import save_image

from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from numeric_encoder import NumericEncoder


# ========== 1. 解析辅助函数 ==========
_re_b50  = re.compile(r"signal_b50\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_re_b800 = re.compile(r"signal_b800\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")

def parse_pair(text: str):
    """
    从 sidecar 文本中解析 signal_b50 / signal_b800
    返回 (b50, b800)，解析失败则返回 (None, None)
    """
    m1 = _re_b50.search(text)
    m2 = _re_b800.search(text)
    if not (m1 and m2):
        return None, None
    def to_float(m):
        return float(m.group(1).rstrip(",;").replace(",", ""))
    try:
        return to_float(m1), to_float(m2)
    except:
        return None, None


# ========== 2. DDIM 采样核心逻辑 ==========
@torch.no_grad()
def ddim_sample(unet, scheduler, zT, cond_emb, guidance_scale=1.0, uncond_emb=None):
    """
    标准的 DDIM 采样循环，支持 CFG
    """
    z = zT
    
    # 打印一下形状，确保是 4 通道
    # print(f"DEBUG: Latent shape: {z.shape}") 

    for t in scheduler.timesteps:
        # 1. 扩展 Latent 以适应 CFG (Batch * 2)
        if guidance_scale > 1.0 and uncond_emb is not None:
            latent_model_input = torch.cat([z] * 2)
            encoder_hidden_states = torch.cat([uncond_emb, cond_emb])
        else:
            latent_model_input = z
            encoder_hidden_states = cond_emb

        # 2. 预测噪声
        noise_pred = unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=encoder_hidden_states
        ).sample

        # 3. CFG 引导计算: pred = uncond + scale * (cond - uncond)
        if guidance_scale > 1.0 and uncond_emb is not None:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 4. DDIM Step
        z = scheduler.step(model_output=noise_pred, timestep=t, sample=z).prev_sample

    return z


# ========== 3. 主函数 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="推理用的 YAML 配置文件")
    args = parser.parse_args()

    # --- 读取配置 ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 123))

    # CFG 强度 (重要参数)
    # 1.0 = 不使用 CFG (普通模式)
    # 4.0 ~ 7.5 = 推荐范围，增强对比度和依从性
    guidance_scale = float(cfg.get("guidance", 1.0))

    # --- 加载 VAE ---
    vae_repo = cfg["vae_repo"]
    print(f"[VAE] Loading from {vae_repo} ...")
    vae = AutoencoderKL.from_pretrained(vae_repo).to(device)
    vae.eval()

    # ★ 关键设定 1：缩放因子 (必须是 0.18215)
    SCALING_FACTOR = float(cfg.get("scaling_factor", 1.0))

    print(f"[Config] Using Scaling Factor: {SCALING_FACTOR}")

    # --- 加载 UNet ---
    arch = cfg["unet_arch"]
    unet_ckpt = cfg["unet_ckpt"]
    print(f"[UNet] Loading from {unet_ckpt} ...")
    
    # 检查通道配置是否正确
    if arch["in_channels"] != 4 or arch["out_channels"] != 4:
        print("⚠️ 警告: 配置文件里的 in_channels/out_channels 不是 4！")
        print("   如果你的模型是重训过的 4 通道版，请去 YAML 改成 4，否则会报错。")

    unet = UNet2DConditionModel(
        sample_size=arch["sample_size"],
        in_channels=arch["in_channels"],     # 应该是 4
        out_channels=arch["out_channels"],   # 应该是 4
        down_block_types=tuple(arch["down_block_types"]),
        up_block_types=tuple(arch["up_block_types"]),
        block_out_channels=tuple(arch["block_out_channels"]),
        layers_per_block=arch["layers_per_block"],
        cross_attention_dim=arch["cross_attention_dim"],
        attention_head_dim=arch["attention_head_dim"],
    ).to(device)
    
    unet.load_state_dict(torch.load(unet_ckpt, map_location="cpu"))
    unet.eval()

    
    


    # --- 加载 NumericEncoder ---
    cond_dim   = int(cfg["cond_dim"])
    hidden     = int(cfg["hidden_size"])
    seq_len    = int(cfg["seq_len"])
    numeric_ckpt = cfg["numeric_ckpt"]

    enc = NumericEncoder(seq_len=seq_len, hidden_size=hidden, cond_dim=cond_dim).to(device)
    enc_sd = torch.load(numeric_ckpt, map_location="cpu")
    
    # 兼容处理
    if "cond_mlp" in enc_sd:
        enc.cond_mlp.load_state_dict(enc_sd["cond_mlp"], strict=True)
    elif any(k.startswith("cond_mlp.") for k in enc_sd.keys()):
        enc.load_state_dict(enc_sd, strict=True)
    else:
        enc.load_state_dict(enc_sd, strict=False)
    enc.eval()

    # ★ 关键设定 2：手动注入归一化参数
    # =======================================================
    # 【★ 请填入你用 calc_cond_stats.py 算出来的真实数值！】
    # 不要留着 0.0 和 1.0，否则模型生成的图可能不听指令
    # =======================================================
    mean_path = cfg["cond_mean_path"]
    std_path  = cfg["cond_std_path"]
    train_mean = np.load(mean_path)
    train_std  = np.load(std_path)

    enc.set_norm(train_mean, train_std)
    print(f"[Encoder] Norm loaded: mean={train_mean}, std={train_std}")


    # --- DDIM Scheduler ---
    sch_cfg = cfg["scheduler"]
    scheduler = DDIMScheduler(
        num_train_timesteps=sch_cfg["num_train_timesteps"],
        beta_start=sch_cfg["beta_start"],
        beta_end=sch_cfg["beta_end"],
        beta_schedule=sch_cfg["beta_schedule"],
        clip_sample=False,
    )
    steps = int(cfg["steps"])
    scheduler.set_timesteps(steps)
    print("[Scheduler] clip_sample =", getattr(scheduler.config, "clip_sample", None))


    print("prediction_type:", getattr(scheduler.config, "prediction_type", None))

    # --- 准备输入 ---
    latent_size     = arch["sample_size"]     # 32
    latent_channels = arch["in_channels"]     # 4 (关键)

    real_dir = cfg["real_dir"]
    out_dir  = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    txts = sorted(glob.glob(os.path.join(real_dir, "*.txt")))
    bs = int(cfg.get("batch_size", 1))

    print(f"[Start] Found {len(txts)} files. Batch={bs}, Steps={steps}, CFG={guidance_scale}")

    # --- 推理循环 ---
    for i in range(0, len(txts), bs):
        batch_txts = txts[i:i+bs]
        stems, conds = [], []
        
        # 1. 解析条件
        for tpath in batch_txts:
            with open(tpath, "r", encoding="utf-8") as f:
                s = f.read().strip()
            b50, b800 = parse_pair(s)
            if b50 is None:
                # 容错：如果解析失败，默认给 0
                b50, b800 = 0.0, 0.0
            
            stem = os.path.splitext(os.path.basename(tpath))[0]
            stems.append(stem)
            conds.append([b50, b800])

        if not stems: continue

        cond = torch.tensor(conds, dtype=torch.float32, device=device)  # [B,2]
        B = cond.shape[0]
        cond_np = cond.detach().cpu().numpy()
        print(f"[COND raw]  b50/b800 = {cond_np}")
        if hasattr(enc, "norm_mean") and hasattr(enc, "norm_std"):
    # 如果你的 encoder 内部保存了 norm 参数
            pass
        cond_norm = (cond - torch.tensor(train_mean, device=device)) / torch.tensor(train_std, device=device)
        print(f"[COND norm] b50/b800 = {cond_norm.detach().cpu().numpy()}")

        # 2. 准备 Embedding
        with torch.no_grad():
            cond_emb = enc(cond).to(device=device, dtype=unet.dtype)    # [B,L,H]
            
            # 构造无条件 Embedding (全0) 用于 CFG
            if guidance_scale > 1.0:
                uncond_emb = torch.zeros_like(cond_emb)
            else:
                uncond_emb = None

        # 3. 初始化噪声 (Latent)
        # 形状应该是 [B, 4, 32, 32]
        zT = torch.randn(B, latent_channels, latent_size, latent_size,
                         device=device, dtype=unet.dtype)

        # 4. 去噪采样（只调用一次）
        z0 = ddim_sample(
            unet,
            scheduler,
            zT,
            cond_emb,
            guidance_scale=guidance_scale,
            uncond_emb=uncond_emb
        )

        print(f"[Config] scaling_factor(from yaml)={SCALING_FACTOR}")
        print(f"[zT] std={zT.std().item():.4f}")
        print(f"[z0 BEFORE] mean={z0.mean().item():.4f} std={z0.std().item():.4f} "
            f"min={z0.min().item():.4f} max={z0.max().item():.4f}")

        # ===== 核心修正：对齐训练时的 latent 分布 =====
        z0 = z0 / z0.std(dim=(1,2,3), keepdim=True).clamp(min=1e-6)

        print(f"[z0 AFTER ] mean={z0.mean().item():.4f} std={z0.std().item():.4f} "
            f"min={z0.min().item():.4f} max={z0.max().item():.4f}")

        # 5. 解码
        with torch.no_grad():
            x = vae.decode(z0 / SCALING_FACTOR).sample
            print(f"[x_decoded] mean/std/min/max="
                f"{x.mean().item():.4f}/{x.std().item():.4f}/"
                f"{x.min().item():.4f}/{x.max().item():.4f}")

            x = (x.clamp(-1, 1) + 1) * 0.5
            print(f"[x_img01] mean/std/min/max="
                f"{x.mean().item():.4f}/{x.std().item():.4f}/"
                f"{x.min().item():.4f}/{x.max().item():.4f}")




            # ==========================================
            # 【新增】强制二值化 (Binarization)
            # ==========================================
            # 因为细胞分割图只需要黑(0)和白(1)
            # 0.5 是阈值，大于 0.5 变白，小于 0.5 变黑
            # x = (x > 0.5).float() 
            # ==========================================

        # 6. 保存
        for idx, stem in enumerate(stems):
            save_path = os.path.join(out_dir, f"{stem}.png")
            save_image(x[idx:idx+1], os.path.join(out_dir, f"{stem}_gray.png"), nrow=1)

            save_image(x[idx:idx+1], save_path, nrow=1)
            print(f"[Saved] {save_path}")

    print(f"[Done] All saved to {out_dir}")


if __name__ == "__main__":
    main()