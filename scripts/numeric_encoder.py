# /workspace/scripts/numeric_encoder.py
import torch
import torch.nn as nn

class NumericEncoder(nn.Module):
    """
    把 (b50, b800) 连续值 → 一段长度为 L 的序列嵌入，维度 768（兼容 SD1.5）
    """
    def __init__(self, seq_len=16, hidden_size=768, cond_dim=2):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim

        # 学习一个“内容基底”序列（类似 CLS+context），不依赖具体数值
        self.base_seq = nn.Parameter(torch.randn(seq_len, hidden_size) * 0.02)

        # 把连续条件映射为对这段序列的偏移（FiLM 风格）
        self.cond_mlp = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, 128), nn.SiLU(),
            nn.Linear(128, seq_len * hidden_size)
        )

        # 统计量（训练前设置，一致用于推理）
        self.register_buffer("cond_mean", torch.zeros(cond_dim))
        self.register_buffer("cond_std", torch.ones(cond_dim))

        # 另外学一个“负提示”基底序列（classifier-free guidance 用）
        self.neg_base_seq = nn.Parameter(torch.randn(seq_len, hidden_size) * 0.02)

    @torch.no_grad()
    def set_norm(self, mean, std):
        self.cond_mean.copy_(torch.as_tensor(mean))
        self.cond_std.copy_(torch.as_tensor(std).clamp_min(1e-6))

    def forward(self, cond):  # cond: [B, cond_dim]（原始数值）
        device = self.base_seq.device
        B = cond.size(0)
        cond = (cond.to(device) - self.cond_mean) / self.cond_std
        delta = self.cond_mlp(cond).view(B, self.seq_len, self.hidden_size)
        base = self.base_seq.unsqueeze(0).expand(B, -1, -1)
        return base + delta  # [B, L, 768]

    def negative(self, batch_size):
        # 负提示的序列（可选也可学一个 cond=0 的映射；这里用独立可学习基底）
        return self.neg_base_seq.unsqueeze(0).expand(batch_size, -1, -1)

    def state_dict_light(self):
        return {
            "base_seq": self.base_seq.data.cpu(),
            "neg_base_seq": self.neg_base_seq.data.cpu(),
            "cond_mlp": self.cond_mlp.state_dict(),
            "cond_mean": self.cond_mean.cpu(),
            "cond_std": self.cond_std.cpu(),
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "cond_dim": self.cond_dim,
        }

    def load_state_dict_light(self, payload):
        self.base_seq.data.copy_(payload["base_seq"].to(self.base_seq.device))
        self.neg_base_seq.data.copy_(payload["neg_base_seq"].to(self.neg_base_seq.device))
        self.cond_mlp.load_state_dict(payload["cond_mlp"])
        self.cond_mean.copy_(payload["cond_mean"].to(self.base_seq.device))
        self.cond_std.copy_(payload["cond_std"].to(self.base_seq.device))
