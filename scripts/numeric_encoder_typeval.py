import torch
import torch.nn as nn

class TypeValEncoder(nn.Module):
    """
    条件 = (b_type, value)
      - b_type: 0 表示 50；1 表示 800（离散）
      - value : 对应数值（连续）
    输出: [B, L, hidden] 作为 UNet cross-attn 的 encoder_hidden_states
    """
    def __init__(self, seq_len=16, hidden_size=768, num_types=2):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_types = num_types

        # 可学习基底序列（与条件无关）
        self.base_seq = nn.Parameter(torch.randn(seq_len, hidden_size) * 0.02)

        # 类型嵌入（离散）
        self.type_emb = nn.Embedding(num_types, hidden_size)
        self.type_proj = nn.Linear(hidden_size, seq_len * hidden_size)

        # 连续数值的 MLP（1 维 → [L,hidden]）
        self.val_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(),
            nn.Linear(64, seq_len * hidden_size)
        )

        # 负提示用的基底序列
        self.neg_base_seq = nn.Parameter(torch.randn(seq_len, hidden_size) * 0.02)

        # 归一化统计（按类型各自的 mean/std）
        self.register_buffer("val_mean_per_type", torch.zeros(num_types))
        self.register_buffer("val_std_per_type",  torch.ones(num_types))

    @torch.no_grad()
    def set_norm_per_type(self, means, stds):
        self.val_mean_per_type.copy_(torch.as_tensor(means).view(-1))
        self.val_std_per_type.copy_(torch.as_tensor(stds).view(-1).clamp_min(1e-6))

    def forward(self, b_type_ids, values):
        """
        b_type_ids: [B]  (0=50, 1=800)
        values    : [B]  原始数值（float）
        """
        device = self.base_seq.device
        B = values.shape[0]

        # 按类型做 value 标准化
        means = self.val_mean_per_type[b_type_ids].to(device)   # [B]
        stds  = self.val_std_per_type[b_type_ids].to(device)    # [B]
        v = ((values.to(device) - means) / stds).unsqueeze(-1)  # [B,1]

        # 类型偏置
        t = self.type_emb(b_type_ids.to(device))                # [B, hidden]
        t = self.type_proj(t).view(B, self.seq_len, self.hidden_size)  # [B,L,H]

        # 数值偏置
        dv = self.val_mlp(v).view(B, self.seq_len, self.hidden_size)   # [B,L,H]

        base = self.base_seq.unsqueeze(0).expand(B, -1, -1)            # [B,L,H]
        return base + t + dv                                            # [B,L,H]

    def negative(self, batch_size):
        return self.neg_base_seq.unsqueeze(0).expand(batch_size, -1, -1)

    def state_dict_light(self):
        return {
            "base_seq": self.base_seq.data.cpu(),
            "neg_base_seq": self.neg_base_seq.data.cpu(),
            "type_emb": self.type_emb.state_dict(),
            "type_proj": self.type_proj.state_dict(),
            "val_mlp": self.val_mlp.state_dict(),
            "val_mean_per_type": self.val_mean_per_type.cpu(),
            "val_std_per_type": self.val_std_per_type.cpu(),
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_types": self.num_types,
        }

    def load_state_dict_light(self, payload):
        self.base_seq.data.copy_(payload["base_seq"].to(self.base_seq.device))
        self.neg_base_seq.data.copy_(payload["neg_base_seq"].to(self.neg_base_seq.device))
        self.type_emb.load_state_dict(payload["type_emb"])
        self.type_proj.load_state_dict(payload["type_proj"])
        self.val_mlp.load_state_dict(payload["val_mlp"])
        self.val_mean_per_type.copy_(payload["val_mean_per_type"].to(self.base_seq.device))
        self.val_std_per_type.copy_(payload["val_std_per_type"].to(self.base_seq.device))
