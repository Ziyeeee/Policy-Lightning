import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)

    def forward(self, x):
        device = x.device
        if self.emb.device != device:
            self.emb = self.emb.to(device)
        emb = x[:, None] * self.emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SinusoidalPosEmbFM(nn.Module):
    def __init__(self, dim, min_period=4e-3, max_period=4.0):
        super().__init__()
        half_dim = dim // 2
        fraction = torch.linspace(0.0, 1.0, half_dim)
        period = min_period * (max_period / min_period) ** fraction
        # Compute the outer product
        scaling_factor = 1.0 / period * 2 * math.pi
        self.register_buffer('scaling_factor', scaling_factor)

    def forward(self, x):
        device = x.device
        if self.scaling_factor.device != device:
            self.scaling_factor = self.scaling_factor.to(device)
        emb = self.scaling_factor[None, :] * x[:, None]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb