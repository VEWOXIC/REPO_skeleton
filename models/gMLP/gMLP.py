# File modified from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/gMLP.py

from .imports import *
from .layers import *


class _SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
        nn.init.normal_(self.spatial_proj.weight, std=1e-6)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


class _gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = _SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class _gMLPBackbone(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, depth=6):
        super().__init__()
        self.model = nn.Sequential(
            *[_gMLPBlock(d_model, d_ffn, seq_len) for _ in range(depth)]
        )

    def forward(self, x):
        return self.model(x)


class gMLP(_gMLPBackbone):
    def __init__(self, cfg):
        c_in = cfg["model"]["c_in"]
        c_out = cfg["model"]["c_out"]
        seq_len = cfg["model"]["seq_len"]
        patch_size = cfg["model"]["patch_size"]
        d_model = cfg["model"]["d_model"]
        d_ffn = cfg["model"]["d_ffn"]
        depth = cfg["model"]["depth"]

        assert seq_len % patch_size == 0, "`seq_len` must be divisibe by `patch_size`"
        super().__init__(d_model, d_ffn, seq_len // patch_size, depth)
        self.patcher = nn.Conv1d(
            c_in, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.head = nn.Linear(d_model, c_out)

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _ = patches.shape
        patches = patches.permute(0, 2, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        # embedding = embedding.mean(dim=1)
        out = self.head(embedding)
        out = out.permute(0, 2, 1)
        return out
