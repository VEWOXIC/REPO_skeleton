# File modified from
# https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResCNN.py

from .imports import *
from .layers import *


class _ResCNNBlock(Module):
    def __init__(
            self,
            ni,
            nf,
            kss=[
                7,
                5,
                3],
            coord=False,
            separable=False,
            zero_norm=False):
        self.convblock1 = ConvBlock(
            ni, nf, kss[0], coord=coord, separable=separable)
        self.convblock2 = ConvBlock(
            nf, nf, kss[1], coord=coord, separable=separable)
        self.convblock3 = ConvBlock(
            nf,
            nf,
            kss[2],
            act=None,
            coord=coord,
            separable=separable,
            zero_norm=zero_norm,
        )

        # expand channels for the sum if necessary
        self.shortcut = ConvBN(ni, nf, 1, coord=coord)
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x


class ResCNN(Module):
    def __init__(self, cfg, coord=False, separable=False, zero_norm=False):
        c_in = cfg["model"]["c_in"]
        c_out = cfg["model"]["c_out"]
        nf = cfg["model"]["nf"]
        coord = cfg["model"]["coord"]
        separable = cfg["model"]["separable"]
        zero_norm = cfg["model"]["zero_norm"]

        self.block1 = _ResCNNBlock(
            c_in,
            nf,
            kss=[7, 5, 3],
            coord=coord,
            separable=separable,
            zero_norm=zero_norm,
        )
        self.block2 = ConvBlock(
            nf,
            nf * 2,
            3,
            coord=coord,
            separable=separable,
            act=nn.LeakyReLU,
            act_kwargs={"negative_slope": 0.2},
        )
        self.block3 = ConvBlock(
            nf * 2, nf * 4, 3, coord=coord, separable=separable, act=nn.PReLU
        )
        self.block4 = ConvBlock(
            nf * 4,
            nf * 2,
            3,
            coord=coord,
            separable=separable,
            act=nn.ELU,
            act_kwargs={"alpha": 0.3},
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.lin = nn.Linear(nf * 2, c_out)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.permute(0, 2, 1)
        # x = self.squeeze(self.gap(x))
        x = self.lin(x)
        x = x.permute(0, 2, 1)
        return x
