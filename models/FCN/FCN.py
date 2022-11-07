# File modified from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/FCN.py

from .imports import *
from .layers import *


class FCN(Module):
    def __init__(self, cfg):

        c_in = cfg["model"]["c_in"]
        c_out = cfg["model"]["c_out"]
        layers = cfg["model"]["layers"]
        kss = cfg["model"]["kss"]

        assert len(layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, layers[0], kss[0])
        self.convblock2 = ConvBlock(layers[0], layers[1], kss[1])
        self.convblock3 = ConvBlock(layers[1], layers[2], kss[2])
        self.gap = GAP1d(1)
        self.fc = nn.Linear(layers[-1], c_out)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        # x = self.gap(x);print(x.size())
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x
