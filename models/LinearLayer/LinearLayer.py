from turtle import forward

import numpy as np
import torch
from torch import nn


class LinearLayer(nn.Module):
    def __init__(self, cfg):
        super(LinearLayer, self).__init__()
        self.num_layer = cfg["model"]["num_layers"]
        self.seq_len = cfg["data"]["lookback"]
        self.pred_len = cfg["data"]["horizon"]
        self.linears = nn.ModuleList()
        for i in range(self.num_layer - 1):
            self.linears.append(nn.Linear(self.seq_len, self.seq_len))
        self.linears.append(nn.Linear(self.seq_len, self.pred_len))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for i in range(self.num_layer - 1):
            x = self.linears[i](x)
        x = self.linears[-1](x)
        return x.permute(0, 2, 1)
