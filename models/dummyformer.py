from turtle import forward
import torch
import numpy as np
from torch import nn

class dummyformer(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.model_cfg=model_cfg
        self.B=model_cfg['batchsize']
        self.LBW=model_cfg['lookback']
        self.H=model_cfg['horizon']
        self.C=model_cfg['channel']
        self.linear=nn.Linear(self.LBW,self.H)

    def forward(self, x):
        x=self.linear(x)
        return x