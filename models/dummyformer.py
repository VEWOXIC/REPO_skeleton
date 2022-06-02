from turtle import forward
import torch
import numpy as np
from torch import nn

class dummyformer(nn.Module):
    def __init__(self, model_cfg):
        super(dummyformer,self).__init__()
        self.model_cfg=model_cfg
        self.LBW=model_cfg['lookback']
        self.H=model_cfg['horizon']
        self.C=model_cfg['channel']
        self.linear=nn.Conv1d(self.LBW,self.H,1)

    def forward(self, x):
        x=self.linear(x)
        return x