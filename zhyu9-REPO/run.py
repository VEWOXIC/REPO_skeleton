import torch
import numpy as np
import json
from experiments.exp_basic import Exp_Basic
import os

if __name__ == '__main__':
    model = 'Autoformer'
    dataset = 'ETTh1'
    cfg_name = 'Autoformer'

    # if os.paths.exists()
    with open('cfgs/exp/Autoformer/Autoformer.json','r') as f:
        cfg =json.load(f)
    # else
    
    exp = Exp_Basic(cfg)
    exp.train()

