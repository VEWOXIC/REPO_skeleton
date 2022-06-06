import torch
import numpy as np
import json
from experiments.exp_basic import Exp_Basic
import os

if __name__ == '__main__':
    model = 'LinearLayer'
    dataset = 'dummy_dataset'
    cfg_name = 'default'

    # if os.paths.exists()
    with open('cfgs/exp/LinearLayer/LinearLayer_dummpy_dataset.json','r') as f:
        cfg =json.load(f)
    # else
    
    exp = Exp_Basic(cfg)
    exp.train()
    exp.test()
