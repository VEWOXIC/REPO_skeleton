import torch
import numpy as np
import json
from experiments.Basic_EXP import Basic_EXP

if __name__ == '__main__':
    with open('./settings/dummy_setting.json','r') as f:
        model_cfg=json.load(f)
    exp=Basic_EXP(model_cfg)
    exp.train()
    exp.test()
