import torch
import numpy as np
import json
from experiments.Basic_EXP import Basic_EXP

if __name__ == '__main__':
    experiment_id='dummyformer'
    with open('./settings/'+experiment_id+'/data_setting.json','r') as f:
        data_cfg=json.load(f)
    with open('./settings/'+experiment_id+'/model_setting.json','r') as f:
        model_cfg=json.load(f)
    with open('./settings/'+experiment_id+'/exp_setting.json','r') as f:
        exp_cfg=json.load(f)
        # TODO: maybe change dict to Dotdict
    exp=Basic_EXP(model_cfg,data_cfg,exp_cfg)
    exp.train()
    exp.test()
