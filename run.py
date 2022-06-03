import torch
import numpy as np
import json
from experiments.exp_basic import Exp_Basic
import os

if __name__ == '__main__':
    """
    cfg system:

    用户提供 模型, dataset, cfg名称(非必要), 一些参数设置(非必要)
    系统读取 cfgs/exps/模型/模型_dataset_cfg名称.json

    cfg名称默认为default, 未提供cfg名称时
    系统读取 cfgs/exps/模型/模型_dataset_default.json, 不存在的话自动创建

    用户提供的参数用于inplace修改cfg内容, 仅针对一次实验, 不保存

    cfg结合exp,model,data为一个整体, 传入模型和exp.py, 书写格式参考cfgs/exps/basic.json

    TODO
    1:  cfgs/exps/basic.json 提供详细注释和说明
    2:  exp文件我还没修改完, 想改的内容写注释了
    
    新增模型时: 
    1. models/模型/README.md 介绍模型, 放一点实验结果(如果有)
    2. 基础的模型设置, 写在cfgs/model/模型.json
    3. 写论文中部分实验使用的cfg, 写在cfgs/exp/模型/
    """

    model='LinearLayer'
    dataset='dummy_dataset'
    cfg_name = 'default'

    # if os.paths.exists()
    with open('cfgs/exp/LinearLayer/LinearLayer_dummpy_dataset.json','r') as f:
        cfg =json.load(f)
    # else
    
    exp=Exp_Basic(cfg)
    exp.train()
    exp.test()
