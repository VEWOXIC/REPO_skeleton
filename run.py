import torch
import numpy as np
import json
from experiments.exp_basic import *
from datetime import datetime
import os
import time

if __name__ == '__main__':
    # if os.paths.exists()
    with open('cfgs/exp/RNN/RNN_ETTh1_test.json','r') as f:
        cfg =json.load(f)
    # else

    exp = Exp_Basic(cfg)
    before_train = datetime.now().timestamp()
    print("===================Train-Start=========================")
    exp.train()
    after_train = datetime.now().timestamp()
    print(f'Training took {(after_train - before_train) / 60} minutes')
    print("===================Train-End=========================")

    before_evaluation = datetime.now().timestamp()
    exp.test()
    after_evaluation = datetime.now().timestamp()
    print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')