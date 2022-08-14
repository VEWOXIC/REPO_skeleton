
import json
import os
import random
from experiments.exp_basic import *
from datetime import datetime

if __name__ == '__main__':
    # if os.paths.exists()
    # with open('cfgs/exp/MTGNN/MTGNN_yellow_taxi_2022-01.json','r') as f:
    # with open('cfgs/exp/MTGNN/MTGNN_wiki_rolling_nips.json','r') as f:
    # with open('cfgs/exp/SCINet/SCINet_wind.json','r') as f:
    # with open('cfgs/exp/MTGNN/MTGNN_ETTh1_example.json','r') as f:
    with open('/work/REPO_skeleton/cfgs/exp/DLinear/DLinear_ETTh1_96_720.json','r') as f:
        cfg =json.load(f)
    # else


    model_save_dir = 'cache/{}/{}/{}'.format(cfg['model']['model_name'], cfg['data']['dataset_name'], cfg['data']['horizon'])
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    model_name = cfg['model']['model_name']
    dataset_name = cfg['data']['dataset_name']
    print("Start training for model:", model_name, " dataset:", dataset_name)
    exp = Exp_Basic(cfg, model_save_dir)
    if cfg['exp']['train']['training'] or not os.path.exists(model_save_dir):
        before_train = datetime.now().timestamp()
        print("===================Train-Start=========================")
        exp.train()
        after_train = datetime.now().timestamp()
        print(f'Training took {(after_train - before_train) / 60} minutes')
        print("===================Train-End=========================")
    else:
        exp.load_model()

    before_evaluation = datetime.now().timestamp()
    exp.test()
    after_evaluation = datetime.now().timestamp()
    print('Test/evaluation took: {} minutes'.format((after_evaluation - before_evaluation) / 60))
