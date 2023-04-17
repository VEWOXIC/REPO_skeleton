import json
import os
import random
from experiments.exp_basic import *
from datetime import datetime
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    
    parser.add_argument('--cfg_file', type=str, required=True, default='NA', help='path to the config file')
    
    args = parser.parse_args()
    # with open('cfgs/exp/MTGNN/MTGNN_metr_la.json','r') as f:
    with open(args.cfg_file,'r') as f:
        cfg =json.load(f)
    # else


    model_save_dir = 'cache/{}/{}/{}'.format(cfg['model']['model_name'], cfg['data']['dataset_name'], cfg['data']['horizon'])
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    model_name = cfg['model']['model_name']
    dataset_name = cfg['data']['dataset_name']
    exp = Exp_Basic(cfg, model_save_dir)
    if cfg['exp']['train']['training'] or not os.path.exists(model_save_dir):
        print("Start training for model:", model_name, " dataset:", dataset_name)
        before_train = datetime.now().timestamp()
        print("===================Train-Start=========================")
        exp.train()
        after_train = datetime.now().timestamp()
        print(f'Training took {(after_train - before_train) / 60} minutes')
        print("===================Train-End=========================")
    else:
        exp.load_model()
    print("Start testing for model:", model_name, " dataset:", dataset_name)
    before_evaluation = datetime.now().timestamp()
    exp.test()
    after_evaluation = datetime.now().timestamp()
    print('Test/evaluation took: {} minutes'.format((after_evaluation - before_evaluation) / 60))