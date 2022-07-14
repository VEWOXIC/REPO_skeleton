
import json
import os
import random
from experiments.exp_basic import *
from datetime import datetime


if __name__ == '__main__':
    with open('cfgs/exp/TransformerModel/TransformerModel_PEMS08_s12h12.json','r') as f:
        cfg =json.load(f)

    model_save_dir = 'cache/{}/{}/{}'.format(cfg['model']['model_name'], cfg['data']['dataset_name'], cfg['data']['horizon'])
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    exp = Exp_Basic(cfg, model_save_dir)
    if cfg['exp']['train']['training'] or not os.path.exists(model_save_dir):
        before_train = datetime.now().timestamp()
        print(f"===================Train-Start=========================")
        exp.train()
        after_train = datetime.now().timestamp()
        print(f'Training took {(after_train - before_train) / 60} minutes')
        print("===================Train-End=========================")
    else:
        exp.load_model()

    before_evaluation = datetime.now().timestamp()
    exp.test()
    after_evaluation = datetime.now().timestamp()
    print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
