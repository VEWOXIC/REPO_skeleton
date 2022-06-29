import json
import os
import random
from experiments.exp_basic import *
from datetime import datetime


if __name__ == '__main__':
    with open('cfgs/exp/MTGNN/MTGNN_ETTh1_example.json','r') as f:
        cfg =json.load(f)

    exp_id = cfg['data']['exp_id']

    if exp_id is None:
        exp_id = int(random.SystemRandom().random() * 100000)
        cfg['data']['exp_id'] = exp_id
    model_cache_file = '/cache/{}/model_cache/{}_{}.m'.format(exp_id, cfg['model']['model_name'], cfg['data']['dataset_name'])
    exp = Exp_Basic(cfg, model_cache_file)
    if cfg['exp']['train']['training'] or not os.path.exists(model_cache_file):
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
