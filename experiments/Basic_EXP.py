import torch
import numpy as np
from torch.utils.data import DataLoader
from data_processing import Data_Handler
from utils import data_utils
import pandas as pd
from torch import nn
from torch import optim

class Basic_EXP(object):
    def __init__(self, model_cfg, data_cfg, exp_cfg) -> None:

        self.model_cfg=model_cfg
        self.data_cfg=data_cfg
        self.exp_cfg=exp_cfg

        self.model=self._build_model(model_cfg)

        self.train_handler,self.test_handler,self.valid_handler = self._load_dataset(data_cfg)

    def _build_model(self,model_cfg):
        if model_cfg['model_name']=='dummyformer':
            from models.dummyformer import dummyformer as model
        elif model_cfg['model_name']=='xxx':
            # import your model here
            pass 
        _model=model(model_cfg).double()
        print(_model)
        return _model

    def _load_dataset(self, data_cfg):
        path='./datasets/dummy_dataset.csv' # should be decided according to data_cfg
        data_dict={'train':[],'valid':[],'test':[]}
        # 1. load file
        df=pd.read_csv(path)
        # 2. preprocessing
        df=data_utils.dummy_normalize(df)
        # 3. train/valid/test split # TODO
        data_dict['train']=df.values[:100]
        data_dict['valid']=df.values[100:150]
        data_dict['test']=df.values[150:] # should be decided according to data_cfg split by ratio or absolute number
        
        train_handler=Data_Handler.Data_Handler(data_dict['train'], data_cfg['horizon'],data_cfg['lookback'])
        test_handler=Data_Handler.Data_Handler(data_dict['test'], data_cfg['horizon'],data_cfg['lookback'])
        valid_handler=Data_Handler.Data_Handler(data_dict['valid'], data_cfg['horizon'],data_cfg['lookback'])

        return train_handler,test_handler,valid_handler

    def _create_loader(self, exp_cfg, datahandler,istrain=False):
        # batchify
        return DataLoader(datahandler,exp_cfg['batchsize'],shuffle=istrain,drop_last=True) # TODO
    def _get_optim(self):
        # TODO: just for demo
        return optim.Adam(self.model.parameters(), lr=self.exp_cfg['lr'])
    def _get_lossfunc(self):
        # TODO: just for demo
        return nn.L1Loss()
    def train(self):
        # TODO: just for demo use, TO BE implemented
        epochs=self.exp_cfg['epochs']
        # TODO: get train and valid loader
        train_loader=self._create_loader(self.exp_cfg, self.train_handler)
        valid_loader=self._create_loader(self.exp_cfg, self.valid_handler)

        # TODO: get loss function and optimizer according to the exp_cfg
        loss_func=self._get_lossfunc()
        optimizer=self._get_optim()
        #

        # train_loop
        for i in range(epochs):
            for input, observation in train_loader:
                optimizer.zero_grad()

                prediction=self.model(input)
                loss=loss_func(observation, prediction)

                loss.backward() # xxxxx
                optimizer.step()

    def test(self):
        pass
