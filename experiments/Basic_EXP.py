import torch
import numpy as np
from torch.utils.data import DataLoader
from data_processing import Data_Handler

class Basic_EXP(object):
    def __init__(self, model_cfg, data_cfg, exp_cfg) -> None:
        
        self.model_cfg=model_cfg
        self.data_cfg=data_cfg
        self.exp_cfg=exp_cfg

        self.model=self._build_model(model_cfg)

        self.train_handler,self.test_handler,self.valid_handler = self._load_dataset(data_cfg)

    def _build_model(self,model_cfg):
        if model_cfg['model_name']=='dummyformer':
            from models import dummyformer as model
        elif model_cfg['model_name']=='xxx':
            # import your model here
            pass 
        _model=model(model_cfg)
        print(_model)
        return _model

    def _load_dataset(self, data_cfg):
        # 1. load file
        # 2. preprocessing
        # 3. train/valid/test split # TODO
        data_dict={'train':[],'valid':[],'test':[]}
        train_handler=Data_Handler.Data_Handler(data_dict['train'])
        test_handler=Data_Handler.Data_Handler(data_dict['test'])
        valid_handler=Data_Handler.Data_Handler(data_dict['valid'])

        return train_handler,test_handler,valid_handler

    def _create_loader(self, exp_cfg, datahandler):
        # batchify
        return DataLoader(datahandler) # TODO
    def _get_optim(self):
        pass
    def train(self):
        train_loader=self._create_loader(self.exp_cfg, )
    def test(self):
        pass
