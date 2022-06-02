import torch
import numpy as np

class Basic_EXP(object):
    def __init__(self, model_cfg, data_cfg, exp_cfg) -> None:

        self.model=self._build_model(model_cfg)

        self.train_loader, self.valid_loader, self.test_loader = self._get_dataloader(data_cfg)

    def _build_model(self,model_cfg):
        if model_cfg['model_name']=='dummyformer':
            from models import dummyformer as model
        elif model_cfg['model_name']=='xxx':
            # import your model here
            pass 
        _model=model(model_cfg)
        print(_model)
        return _model
    def _get_dataloader(self, data_cfg):

        return train_loader, valid_loader, test_loader
    def _get_optim(self):
        pass
    def train(self):
        pass
    def test(self):
        pass
