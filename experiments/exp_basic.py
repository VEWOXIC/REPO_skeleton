import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import numpy as np
import models
from utils.metrics import metric
from data_processing.data_handler import get_dataset

class Exp_Basic(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = cfg['exp']['device']
        self.model = self._build_model()
        self.train_handler,self.test_handler,self.valid_handler = self._load_dataset()
        
    def _build_model(self,cfg):
        return models.__dict__[cfg['model']['model_name']](cfg).float()

    def _create_loader(self,flag="train"):
        return DataLoader(get_dataset(self.cfg, flag),self.cfg["exp"][flag]['batchsize'],shuffle=self.cfg["exp"][flag]['shuffle'],drop_last=self.cfg["exp"][flag]['drop_last'])

    def _get_optim(self):
        # TODO: just for demo， 从utils选择
        return optim.Adam(self.model.parameters(), lr=self.exp_cfg['lr'])

    def _get_lossfunc(self):
        # TODO: just for demo， 从utils选择
        return nn.L1Loss()

    def train(self):
        # TODO: just for demo use, TO BE implemented
        epochs=self.cfg['exp']['train']['epochs']
        # TODO: get train and valid loader
        train_loader=self._create_loader("train")
        valid_loader=self._create_loader("valid")

        # TODO: get loss function and optimizer according to the exp_cfg
        loss_func=self._get_lossfunc()
        optimizer=self._get_optim()

        # train_loop
        for epoch in range(epochs):
            for input, target in train_loader:
                input = input.float().to(self.device)
                target = target.float().to(self.device)

                optimizer.zero_grad()
                prediction = self.model(input)
                loss = loss_func(target, prediction)

                loss.backward() 
                optimizer.step()

    def test(self):
        test_loader =self._create_loader("test")
        # train_loop
        self.model.eval()
        preds, trues = [], []
        for input, target in test_loader:
            prediction = self.model(input)
            prediction = prediction.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            preds.append(prediction)
            trues.append(target)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = np.array(trues)
        trues = np.array(trues).reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

