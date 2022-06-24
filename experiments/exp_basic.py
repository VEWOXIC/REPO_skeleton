import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import numpy as np
import models
import pandas as pd
from utils.metrics import metric
from data_processing.data_handler import get_dataset
from datetime import datetime
import time
import utils
class Exp_Basic(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg['exp']['device'])
        self.model = self._build_model()
        self.model.to(self.device)
        
    def _build_model(self):
        return models.__dict__[self.cfg['model']['model_name']](self.cfg).float()


    def _create_loader(self,flag="train"):
        dataset = get_dataset(self.cfg, flag)
        batch_size = self.cfg["exp"][flag]['batchsize']
        shuffle = self.cfg["exp"][flag]['shuffle']
        drop_last = self.cfg["exp"][flag]['drop_last']
        return DataLoader(dataset,batch_size,shuffle=shuffle,drop_last=drop_last)

    def _get_optim(self):
        # TODO: just for demo， 从utils选择
        return optim.Adam(self.model.parameters(), lr=self.cfg['exp']['train']['lr'])

    def _get_lossfunc(self):
        # TODO: just for demo， 从utils选择
        return nn.L1Loss()

    def train(self):
        # TODO: just for demo use, TO BE implemented
        epochs = self.cfg['exp']['train']['epochs']
        # TODO: get train and valid loader
        train_loader = self._create_loader("train")
        valid_loader = self._create_loader("valid")

        # TODO: get loss function and optimizer according to the exp_cfg
        loss_func = self._get_lossfunc()
        optimizer = self._get_optim()

        # train_loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            cnt = 0
            loss_total = 0
            if self.cfg['model']['model_name'] == 'MTGNN':
            	for input, target, input_time, target_time in train_loader:
                    input = input.numpy()
                    target = target.numpy()
                    input = np.expand_dims(input, axis=-1)
                    target = np.expand_dims(target, axis=-1)
                    input = [input]
                    target = [target]
                    input.append(input_time)
                    target.append(target_time)
                    
                    input = np.concatenate(input,axis=-1)
                    target = np.concatenate(target,axis=-1)
                    trainx = torch.from_numpy(input).to(self.device)
                    trainx = trainx.transpose(1,3)
                    trainy = torch.from_numpy(target).to(self.device)
                    trainy = trainy.transpose(1,3)
                    if cnt%100 == 0:
                   	    perm = np.random.permutation(range(self.cfg['model']['num_nodes']))
                    num_sub = int(self.cfg['model']['num_nodes'])
                    idx = perm[0:]
                    idx = torch.tensor(idx).to(self.device)
                    tx = trainx[:,:,idx,:]
                    ty = trainy[:,:,idx,:]
                    tx = tx.float()
                    ty = ty.float()
                    optimizer.zero_grad()
                    prediction = self.model(tx,idx)
                    ty = ty[:,0,:,:]
                    prediction = prediction.transpose(1,3)
                    real = torch.unsqueeze(ty,dim=1)
                    loss = loss_func(real, prediction)
                    cnt += 1
                    loss.backward()
                    optimizer.step()
                    loss_total += float(loss)
            else:
                for input, target in train_loader:
                    input = input.float().to(self.device)
                    target = target.float().to(self.device)
                    optimizer.zero_grad()
                    prediction = self.model(input)
                    loss = loss_func(target, prediction)
                    cnt += 1
                    loss.backward() 
                    optimizer.step()
                    loss_total += float(loss)
            print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} '.format(epoch, (
                    time.time() - epoch_start_time), loss_total / cnt))
            if (epoch+1)%5 == 0 and epoch != 0:
                self.test()
        after_train = datetime.now().timestamp()
        print('Training took {(after_train - before_train) / 60} minutes')
        print("===================Normal-End=========================")


    def test(self):
        test_loader =self._create_loader("test")
        # train_loop
        self.model.eval()
        preds, trues = [], []
        if self.cfg['model']['model_name'] == "MTGNN":
            for input, target, input_time, target_time in test_loader:
                input = input.numpy()
                target = target.numpy()
                input = np.expand_dims(input, axis=-1)
                target = np.expand_dims(target, axis=-1)
                input = [input]
                target = [target]
                input.append(input_time)
                target.append(target_time)
                input = np.concatenate(input,axis=-1)
                target = np.concatenate(target,axis=-1)
                testx = torch.from_numpy(input).to(self.device)
                testx = testx.transpose(1,3)
                testy = torch.from_numpy(target).to(self.device)
                testy = testy.transpose(1,3)
                testy = testy[:,0,:,:]
                testx = testx.float()
                testy = testy.float()
                output = self.model(testx)
                output = output.transpose(1,3)
                real = torch.unsqueeze(testy, dim=1)
                prediction = output.detach().cpu().numpy()
                target = real.detach().cpu().numpy()
                preds.append(prediction)
                trues.append(target)
        else:
            for input, target in test_loader:
                input = input.float().to(self.device)
                target = target.float().to(self.device)
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
        print("------------TEST result:------------")
        print("mae:", mae, " mse:",mse," rmse:",rmse)
        print("mape:",mape," mspe:",mspe," rse:",rse)
        print("corr:",corr)
