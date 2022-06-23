import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import numpy as np
import models
from utils import trainer
from utils.metrics import metric
from data_processing.Data_Handler import get_dataset
import time
from utils.trainer import *

class Exp_Basic(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = cfg['exp']['device']
        self.model = self._build_model()
        
        
    def _build_model(self):
        return models.gtnet(self.cfg)

    def _create_loader(self,flag="train"):
        print("create loader...")
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
        
        print("start training...")
        his_loss =[]
        val_time = []
        train_time = []
        # train_loop
        for epoch in range(epochs):
            
            epoch_start_time = time.time()
            cnt = 0
            loss_total = 0
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
                for j in range(self.cfg['exp']['num_split']):
                    if j != self.cfg['exp']['num_split']-1:
                        id = perm[j * num_sub:(j + 1) * num_sub]
                    else:
                        id = perm[j * num_sub:]
            
                id = torch.tensor(id).to(self.device).long()
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
                
                tx = tx.float()
                ty = ty.float()
                optimizer.zero_grad()
                prediction = self.model(tx,id)
                ty = ty[:,0,:,:]
                prediction = prediction.transpose(1,3)
                
                real = torch.unsqueeze(ty,dim=1)
                loss = loss_func(real, prediction)
                loss.backward()
                optimizer.step()
                cnt += 1
                loss_total += float(loss)
            epoch_end_time = time.time()
            print("Epoch: {}/{}, loss: {}, time: {}".format(epoch, epochs, loss_total/cnt, epoch_end_time-epoch_start_time)) 
            # else:
            #     for input, target in train_loader:

            #         input = input.float().to(self.device)
            #         target = target.float().to(self.device)

            #         optimizer.zero_grad()
            #         print("exp: input: ", input.shape)
            #         prediction = self.model(input)
            #         loss = loss_func(target, prediction)
            
            
        print("train finished")
        print("validating...")
        
        loss_total = 0
        cnt = 0
        for input, target, input_time, target_time in valid_loader:
            
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
            validx = torch.from_numpy(input).to(self.device)
            validx = validx.transpose(1,3)
            validy = torch.from_numpy(target).to(self.device)
            validy = validy.transpose(1,3)
            

            
            if cnt%100 == 0:
                perm = np.random.permutation(range(self.cfg['model']['num_nodes']))
                
            idx = perm[0:]
            idx = torch.tensor(idx).to(self.device).long()
            
            vx = validx[:, :, idx, :]
            vy = validy[:, :, idx, :]
            vx = vx.float()
            vy = vy.float()

            with torch.no_grad():
                predict = self.model(vx,idx)
                predict = predict.transpose(1,3)
            vy = vy[:,0,:,:]
            
            real = torch.unsqueeze(vy,dim=1)
            loss = loss_func(real, predict)
            loss_total += float(loss)

            loss.backward()
            cnt += 1
            
        print("validation finished")
        print("validation_loss: {}".format(loss_total/cnt))
        


                
            
    def test(self):
        print("start testing...")
        test_loader =self._create_loader("test")
        # train_loop
        self.model.eval()
        preds, trues = [], []
        for input, target , input_time, target_time in test_loader:
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
            with torch.no_grad():
                preds = self.model(testx)
                preds = preds.transpose(1, 3)
            preds.append(preds)
            trues.append(target)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = np.array(trues)
        trues = np.array(trues).reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

