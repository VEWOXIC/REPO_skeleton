import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import numpy as np
import models
from utils.metrics import metric
from data_processing.Data_Handler import get_dataset
import time

class Exp_Basic(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg['exp']['device'])
        self.model = self._build_model()
        self.model.to(self.device)
        
    def _build_model(self):
        return models.__dict__[self.cfg['model']['model_name']](self.cfg)

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
        # TODO: just for demo, TO BE implemented
        epochs = self.cfg['exp']['train']['epochs']
        # TODO: get train and valid loader
        train_loader = self._create_loader("train")
        valid_loader = self._create_loader("valid")

        # TODO: get loss function and optimizer according to the exp_cfg
        loss_func = self._get_lossfunc()
        #optimizer = self._get_optim()

        # train_loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            loss_total = 0
            iter_count = 0

            for input, target, input_time, target_time in train_loader:
                    input.float().to(self.device), target.float().to(self.device), input_time.float().to(self.device), target_time.float().to(self.device)

                optimizer.zero_grad()
                print(input.size())
                prediction = self.model(input) if not self.cfg['model']['UseTimeFeature'] else self.model(input,input_time,target_time)
                loss = loss_func(target, prediction)
                iter_count += 1
                loss.backward() 
                optimizer.step()
                loss_total += float(loss)

                # if self.cfg['model']['model_name'] == 'MTGNN':
                    
                #     for input, target, input_time, target_time in train_loader:
                #         input = input.numpy()
                #         target = target.numpy()
                #         input = np.expand_dims(input, axis=-1)
                #         target = np.expand_dims(target, axis=-1)
                #         input = [input]
                #         target = [target]
                #         input.append(input_time)
                #         target.append(target_time)
                        
                #         input = np.concatenate(input,axis=-1)
                #         target = np.concatenate(target,axis=-1)
                #         trainx = torch.from_numpy(input).to(self.device)
                #         trainx = trainx.transpose(1,3)
                #         trainy = torch.from_numpy(target).to(self.device)
                #         trainy = trainy.transpose(1,3)
                #         if iter_count%100 == 0:
                #        	    perm = np.random.permutation(range(self.cfg['model']['num_nodes']))
                #         num_sub = int(self.cfg['model']['num_nodes'])
                #         idx = perm[0:]
                #         idx = torch.tensor(idx).to(self.device)
                #         tx = trainx[:,:,idx,:]
                #         ty = trainy[:,:,idx,:]
                #         tx = tx.float()
                #         ty = ty.float()
                #         optimizer.zero_grad()
                #         prediction = self.model(tx,idx)
                #         ty = ty[:,0,:,:]
                #         prediction = prediction.transpose(1,3)
                #         real = torch.unsqueeze(ty,dim=1)
                #         loss = loss_func(real, prediction)
                #         iter_count += 1
                #         loss.backward()
                #         optimizer.step()
                #         loss_total += float(loss)

            print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} '.format(epoch, (
                    time.time() - epoch_start_time), loss_total / iter_count))

            self.test(valid_loader)

    def test(self, data_loader=None):
        if data_loader is None:
            data_loader = self._create_loader("test")

        self.model.eval()
        preds, trues = [], []

        for input, target, input_time, target_time in data_loader:
            input, target, input_time, target_time = \
                input.float().to(self.device), target.float().to(self.device), input_time.float().to(self.device), target_time.float().to(self.device)
            
            prediction = self.model(input) if not self.cfg['model']['UseTimeFeature'] else self.model(input,input_time,target_time)
            prediction = prediction.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            preds.append(prediction)
            trues.append(target)

        # if self.cfg['model']['model_name'] == "MTGNN":
        #     for input, target, input_time, target_time in loader:
        #         input = input.numpy()
        #         target = target.numpy()
        #         input = np.expand_dims(input, axis=-1)
        #         target = np.expand_dims(target, axis=-1)
        #         input = [input]
        #         target = [target]
        #         input.append(input_time)
        #         target.append(target_time)
        #         input = np.concatenate(input,axis=-1)
        #         target = np.concatenate(target,axis=-1)
        #         testx = torch.from_numpy(input).to(self.device)
        #         testx = testx.transpose(1,3)
        #         testy = torch.from_numpy(target).to(self.device)
        #         testy = testy.transpose(1,3)
        #         testy = testy[:,0,:,:]
        #         testx = testx.float()
        #         testy = testy.float()
        #         output = self.model(testx)
        #         output = output.transpose(1,3)
        #         real = torch.unsqueeze(testy, dim=1)
        #         prediction = output.detach().cpu().numpy()
        #         target = real.detach().cpu().numpy()
        #         preds.append(prediction)
        #         trues.append(target)

        preds, trues = np.array(preds),np.array(trues)
        preds, trues = preds.reshape(-1, preds.shape[-2], preds.shape[-1]), trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print("------------TEST result:------------")
        print("mae:", mae, " mse:",mse," rmse:",rmse)
        # print("mape:",mape," mspe:",mspe," rse:",rse)
        # print("corr:",corr)
