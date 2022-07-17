import torch
from torch.utils.data import DataLoader
import numpy as np
import models
from utils.metrics import metric
from data_processing.Data_Handler import get_dataset
import utils.exp_utils
import time

class Exp_Basic(object):
    def __init__(self, cfg, file_dir) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg['exp']['device'])
        self.file_dir = file_dir
        self.model = self._build_model()
        self.model.to(self.device)
        self.loss_func = self._get_lossfunc()
        self.optimizer = self._get_optim()
    


    def _build_model(self):
        return models.__dict__[self.cfg['model']['model_name']](self.cfg).float()

    def _create_loader(self,flag="train"):
        dataset = get_dataset(self.cfg, flag)
        batch_size = self.cfg["exp"][flag]['batchsize']
        shuffle = self.cfg["exp"][flag]['shuffle']
        drop_last = self.cfg["exp"][flag]['drop_last']
        return DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)

    def _get_optim(self):
        return utils.exp_utils.build_optimizer(self.cfg, self.model)

    def _get_lossfunc(self):
        return utils.exp_utils.build_train_loss(self.cfg)

    def load_model(self):
        self.model, self.optimizer = utils.exp_utils.load_model(self.file_dir, self.model, self.optimizer)

    def train(self):
        # TODO: just for demo, TO BE implemented
        epochs = self.cfg['exp']['train']['epochs']
        # TODO: get train and valid loader
        train_loader = self._create_loader("train")
        valid_loader = self._create_loader("valid")
        min_val_loss = float('inf')
        early_stopping = utils.exp_utils.EarlyStopping(self.cfg)
        
        # train_loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            loss_total = 0
            iter_count = 0

            for input, target, input_time, target_time in train_loader:
                input, target, input_time, target_time = \
                    input.float().to(self.device), target.float().to(self.device), input_time.float().to(self.device), target_time.float().to(self.device)

                self.optimizer.zero_grad()
                prediction = self.model(input) if not self.cfg['model']['UseTimeFeature'] else self.model(input,input_time,target_time)
                loss = self.loss_func(target, prediction)
                iter_count += 1
                loss.backward() 
                self.optimizer.step()
                loss_total += float(loss)
                

            print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} '.format(epoch, (
                    time.time() - epoch_start_time), loss_total / iter_count))
            
            val_loss, self.metrics = self.test(valid_loader)
            early_stopping(val_loss, self.model, self.optimizer, self.file_dir)
            print() 
            if early_stopping.early_stop:
                print("Early stopping")
                break
            #if val_loss < min_val_loss:
            #    if self.cfg['exp']['train']['saved_model']:
            #        print('Validate loss decreases from {:.4f} to {:.4f}, saving to {}'.format(min_val_loss, val_loss, self.file_dir + '/' + 'checkpoints'))
            #        utils.exp_utils.save_model(self.cfg, self.file_dir, self.model, self.optimizer, self.metrics)
            #        min_val_loss = val_loss
              
        print("Loading the best model.....") 
        self.load_model()
        

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


        preds, trues = np.array(preds),np.array(trues)
        preds, trues = preds.reshape(-1, preds.shape[-2], preds.shape[-1]), trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print("------------TEST result:------------")
        print("mae:", mae, " mse:",mse," rmse:",rmse)
        return mae, [metric(preds, trues)]
        # print("mape:",mape," mspe:",mspe," rse:",rse)
        # print("corr:",corr)
