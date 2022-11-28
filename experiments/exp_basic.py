import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
import utils.exp_utils
from data_processing.Data_Handler import get_dataset
from utils.metrics import metric


class Exp_Basic(object):
    def __init__(self, cfg, model_save_dir) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg["exp"]["device"])
        self.file_dir = model_save_dir
        self.model = self._build_model()
        self.model.to(self.device)
        self.loss_func = self._get_lossfunc()
        self.optimizer = self._get_optim()

    def _build_model(self):
        return models.__dict__[
            self.cfg["model"]["model_name"]](
            self.cfg).float()

    def _create_loader(self, flag="train"):
        self.dataset = get_dataset(self.cfg, flag)
        batch_size = self.cfg["exp"][flag]["batchsize"]
        shuffle = self.cfg["exp"][flag]["shuffle"]
        drop_last = self.cfg["exp"][flag]["drop_last"]
        return DataLoader(
            self.dataset, batch_size, shuffle=shuffle, drop_last=drop_last
        )

    def _get_optim(self):
        return utils.exp_utils.build_optimizer(self.cfg, self.model)

    def _get_lossfunc(self):
        return utils.exp_utils.build_train_loss(self.cfg)

    def load_model(self):
        self.model, self.optimizer = utils.exp_utils.load_model(
            self.file_dir, self.model, self.optimizer
        )

    def train(self):
        # TODO: just for demo, TO BE implemented
        epochs = self.cfg["exp"]["train"]["epochs"]
        # TODO: get train and valid loader
        train_loader = self._create_loader("train")
        valid_loader = self._create_loader("valid")
        min_val_loss = float("inf")
        early_stopping = utils.exp_utils.EarlyStopping(self.cfg)

        # train_loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            loss_total = 0
            iter_count = 0
            self.adjust_learning_rate(self.optimizer, epoch, self.cfg)
            process = tqdm(train_loader)
            for input, target, input_time, target_time in process:
                input, target, input_time, target_time = (
                    input.float().to(self.device),
                    target.float().to(self.device),
                    input_time.float().to(self.device),
                    target_time.float().to(self.device),
                )

                self.optimizer.zero_grad()
                prediction = (
                    self.model(input)
                    if not self.cfg["model"]["UseTimeFeature"]
                    else self.model(input, target, input_time, target_time)
                )
                loss = self.loss_func(prediction, target)
                iter_count += 1
                loss.backward()
                self.optimizer.step()
                loss_total += float(loss)

            print(
                "| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} ".format(
                    epoch, (time.time() - epoch_start_time), loss_total / iter_count
                )
            )

            val_loss, self.metrics = self.test(valid_loader)
            early_stopping(val_loss, self.model, self.optimizer, self.file_dir)
            print()
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # if val_loss < min_val_loss:
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
        process = tqdm(data_loader)
        for input, target, input_time, target_time in process:
            input, target, input_time, target_time = (
                input.float().to(self.device),
                target.float().to(self.device),
                input_time.float().to(self.device),
                target_time.float().to(self.device),
            )

            prediction = (
                self.model(input)
                if not self.cfg["model"]["UseTimeFeature"]
                else self.model(input, target, input_time, target_time)
            )
            prediction = prediction.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            preds.append(prediction)
            trues.append(target)

        preds, trues = np.array(preds), np.array(trues)
        preds, trues = preds.reshape(
            -1, preds.shape[-2], preds.shape[-1]
        ), trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        print("------------TEST result:------------")
        print("norm mae:", mae, " norm mse:", mse, " norm rmse:", rmse)
        print("mape:", mape, " mspe:", mspe, " rse:", rse, " corr:", corr)

        preds, trues = self.denormalized(preds, trues)
        de_mae, de_mse, de_rmse, de_mape, de_mspe, de_rse, de_corr = metric(
            preds, trues
        )
        print(
            "denorm mae:",
            de_mae,
            " denorm mse:",
            de_mse,
            " denorm rmse:",
            de_rmse)
        print(
            "denorm_mape",
            de_mape,
            " denorm_mspe",
            de_mspe,
            " denorm_rse:",
            de_rse,
            " denorm_corr:",
            de_corr,
        )
        return mae, [metric(preds, trues)]
        # print("mape:",mape," mspe:",mspe," rse:",rse)
        # print("corr:",corr)

    def denormalized(self, preds, trues):
        if self.cfg["data"]["normalize"] == 1:
            preds *= np.max(self.dataset.train_data)
            trues *= np.max(self.dataset.train_data)
        elif self.cfg["data"]["normalize"] == 2:
            for i in range(self.cfg["data"]["channel"]):
                preds[:, :, i] *= self.dataset.scale[i]
                trues[:, :, i] *= self.dataset.scale[i]
        elif self.cfg["data"]["normalize"] == 3:
            for i in range(self.cfg["data"]["channel"]):
                preds[:, :, i] *= self.dataset.scale[i]
                trues[:, :, i] *= self.dataset.scale[i]
                preds[:, :, i] += self.dataset.bias[i]
                trues[:, :, i] += self.dataset.bias[i]
        return preds, trues

    def adjust_learning_rate(self, optimizer, epoch, cfg):
        lr = cfg["exp"]["train"]["lr"]
        lr_adj = cfg["exp"]["train"]["lr_adj"]
        if lr_adj == 1:
            lr_adjust = {epoch: lr * (0.95 ** (epoch // 1))}

        elif lr_adj == 2:
            lr_adjust = {
                0: 0.0001,
                5: 0.0005,
                10: 0.001,
                20: 0.0001,
                30: 0.00005,
                40: 0.00001,
                70: 0.000001,
            }

        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            print("Updating learning rate to {}".format(lr))
        else:
            for param_group in optimizer.param_groups:
                lr = param_group["lr"]
        return lr
