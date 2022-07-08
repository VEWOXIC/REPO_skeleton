from functools import partial
import torch
from torch import optim
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score
import os 
import torch.nn as nn
from numpy import inf

# optimizer selection
def build_optimizer(cfg, model):
    if cfg['exp']['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg['exp']['train']['lr'])
    elif cfg['exp']['train']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg['exp']['train']['lr'])
    elif cfg['exp']['train']['optimizer'] == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg['exp']['train']['lr'])
    elif cfg['exp']['train']['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg['exp']['train']['lr'])
    elif cfg['exp']['train']['optimizer'] == 'sparse_adam':
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=cfg['exp']['train']['lr'])
    else:
        print('Received unrecognized optimizer, set default Adam optimizer')
        optimizer = optim.Adam(model.parameters(), lr=cfg['exp']['train']['lr'])
    return optimizer


# loss_function selection
def build_train_loss(cfg):
    if cfg['exp']['train']['loss'] == 'mae':
        lf = masked_mae_torch
    elif cfg['exp']['train']['loss'] == 'mse':
        lf = masked_mse_torch
    elif cfg['exp']['train']['loss'] == 'rmse':
        lf = masked_rmse_torch
    elif cfg['exp']['train']['loss'] == 'mape':
        lf = masked_mape_torch
    elif cfg['exp']['train']['loss'] == 'logcosh':
        lf = log_cosh_loss
    elif cfg['exp']['train']['loss'] == 'huber':
        lf = huber_loss
    elif cfg['exp']['train']['loss'] == 'quantile':
        lf = quantile_loss
    elif cfg['exp']['train']['loss'] == 'masked_mae':
        lf = partial(masked_mae_torch, null_val=0)
    elif cfg['exp']['train']['loss'] == 'masked_mse':
        lf = partial(masked_mse_torch, null_val=0)
    elif cfg['exp']['train']['loss'] == 'masked_rmse':
        lf = partial(masked_rmse_torch, null_val=0)
    elif cfg['exp']['train']['loss'] == 'masked_mape':
        lf = partial(masked_mape_torch, null_val=0)
    elif cfg['exp']['train']['loss'] == 'r2':
        lf = r2_score_torch
    elif cfg['exp']['train']['loss'] == 'evar':
        lf = explained_variance_score_torch
    else:
        print('Received none train loss func and will use the loss func defined in the model.')
        lf = masked_mae_torch
    return lf

# save model
def save_model(cfg, cache_name, model, optimizer, best_metrics):
    torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metrics': best_metrics
                },
                cache_name + '/' + 'checkpoints.pth'
                )

# load model
def load_model(cache_name, model, optimizer):
    print("Loaded model at " + cache_name + '/' + 'checkpoints.pth')
    checkpoint = torch.load(cache_name + '/' + 'checkpoints.pth')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

# early stop
class EarlyStopping:
    def __init__(self, cfg):
        self.patience = cfg['exp']['train']['patience']
        self.verbose = cfg['exp']['train']['verbose']
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = cfg['exp']['train']['delta']

    def __call__(self, val_loss, model, optimizer, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                },
                path + '/' + 'checkpoints.pth'
                )
        self.val_loss_min = val_loss


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= (mask.mean())
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_mae_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask) 
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) if torch.mean(loss) != inf else 0


def log_cosh_loss(preds, labels):
    loss = torch.log(torch.cosh(preds - labels))
    return torch.mean(loss)


def huber_loss(preds, labels, delta=1.0):
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res))
    # lo = torch.nn.SmoothL1Loss()
    # return lo(preds, labels)


def quantile_loss(preds, labels, delta=0.25):
    condition = torch.ge(labels, preds)
    large_res = delta * (labels - preds)
    small_res = (1 - delta) * (preds - labels)
    return torch.mean(torch.where(condition, large_res, small_res))


def masked_mape_torch(preds, labels, null_val=np.nan, eps=0):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val) and eps != 0:
        loss = torch.abs((preds - labels) / (labels + eps))
        return torch.mean(loss)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels,
                                       null_val=null_val))


def r2_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return r2_score(labels, preds)


def explained_variance_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return explained_variance_score(labels, preds)


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels,
                                 null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(
            preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def r2_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return r2_score(labels, preds)


def explained_variance_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return explained_variance_score(labels, preds)

