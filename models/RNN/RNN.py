import numpy as np
import torch
import torch.nn as nn
import random
from logging import getLogger
from torch.nn import init
import torch.nn.functional as F


class _RNN_Base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # def __init__(self, c_in, c_out, hidden_size=100, n_layers=1, bias=True, rnn_dropout=0, bidirectional=False, fc_dropout=0.):
        self.c_in = cfg['model']['c_in']
        self.c_out = cfg['model']['c_out']
        self.hidden_size = cfg['model']['hidden_size']
        self.fc_dropout = cfg['model']['fc_dropout']
        self.bidirectional = cfg['model']['bidirectional']
        self.rnn_dropout = cfg['model']['rnn_dropout']
        self.bias = cfg['model']['bias']
        self.n_layers = cfg['model']['n_layers']
        
        self.rnn = self._cell(self.c_in, self.hidden_size, num_layers=self.n_layers, bias=self.bias, batch_first=True, dropout=self.rnn_dropout, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(self.fc_dropout) #if self.fc_dropout else noop
        self.fc = nn.Linear(self.hidden_size * (1 + self.bidirectional), self.c_out)

    def forward(self, x):
        x = x.transpose(2,1)    # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        output, _ = self.rnn(x) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        # output = output[:, -1]# output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        output = self.fc(self.dropout(output))
        output = output.transpose(2,1)
        return output

class RNN(_RNN_Base):
    _cell = nn.RNN

class LSTM(_RNN_Base):
    _cell = nn.LSTM

class GRU(_RNN_Base):
    _cell = nn.GRU



class RNN(nn.Module):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.num_nodes = cfg['data']['num_nodes']
        self.feature_dim = cfg['data']['feature_dim']
        self.output_dim = cfg['data']['output_dim']

        self.input_window = cfg['data']['lookback']
        self.output_window = cfg['data']['horizon']
        self.device = cfg['model']['device']
        #self._logger = getLogger()
        self._scalar = cfg['data']['scalar']

        self.rnn_type = cfg['model']['rnn_type']
        self.hidden_size = cfg['model']['hidden_size']
        self.num_layers = cfg['model']['num_layers']
        self.dropout = cfg['model']['dropout']
        self.bidirectional = cfg['model']['bidirectional']
        self.teacher_forcing_ratio = cfg['model']['teacher_forcing_ratio']
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.input_size = self.num_nodes * self.feature_dim

        #self._logger.info('You select rnn_type {} in RNN!'.format(self.rnn_type))
        if self.rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, dropout=self.dropout,
                              bidirectional=self.bidirectional)
        elif self.rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, dropout=self.dropout,
                               bidirectional=self.bidirectional)
        elif self.rnn_type.upper() == 'RNN':
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, dropout=self.dropout,
                              bidirectional=self.bidirectional)
        else:
            raise ValueError('Unknown RNN type: {}'.format(self.rnn_type))
        self.fc = nn.Linear(self.hidden_size * self.num_directions, self.num_nodes * self.output_dim)

    def forward(self, input, target, input_time, target_time):
        input = input.cpu()
        input_time = input_time[:, :, 0].cpu()
        input_time = np.expand_dims(input_time, axis=-1)
        input_time = np.tile(input_time, 7)
        input_time = np.expand_dims(input_time, axis=-1)
        input = np.expand_dims(input, axis=-1)
        input = [input]
        idx = np.arange(self.cfg['model']['num_nodes'])
        idx = torch.tensor(idx).to(self.device)
        input.append(input_time)
        input = np.concatenate(input, axis=-1)
        trainx = torch.from_numpy(input).to(self.device)
        trainx = trainx.permute(1, 0, 2, 3)
        src = trainx.float()


        target = target.cpu()
        target_time = target_time[:, :, 0].cpu()
        target_time = np.expand_dims(target_time, axis=-1)
        target_time = np.tile(target_time, 7)
        target_time = np.expand_dims(target_time, axis=-1)
        target = np.expand_dims(target, axis=-1)
        target = [target]
        idx = np.arange(self.cfg['model']['num_nodes'])
        idx = torch.tensor(idx).to(self.device)
        target.append(target_time)
        target = np.concatenate(target, axis=-1)
        trainy = torch.from_numpy(target).to(self.device)
        trainy = trainy.permute(1, 0, 2, 3)
        target = trainy.float()

        print("src",src.size())
        
        #src = [input]  # [batch_size, input_window, num_nodes, feature_dim]
        #target = batch['y']  # [batch_size, output_window, num_nodes, feature_dim]
        #src = src.permute(1, 0, 2, 3)  # [input_window, batch_size, num_nodes, feature_dim]
        #target = target.permute(1, 0, 2, 3)  # [output_window, batch_size, num_nodes, output_dim]

        batch_size = src.shape[1]
        src = src.reshape(self.input_window, batch_size, self.num_nodes * self.feature_dim)
        #print("src",src.size())
        # src = [self.input_window, batch_size, self.num_nodes * self.feature_dim]
        outputs = []
        for i in range(self.output_window):
            # src: [input_window, batch_size, num_nodes * feature_dim]
            out, _ = self.rnn(src)
            #print("out",out.size())
            # out: [input_window, batch_size, hidden_size * num_directions]
            out = self.fc(out[-1])
            #print("out",out.size())
            # out: [batch_size, num_nodes * output_dim]
            out = out.reshape(batch_size, self.num_nodes, self.output_dim)
            #print("out",out.size())
            # out: [batch_size, num_nodes, output_dim]
            outputs.append(out.clone())
            if self.output_dim < self.feature_dim:  # output_dim可能小于feature_dim
                out = torch.cat([out, target[i, :, :, self.output_dim:]], dim=-1)
            # out: [batch_size, num_nodes, feature_dim]
            if self.training and random.random() < self.teacher_forcing_ratio:
                src = torch.cat((src[1:, :, :], target[i].reshape(
                    batch_size, self.num_nodes * self.feature_dim).unsqueeze(0)), dim=0)
            else:
                #print("out",out.size())
                outt = out.reshape(batch_size, self.num_nodes * self.feature_dim)
                #print("outt",outt.size())
                outt = outt.unsqueeze(0)
                #print("outt",outt.size())
                src = torch.cat((src[1:, :, :], outt), dim=0)
        outputs = torch.stack(outputs)
        # outputs = [output_window, batch_size, num_nodes, output_dim]
        outputs = outputs.permute(1, 0, 2, 3)
        #print("outputs",outputs.size())
        outputs = torch.squeeze(outputs)
        print("outputs",outputs.size())
        return outputs

    # def calculate_loss(self, batch):
    #     y_true = batch['y']
    #     y_predicted = self.predict(batch)
    #     y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
    #     y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
    #     return loss.masked_mae_torch(y_predicted, y_true, 0)

    # def predict(self, batch):
    #     return self.forward(batch)

