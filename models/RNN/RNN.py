import torch.nn as nn
from torch.nn import init

import torch.nn.functional as F
import numpy as np

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