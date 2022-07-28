import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger


class SANN(nn.Module):
    def __init__(self, n_inp, n_out, t_inp, t_out, n_points, past_t, hidden_dim, dropout):
        super(SANN, self).__init__()
        # Variables
        self.n_inp = n_inp
        self.n_out = n_out
        self.t_inp = t_inp
        self.t_out = t_out
        self.n_points = n_points
        self.past_t = past_t
        self.hidden_dim = hidden_dim
        # Convolutional layer
        self.conv_block = AgnosticConvBlock(n_inp, n_points, past_t, hidden_dim, num_conv=1)
        self.convT = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, n_points))
        # Regressor layer
        self.regressor = ConvRegBlock(t_inp, t_out, n_points, hidden_dim)
        # Dropout
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        N, C, T, S = x.size()
        # Padding
        xp = F.pad(x, pad=(0, 0, self.past_t - 1, 0))
        # NxCxTxS ---> NxHxTx1
        out = self.conv_block(xp)
        out = out.view(N, self.hidden_dim, T, 1)
        # NxHxTx1 ---> NxHxTxS
        out = self.convT(out)
        # 2D dropout
        out = self.drop(out)
        # NxHxTxS ---> NxC'xT'xS
        out = self.regressor(out.reshape(N, -1, S))
        return out.view(N, self.n_out, self.t_out, self.n_points)


class AgnosticConvBlock(nn.Module):
    def __init__(self, n_inp, n_points, past_t, hidden_dim, num_conv):
        super(AgnosticConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=n_inp, out_channels=hidden_dim, kernel_size=(past_t, n_points), bias=True),
                  nn.BatchNorm2d(num_features=hidden_dim, affine=True, track_running_stats=True),
                  nn.ReLU()]
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)


class ConvRegBlock(nn.Module):
    def __init__(self, t_inp, t_out, n_points, hidden_dim):
        super(ConvRegBlock, self).__init__()
        layers = [nn.Conv1d(in_channels=hidden_dim * t_inp, out_channels=t_out, kernel_size=1, bias=True),
                  nn.BatchNorm1d(num_features=t_out, affine=True, track_running_stats=True)]
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)


class ATDM(nn.Module):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        # get data feature
        self.num_nodes = cfg['model']['num_nodes']
        self.adj_mx = np.ones((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.feature_dim = cfg['model']['feature_dim']
        self.output_dim = cfg['model']['output_dim']
        self._scaler = cfg['data']['scalar']
        # get model config
        self.hidden_size = cfg['model']['hidden_size']
        self.input_window = cfg['data']['lookback']
        self.output_window = cfg['data']['horizon']
        self.past_t = cfg['model']['past_t']
        self.dropout = cfg['model']['dropout']
        self.input_window = cfg['data']['lookback']
        self.output_window = cfg['data']['horizon']
        self.device = cfg['exp']['device']

        # init logger
        self._logger = getLogger()
        # define the model structure
        self.sann = SANN(self.feature_dim, self.output_dim, self.input_window, self.output_window, self.num_nodes,
                         self.past_t, self.hidden_size, self.dropout)

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
        trainx = trainx.permute(0, 3, 1, 2) # B x 1 x input_window x num_nodes
        inputs = trainx.float()
        output_y = self.sann(inputs)  # bz x 1 x T' x S
        output_y = output_y.permute(0, 2, 3, 1)  # bz x T' x S x 1
        output = torch.squeeze(output_y)
        return output

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)
