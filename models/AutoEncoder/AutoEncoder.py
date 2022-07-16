import numpy as np
import torch
import torch.nn as nn
from logging import getLogger


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.num_nodes = cfg['model']['num_nodes']
        self.feature_dim = cfg['model']['feature_dim']
        self.output_dim = cfg['model']['output_dim']

        self.input_window = cfg['data']['lookback']
        self.output_window = cfg['data']['horizon']
        self.device = cfg['model']['device']
        self._logger = getLogger()
        self._scaler = cfg['data']['scalar']

        self.encoder = nn.Sequential(
            nn.Linear(self.input_window * self.num_nodes * self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_window * self.num_nodes * self.output_dim)
        )

    def forward(self, input, target, input_time, target_time):
        input = input.cpu() # [batch_size, input_window, num_nodes, feature_dim]
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
        trainx = trainx.reshape(-1, self.input_window * self.num_nodes * self.feature_dim)
        x = trainx.float()
        # [batch_size, output_window * num_nodes * feature_dim]
        x = self.encoder(x)  # [batch_size, 16]
        x = self.decoder(x)
        # [batch_size, output_window * num_nodes * output_dim]
        outputs = x.reshape(-1, self.output_window, self.num_nodes, self.output_dim)
        outputs = torch.squeeze(outputs)
        return outputs
