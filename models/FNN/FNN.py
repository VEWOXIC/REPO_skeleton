import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        self._logger = getLogger()

        self.num_nodes = cfg["model"]["num_nodes"]
        self.feature_dim = cfg["model"]["feature_dim"]
        self.output_dim = cfg["model"]["output_dim"]

        self.input_window = cfg["data"]["lookback"]
        self.output_window = cfg["data"]["horizon"]
        self.device = cfg["model"]["device"]
        # self._logger = getLogger()
        self._scaler = cfg["data"]["scalar"]
        self.hidden_size = cfg["model"]["hidden_size"]

        self.fc1 = nn.Linear(
            self.input_window *
            self.feature_dim,
            self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(
            self.hidden_size,
            self.output_window *
            self.output_dim)

    def forward(self, input, target, input_time, target_time):
        # [batch_size, input_window, num_nodes, feature_dim]
        input = input.cpu()
        input_time = input_time.cpu()
        input_time = np.expand_dims(input_time, axis=-1)
        input = np.expand_dims(input, axis=-1)
        input = [input]
        idx = np.arange(self.cfg["model"]["num_nodes"])
        idx = torch.tensor(idx).to(self.device)
        input.append(input_time)
        input = np.concatenate(input, axis=-1)
        trainx = torch.from_numpy(input).to(self.device)
        batch_size = input.shape[0]
        trainx = trainx.permute(0, 2, 1, 3)
        trainx = trainx.reshape(batch_size, self.num_nodes, -1)
        inputs = trainx.float()

        outputs = self.fc1(inputs)
        outputs = self.relu(outputs)
        outputs = self.fc2(outputs)
        outputs = outputs.reshape(
            batch_size, self.num_nodes, self.output_window, self.output_dim
        )
        outputs = outputs.permute(0, 2, 1, 3)
        outputs = torch.squeeze(outputs)
        print("outputs", outputs.size())
        return outputs
