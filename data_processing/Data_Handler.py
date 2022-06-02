import torch
import numpy as np
from torch.utils.data import Dataset

class Data_Handler(Dataset):
    def __init__(self,data, horizon, lookback) -> None:
        super().__init__()
        self.data=data
        self.lookback=lookback
        self.horizon=horizon
        # need to be implemented
    def __getitem__(self, index):
        # to be implemented
        
        return self.data[index:index+self.lookback], self.data[index+self.lookback:index+self.lookback+self.horizon]
    def __len__(self):
        return self.data.shape[0]-self.horizon-self.lookback+1
