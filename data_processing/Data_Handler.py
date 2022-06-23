import torch
import numpy as np
from torch.utils.data import Dataset
from utils import data_utils
import pandas as pd
from torch.autograd import Variable


def get_dataset(cfg, flag):
    # 可能要选择
    return  Dataset_Custom(cfg, flag)

# 如果想一个dataloader对应所有数据，这里需要非常多的函数支持
class Dataset_Custom(Dataset):
    def __init__(self, cfg, flag) -> None:
        super().__init__()
        self.cfg = cfg
        self.flag = flag
        self.lookback = cfg['data']['lookback']
        self.horizon = cfg['data']['horizon']
        self.__read_data__()
        
    def add_time_dim_in_MTGNN(self):
        print("adding time...")
        num_samples, num_nodes = self.data.shape
        data_tmp = np.expand_dims(self.data.values, axis=-1)
        data_list = [data_tmp]
        time_ind = (self.data.index.values.astype("datetime64[h]") - self.data.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        self.time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        
    def __read_data__(self):
        print("data handler: read data...")
        self.scaler = data_utils.get_scaler(self.cfg['data']['scalar'])
        path = self.cfg["data"]['path']
        self.data = pd.read_csv(path,usecols=[1,2,3,4,5,6,7])
        
        if self.cfg['model']['model_name'] == 'MTGNN':
            
            self.add_time_dim_in_MTGNN()
            
        num_train = int(len(self.data) * self.cfg["data"]["train_ratio"])
        num_test = int(len(self.data) * self.cfg["data"]["test_ratio"])
        num_vali = len(self.data) - num_train - num_test
        boarder = {'train':[0,num_train],'valid':[num_train,num_train+num_vali],'test':[num_train+num_vali,len(self.data)-1]}
        train_data = self.data[boarder["train"][0]: boarder["train"][1]].values
        self.scaler.fit(train_data)
        self.data = self.data[boarder[self.flag][0]: boarder[self.flag][1]]
        self.data = self.scaler.transform(self.data.values)
        # 单变量/多变量
        print("data handler: read data done.")
        
    def __getitem__(self, index):
        # some model use time stamp
        # to be implemented
        return self.data[index:index+self.lookback], self.data[index+self.lookback:index+self.lookback+self.horizon], self.time_in_day[index:index+self.lookback], self.time_in_day[index+self.lookback:index+self.lookback+self.horizon]
    
    def __len__(self):
        return self.data.shape[0]-self.horizon-self.lookback+1