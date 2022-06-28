import torch
import numpy as np
from torch.utils.data import Dataset
from utils import data_utils
from utils.timefeatures import time_features
import pandas as pd


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
        self.timeStampFreq = cfg['data']['freq'] # Choose a time feature

        self.__read_data__()
        # need to be implemented
        # transformer based methods have label_len

    def add_timeFeature(self, data):
        data['date'] = pd.to_datetime(data.date)
        data_stamp = time_features(pd.to_datetime(data['date'].values), freq=self.timeStampFreq)
        data_stamp = data_stamp.transpose(1, 0)
        return data_stamp # cfg['data']['freq']==“h" -> data_stamp = [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear] MTGNN就拿第一个

    def __read_data__(self):
        self.scaler = data_utils.get_scaler(self.cfg['data']['scalar'])
        path = self.cfg["data"]['path']
        self.data = pd.read_csv(path)

        num_train = int(len(self.data) * self.cfg["data"]["train_ratio"])
        num_test = int(len(self.data) * self.cfg["data"]["test_ratio"])
        num_vali = len(self.data) - num_train - num_test
        boarder = {'train':[0,num_train],'valid':[num_train,num_train+num_vali],'test':[num_train+num_vali,len(self.data)-1]}


        self.data_stamp = self.add_timeFeature(self.data[['date']][boarder[self.flag][0]:boarder[self.flag][1]])

        self.data = self.data.drop(self.data.columns[[i for i in range(self.data.shape[1]-self.cfg['data']['channel'])]] ,axis = 1)

        train_data = self.data[boarder["train"][0]: boarder["train"][1]].values
        self.scaler.fit(train_data)
        self.data = self.data[boarder[self.flag][0]: boarder[self.flag][1]]
        self.data = self.scaler.transform(self.data.values)

        # 单变量/多变量

    def __getitem__(self, index):
        # some model use time stamp
        x = self.data[index:index+self.lookback]
        y = self.data[index+self.lookback:index+self.lookback+self.horizon]

        timestamp_x = self.data_stamp[index:index+self.lookback]
        timestamp_y = self.data_stamp[index+self.lookback:index+self.lookback+self.horizon]

        return x, y, timestamp_x, timestamp_y

    def __len__(self):
        return self.data.shape[0]-self.horizon-self.lookback+1