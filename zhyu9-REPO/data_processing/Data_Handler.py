import torch
import numpy as np
from torch.utils.data import Dataset
from utils import data_utils
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import os

def get_dataset(cfg, flag):
    # 可能要选择
    if cfg['data']["dataset_name"] == "ETTH1":
        return Dataset_ETTh(cfg, flag)
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
        # need to be implemented
        # transformer based methods have label_len
    
    def __read_data__(self):
        self.scaler = data_utils.get_scaler(self.cfg['data']['scalar'])
        path = self.cfg["data"]['path']
        self.data = pd.read_csv(path)
        
        num_train = int(len(self.data) * self.cfg["data"]["train_ratio"])
        num_test = int(len(self.data) * self.cfg["data"]["test_ratio"])
        num_vali = len(self.data) - num_train - num_test
        boarder = {'train':[0,num_train],'valid':[num_train,num_train+num_vali],'test':[num_train+num_vali,len(self.data)-1]}

        train_data = self.data[boarder["train"][0]: boarder["train"][1]].values
        self.scaler.fit(train_data)
        self.data = self.data[boarder[self.flag][0]: boarder[self.flag][1]]
        self.data = self.scaler.transform(self.data.values)
        # 单变量/多变量
        
    def __getitem__(self, index):
        # some model use time stamp
        # to be implemented
        return self.data[index:index+self.lookback], self.data[index+self.lookback:index+self.lookback+self.horizon]
    
    def __len__(self):
        return self.data.shape[0]-self.horizon-self.lookback+1



      




      
   
      

  

      

   
  
class Dataset_ETTh(Dataset):
    def __init__(self, cfg, flag):
    # def __init__(self, root_path, flag='train', size=None, 
    #              features='S', data_path='ETTh1.csv', 
    #              target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        
        root_path = cfg['data']['root_path']
        
        size = cfg['data']['size']
        features = cfg['data']['features']
        data_path = cfg['data']['data_path']
        
        target = cfg['data']['target']
        scale = cfg['data']['scale']
        timeenc = cfg['data']['timeenc']
        freq = cfg['data']['freq']
        cols = cfg['data']['cols']
        inverse = cfg['data']['inverse']
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = cfg['data']['input_len']
            self.label_len = cfg['data']['input_len']
            self.pred_len = cfg['data']['output_len']
        else:
            self.seq_len = cfg['data']['input_len']
            self.label_len = cfg['data']['input_len']
            self.pred_len = cfg['data']['output_len']
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            # data = self.scaler.fit_transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        # r_end = r_begin + self.label_len + self.pred_len
        r_end = r_begin + 24
        
        # print(r_begin)
        # print(r_end)

        seq_x = self.data_x[s_begin:s_end]  # 0 - 24
        seq_y = self.data_y[r_begin:r_end] # 0 - 48
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y
    
   
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)