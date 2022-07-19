from locale import normalize
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import data_utils
from utils.timefeatures import time_features
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
        self.timeStampFreq = cfg['data']['freq'] # Choose a time feature
        self.normalize = cfg['data']['normalize']

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



        file_dir = path.split('/')
        file_name = file_dir[-1]
        file_type = file_name.split('.')[-1]
        if file_type == 'csv':
            self.data = pd.read_csv(path)
        elif file_type == 'txt':
            fin = open(path)
            rawdat = np.loadtxt(fin, delimiter=',')
            self.data = pd.DataFrame(rawdat)
        elif file_type == 'npz':
            data = np.load(path)
            data = data['data'][:,:,0]
            self.data = pd.DataFrame(data)


        self.data = self.data.fillna(method='ffill', limit=len(self.data)).fillna(method='bfill', limit=len(self.data)).values      
        
        #normalize data from SCINet's financial dataloader
        self.scale = np.ones(self.data.shape[1])
        self.bias =  np.zeros(self.data.shape[1])
        self.scale = torch.from_numpy(self.scale).float()
        self.bias = torch.from_numpy(self.bias).float()
        self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)
        self.bias = self.bias.cuda()
        self.bias = Variable(self.bias)
        
        
        if(self.normalize == 0):#dafault
            print("default norm")
        elif(self.normalize == 1):#max
            print("normalized by the maximum value of entire matrix.")
            self.data = self.data / np.max(self.data)
        elif(self.normalize == 2):
            print("# normlized by the maximum value of each row (sensor).")
            for i in range(self.data.shape[1]):
                self.scale[i] = np.max(np.abs(self.data[:, i]))
                print("scale[", i, "]:", self.scale[i])
                self.data[:, i] = self.data[:, i] / self.scale[i].cpu().numpy()
        elif (self.normalize == 3):
            print("normlized by the mean/std value of each row (sensor).")
            for i in range(self.data.shape[1]):
                self.scale[i] = np.std(self.data[:, i]) #std
                self.bias[i] = np.mean(self.data[:, i]) #mean
                self.data[:, i] = (self.data[:, i] - self.bias[i].cpu().numpy()) / self.scale[i].cpu().numpy()
        
        self.data = pd.DataFrame(self.data)
        
        num_train = int(len(self.data) * self.cfg["data"]["train_ratio"])
        num_test = int(len(self.data) * self.cfg["data"]["test_ratio"])
        num_vali = len(self.data) - num_train - num_test
        boarder = {'train':[0,num_train],'valid':[num_train,num_train+num_vali],'test':[num_train+num_vali,len(self.data)-1]}

        if self.cfg['model']['UseTimeFeature']:
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

        if self.cfg['model']['UseTimeFeature']:
            timestamp_x = self.data_stamp[index:index+self.lookback]
            timestamp_y = self.data_stamp[index+self.lookback:index+self.lookback+self.horizon]
        else:
            timestamp_x = 0
            timestamp_y = 0
            
        return x, y, timestamp_x, timestamp_y

    def __len__(self):
        return self.data.shape[0]-self.horizon-self.lookback+1
