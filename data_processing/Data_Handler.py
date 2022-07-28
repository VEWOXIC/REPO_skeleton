from distutils.dep_util import newer
from locale import normalize
from sqlite3 import Timestamp
import sklearn
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

    def add_timeFeature(self,data):# add time stamp to the data, and drop the date column(s)
        if(self.cfg['data']['path'] == "./datasets/ETTh1.csv"):
            data['date'] = pd.to_datetime(data.date)
            data_stamp = time_features(pd.to_datetime(data['date'].values), freq=self.timeStampFreq)
            data_stamp = data_stamp.transpose(1, 0)
            #drop the first column
            self.data = self.data.drop(['date'], axis=1)
            return data_stamp, self.data # cfg['data']['freq']==“h" -> data_stamp = [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear] MTGNN就拿第一个
        elif(self.cfg['data']['path'] == "./datasets/yellow_taxi_2022-01.csv"):
            # print("Add time feature for yellow_taxi_2022-01.csv")
            # print("original data shape:", data.shape)
            data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])
            data_stamp0 = time_features(pd.to_datetime(data['tpep_pickup_datetime'].values), freq=self.timeStampFreq)
            data_stamp0 = data_stamp0.transpose(1, 0)
            self.data = self.data.drop(['tpep_pickup_datetime'], axis=1)
            data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])
            data_stamp1 = time_features(pd.to_datetime(data['tpep_dropoff_datetime'].values), freq=self.timeStampFreq)
            data_stamp1 = data_stamp1.transpose(1, 0)
            self.data = self.data.drop(['tpep_dropoff_datetime'], axis=1)
            return np.concatenate((data_stamp0, data_stamp1), axis=1), self.data
        elif(self.cfg['data']['path'] == "./datasets/wiki_rolling_nips_train.csv"):
            # add time featrue in first column
            data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
            data_stamp = time_features(pd.to_datetime(data.iloc[:, 0].values), freq=self.timeStampFreq)
            data_stamp = data_stamp.transpose(1, 0)
            #drop the first column
            self.data = self.data.drop(self.data.columns[0], axis=1)          
            return data_stamp, self.data
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
        elif file_type == 'parquet':
            data = pd.read_parquet(path)
            self.data = pd.DataFrame(data) 
        else:
            print("Error: file type not supported")
            exit()

        self.data = self.data.fillna(method='ffill')
            
        num_train = int(len(self.data) * self.cfg["data"]["train_ratio"])
        num_test = int(len(self.data) * self.cfg["data"]["test_ratio"])
        num_vali = int(len(self.data) * self.cfg["data"]["valid_ratio"])
        self.boarder = {'train':[0,num_train],'valid':[num_train + 1,num_train+num_vali],'test':[num_train+num_vali + 1,num_train + num_vali + num_test]}

        if self.cfg['model']['UseTimeFeature']:
            self.data_stamp , self.data= self.add_timeFeature(self.data)
            
        self.train_data = self.data[self.boarder["train"][0]: self.boarder["train"][1]].values
        self.data = self.data[self.boarder[self.flag][0]: self.boarder[self.flag][1]].values
        self._normalize()
        print("data after process is:", self.data.shape, self.data)
        
    def _normalize(self):
        self.scale = np.ones(self.data.shape[1])
        self.bias =  np.zeros(self.data.shape[1])
        if (self.normalize == 0):
            self.data = self.data
        if (self.normalize == 1):
            # normalized by the maximum value of entire matrix.
            self.data = self.data / np.max(self.train_data)
        if (self.normalize == 2):
            # normlized by the maximum value of each row (sensor).
            for i in range(self.cfg['data']['channel']):
                self.scale[i] = np.max(np.abs(self.train_data[:, i]))
                self.data[:, i] = self.data[:, i] /  self.scale[i]
        if (self.normalize == 3):
            # normlized by the mean/std value of each row (sensor).
            for i in range(self.cfg['data']['channel']):
                self.scale[i] = np.std(self.train_data[:, i]) #std
                self.bias[i] = np.mean(self.train_data[:, i])
                self.data[:, i] = (self.data[:, i] - self.bias[i]) / self.scale[i]

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