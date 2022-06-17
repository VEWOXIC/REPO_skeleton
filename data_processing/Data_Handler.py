import torch
import numpy as np
from torch.utils.data import Dataset
from utils import data_utils
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def get_dataset(cfg, flag):
    # 可能要选择
    if cfg['data']['dataset_name']=="ETTh1":
        return  Dataset_Custom1(cfg, flag)
    elif cfg['data']['dataset_name']=="dummy_dataset":
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

class Dataset_Custom1(Dataset):
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
        self.data = self.data.iloc[:, 1:]
        
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

class TimeSeriesImputer(TransformerMixin):
    def __init__(self, method: str = 'linear', fail_save: TransformerMixin = SimpleImputer()):
        self.method = method
        self.fail_save = fail_save

    def fit(self, data):
        if self.fail_save:
            self.fail_save.fit(data)
        return self

    def transform(self, data):
        # Interpolate missing values in columns
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        data = data.interpolate(method=self.method, limit_direction='both')
        # spline or time may be better?

        if self.fail_save:
            data = self.fail_save.transform(data)

        return data

class TimeSeriesPreprocessor(TransformerMixin):
    def __init__(self, look_back_window: int, forecast_horizon: int, stride: int, diff_order: int,
                 relative_diff: bool = True, splitXy: bool = True, scaling: str = 'minmax'):
        if not (look_back_window > 0 and forecast_horizon > 0 and stride > 0):
            raise ValueError('look_back_window, forecast_horizon and stride must be positive')
        # if stride < look_back_window + forecast_horizon:
        #     raise PendingDeprecationWarning('Setting stride to less than look_back_window + forecast_horizon may not'
        #                                     'be supported in the future due to potential data leak.')
        super().__init__()
        self.look_back_window = look_back_window
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.diff_order = diff_order
        self.relative_diff = relative_diff
        self.splitXy = splitXy
        self.interpolation_imputer = TimeSeriesImputer(method='linear')

        if scaling == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling == 'standard':
            self.scaler = StandardScaler()
        elif scaling == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError('Invalid value for scaling')

    def fit_transform(self, data, **fit_params):
        # Fill missing values via interpolation
        data = self.interpolation_imputer.fit_transform(data)

        # Differencing
        diff = np.array(data)
        for d in range(1, self.diff_order + 1):
            diff = difference(diff, relative=self.relative_diff)
            data = np.append(data, np.pad(diff, pad_width=((d, 0), (0, 0))), axis=1)
        if self.diff_order > 0:
            data = data[:, diff.shape[1]:]

        # Scale
        # if self.diff_order < 1:
        data = self.scaler.fit_transform(data)

        if not self.splitXy:
            return data

        # Extract X, y from time series
        X, y = split_sequence(data, self.look_back_window, self.forecast_horizon, self.stride)

        return X, y

    def transform(self, data):
        # Fill missing values via interpolation
        data = self.interpolation_imputer.transform(data)

        # Differencing
        diff = np.array(data)
        for d in range(1, self.diff_order + 1):
            diff = difference(diff, relative=self.relative_diff)
            data = np.append(data, np.pad(diff, pad_width=((d, 0), (0, 0))), axis=1)
        if self.diff_order > 0:
            data = data[:, diff.shape[1]:]

        # Scale
        # if self.diff_order < 1:
        data = self.scaler.transform(data)

        if not self.splitXy:
            return data

        # Extract X, y
        X, y = split_sequence(data, self.look_back_window, self.forecast_horizon, self.stride)

        return X, y

def difference(dataset, interval=1, relative=False, min_price=1e-04):
    delta = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        if relative:
            prev_price = dataset[i - interval]
            prev_price[prev_price == 0] = min_price
            value /= prev_price
        delta.append(value)
    return np.asarray(delta)
    
def split_sequence(sequence, look_back_window: int, forecast_horizon: int, stride: int = 1):
    X, y = [], []
    for i in range(0, len(sequence), stride):
        # find the end x and y
        end_ix = i + look_back_window
        end_iy = end_ix + forecast_horizon

        # check if there is enough elements to fill this x, y pair
        if end_iy > len(sequence):
            break

        X.append(sequence[i:end_ix])
        y.append(sequence[end_iy - 1 if forecast_horizon == 1 else end_ix:end_iy])
    return np.asarray(X), np.asarray(y)