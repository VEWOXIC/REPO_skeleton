from distutils.dep_util import newer
from locale import normalize
from sqlite3 import Timestamp

import numpy as np
import pandas as pd
import sklearn
from click import echo
from torch.autograd import Variable
from torch.utils.data import Dataset

from utils import data_utils
from utils.timefeatures import time_features


def get_dataset(cfg, flag):
    # 可能要选择
    return Dataset_Custom(cfg, flag)


# 如果想一个dataloader对应所有数据，这里需要非常多的函数支持


class Dataset_Custom(Dataset):
    def __init__(self, cfg, flag) -> None:
        super().__init__()
        self.cfg = cfg
        self.flag = flag
        self.lookback = cfg["data"]["lookback"]
        self.horizon = cfg["data"]["horizon"]
        self.timeStampFreq = cfg["data"]["freq"]  # Choose a time feature
        self.normalize = cfg["data"]["normalize"]
        self.__read_data__()
        # need to be implemented
        # transformer based methods have label_len

    def add_timeFeature(
        self, data
    ):  # add time stamp to the data, and drop the date column(s)
        num_samples, num_nodes = self.data.shape
        if (
            self.cfg["data"]["dataset_name"] == "metr-la"
            or self.cfg["data"]["dataset_name"] == "pems-bay"
        ):
            if self.cfg["data"]["add_time_in_day"]:
                time_ind = (
                    self.data.index.values
                    - self.data.index.values.astype("datetime64[D]")
                ) / np.timedelta64(1, "D")
                time_in_day = np.tile(
                    time_ind, [
                        1, num_nodes, 1]).transpose(
                    (2, 1, 0))
                time_in_day = time_in_day.squeeze()
                return time_in_day
            if self.cfg["data"]["add_time_in_day"]:
                day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
                day_in_week[np.arange(num_samples), :,
                            self.data.index.dayofweek] = 1
                return day_in_week

        if (
            self.cfg["data"]["dataset_name"] == "ETTh1"
            or self.cfg["data"]["dataset_name"] == "ETTh2"
            or self.cfg["data"]["dataset_name"] == "ECL"
            or self.cfg["data"]["dataset_name"] == "WTH"
        ):
            data["date"] = pd.to_datetime(data.date)
            data_stamp = time_features(
                pd.to_datetime(data["date"].values), freq=self.timeStampFreq
            )
            data_stamp = data_stamp.transpose(1, 0)
        elif self.cfg["data"]["dataset_name"] == "yellow_taxi_2022-01":
            data["tpep_pickup_datetime"] = pd.to_datetime(
                data["tpep_pickup_datetime"])
            data_stamp0 = time_features(
                pd.to_datetime(data["tpep_pickup_datetime"].values),
                freq=self.timeStampFreq,
            )
            data_stamp0 = data_stamp0.transpose(1, 0)
            self.data = self.data.drop(["tpep_pickup_datetime"], axis=1)
            data["tpep_dropoff_datetime"] = pd.to_datetime(
                data["tpep_pickup_datetime"])
            data_stamp1 = time_features(
                pd.to_datetime(data["tpep_dropoff_datetime"].values),
                freq=self.timeStampFreq,
            )
            data_stamp1 = data_stamp1.transpose(1, 0)
            data_stamp = np.concatenate((data_stamp0, data_stamp1), axis=1)
        elif self.cfg["data"]["dataset_name"] == "wiki_rolling_nips_train":
            # add time featrue in first column
            data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
            data_stamp = time_features(
                pd.to_datetime(data.iloc[:, 0].values), freq=self.timeStampFreq
            )
            data_stamp = data_stamp.transpose(1, 0)
            # drop the first column
        else:
            data_stamp = np.zeros(num_samples)
            data_stamp = np.expand_dims(data_stamp, axis=0)
            data_stamp = np.transpose(data_stamp)

        if self.cfg["data"]["freq"] == "h":
            data_stamp = data_stamp[:, 0]
            data_stamp = np.expand_dims(data_stamp, axis=-1)
            data_stamp = np.tile(data_stamp, self.cfg["data"]["channel"])
        elif self.cfg["data"]["freq"] == "w":
            data_stamp = data_stamp[:, 1]
            data_stamp = np.expand_dims(data_stamp, axis=-1)
            data_stamp = np.tile(data_stamp, self.cfg["data"]["channel"])
        elif self.cfg["data"]["freq"] == "m":
            data_stamp = data_stamp[:, 2]
            data_stamp = np.expand_dims(data_stamp, axis=-1)
            data_stamp = np.tile(data_stamp, self.cfg["data"]["channel"])
        elif self.cfg["data"]["freq"] == "y":
            data_stamp = data_stamp[:, 3]
            data_stamp = np.expand_dims(data_stamp, axis=-1)
            data_stamp = np.tile(data_stamp, self.cfg["data"]["channel"])
        return data_stamp

    def __read_data__(self):
        self.scaler = data_utils.get_scaler(self.cfg["data"]["scalar"])
        path = self.cfg["data"]["path"]

        file_dir = path.split("/")
        file_name = file_dir[-1]
        file_type = file_name.split(".")[-1]
        if file_type == "csv":
            self.data = pd.read_csv(path)
        elif file_type == "txt":
            fin = open(path)
            rawdat = np.loadtxt(fin, delimiter=",")
            self.data = pd.DataFrame(rawdat)
        elif file_type == "npz":
            data = np.load(path)
            data = data["data"][:, :, 0]
            self.data = pd.DataFrame(data)
        elif file_type == "parquet":
            data = pd.read_parquet(path)
            self.data = pd.DataFrame(data)
        elif file_type == "h5":
            self.data = pd.read_hdf(path)
        else:
            print("Error: file type not supported")
            exit()

        self.data = self.data.fillna(method="ffill")

        num_train = int(len(self.data) * self.cfg["data"]["train_ratio"])
        num_test = int(len(self.data) * self.cfg["data"]["test_ratio"])
        num_vali = len(self.data) - num_train - num_test
        boarder = {
            "train": [0, num_train],
            "valid": [num_train, num_train + num_vali],
            "test": [num_train + num_vali, len(self.data) - 1],
        }

        if self.cfg["model"]["UseTimeFeature"]:
            self.data_stamp = self.add_timeFeature(self.data)
            self.data_stamp = self.data_stamp[
                boarder[self.flag][0]: boarder[self.flag][1]
            ]
        self.data = self.data.drop(self.data.columns[[i for i in range(
            self.data.shape[1] - self.cfg["data"]["channel"])]], axis=1, )
        self.train_data = self.data[boarder["train"]
                                    [0]: boarder["train"][1]].values
        self.data = self.data[boarder[self.flag]
                              [0]: boarder[self.flag][1]].values
        self.data = np.nan_to_num(self.data)
        self._normalize()

    def _normalize(self):
        self.scale = np.ones(self.data.shape[1])
        self.bias = np.zeros(self.data.shape[1])
        if self.normalize == 0:
            self.data = self.data
        if self.normalize == 1:
            # normalized by the maximum value of entire matrix.
            self.data = self.data / np.max(self.train_data)

        if self.normalize == 2:
            # normlized by the maximum value of each row (sensor).
            for i in range(self.cfg["data"]["channel"]):
                self.scale[i] = np.max(np.abs(self.train_data[:, i]))
                self.data[:, i] = self.data[:, i] / self.scale[i]

        if self.normalize == 3:
            # normlized by the mean/std value of each row (sensor).
            for i in range(self.cfg["data"]["channel"]):
                self.scale[i] = np.std(self.train_data[:, i])  # std
                self.bias[i] = np.mean(self.train_data[:, i])
                self.data[:, i] = (self.data[:, i] -
                                   self.bias[i]) / self.scale[i]

        # 单变量/多变量

    def __getitem__(self, index):
        # some model use time stamp
        # seq_len->lookback, seq_len-label_len->lookback,
        # seq_len+pred_len->horizon
        x = self.data[index: index + self.lookback]
        y = self.data[index +
                      self.lookback: index +
                      self.lookback +
                      self.horizon]
        if self.cfg["model"]["UseTimeFeature"]:
            timestamp_x = self.data_stamp[index: index + self.lookback]
            timestamp_y = self.data_stamp[
                index + self.lookback: index + self.lookback + self.horizon
            ]
        else:
            timestamp_x = 0
            timestamp_y = 0

        return x, y, timestamp_x, timestamp_y

    def __len__(self):
        return self.data.shape[0] - self.horizon - self.lookback + 1
