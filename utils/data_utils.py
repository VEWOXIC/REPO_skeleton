from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

from typing import List
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

def get_scaler(scalar_name):
    if(scalar_name == 'standard'):
        return StandardScaler()
    elif(scalar_name == 'minmax'):
        return MinMaxScaler()
    elif(scalar_name == 'maxabs'):
        return MaxAbsScaler()
    elif(scalar_name == 'robust'):
        return RobustScaler()
    else:
        return StandardScaler()

