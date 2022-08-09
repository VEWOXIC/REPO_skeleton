import os
from sklearn.preprocessing import StandardScaler

def get_scaler(scalar_name):
    return StandardScaler()

def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

