import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
import warnings
import h5py
from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path="./data/ETT", flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, percent=100):
        if size == None:
            self.seq_len = 36  # 24 * 4 * 4
            self.label_len = 240  # 24 * 4
            self.pred_len = 240  # 24 * 4
        else:
            self.seq_len = size[0]  # size[0]
            self.label_len = size[1]  # size[1]
            self.pred_len = size[2]  # size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.tot_len = self.__read_data__()
        self.embed_path = f"/media/sdb/TSLAB/wangziyin/S2IP-LLM-main/Long-term_Forecasting/Embeddings/{data_path}/{flag}/"

    def __read_data__(self):
        # 读取原始数据集
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 获取数据的总列数
        total_columns = df_raw.shape[1]
        # 17860  3827  weibo2016
        # 59755  12804 aps2017
        # 36583 7316   twitter
        # 4821  1034   weibo2021
        #
        # 按照7:3:3的比例划分列索引，确定训练集、测试集、验证集的边界列索引
        train_columns_end = int(total_columns * 0.7)  # int(total_columns * 0.7)
        test_columns_end = int(total_columns * 0.85)
        valid_columns_end = total_columns

        border1s = [0, train_columns_end]
        border2s = [train_columns_end, test_columns_end]
        border3s = [test_columns_end, valid_columns_end]

        # 由于所有列都是单一特征，直接使用原数据框
        df_data = df_raw

        # 不需要数据标准化处理，直接将原数据赋值给data
        data = df_data.values

        # 根据当前的set_type（训练、测试、验证）提取对应的数据子集
        if self.set_type == 0:
            self.data_x = data[:self.seq_len, border1s[0]:border2s[0]]
            self.data_y = data[self.seq_len:, border1s[0]:border2s[0]]
            self.tot_len = train_columns_end
        elif self.set_type == 1:
            self.data_x = data[:self.seq_len, border2s[0]:border3s[0]]
            self.data_y = data[self.seq_len:, border2s[0]:border3s[0]]
            self.tot_len = test_columns_end - train_columns_end
        elif self.set_type == 2:
            self.data_x = data[:self.seq_len, border3s[0]:]
            self.data_y = data[self.seq_len:, border3s[0]:]
            self.tot_len = total_columns - test_columns_end
        return self.tot_len

    def __getitem__(self, index):
        feat_id = index
        s_begin = 0

        s_end = s_begin + self.seq_len
        r_begin = 0
        r_end = r_begin + self.label_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]

        embeddings_stack = []
        file_path = os.path.join(self.embed_path, f"{index}.h5")

        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as hf:
                data = hf['embeddings'][:]
                tensor = torch.from_numpy(data)
                embeddings_stack.append(tensor.squeeze(0))
        else:
            raise FileNotFoundError(f"No embedding file found at {file_path}")

        embeddings = torch.stack(embeddings_stack, dim=-1)

        return seq_x, seq_y, embeddings

    def __len__(self):
        return self.tot_len

    def inverse_transform(self, data):
        return data