from shannon_store import DataStorageBase
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from shannon_store import DataStorageBase
import numpy as np
import os
import pandas as pd


class LoadAllStockOneDayData(Dataset):
    def __init__(self, date_path = '/home/dongyikai/stock_date.txt', stock_num_path = '/home/dongyikai/stock_num.txt', window_size = 100, output_window = 1, pred_n = 60, pred_m = 600, pred_len = 3, label_len = 48):
        super().__init__()
        self.date_path = date_path
        self.stock_num_path = stock_num_path
        self.feature = []
        self.label = []
        self.window_size = window_size
        self.output_window = output_window
        self.pred_n = pred_n
        self.pred_m = pred_m
        self.label_len = label_len
        self.pred_len = pred_len
        self.openfile()
        
    def __getitem__(self, index):
        index1 = index // 135
        index2 = int(index % (135))
        feature = self.feature[index1][:, index2*100:index2*100+self.window_size].transpose(0, 1)
        feature2 = self.feature[index1][:, index2+self.window_size-self.label_len:index2+self.window_size+self.pred_len]
        # feature2 = None
        label = self.label[index1][index2 * 100]
        return feature, feature2, label
        
    def __len__(self):
        return 135 * self.stock_num

    def openfile(self):
        fp = DataStorageBase('/mnt/weka/home/shannon_public/StockData/1000ms_aligned/')
        data_list = ['MidPrice', 'AskPrice', 'AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5', 'AskPrice6', 'AskPrice7', 'AskPrice8', 'AskPrice9', 'AskPrice10',\
            'BidPrice', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5', 'BidPrice6', 'BidPrice7', 'BidPrice8', 'BidPrice9', 'BidPrice10', 'AskVol', \
            'AskVol2', 'AskVol3', 'AskVol4', 'AskVol5', 'AskVol6', 'AskVol7', 'AskVol8', 'AskVol9', 'AskVol10', 'BidVol', 'BidVol2', 'BidVol3', \
            'BidVol4', 'BidVol5', 'BidVol6', 'BidVol7', 'BidVol8', 'BidVol9', 'BidVol10']
        data_list1 = ['MidPrice', 'AskPrice', 'AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5', 'AskPrice6', 'AskPrice7', 'AskPrice8', 'AskPrice9', 'AskPrice10',\
            'BidPrice', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5', 'BidPrice6', 'BidPrice7', 'BidPrice8', 'BidPrice9', 'BidPrice10']
        with open(self.date_path, 'r') as f1:
            for date in f1.readlines():
                date = date.strip('\n\t').split()[0]
                with open(self.stock_num_path, 'r') as f2:
                    for stock_id in f2.readlines():
                        stock_id = stock_id.strip('\n\t').split()[0]
                        if fp.is_exists(stock_id, date) == False:
                            print("no stock: " + stock_id + ' ' + date)
                            continue
                        feature = fp.get_data(stock_id, date, 'TAQ', data_list1)
                        
                        feature = self.dict_to_numpy(feature)
                        label = self.generate_label(feature)
                        feature = self.normalize_feature(feature)
                        
                        # label = fp.get_data(stock_id, date, 'Label', ['sigma_lag_600'])
                        # if not isinstance(label, pd.DataFrame):
                        #     print("no stock label: " + stock_id + ' ' + date)
                        #     continue
                        # label = label['sigma_lag_600']
                        # feature, label = self.changefeature(feature, label)
                        self.feature += [feature]
                        self.label += [label]
        # print(self.label)
        self.feature = np.array(self.feature)
        self.label = np.array(self.label)
        self.feature = torch.tensor(self.feature, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.float32)
        fp.close()
        print(self.label.shape)
        self.stock_num = self.feature.shape[0]
    def generate_label(self, feature):
        index = np.array(range(feature.shape[1] - self.window_size - self.pred_m))
        ratio = feature[0, index + self.window_size + self.pred_n] / feature[0, index + self.window_size] - 1
        fea = feature[0, :].reshape(-1)
        import pandas as pd
        maxi = np.array(pd.Series(fea).rolling(self.window_size + self.pred_m).max().dropna().tolist())[:-1] / feature[0, index + self.window_size] - 1
        mini = np.array(pd.Series(fea).rolling(self.window_size + self.pred_m).min().dropna().tolist())[:-1] / feature[0, index + self.window_size] - 1
        label = np.array([ratio, maxi, mini]).transpose()


        # index2 = 0
        # label = []
        # while index2 + self.window_size + self.pred_m < feature.shape[1]:
        #     ratio = feature[0, index2+self.window_size + self.pred_n] / feature[0, index2+self.window_size] - 1
        #     maxi = max(feature[0, index2:index2+self.window_size + self.pred_m]) / feature[0, index2+self.window_size] - 1
        #     mini = min(feature[0, index2:index2+self.window_size + self.pred_m]) / feature[0, index2+self.window_size] - 1
        #     index2 += 1
        #     label.append([ratio, maxi, mini])
        return label
    def dict_to_numpy(self, feature):
        ans = []
        for k, v in feature.items():
            ans.append(v)
        return np.array(ans)

    def changefeature(self, feature, label):
        ans_f = []
        ans_l = []
        for index in range(self.window_size, len(label) - self.window_size):
            ans_f.append(feature[:, index - self.window_size:index])
            ans_l.append(label[index])
        return ans_f, ans_l

    def normalize_feature(self, feature):
        maxi, mini = max(feature[0, :]), min(feature[0, :])
        feature[:21, :] = 2 * feature[:21, :]/ (maxi - mini + 1e-5) - (maxi + mini) / (maxi - mini + 1e-5)
        return feature



class OneStockAllDayData(pl.LightningDataModule):
    def __init__(self, date_path = 'stock_date.txt', stock_num_path = 'stock_num.txt', window_size = 100, output_window = 1, batch_size = 128, pred_n = 60, pred_m = 600, pred_len = 3, label_len = 48):
        super().__init__()
        self.date_path = date_path
        self.stock_num_path = stock_num_path
        self.window_size = window_size
        self.output_window = output_window
        self.batch_size = batch_size
        self.pred_n = pred_n
        self.pred_m = pred_m
    def setup(self, stage):
        from torch.utils.data import random_split
        dataset = LoadAllStockOneDayData(self.date_path, self.stock_num_path, self.window_size, self.output_window, self.pred_n, self.pred_m, pred_len = 3, label_len = 48)
        total_len = len(dataset)
        train_len = int(total_len*0.9)
        val_len = int(train_len*0.9)
        intrain, self.test_set = random_split(dataset, [train_len, total_len - train_len])
        if stage == 'fit':
            self.train_set, self.val_set = random_split(intrain, [val_len, train_len - val_len])

    def train_dataloader(self) :
        return DataLoader(self.train_set, self.batch_size, shuffle=True)
    def val_dataloader(self) :
        return DataLoader(self.val_set, self.batch_size, shuffle=False)
    def test_dataloader(self) :
        return DataLoader(self.test_set, self.batch_size, shuffle=False)
