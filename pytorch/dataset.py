from unittest import TestCase
import torch
from torch.utils.data import DataLoader, Dataset
import transformers as trs
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class QNAIDataset(Dataset):

    def __init__(self, csv_file, mode):
        super(QNAIDataset, self).__init__()

        print("Load csv...")

        data = pd.read_csv(csv_file)
        labels = np.array(data.loc[:, ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']])
        features = data.loc[:, 'Review']
        del data
        print("Load successfully!")

        x_train, x_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.2)
        self.mode = mode
        self.x_train = list(x_train)
        self.x_test = list(x_test)

    def __len__(self):
        if self.mode == "train":
            return len(self.x_train)
        elif self.mode == "test":
            return len(self.x_test)

    def __getitem__(self, idx) :
        if self.mode == "train":
            return (self.x_train[idx], self.y_train[idx])
        elif self.mode == "test":
            return (self.x_test[idx], self.y_test[idx])

def get_data_loader(dataset: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

