from unittest import TestCase
import torch
from torch.utils.data import DataLoader, Dataset
import transformers as trs
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class TextProcessing:
    def __init__(self):
        self.tokenizer = trs.AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.phobert = trs.AutoModel.from_pretrained("vinai/phobert-base")

    def tokenize(self, text):
        return self.tokenizer(text, truncation=True, padding=True)

    def read_csv(self, name, test_size=0.2):
        # Convert CSV to List
        data = pd.read_csv(name)
        labels = np.array(data.loc[:, ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']])
        features = data.loc[:, 'Review']
        del data
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=test_size)
        return list(X_train), list(X_test), Y_train, Y_test

    def tokenize_bert(self, text):
        out = self.tokenizer(text, truncation=True, padding=True)
        out = self.phobert(out)
        return out

class CustomDataset(Dataset):
    def __init__(self, csv_file) -> None:
        super(CustomDataset, self).__init__()
        preprocessor = TextProcessing()
        
        #   Labels format is not binary, but score. E.x: [3,2,1,0,0,1] . Therefore, need to 
        #   re-format in training
        self.x_train, self.x_test, self.y_train, self.y_test = preprocessor.read_csv(csv_file)
        self.x_train = preprocessor.tokenize_bert(self.x_train)
        self.x_test = preprocessor.tokenize_bert(self.x_test)


    def __len__(self):
        pass
    
    def __getitem__(self):
        pass

class QNAIDataset(CustomDataset):
    def __init__(self, mode, size) -> None:
        super(QNAIDataset, self).__init__()
        if mode == "train":
            self.data = self.x_train[:size][1] 
            self.label = self.y_train[:size]
        elif mode == "test":
            self.data = self.x_test[:size][1]
            self.label = self.y_test[:size]
    def __len__(self):
        return self.data
    
    def __getitem__(self, idx):
        return (self.x_train[idx], self.label[idx])