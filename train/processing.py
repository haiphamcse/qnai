import transformers as trs
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
'''
USAGE: 
    Prepare dataset from csv file
    Tokenized and Bertized dataset before feeding into model

NOTES:
    Haven't remove icons and colon/semicolons yet -> Will implement later
'''

class TextProcessing:
    def __init__(self):
        self.tokenizer = trs.AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.phobert = trs.TFAutoModel.from_pretrained("vinai/phobert-base")

    def tokenize(self, text):
        return self.tokenizer(text, truncation=True, padding=True,  return_tensors="tf")

    def read_csv(self, name, test_size=0.2):
        # Convert CSV to List
        data = pd.read_csv(name)
        labels = np.array(data.loc[:, ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']])
        features = data.loc[:, 'Review']
        del data
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=test_size)
        return list(X_train), list(X_test), Y_train, Y_test

    def tokenize_bert(self, text):
        out = self.tokenizer(text, truncation=True, padding=True, return_tensors="tf")
        out = self.phobert(out)
        return out