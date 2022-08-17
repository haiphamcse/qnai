import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoTokenizer

class TextProcessing:
    def __init__(self, tokenizer, max_length=256, shuffle=False):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.shuffle = shuffle
        self.train_features = None
        self.test_features = None
        self.train_labels = None
        self.test_labels = None

    def read_csv(self, name):
        #Convert CSV to List
        data = pd.read_csv(name)
        labels = np.array(data.loc[:, ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']])
        features = data.loc[:, 'Review']
        del data
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2)
        self.train_features, self.test_features, self.train_labels, self.test_labels = list(X_train), list(X_test), list(Y_train), list(Y_test)

    def process(self):
        #Encode string using tokenizer
        train_encodings = self.tokenizer(self.train_features, truncation=True, padding=True)
        val_encodings = self.tokenizer(self.test_features, truncation=True, padding=True)

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            self.train_labels
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            self.test_labels
        ))
        return train_dataset, val_dataset