# import sklearn as sk
# import sklearn.model_selection
# import transformers as trs
# import pandas as pd
# import numpy as np
# import datasets as dt
# import tensorflow as tf
#
# def read_csv(name):
#     data = pd.read_csv(name)
#     labels = np.array(data.loc[:, ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']])
#     features = data.loc[:, 'Review']
#     del data
#     X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(features, labels, test_size=0.2)
#     return list(X_train), list(X_test), list(Y_train), list(Y_test)
#
#
# def create_tf_dataset(train_encodings, train_labels):
#     train_dataset = tf.data.Dataset.from_tensor_slices((
#         dict(train_encodings),
#         train_labels))
#     return train_dataset
#
# def preprocess(data, tokenizer):
#     encoded_inputs = tokenizer(data, padding=True, truncation=True, return_tensors="tf")
#     return encoded_inputs
#
# def read_dataset(name):
#     class_names = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']
#     emotion_features = dt.Features({'text': dt.Value(dtype='string'), 'label': dt.ClassLabel(names=class_names)})
#     dataset = dt.load_dataset('csv',data_files=name, column_names=['text', 'label'], features=emotion_features)
#     return dataset
#

from tkinter.tix import Tree
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import transformers as trs
from transformers import AutoTokenizer
import re
import underthesea
import datasets as ds

strip_special_chars = re.compile("[^\w0-9 ]+")


class TextProcessing:
    def __init__(self, tokenizer, max_length=256, shuffle=False):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, is_fast=True)
        print(self.tokenizer)
        self.max_length = max_length
        self.shuffle = shuffle
        self.train_features = None
        self.test_features = None
        self.train_labels = None
        self.test_labels = None

    def read_csv(self, name):
        # Convert CSV to List
        data = pd.read_csv(name)
        labels = np.array(data.loc[:, ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']])
        features = data.loc[:, 'Review']
        del data
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2)
        self.train_features, self.test_features, self.train_labels, self.test_labels = list(X_train), list(
            X_test), Y_train, Y_test

    def _clean_sentences(self, string):
        string = string.lower().replace("<br />", " ")
        return re.sub(strip_special_chars, "", string.lower())

    def process(self):
        #   Clean sentences --> just include digits and numbers. Remove commas, dots, etc...
        self.train_features = [self._clean_sentences(sentence) for sentence in
                               self.train_features]  # --> Return list: contain each sentence in train set
        self.test_features = [self._clean_sentences(sentence) for sentence in
                              self.test_features]  # --> Return list: containt each sentence in test set

        #   Tokenize sentences
        self.train_tokens = [underthesea.word_tokenize(sentence) for sentence in
                             self.train_features]  # --> Nested List: contain each list of tokenized words
        self.test_tokens = [underthesea.word_tokenize(sentence) for sentence in self.test_features]


        # print(self.train_tokens[:5])

        #   Encode string using tokenizer
        #DH-Temporary disable underthesea tokenizer
        train_encodings = [self.tokenizer(train_features, truncation=True, padding=True,  return_tensors="tf") for train_features in
                           self.train_features]  # --> Return word indices of each tokenized words list
        val_encodings = [self.tokenizer(test_features, truncation=True, padding=True,  return_tensors="tf") for test_features in
                         self.test_features]  # --> Return word indices

        train_dataset = tf.data.Dataset.from_tensor_slices((
            train_encodings,
            self.train_labels
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
          val_encodings,
            self.test_labels
        ))
        return train_encodings, val_encodings, self.train_labels, self.test_labels, train_dataset, val_dataset


class TFTextProcessing:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    def clean_function(self, examples):
        return re.sub(strip_special_chars, "", examples['Review'].lower().replace("<br />", " "))

    def tokenize_function(self, examples):
        return self.tokenizer(examples["Review"], truncation=True, padding=True,  return_tensors="tf")

    def process(self, name):
        datasets = ds.load_dataset("csv", data_files=name)
        datasets.map(self.clean_function, batched=True)
        datasets.map(self.tokenize_function, batched=True)
        print(datasets)


