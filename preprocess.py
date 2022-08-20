import sklearn as sk
import sklearn.model_selection
import transformers as trs
import pandas as pd
import numpy as np
import datasets as dt
import tensorflow as tf

def read_csv(name):
    data = pd.read_csv(name)
    labels = np.array(data.loc[:, ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']])
    features = data.loc[:, 'Review']
    del data
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(features, labels, test_size=0.2)
    return list(X_train), list(X_test), list(Y_train), list(Y_test)


def create_tf_dataset(train_encodings, train_labels):
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels))
    return train_dataset

def preprocess(data, tokenizer):
    encoded_inputs = tokenizer(data, padding=True, truncation=True, return_tensors="tf")
    return encoded_inputs

def read_dataset(name):
    class_names = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']
    emotion_features = dt.Features({'text': dt.Value(dtype='string'), 'label': dt.ClassLabel(names=class_names)})
    dataset = dt.load_dataset('csv',data_files=name, column_names=['text', 'label'], features=emotion_features)
    return dataset

