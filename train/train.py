from asyncio import constants
from processing import TextProcessing
import tensorflow as tf
import numpy as np
import model

'''
USAGE: run train.py,
NOTES:
    Will add argument passer later
    Preprocessing ENTIRE DATASET may cause OOM -> Reduce size of X_train, X_test to mitigate this while testing
'''

def balance_loss(y: tf.Tensor, y_pred: tf.Tensor, alpha = 0.5, beta = 0.5) -> tf.Tensor:

    loss =  - (1.-y_pred)**alpha * y * tf.math.log(y_pred) -  (y_pred) ** beta * (1.-y)* tf.math.log(1. - y_pred)
    loss = tf.reduce_sum(loss, axis = 1) / 6
    loss = tf.reduce_mean(loss, keepdims=True)
    return loss

def preprocess(data, train_size=25, test_size=20):
    processor = TextProcessing()
    X_train, X_test, Y_train, Y_test = processor.read_csv(data)
    
    #   Tokenize text
    X_train = processor.tokenize_bert(X_train[0:train_size])
    X_test = processor.tokenize_bert(X_test[0:test_size])

    #   Get label from grounth truth
    Y_train = (np.array(Y_train[:train_size]) > 0).astype(np.float64)
    Y_test = (np.array(Y_test[:test_size]) > 0).astype(np.float64)
    
    return X_train, X_test, Y_train[:train_size], Y_test[:test_size]


def train(model, epochs: int, batch_size: int, lr: float, save_path, beta_adam = [0.9, 0.99]):

    print("TRAINING")
    train_features, test_features, train_labels, test_labels = preprocess("data.csv")

    optimizer = tf.keras.optimizers.Adagrad()
    model.compile(optimizer = optimizer, loss = balance_loss)   
    history = model.fit(x = train_features[1], y = train_labels, validation_data = (test_features[1], test_labels), 
                        epochs = epochs, batch_size = batch_size,
                        callbacks = tf.keras.callbacks.ModelCheckpoint(save_path, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min'))

    return history

if __name__ == '__main__':
    
    model = model.QHD_Model(seperate = False)

    history = train(model, 100, 5, 0.01, "ckpt")

    # y = tf.constant([[1., 0., 1., 0., 1., 1.],
                    # [0., 1., 1., 0., 0., 1.],
                    # [1., 1., 1., 0., 1., 0.]])
    # print(y)
    # y_pred = tf.math.sigmoid(tf.random.normal((3,6)))

    # loss = balance_loss(y, y_pred)
    # print(loss)