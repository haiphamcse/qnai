from processing import TextProcessing
import tensorflow as tf
import numpy as np
'''
USAGE: run train.py,
NOTES:
    Will add argument passer later
    Preprocessing ENTIRE DATASET may cause OOM -> Reduce size of X_train, X_test to mitigate this while testing
'''
def my_metric_fn(y_true, y_pred):
    f1_true = np.where(y_true > 0, 1, y_true)
    f1_pred = np.where(y_pred >0, 1, y_pred)
    f1 = np.bitwise_and(f1_true, f1_pred)
    print("F1", f1)
    #R2 - Score
    r2_all = ((y_true - y_pred)**2)/16
    r2 = 1 - (r2_all)
    print("R2", r2)
    return (1/6)*np.sum( np.multiply(f1, r2) )

def preprocess(data, train_size=50, test_size=20):
    processor = TextProcessing()
    X_train, X_test, Y_train, Y_test = processor.read_csv(data)
    X_train = processor.tokenize_bert(X_train[0:train_size])
    X_test = processor.tokenize_bert(X_test[0:test_size])
    return X_train, X_test, Y_train, Y_test

def train():
    print("TRAINING")
    train_features, test_features, train_labels, test_labels = preprocess("data.csv")
    print(test_features[1])

    shallow_mlp_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(6, activation="sigmoid"),
        ]  # More on why "sigmoid" has been used here in a moment.
    )

    y_pred = np.array([[1, 0, 3, 4, 2, 0], [0, 2, 2, 5, 1, 0]])
    y_true = np.array([[1, 1, 3, 5, 3, 1], [0, 0, 3, 4, 0, 0]])

    print(my_metric_fn(y_true, y_pred))

    # shallow_mlp_model.compile(optimizer="Adam", loss=tf.keras.losses.BinaryCrossentropy())
    # shallow_mlp_model.fit(test_features[1], test_labels[0:20], epochs= 5)
if __name__ == '__main__':
    train()