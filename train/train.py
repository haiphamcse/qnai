from processing import TextProcessing

'''
USAGE: run train.py,
NOTES:
    Will add argument passer later
    Preprocessing ENTIRE DATASET may cause OOM -> Reduce size of X_train, X_test to mitigate this while testing
'''


def preprocess(data, train_size=50, test_size=20):
    processor = TextProcessing()
    X_train, X_test, Y_train, Y_test = processor.read_csv(data)
    X_train = processor.tokenize_bert(X_train[0:train_size])
    X_test = processor.tokenize_bert(X_test[0:test_size])
    return X_train, X_test, Y_train, Y_test

def train():
    print("TRAINING")
    train_features, test_features, train_labels, test_labels = preprocess("data.csv")
    print(test_features)

if __name__ == '__main__':
    train()