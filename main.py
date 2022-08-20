
# Press the green button in the gutter to run the script.
import preprocess
import model
import transformers as trs
import tensorflow as tf
import numpy as np
if __name__ == '__main__':
    print("MAIN")
    processor = preprocess.TextProcessing("vinai/phobert-base")
    processor.read_csv("data.csv")

    m = model.QHD_PhoBert()

    train_encodings, val_encodings, train_labels, test_labels,train_dataset, val_dataset = processor.process()
    print(m(train_encodings[0]))