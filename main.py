
# Press the green button in the gutter to run the script.
import preprocess
import model
import transformers as trs
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)
if __name__ == '__main__':
    print("MAIN")
    processor = preprocess.TextProcessing("vinai/phobert-base")
    processor.read_csv("data.csv")

    with tf.device('/cpu:0'):
        m = model.QHD_PhoBert()

    train_encodings, val_encodings, train_labels, test_labels,train_dataset, val_dataset = processor.process()
    #xxx_encodings: list of batch(transformers.tokenization_utils_base.BatchEncoding)
    # for i in val_dataset.as_numpy_iterator():
    #     for j in np.nditer(i[0]):
    #         print(j)
    #     print(i[1])
    model.train(m, train_dataset)