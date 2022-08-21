
# Press the green button in the gutter to run the script.
import preprocess
import model
import transformers as trs
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

if __name__ == '__main__':
    print("MAIN")
    processor = preprocess.TextProcessing("vinai/phobert-base")
    processor.read_csv("data.csv")

    m = model.QHD_PhoBert()

    train_encodings, val_encodings, train_labels, test_labels,train_dataset, val_dataset = processor.process()
    train_labels = np.array(train_labels, dtype='float64')
    print(train_encodings)    
    exit(0)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    m.compile(optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
    m.fit(train_encodings, train_labels, epochs = 2, batch_size = 1)

    
    
    
    
    
    
    
    
    
    
    
    
    
    # print(train_labels)
    #xxx_encodings: list of batch(transformers.tokenization_utils_base.BatchEncoding)
    # for i in val_dataset.as_numpy_iterator():
    #     for j in np.nditer(i[0]):
    #         print(j)
    #     print(i[1])



    # test = [
    #     "Thầy là nhất",
    #     "Tôi yêu biển đảo trường sa",
    #     "Cách mạng là con đường giúp đất nước phát triển kinh tế, xây dựng nhà nước"
    # ]

    # test_tokenized = processor.tokenizer(test, truncation=True, padding=True, return_tensors="tf")
    # out = m(val_encodings[0])
    # print(out)