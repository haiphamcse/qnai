
# Press the green button in the gutter to run the script.
import model

# processor = preprocess.TextProcessing("vinai/phobert-base")
    # processor.read_csv("data.csv")
    # train_encodings, val_encodings, train_labels, val_labels, train_dataset, val_dataset = processor.process()
    #
    # m = model.QHD_PhoBert()
    #
    #
    # #xxx_encodings: list of batch(transformers.tokenization_utils_base.BatchEncoding)
    # # for i in val_dataset.as_numpy_iterator():
    # #     for j in np.nditer(i[0]):
    # #         print(j)
    # #     print(i[1])
    #
    # test = [
    #     "Thầy là nhất",
    #     "Tôi yêu biển đảo trường sa",
    #     "Cách mạng là con đường giúp đất nước phát triển kinh tế, xây dựng nhà nước"
    # ]
    #
    # test_tokenized = processor.tokenizer(test, truncation=True, padding=True, return_tensors="tf")
    # print(len(val_encodings), len(val_labels))
    #
    # # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # #
    # # print("BEFORE", val_labels.shape)
    # # n_val_labels = val_labels.reshape(1, 738, 6)
    # # print("AFTER", n_val_labels.shape)
    #
    # print(m(val_encodings[0]))
if __name__ == '__main__':
    model = model.pipeline()