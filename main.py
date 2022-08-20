
# Press the green button in the gutter to run the script.
import preprocess
import model
import transformers as trs
import tensorflow as tf
if __name__ == '__main__':
    print("MAIN")
    X_train, X_test, Y_train, Y_test = preprocess.read_csv("data.csv")

    tokenizers = trs.AutoTokenizer.from_pretrained("vinai/phobert-base")
    sen = tokenizers("một anh trai gõ cửa kính, bảo cho em xin tiền gửi xe ô tô. Khi mình nói là ăn bún ở đây thì anh ấy bảo là vẫn phải trả tiền. Mình thề là k bao giờ quay lại đây!", return_tensors="tf", truncation=True, padding=True)
    m = model.QHD_PhoBert()
    print(m(sen))
    m.summary()