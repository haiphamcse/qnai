import transformers as trs
import tensorflow as tf


def train():
    model = trs.TFAutoModel.from_pretrained("vinai/phobert-base")
    tok = trs.AutoTokenizer.from_pretrained("vinai/phobert-base", is_fast=True)
    sentence = ["Sau khi thay máu hàng loạt vị trí lãnh đạo, Bamboo Airways chia sẻ về các khoản nợ cũng như trả lời dư luận về câu hỏi hãng hủy loạt chuyến bay chặng Úc",
                "Tình yêu đát nước vô bờ"
                ]
    output = tok(sentence, truncation=True, padding=True,  return_tensors="tf")
    output = model(output)
    #TFBaseModelOutputWithPoolingAndCrossAttentions,

    output = tf.nn.dropout(output[0], rate=0.2)
    print(output)
    output = tf.keras.layers.Dense(768)(output)
    #out = (batch, x, 768)
    #out[:,0,:] = (batch, 768) -> WHY?
    print(output[:,0,:])


if __name__ == '__main__':
    train()