import tensorflow as tf
import transformers as trs
import numpy as np
'''
USAGE: This file is for INFERENCE ONLY, TRAINING will be done in TRAIN.py
ONLY use this AFTER TRAINING and EXPORTING WEIGHTS is COMPLETE
'''

def balance_loss(y: tf.Tensor, y_pred: tf.Tensor, alpha = 0.5, beta = 0.5) -> tf.Tensor:

    loss =  - (1.-y_pred)**alpha * y * tf.math.log(y_pred) -  (y_pred) ** beta * (1.-y)* tf.math.log(1. - y_pred)
    loss = tf.reduce_sum(loss, axis = 1) / 6
    loss = tf.reduce_mean(loss, keepdims=True)
    return loss

class ReviewClassifierModel(tf.Module):
    def __init__(self):
        super(ReviewClassifierModel, self).__init__()
        self.tokenizer = trs.AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.phobert = trs.TFAutoModel.from_pretrained("vinai/phobert-base")

        self.classifier = None

    def setup_classifier(self):
        #TEMPORARY, WILL ADD CONFIG TO CLASSIFIER LATER
        self.classifier = tf.keras.models.load_model("model", custom_objects={"balance_loss":balance_loss})

    def __call__(self, text):
        #Single input
        out = self.tokenizer(text, truncation=True, padding=True,  return_tensors="tf")
        out = self.phobert(out)
        out = self.classifier.predict(out[1])
        #HAVEN'T HANDLE YET
        out = np.reshape(out, (6))

        print(out[5])
        return out




