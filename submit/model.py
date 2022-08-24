import tensorflow as tf
import transformers as trs

'''
USAGE: This file is for INFERENCE ONLY, TRAINING will be done in TRAIN.py
ONLY use this AFTER TRAINING and EXPORTING WEIGHTS is COMPLETE
'''


class ReviewClassifierModel(tf.Module):
    def __init__(self):
        super(ReviewClassifierModel, self).__init__()
        self.tokenizer = trs.AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.phobert = trs.TFAutoModel.from_pretrained("vinai/phobert-base")

        self.classifier = None

    def setup_classifier(self):
        #TEMPORARY, WILL ADD CONFIG TO CLASSIFIER LATER
        self.classifier = tf.keras.Sequential(
        )

    def __call__(self, text):
        #Single input
        out = self.tokenizer(text, truncation=True, padding=True,  return_tensors="tf")
        out = self.phobert(out)
        out = self.classifier(out)
        return out




