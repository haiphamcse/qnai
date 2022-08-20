import transformers as trs
import pandas as pd
import numpy as np
import tensorflow as tf


class QHD_PhoBert(tf.keras.Model):
    def __init__(self, config=trs.AutoConfig.from_pretrained("vinai/phobert-base"), *inputs, **kwargs):
        super(QHD_PhoBert, self).__init__(*inputs, **kwargs)
        #main PhoBert Layer
        self.roberta = trs.TFRobertaMainLayer(config, name="roberta")
        self.dense = tf.keras.layers.Dense(768)
        self.classifier = tf.keras.layers.Dense(6, activation="softmax", name="classification")

    # a single Tensor with input_ids only and nothing else: model(inputs_ids)
    # a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: model([input_ids, attention_mask]) or model([input_ids, attention_mask, token_type_ids])
    def call(self, input_ids, **kwargs):
        outputs = self.roberta(input_ids, **kwargs)
        #MAY CAUSE ERROR WHEN INSERTING ENTIRE BATCH AS INPUTS
        #By transformers doc, the output[1] of Roberta layer is the padded output
        outputs = outputs[1]
        outputs = self.dense(outputs)
        outputs = self.classifier(outputs)
        return outputs
