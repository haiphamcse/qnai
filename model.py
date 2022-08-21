import transformers as trs
import pandas as pd
import numpy as np
import tensorflow as tf


class QHD_PhoBert(tf.keras.Model):
    def __init__(self, config=trs.AutoConfig.from_pretrained("vinai/phobert-base"), *inputs, **kwargs):
        super(QHD_PhoBert, self).__init__(*inputs, **kwargs)
        #main PhoBert Layer
        self.roberta = trs.TFRobertaMainLayer(config, name="roberta")
        self.roberta.trainable = False
        
        self.dense = tf.keras.layers.Dense(768)
        self.classifier = tf.keras.layers.Dense(6, activation="softmax", name="classification")
        self.train_loss_tracker = tf.keras.metrics.Mean(name = 'train_loss')
        self.val_loss_tracker = tf.keras.metrics.Mean(name = 'val_loss')

    # a single Tensor with input_ids only and nothing else: model(inputs_ids)
    # a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: model([input_ids, attention_mask]) or model([input_ids, attention_mask, token_type_ids])
    def call(self, inputs, **kwargs):
        """
        Params: 
            inputs: output of pretrained tokenizer
        """
        print("Model called")
        outputs = self.roberta(inputs, **kwargs)
        #MAY CAUSE ERROR WHEN INSERTING ENTIRE BATCH AS INPUTS
        #By transformers doc, the output[1] of Roberta layer is the padded output
        outputs = outputs[1] #  (batch_size, )
        outputs = self.dense(outputs)
        outputs = self.classifier(outputs)

        return outputs

    def train_step(self, train_x,  train_y, **kwargs):
        """
        :Params: 
            train_x: {'input_ids': list, 'attention_mask': list, 'token_ids': list}
            val_x: (batch, # classes)

        """
        #   Training
        print(train_x)
        with tf.GradientTape() as tape:
            outputs = self(train_x, training = True)
            print("Train y : {} -- Output y : {}".format(train_y.shape, outputs.shape))
            exit(0)
            train_loss = self.compiled_loss(train_y, outputs)

        grads = tape.gradient(train_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.train_loss_tracker.update_state(train_loss)
        self.compiled_metrics.update_state(train_y, outputs)

        #   Validate
        # self.trainable = False
        # val_outputs = self.call(val_x)
        # val_loss = self.loss(val_outputs, val_y)
        
        return {
            m.name: m.result() for m in self.metrics    
        }

def train(model, train_dataset):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits = True))
    model.fit(train_dataset.shuffle(1000).batch(16), epochs=2, batch_size=16)
