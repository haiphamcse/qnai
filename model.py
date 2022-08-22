import transformers as trs
import pandas as pd
import numpy as np
import tensorflow as tf
import preprocess


def pipeline(model=None, datafile="data.csv", config="vinai/phobert-base", train_size=60, test_size=20):
    v_phobert = trs.TFAutoModel.from_pretrained(config)
    v_tokenizer = trs.AutoTokenizer.from_pretrained(config)
    processor = preprocess.TextProcessing(config)
    processor.read_csv(datafile)

    out = v_tokenizer(processor.train_features[0:train_size], truncation=True, padding=True,  return_tensors="tf")
    out = v_phobert(out)

    out_test = v_tokenizer(processor.test_features[0:test_size], truncation=True, padding=True,  return_tensors="tf")
    out_test = v_phobert(out_test)

    print(out["pooler_output"])
    x = out["pooler_output"]
    y = processor.train_labels[0:train_size]

    x_test = out_test["pooler_output"]
    y_test = processor.train_labels[0:test_size]

    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(768, input_shape=(768,)),
         tf.keras.layers.Dense(6, activation="relu")]
    )

    model.compile(optimizer='Adam', loss=tf.keras.losses.MeanSquaredError(), metrics=tf.keras.metrics.Accuracy())
    model.fit(x, y, validation_data =(x_test, y_test), batch_size=32, epochs=200)

    test = v_tokenizer(processor.test_features[22], truncation=True, padding=True,  return_tensors="tf")
    test = v_phobert(test)
    test_predict = test["pooler_output"]
    print(model.predict(test_predict))
    print(processor.test_labels[22])
    return model

# class QHD_PhoBert(tf.keras.Model):
#     def __init__(self, config=trs.AutoConfig.from_pretrained("vinai/phobert-base"), *inputs, **kwargs):
#         super(QHD_PhoBert, self).__init__(*inputs, **kwargs)
#         #main PhoBert Layer
#         self.roberta = trs.TFRobertaMainLayer(config, name="roberta")
#         self.roberta.trainable = False
#         self.dense = tf.keras.layers.Dense(768)
#         self.classifier = tf.keras.layers.Dense(6, activation="softmax", name="classification")
#
#     # a single Tensor with input_ids only and nothing else: model(inputs_ids)
#     # a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: model([input_ids, attention_mask]) or model([input_ids, attention_mask, token_type_ids])
#     def call(self, inputs, **kwargs):
#         outputs = self.roberta(inputs, **kwargs)
#         #MAY CAUSE ERROR WHEN INSERTING ENTIRE BATCH AS INPUTS
#         #By transformers doc, the output[1] of Roberta layer is the padded output
#         outputs = outputs[1]
#         outputs = self.dense(outputs)
#         outputs = self.classifier(outputs)
#         return outputs
#
#     def train_step(self, x, y):
#         # Unpack the data. Its structure depends on your model and
#         # on what you pass to `fit()`.
#         print(x)
#         with tf.GradientTape() as tape:
#             y_pred = self(x)  # Forward pass
#             # Compute the loss value
#             # (the loss function is configured in `compile()`)
#             loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
#
#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         # Update metrics (includes the metric that tracks the loss)
#         self.compiled_metrics.update_state(y, y_pred)
#         # Return a dict mapping metric names to current value
#         return {m.name: m.result() for m in self.metrics}
#
#     def custom_train(self, x, y):
#         optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#         mse_loss_fn = tf.keras.losses.MeanSquaredError()
#         loss_metric = tf.keras.metrics.Mean()
#         epochs = 2
#
#         # Iterate over epochs.
#         for epoch in range(epochs):
#             print("Start of epoch %d" % (epoch,))
#             for i in range(0, len(x)):
#                 with tf.GradientTape() as tape:
#                     tape.watch(self.trainable_weights)
#                     y_ = self.call(x[i])
#                     print("Y_", y_)
#                     loss = mse_loss_fn(y_ , y[i])
#                 grads = tape.gradient(loss, self.trainable_weights)
#                 optimizer.apply_gradients(zip(grads, self.trainable_weights))
#                 loss_metric(loss)
#
#                 if i % 100 == 0:
#                     print("step %d: mean loss = %.4f" % (i, loss_metric.result()))
