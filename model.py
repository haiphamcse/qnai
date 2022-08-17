import tensorflow as tf
from tensorflow import keras
from transformers import TFAutoModelForSequenceClassification
from transformers import TFTrainer, TFTrainingArguments
import os
class BERT(tf.Module):
    def __init__(self, bert_name, num_labels=6):
        '''

        :param bert_name: name of bert model
        :param num_labels: number of class to classify
        '''
        super(BERT, self).__init__()
        self.num_labels = num_labels
        self.bert = TFAutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=self.num_labels)

        #Freeze all layers except the last one
        for layer in self.bert.layers[0:-1]:
            layer.trainable = False

        self.bert.summary()

    def train(self, train_dataset, val_dataset, batch_size=16, epochs=3, learning_rate=5e-5):
        print("Training")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        checkpoint = keras.callbacks.ModelCheckpoint("best_model", save_best_only=True)
        self.bert.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy())  # can also use any keras loss fn
        self.bert.fit(train_dataset.shuffle(1000).batch(batch_size), epochs=epochs, batch_size=batch_size, validation_data=val_dataset.batch(batch_size), callbacks =[checkpoint])
        print("DONE")

    def predict(self, sentence):
        '''
        HAVEN'T IMPLEMENT YET
        :param sentence:
        :return:
        '''
        print("Prediction")

