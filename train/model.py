import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Layer, Input

class QHD_Model(tf.keras.Model):
    def __init__(self, seperate = False):
        """
        :params:
            seperate: bool - > if one use 6 seperated output layers
        """
        super(QHD_Model, self).__init__()
        self.mode = seperate
        self.train_loss_tracker = tf.keras.metrics.Mean()
        self.val_loss_tracker = tf.keras.metrics.Mean()

        if not seperate:
            self.kernels = tf.keras.Sequential([
                Dense(768, activation = 'relu'),
                Dense(6, activation = 'sigmoid')
            ])

        else:
            self.kernels = []
            self.kernels.extend( [tf.keras.Sequential([
                    Dense(768, activation = 'relu'),
                    Dense(1, activation = 'sigmoid')
                ]) for i in range(6)]
            )
    
    def call(self, x):

        #   Not seperated
        if not self.mode:
            out = self.kernels(x)
        
        #   Seperatedly
        else:
            out = self.kernels[0](x)
            for i in range(1,len(self.layers)):
                out = tf.concat((out, self.layers[i](x)), axis = 1)
        return out
    
    