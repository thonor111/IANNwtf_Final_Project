import tensorflow as tf
import tensorflow.keras as K


class Discriminator(K.Model):

    def __init__(self):
        '''
        Initializes the discriminator
        '''
        super(Discriminator, self).__init__()

        clipping_value = 0.01

        self.input_layer = K.layers.Dense(100, kernel_constraint=tf.keras.constraints.MinMaxNorm(-clipping_value,
                                                                                                 clipping_value))
        self.res_blocks = []
        for i in range(40):
            res_block = [
                K.layers.Dense(100, activation='relu',
                               kernel_constraint=tf.keras.constraints.MinMaxNorm(-clipping_value, clipping_value)),
                K.layers.Dense(100, activation=None,
                               kernel_constraint=tf.keras.constraints.MinMaxNorm(-clipping_value, clipping_value)),
                K.layers.Add()
            ]
            self.res_blocks.append(res_block)

        self.out = K.layers.Dense(2, activation=None, kernel_constraint=tf.keras.constraints.MinMaxNorm(-clipping_value,
                                                                                                        clipping_value))  # linear activation for the WGAN

    # @tf.function
    def call(self, inputs, training):
        x = self.input_layer(inputs, training=training)
        inputs = x
        for res_block in self.res_blocks:
            x = res_block[0](x, training=training)
            x = res_block[1](x, training=training)
            x = res_block[2]((x, inputs), training=training)
        x = self.out(x, training=training)
        return x
