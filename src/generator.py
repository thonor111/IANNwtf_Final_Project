import tensorflow as tf
import tensorflow.keras as K


class Generator(K.Model):

    def __init__(self, state_size=250):
        '''
        Initializes the generator
        '''
        super(Generator, self).__init__()

        self.input_layer = K.layers.Dense(100)
        self.res_blocks = []
        for i in range(40):
            res_block = [
                K.layers.Dense(100, activation='relu'),
                K.layers.Dropout(0.5),
                K.layers.Dense(100),
                K.layers.Dropout(0.5),
                K.layers.Add()
            ]
            self.res_blocks.append(res_block)

        self.out = K.layers.Dense(state_size, activation='sigmoid')

    @tf.function
    def call(self, inputs, training):
        x = self.input_layer(inputs, training=training)
        inputs = x
        for res_block in self.res_blocks:
            x = res_block[0](x, training=training)
            x = res_block[1](x, training=training)
            x = res_block[2](x, training=training)
            x = res_block[3](x, training=training)
            x = res_block[4]((x, inputs), training=training)
        x = self.out(x, training=training)
        return x
