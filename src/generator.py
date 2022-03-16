import tensorflow as tf
import tensorflow.keras as K


class Generator(K.Model):

    def __init__(self):
        '''
        Initializes the generator
        '''
        super(Generator, self).__init__()

        self.embedding_size = 100

        self.input_layer = K.layers.Dense(100)
        self.res_blocks = []
        for i in range(40):
            res_block = [
                K.layers.Dense(100, activation='relu'),
                K.layers.Dense(100),
                K.layers.Add()
            ]
            self.res_blocks.append(res_block)

        self.out = K.layers.Dense(self.embedding_size, activation='sigmoid')

    @tf.function
    def call(self, inputs, training):
        x = self.input_layer(inputs, training=training)
        inputs = x
        for res_block in self.res_blocks:
            x = res_block[0](x, training=training)
            x = res_block[1](x, training=training)
            x = res_block[2]((x, inputs), training = training)
        x = self.out(x, training=training)
        return x
