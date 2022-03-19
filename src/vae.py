import tensorflow as tf
import tensorflow.keras as K


class VariationalAutoEncoder(K.Model):

    def __init__(self, vocab_size, embedding_size):

        super(VariationalAutoEncoder, self).__init__()

        self.encoder = Encoder(vocab_size, embedding_size)
        self.decoder = Decoder(vocab_size, embedding_size)

    @tf.function
    def call(self, inputs, training):
        
        x = Encoder(inputs, training=training)
        y = Decoder(x, training=training)

        return y

class Encoder(K.Model):

    def __init__(self, vocab_size, embedding_size):

        super(Encoder, self).__init__()

        self.embedding_layer = K.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)
        self.lstm_layer = K.layers.LSTM(100)
        self.dense_layer = K.layers.Dense(600)

    @tf.function
    def call(self, inputs, training):

        x = self.embedding_layer(inputs)
        x = self.lstm_layer(x, training=training)
        y = self.dense_layer(x, training=training)

        return y

class Decoder(K.Model):

    def __init__(self, vocab_size, embedding_size):

        super(Decoder, self).__init__()

        self.embedding_layer = K.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)
        self.lstm_layer = K.layers.LSTM(600)
        self.dense_layer = K.layers.Dense(vocab_size)

    @tf.function
    def call(self, inputs, training):

        x = self.embedding_layer(inputs)
        x = self.lstm_layer(x, training=training)
        y = self.dense_layer(x, training=training)

        return y


