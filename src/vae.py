import tensorflow as tf
import tensorflow.keras as K


class VariationalAutoEncoder(K.Model):

    def __init__(self, vocab_size, embedding_size):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = Encoder(vocab_size, embedding_size)
        self.decoder = Decoder(vocab_size, embedding_size)

    @tf.function
    def call(self, inputs, training):
        x = self.encoder(inputs, training=training)
        y = self.decoder(inputs, x, training=training)

        return y

    @tf.function
    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)


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
        self.lstm_layer = K.layers.LSTM(600, return_sequences=True)
        self.dense_layer = K.layers.Dense(vocab_size, activation='softmax')

    @tf.function
    def call(self, inputs, states, training):
        x = self.embedding_layer(inputs)
        x = self.lstm_layer(x, initial_state=[states, tf.zeros_like(states)], training=training)
        y = self.dense_layer(x, training=training)

        return y
