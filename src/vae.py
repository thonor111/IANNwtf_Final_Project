import tensorflow as tf
import tensorflow.keras as K
import numpy as np


class VariationalAutoEncoder(K.Model):

    def __init__(self, vocab_size, state_size):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = Encoder(state_size)
        self.decoder = Decoder(vocab_size)

    # @tf.function
    def call(self, inputs, training):
        x = self.encoder(inputs, training=training)
        y = self.decoder(inputs, x, training=training)

        return y

    # @tf.function
    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)


class Encoder(K.Model):

    def __init__(self, state_size):
        super(Encoder, self).__init__()

        # self.embedding_layer = K.layers.Embedding(input_dim=vocab_size, output_dim=state_size)
        self.lstm_layer = K.layers.LSTM(100)
        self.dense_layer = K.layers.Dense(state_size)

    # @tf.function
    def call(self, inputs, training):
        # x = self.embedding_layer(inputs)
        x = self.lstm_layer(inputs, training=training)
        y = self.dense_layer(x, training=training)

        return y


class Decoder(K.Model):

    def __init__(self, vocab_size):
        super(Decoder, self).__init__()

        # self.embedding_layer = K.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)
        self.lstm_layer = K.layers.LSTM(600, return_sequences=True, return_state=True)
        self.dense_layer = K.layers.Dense(vocab_size, activation='softmax')

    # @tf.function
    def call(self, inputs, states, training):
        # x = self.embedding_layer(inputs)
        x, _, _ = self.lstm_layer(inputs, initial_state=[states, tf.zeros_like(states)], training=training)
        y = self.dense_layer(x, training=training)

        return y

    # @tf.function
    def generate_sentence(self, starting_input, states, training=False):
        sentence = tf.cast(tf.squeeze(starting_input, (2)), tf.int64).numpy()
        max_sentence_length = 250
        current_sentence_length = 0
        memory_state = states
        carry_state = tf.zeros_like(states)
        sentence_ended = False
        while current_sentence_length < max_sentence_length and not sentence_ended:
            # x = self.embedding_layer(starting_input)
            x, memory_state, carry_state = self.lstm_layer(starting_input, initial_state=[memory_state, carry_state], training=training)
            y = self.dense_layer(x, training=training)
            starting_input = tf.argmax(y, axis=2)
            sentence = np.concatenate((sentence, starting_input.numpy()), axis=1)
            starting_input = tf.expand_dims(starting_input, -1)
            starting_input = tf.cast(starting_input, tf.float32)
            current_sentence_length += 1
        return sentence