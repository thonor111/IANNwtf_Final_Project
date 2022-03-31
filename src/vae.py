import tensorflow as tf
import tensorflow.keras as K
import numpy as np


class VariationalAutoEncoder(K.Model):

    def __init__(self, vocab_size, state_size):
        """
        Initializes the VAE
        :param vocab_size: The size of the vocabulary used in the sentences
        :param state_size: The size of the states to be used by the decoder
        """
        super(VariationalAutoEncoder, self).__init__()

        self.embedding_size = 100

        self.encoder = Encoder(state_size, vocab_size, self.embedding_size)
        self.decoder = Decoder(vocab_size, self.embedding_size)

    def call(self, inputs, training):
        """
        Encodes the input and decodes the encoding again
        :param inputs:
        :param training: Whether the model gets trained
        :return: The decoded input after getting encoded
        """
        x = self.encoder(inputs, training=training)
        y = self.decoder(inputs, x, training=training)

        return y

    def encode(self, inputs, training=False):
        """
        Only calls the encoder of the VAE
        :param inputs: Inputs to the Encoder
        :param training: Whether the Encoder gets trained
        :return: Output of the Encoder
        """
        return self.encoder(inputs, training=training)


class Encoder(K.Model):

    def __init__(self, state_size, vocab_size, embedding_size):
        """
        Initializes the Encoder
        :param vocab_size: The size of the vocabulary used in the sentences
        :param state_size: The size of the states to be used by the decoder
        :param embedding_size: The size of the embedding to be used on the inputs
        """
        super(Encoder, self).__init__()

        self.embedding_layer = K.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)
        self.lstm_layer = K.layers.LSTM(100, return_state=True, dropout=0.5)
        self.dense_layer = K.layers.Dense(state_size, activation='sigmoid')

    def call(self, inputs, training):
        """
        Encodes the input
        :param inputs: Inputs to be decoded
        :param training: Whether the Encoder gets trained
        :return: The encoded input
        """
        x = self.embedding_layer(inputs)
        x = tf.squeeze(x)
        x, hidden_state, cell_state = self.lstm_layer(x, training=training)
        y = self.dense_layer(tf.concat((x, cell_state), axis=1), training=training)

        return y


class Decoder(K.Model):

    def __init__(self, vocab_size, embedding_size):
        """
        Initializes the Decoder
        :param vocab_size: The size of the vocabulary used in the sentences
        :param embedding_size: The size of the embedding to be used on the inputs
        """
        super(Decoder, self).__init__()

        self.embedding_layer = K.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)
        self.lstm_layer = K.layers.LSTM(600, return_sequences=True, return_state=True)
        self.dropout = K.layers.Dropout(0.5)
        self.dense_layer = K.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs, states, training):
        """
        Decodes the given states given the current input
        :param inputs: The input the Encoder got
        :param states: The encoded states
        :param training: Whether the decoder gets trained
        :return: A Tensor with probabilities for each word in the vocab to follow the input
        """
        x = self.embedding_layer(inputs)
        x = tf.squeeze(x)
        x, _, _ = self.lstm_layer(x, initial_state=[states, tf.zeros_like(states)], training=training)
        x = self.dropout(x, training=training)
        y = self.dense_layer(x, training=training)

        return y

    def generate_sentence(self, starting_input, states, training=False):
        """
        Generates a sentence based on the given starting input and states encoding this sentence
        :param starting_input: The token resembling [START]
        :param states: The states used to build this sentence
        :param training: Whether the Decoder gets trained
        :return: The generated sentence
        """
        sentence = tf.cast(tf.squeeze(starting_input, (2)), tf.int64).numpy()
        max_sentence_length = 50
        current_sentence_length = 0
        memory_state = states
        carry_state = tf.zeros_like(states)
        while current_sentence_length < max_sentence_length:
            x = self.embedding_layer(starting_input)
            x = tf.squeeze(x)
            x = tf.expand_dims(x, -2)
            x, memory_state, carry_state = self.lstm_layer(x, initial_state=[memory_state, carry_state],
                                                           training=training)
            y = self.dense_layer(x, training=training)
            starting_input = tf.argmax(y, axis=2)
            sentence = np.concatenate((sentence, starting_input.numpy()), axis=1)
            starting_input = tf.expand_dims(starting_input, -1)
            starting_input = tf.cast(starting_input, tf.float32)
            current_sentence_length += 1
        return sentence
