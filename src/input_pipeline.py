import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
import sentencepiece as sp

class Input_Pipeline():

    def __init__(self):
        self.tokenizer_layer = tf.keras.layers.TextVectorization(output_sequence_length=250)
        self.start_token = 0
        self.end_token = 0

    def train_tokenizer(self, data):
        plain_text = data.map(lambda text, sentiment: text)
        self.tokenizer_layer.adapt(plain_text)
        self.start_token = self.tokenizer_layer.vocabulary_size()
        self.end_token = self.tokenizer_layer.vocabulary_size() + 1

    def prepare_data(self, data):
        # tokenizing the text
        data = data.map(self.tokenize_data)
        # standard pipeline
        data = data.cache().shuffle(1000).batch(10).prefetch(20)

        return data


    def tokenize_data(self, text, label):
        # text = tf.expand_dims(text, -1)
        return self.tokenizer_layer(text), label


    def prepare_data_GAN(self, data):
        # tokenizing the text
        data = data.map(self.tokenize_data)
        # adding start and end to every sentence
        data = data.map(lambda embedding, sentiment: (tf.concat((tf.constant(self.start_token, dtype=tf.int64,
                                                                             shape=(1,)), embedding,
                                                                 tf.constant(self.end_token, dtype=tf.int64,
                                                                             shape=(1,))), axis=0), sentiment))

        # adding noise as input for the GAN
        data = data.map(lambda embedding, sentiment: (embedding, sentiment, tf.random.uniform(shape=[100])))

        data = data.map(
            lambda embedding, sentiment, noise: (embedding, tf.roll(embedding, shift=-1, axis=0), sentiment, noise))


        # standard pipeline
        data = data.cache().shuffle(1000).batch(3).prefetch(20)

        return data
