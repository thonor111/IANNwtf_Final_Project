import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text

tokenizer_layer = tf.keras.layers.TextVectorization(output_sequence_length=250)


def prepare_data(data):
    # tokenizing the text
    plain_text = data.map(lambda text, sentiment: text)
    tokenizer_layer.adapt(plain_text)
    data = data.map(tokenize_data)
    # standard pipeline
    data = data.cache().shuffle(1000).batch(10).prefetch(20)

    return data


def tokenize_data(text, label):
    # text = tf.expand_dims(text, -1)
    return tokenizer_layer(text), label


def prepare_data_GAN(data):
    # tokenizing the text
    plain_text = data.map(lambda text, sentiment: text)
    tokenizer_layer.adapt(plain_text)
    data = data.map(tokenize_data)
    # adding noise as input for the GAN
    data = data.map(lambda embedding, sentiment: (embedding, sentiment, tf.random.uniform(shape=[100])))

    # standard pipeline
    data = data.cache().shuffle(1000).batch(10).prefetch(20)

    return data
