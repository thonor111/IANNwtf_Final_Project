import tensorflow as tf
import tensorflow_datasets as tfds


def prepare_data(data):
    # standard pipeline
    data = data.cache().shuffle(1000).batch(10).prefetch(20)

    return data


def prepare_data_GAN(data):
    # adding noise as input for the GAN
    data = data.map(lambda embedding, sentiment: (embedding, sentiment, tf.random.uniform(shape=[100])))

    # standard pipeline
    data = data.cache().shuffle(1000).batch(10).prefetch(20)

    return data
