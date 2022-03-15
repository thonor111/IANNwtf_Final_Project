import tensorflow as tf
import tensorflow_datasets as tfds

def prepare_data(data):
    
    # standard pipeline
    data = data.cache().shuffle(1000).batch(10).prefetch(20)

    return data