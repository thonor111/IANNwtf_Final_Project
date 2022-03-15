import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import tensorflow_datasets as tfds
import input_pipeline

tf.keras.backend.clear_session()

# load dataset and split into to train (25,000) examples and test (2,500 examples) sets
train_data, test_data = tfds.load('imdb_reviews', split=['train', 'test[:10%]'], as_supervised=True)

# preprocess datasplits
train_data = train_data.apply(input_pipeline.prepare_data)
test_data = test_data.apply(input_pipeline.prepare_data)