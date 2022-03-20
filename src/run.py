import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import tensorflow_datasets as tfds
from input_pipeline import Input_Pipeline
import training_loop
from generator import Generator
from discriminator import Discriminator
from vae import VariationalAutoEncoder

tf.keras.backend.clear_session()

# load dataset and split into to train (25,000) examples and test (2,500 examples) sets
train_data, test_data = tfds.load('imdb_reviews', split=['train', 'test[:10%]'], as_supervised=True)

input_pipeline = Input_Pipeline()

input_pipeline.train_tokenizer(train_data)

# preprocess datasplits
# using the GAn version to be able to test the GAN
train_data = train_data.apply(input_pipeline.prepare_data_GAN)
test_data = test_data.apply(input_pipeline.prepare_data_GAN)

##################################################################
# Training of the Autoencoder
##################################################################

num_epochs_vae = 10
alpha_vae = 0.001

# Initialize Model
vocab_size = input_pipeline.tokenizer_layer.vocabulary_size()
embedding_size = 250
vae = VariationalAutoEncoder(vocab_size=vocab_size, embedding_size=embedding_size)

# optimizer
optimizer_vae = K.optimizers.Adam(alpha_vae)

# loss function
loss_function_vae = K.losses.SparseCategoricalCrossentropy()

# initialize lists for later visualization.
train_losses_vae = []
test_losses_vae = []

# We train for num_epochs epochs.
for epoch in range(num_epochs_vae):

    # training (and checking in with training)
    epoch_losses_vae = []
    for embedding, sentiment, noise in train_data:
        train_loss_vae = training_loop.train_step_vae(vae=vae,
                                                      input=embedding,
                                                      target=embedding,
                                                      loss_function=loss_function_vae,
                                                      optimizer=optimizer_vae)
        epoch_losses_vae.append(train_loss_vae)
        print(train_loss_vae)

    # track training loss
    train_losses_vae.append(tf.reduce_mean(epoch_losses_vae))


##################################################################
# Training of the GAN
##################################################################

# Hyperparameters
num_epochs_gan = 10
alpha_generator = 0.00005
alpha_discriminator = 0.00005

# Initialize Model
generator = Generator(embedding_size=embedding_size)
discriminator = Discriminator()

# optimizer
optimizer_generator = K.optimizers.RMSprop(alpha_generator)
optimizer_discriminator = K.optimizers.RMSprop(alpha_discriminator)

# initialize lists for later visualization.
train_losses_discriminator = []
test_losses_discriminator = []
train_losses_generator = []
test_losses_generator = []

# We train for num_epochs epochs.
learning_step = 0
for epoch in range(num_epochs_gan):

    # training (and checking in with training)
    epoch_losses_discriminator = []
    epoch_losses_generator = []
    for embedding, sentiment, noise in train_data:
        learning_step += 1
        train_loss_discriminator, train_loss_generator = training_loop.train_step_gan(generator, discriminator,
                                                                                      encoded_sentence=embedding,
                                                                                      gaussian=noise,
                                                                                      sentiment=sentiment,
                                                                                      optimizer_generator=optimizer_generator,
                                                                                      optimizer_discriminator=optimizer_discriminator,
                                                                                      learning_step=learning_step)
        epoch_losses_discriminator.append(train_loss_discriminator)
        epoch_losses_generator.append(train_loss_generator)

    # track training loss
    train_losses_discriminator.append(tf.reduce_mean(epoch_losses_discriminator))
    train_losses_generator.append(tf.reduce_mean(epoch_losses_generator))
