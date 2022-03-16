import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import tensorflow_datasets as tfds
import input_pipeline
import training_loop
from generator import Generator
from discriminator import  Discriminator
tf.keras.backend.clear_session()

# load dataset and split into to train (25,000) examples and test (2,500 examples) sets
train_data, test_data = tfds.load('imdb_reviews', split=['train', 'test[:10%]'], as_supervised=True)

# preprocess datasplits
# using the GAn version to be able to test the GAN
train_data = train_data.apply(input_pipeline.prepare_data_GAN)
test_data = test_data.apply(input_pipeline.prepare_data_GAN)

##################################################################
# Training of the Autoencoder
##################################################################


##################################################################
# Training of the GAN
##################################################################

# Hyperparameters
num_epochs = 10
alpha_generator = 0.00005
alpha_discriminator = 0.00005

# Initialize Model
generator = Generator()
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
for epoch in range(num_epochs):


    # training (and checking in with training)
    epoch_losses_discriminator = []
    epoch_losses_generator = []
    for embedding, sentiment, noise in train_data:
        learning_step += 1
        train_loss_discriminator, train_loss_generator = training_loop.train_step(generator, discriminator,
                                                                                            encoded_sentence=embedding,
                                                                                            gaussian=noise, sentiment=sentiment,
                                                                                            optimizer_generator=optimizer_generator,
                                                                                            optimizer_discriminator=optimizer_discriminator,
                                                                                            learning_step=learning_step)
        epoch_losses_discriminator.append(train_loss_discriminator)
        epoch_losses_generator.append(train_loss_generator)

    # track training loss
    train_losses_discriminator.append(tf.reduce_mean(epoch_losses_discriminator))
    train_losses_generator.append(tf.reduce_mean(epoch_losses_generator))


