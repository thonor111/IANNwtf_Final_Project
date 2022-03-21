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

input_pipeline.train_tokenizer(train_data.take(100))

# preprocess datasplits
# using the GAN version to be able to test the GAN
train_data = train_data.apply(input_pipeline.prepare_data_GAN)
test_data = test_data.apply(input_pipeline.prepare_data_GAN)

##################################################################
# Training of the Autoencoder
##################################################################

num_epochs_vae = 1
alpha_vae = 0.001

# Initialize Model
vocab_size = input_pipeline.vocab_size
embedding_size = 252
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
    for embedding, target, sentiment, noise in train_data.take(5):
        train_loss_vae = training_loop.train_step_vae(vae=vae,
                                                      input=embedding,
                                                      target=target,
                                                      loss_function=loss_function_vae,
                                                      optimizer=optimizer_vae)
        epoch_losses_vae.append(train_loss_vae)

    # track training loss
    train_losses_vae.append(tf.reduce_mean(epoch_losses_vae))
    print(f"Epoch {epoch} of the VAE ending with an average loss of {tf.reduce_mean(epoch_losses_vae)}")

##################################################################
# Training of the GAN
##################################################################

# Hyperparameters
num_epochs_gan = 1
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
    for embedding, target, sentiment, noise in train_data.take(5):
        learning_step += 1
        encoded_sentence = vae.encode(embedding)
        train_loss_discriminator, train_loss_generator = training_loop.train_step_gan(generator, discriminator,
                                                                                      encoded_sentence=encoded_sentence,
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
    print(
        f"Epoch {epoch} of the GAN ending with an average Generator loss of {tf.reduce_mean(epoch_losses_generator)} and an average discriminator loss of {tf.reduce_mean(epoch_losses_generator)}")


for embedding, target, sentiment, noise in test_data.take(2):
    generated_states = generator(noise, training=False)
    print(f"Shape of the generated states: {generated_states.numpy().shape}")
    generated_text = vae.decoder.generate_sentence(
        starting_input=tf.constant(input_pipeline.start_token, dtype=tf.int64,
                                   shape=(3,1)), states=generated_states, training=False)
    print(generated_text)

    text = input_pipeline.tokenizer_layer(generated_text)

    print(text)

    vocabulary = input_pipeline.tokenizer_layer.get_vocabulary(include_special_tokens=False)

    generated_text_string = []
    for index in generated_text:
        generated_text.append(vocabulary[index.numpy()])
    print(generated_text_string)
