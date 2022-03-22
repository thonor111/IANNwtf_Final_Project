import tensorflow as tf
import tensorflow.keras as K
import tensorflow_datasets as tfds
from input_pipeline import InputPipeline
import training_loop
from generator import Generator
from discriminator import Discriminator
from vae import VariationalAutoEncoder

tf.keras.backend.clear_session()

# load dataset and split into to train (25,000) examples and test (2,500 examples) sets
train_data, test_data = tfds.load('imdb_reviews', split=['train', 'test[:10%]'], as_supervised=True)

input_pipeline = InputPipeline()

input_pipeline.train_tokenizer(train_data)

# preprocess datasplits
# using the GAN version to be able to test the GAN
train_data = train_data.apply(input_pipeline.prepare_data)
test_data = test_data.apply(input_pipeline.prepare_data)

##################################################################
# Training of the AutoEncoder
##################################################################

num_epochs_vae = 5
alpha_vae = 0.001

# Initialize Model
vocab_size = input_pipeline.vocab_size
state_size = 600
vae = VariationalAutoEncoder(vocab_size=vocab_size, state_size=state_size)

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
    for embedding, target, sentiment, noise in train_data.take(50):
        train_loss_vae = training_loop.train_step_vae(vae=vae,
                                                      inputs=embedding,
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
num_epochs_gan = 5
alpha_generator = 0.00005
alpha_discriminator = 0.00005

# Initialize Model
generator = Generator(state_size=state_size)
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
    for embedding, target, sentiment, noise in train_data.take(100):
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
    print(f"Epoch {epoch} of the GAN ending with an average Generator loss of {tf.reduce_mean(epoch_losses_generator)} and an average discriminator loss of {tf.reduce_mean(epoch_losses_discriminator)}")

print('Generated Sentences:')
for embedding, target, sentiment, noise in test_data.take(2):
    generated_states = generator(noise, training=False)
    start_input = tf.constant(input_pipeline.start_token, dtype=tf.float32, shape=(input_pipeline.batch_size, 1, 1))

    generated_text = vae.decoder.generate_sentence(
        starting_input=start_input, states=generated_states, training=False)

    text = input_pipeline.tokenizer.detokenize(generated_text)

    for sentence_array in text.numpy():
        sentence = ''
        for index, word in enumerate(sentence_array):
            sentence += word.decode('utf-8')
            sentence += ' '
            if (word.decode('utf-8') == '[END]') or (index == len(sentence_array) - 1):
                break
        print(sentence, '\n')
