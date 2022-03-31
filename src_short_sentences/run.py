import tensorflow as tf
import tensorflow.keras as K
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
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
train_data = train_data.apply(input_pipeline.prepare_data)
test_data = test_data.apply(input_pipeline.prepare_data)

train_vae = True
train_gan = True


##################################################################
# Training of the AutoEncoder
##################################################################

num_epochs_vae = 3
alpha_vae = 0.001

# Initialize the VAE
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

print(f"Testing loss of the VAE before training: {training_loop.test_step_vae(vae, test_data.take(5), loss_function_vae)}")

if train_vae:
    # We train for num_epochs epochs.
    for epoch in range(num_epochs_vae):

        # training (and checking in with training)
        epoch_losses_vae = []
        for embedding, target, sentiment, noise in train_data.take(1000):
            train_loss_vae = training_loop.train_step_vae(vae=vae,
                                                          inputs=embedding,
                                                          target=target,
                                                          loss_function=loss_function_vae,
                                                          optimizer=optimizer_vae)
            epoch_losses_vae.append(train_loss_vae)

        # track training loss
        train_losses_vae.append(tf.reduce_mean(epoch_losses_vae))
        # track test loss
        test_losses_vae.append(training_loop.test_step_vae(vae, test_data.take(10), loss_function_vae))
        print(f"Epoch {epoch} of the VAE ending with an average training loss of {tf.reduce_mean(epoch_losses_vae)}")
        vae.save_weights('saved_models/weights/vae')
else:
    vae.load_weights('saved_models/weights/vae')

##################################################################
# Training of the GAN
##################################################################

# Hyperparameters
num_epochs_gan = 5
alpha_generator = 0.00005
alpha_discriminator = 0.00005

# Initialize GAN
generator = Generator(state_size=state_size)
discriminator = Discriminator()

# optimizer
optimizer_generator = K.optimizers.RMSprop(alpha_generator)
optimizer_discriminator = K.optimizers.RMSprop(alpha_discriminator)

# loss function
loss_function_sentiment = K.losses.MeanSquaredError()

# initialize lists for later visualization.
train_losses_discriminator = []
test_losses_discriminator = []
train_losses_generator = []
test_losses_generator = []

test_loss_generator, test_loss_discriminator = training_loop.test_step_gan(generator, discriminator, test_data.take(5), vae, loss_function_sentiment)
print(f"Test loss Generator: {test_loss_generator}, test loss discriminator: {test_loss_discriminator}")

if train_gan:
    # We train for num_epochs epochs.
    learning_step = 0
    for epoch in range(num_epochs_gan):

        # training (and checking in with training)
        epoch_losses_discriminator = []
        epoch_losses_generator = []
        for embedding, target, sentiment, noise in train_data.take(1000):
            learning_step += 1
            encoded_sentence = vae.encode(embedding)
            train_loss_discriminator, train_loss_generator = training_loop.train_step_gan(generator, discriminator,
                                                                                          encoded_sentence=encoded_sentence,
                                                                                          gaussian=noise,
                                                                                          sentiment=sentiment,
                                                                                          optimizer_generator=optimizer_generator,
                                                                                          optimizer_discriminator=optimizer_discriminator,
                                                                                          learning_step=learning_step,
                                                                                          loss_function_sentiment=loss_function_sentiment)
            epoch_losses_discriminator.append(train_loss_discriminator)
            epoch_losses_generator.append(train_loss_generator)

        # track training loss
        train_losses_discriminator.append(tf.reduce_mean(epoch_losses_discriminator))
        # track test loss
        train_losses_generator.append(tf.reduce_mean(epoch_losses_generator))
        test_loss_generator, test_loss_discriminator = training_loop.test_step_gan(generator, discriminator,
                                                                                   test_data.take(10), vae,
                                                                                   loss_function_sentiment)
        test_losses_generator.append(test_loss_generator)
        test_losses_discriminator.append(test_loss_discriminator)
        print(f"Epoch {epoch} of the GAN ending with an average Generator training loss of {tf.reduce_mean(epoch_losses_generator)} and an average discriminator training loss of {tf.reduce_mean(epoch_losses_discriminator)}")
        generator.save_weights('saved_models/weights/generator')
        discriminator.save_weights('saved_models/weights/discriminator')
else:
    generator.load_weights('saved_models/weights/generator')
    discriminator.load_weights('saved_models/weights/discriminator')

test_loss_generator, test_loss_discriminator = training_loop.test_step_gan(generator, discriminator, test_data.take(5), vae, loss_function_sentiment)

##################################################################
# Generating Sentences
##################################################################

print('Generated Sentences:')
for embedding, target, sentiment, noise in test_data.take(50):
    sentiment_vector = tf.transpose(tf.multiply(tf.transpose(tf.ones_like(noise)), tf.cast(sentiment, tf.float32)))
    generator_input = tf.concat((noise, sentiment_vector), axis=-1)
    generated_states = generator(generator_input, training=False)
    start_input = tf.constant(input_pipeline.start_token, dtype=tf.float32, shape=(input_pipeline.batch_size, 1, 1))

    generated_text = vae.decoder.generate_sentence(
        starting_input=start_input, states=generated_states, training=False)

    text = input_pipeline.tokenizer.detokenize(generated_text)

    for index, sentence_array in enumerate(text.numpy()):
        sentence = ''
        for i, word in enumerate(sentence_array):
            sentence += word.decode('utf-8')
            sentence += ' '
            if (word.decode('utf-8') == '[END]') or (i == len(sentence_array) - 1):
                break
        if sentiment.numpy()[index] == 0:
            print('Sentiment: Negative; Sentence: ')
        else:
            print('Sentiment: Positive; Sentence: ')
        print(sentence, '\n')

##################################################################
# Plotting of the Losses during Training
##################################################################

if train_vae and train_gan:
    plt.figure()
    line1, = plt.plot(train_losses_vae)
    line2, = plt.plot(test_losses_vae)
    line3, = plt.plot(train_losses_discriminator)
    line4, = plt.plot(test_losses_discriminator)
    line5, = plt.plot(train_losses_generator)
    line6, = plt.plot(test_losses_generator)
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.legend((line1, line2, line3, line4, line5, line6), ("Train losses VAE", "Test Losses VAE", "Train losses Discriminator", "Test losses Discriminator", "Train losses Generator", "Test losses Generator"))
    plt.show()

if train_vae:
    plt.figure()
    line1, = plt.plot(train_losses_vae)
    line2, = plt.plot(test_losses_vae)
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.legend((line1, line2), ("Train losses VAE", "Test Losses VAE"))
    plt.show()

if train_gan:
    plt.figure()
    line1, = plt.plot(train_losses_discriminator)
    line2, = plt.plot(test_losses_discriminator)
    line3, = plt.plot(train_losses_generator)
    line4, = plt.plot(test_losses_generator)
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.legend((line1, line2, line3, line4), ("Train losses Discriminator", "Test losses Discriminator", "Train losses Generator", "Test losses Generator"))
    plt.show()

##################################################################
# Training of the Latent space of the Encoded Sentences
##################################################################

number_points = 50
embeddings = None
types = None
for embedding, target, sentiment, noise in test_data.take(number_points):
    sentiment_vector = tf.transpose(tf.multiply(tf.transpose(tf.ones_like(noise)), tf.cast(sentiment, tf.float32)))
    generator_input = tf.concat((noise, sentiment_vector), axis=-1)
    generation = generator(generator_input, training=False)
    encoded_sentence = vae.encode(embedding)
    sentiment = sentiment.numpy()
    if embeddings is None:
        embeddings = np.concatenate((generation, encoded_sentence), axis = 0)
        types = [sentiment[0], sentiment[1], 2 + sentiment[0], 2 + sentiment[1]]
    else:
        embeddings = np.concatenate((embeddings, generation), axis = 0)
        embeddings = np.concatenate((embeddings, encoded_sentence), axis = 0)
        types = np.concatenate((types, [sentiment[0], sentiment[1], 2 + sentiment[0], 2 + sentiment[1]]), axis=0)
points_embedded = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(embeddings)

label = ["Generated negative", "Generated positive", "Real negative", "Real positive"]
figure, ax = plt.subplots()
scatter = ax.scatter(points_embedded[:,0], points_embedded[:,1], c=types)
handels, labels = scatter.legend_elements()
legend = ax.legend(handels, label, loc="upper right")
ax.add_artist(legend)
plt.legend()
plt.show()
