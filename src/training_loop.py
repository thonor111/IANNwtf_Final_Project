import tensorflow as tf
import tensorflow.keras as K

# Training of the Autoencoder




# Training of the GAN

def train_step(generator, discriminator, encoded_sentence, gaussian, sentiment, optimizer_generator, optimizer_discriminator, learning_step):
    '''
    Performs the training step
    Args:
      generator: the generator to be trained
      discriminator: the discriminator to be trained
      encoded_sentence: the encoded embedding of the given sentences
      gaussian: the input of the generator
      sentiment: The sentiment of the sentence to given sentence
      loss_function: the loss_function to be used
      optimizer_generator: the optimizer to be used by the generator
      optimizer_discriminator: the optimizer to be used by the discriminator
    Returns:
      loss: the loss of the current epoch
  '''

    with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
        # generation = generator(tf.concat((gaussian, sentiment), axis = 0), training=True)
        generation = generator(gaussian, training=True)
        prediction_fake, prediction_fake_sentiment = discriminator(generation, training=True)
        prediction_real, prediction_real_sentiment = discriminator(encoded_sentence, training=True)
        loss_generator = tf.math.negative(tf.reduce_mean(prediction_fake))
        loss_discriminator = tf.reduce_mean(prediction_fake - prediction_real)
    # calculating the gradients
    gradients_discriminator = discriminator_tape.gradient(loss_discriminator, discriminator.trainable_variables)
    gradients_generator = generator_tape.gradient(loss_generator, generator.trainable_variables)
    # changing the weights
    optimizer_discriminator.apply_gradients(zip(gradients_discriminator, discriminator.trainable_variables))
    if learning_step % 5 == 0:
        optimizer_generator.apply_gradients(zip(gradients_generator, generator.trainable_variables))
    return (loss_discriminator, loss_generator)