import tensorflow as tf


##################################################################
# Training of the AutoEncoder
##################################################################

# @tf.function
def train_step_vae(vae, inputs, target, loss_function, optimizer):
    """
    Performs the training step of the VAE
    Args:
      vae: the VAE to be trained
      inputs: The input of the encoder
      target: The target of the decoder
      loss_function: the loss_function to be used
      optimizer: the optimizer to be used for learning
    Returns:
      loss: the loss of the current epoch
    """

    with tf.GradientTape() as tape:
        # calculating the prediction of the vae
        prediction = vae(inputs, training=True)
        # calculating the loss of the prediction to the target
        loss = loss_function(target, prediction)
    # calculating the gradient
    gradients = tape.gradient(loss, vae.trainable_variables)
    # changing the weights
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))

    return loss


##################################################################
# Training of the GAN
##################################################################

def train_step_gan(generator, discriminator, encoded_sentence, gaussian, sentiment, optimizer_generator,
                   optimizer_discriminator, learning_step, loss_function_sentiment):
    '''
    Performs the training step of the GAN
    Args:
      generator: the generator to be trained
      discriminator: the discriminator to be trained
      encoded_sentence: the encoded embedding of the given sentences
      gaussian: the input of the generator
      sentiment: The sentiment of the sentence to given sentence
      optimizer_generator: the optimizer to be used by the generator
      optimizer_discriminator: the optimizer to be used by the discriminator
      learning_step: A counter used to achieve a different number of learning steps for the generator and discriminator
      loss_function_sentiment: The loss_function to use for the loss of the sentiment prediction
    Returns:
      loss: the loss of the current epoch
  '''

    with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
        sentiment_vector = tf.transpose(tf.multiply(tf.transpose(tf.ones_like(gaussian)), tf.cast(sentiment, tf.float32)))
        generator_input = tf.concat((gaussian, sentiment_vector), axis=-1)
        generation = generator(generator_input, training=True)
        # generation = generator(gaussian, training=True)
        predictions_fake = discriminator(generation, training=True)
        prediction_fake, prediction_fake_sentiment = tf.transpose(predictions_fake)
        predictions_real = discriminator(encoded_sentence, training=True)
        prediction_real, prediction_real_sentiment = tf.transpose(predictions_real)
        loss_generator = tf.math.negative(tf.reduce_mean(prediction_fake))
        loss_discriminator = tf.reduce_mean(prediction_fake - prediction_real)
        # Adding the Losses for the Sentiment
        loss_generator = tf.add(loss_generator, loss_function_sentiment(prediction_fake_sentiment, tf.cast(sentiment, tf.float32)))
        loss_discriminator = tf.add(loss_discriminator, loss_function_sentiment(prediction_real_sentiment, tf.cast(sentiment, tf.float32)))
    # calculating the gradients
    gradients_discriminator = discriminator_tape.gradient(loss_discriminator, discriminator.trainable_variables)
    gradients_generator = generator_tape.gradient(loss_generator, generator.trainable_variables)
    # changing the weights
    optimizer_discriminator.apply_gradients(zip(gradients_discriminator, discriminator.trainable_variables))
    if learning_step % 5 == 0:
        optimizer_generator.apply_gradients(zip(gradients_generator, generator.trainable_variables))
    return (loss_discriminator, loss_generator)


##################################################################
# Testing of the AutoEncoder
##################################################################

def test_step_vae(vae, test_data, loss_function):
    """
    Performs the training step of the VAE
    Args:
      vae: the VAE to be trained
      test_data: THe data to test the vae on
      loss_function: the loss_function to be used
    Returns:
      loss: the loss of the current epoch
    """
    losses = []
    for inputs, target, sentiment, noise in test_data:
        # calculating the prediction of the vae
        prediction = vae(inputs, training=False)
        # calculating the loss of the prediction to the target
        loss = loss_function(target, prediction)
        # calculating the gradient
        losses.append(loss)
    loss_average = tf.reduce_mean(losses)

    return loss_average


##################################################################
# Testing of the GAN
##################################################################

def test_step_gan(generator, discriminator, test_data, vae, loss_function_sentiment):
    '''
    Performs the training step of the GAN
    Args:
      generator: the generator to be trained
      discriminator: the discriminator to be trained
      test_data: the data to test the gan on
      vae: The vae to use for the encoding of the input
      loss_function_sentiment: The loss_function to use for the loss of the sentiment prediction
    Returns:
      loss: the loss of the current epoch
  '''
    losses_generator = []
    losses_discriminator = []
    for inputs, target, sentiment, noise in test_data:
        sentiment_vector = tf.transpose(tf.multiply(tf.transpose(tf.ones_like(noise)), tf.cast(sentiment, tf.float32)))
        generator_input = tf.concat((noise, sentiment_vector), axis=-1)
        generation = generator(generator_input, training=False)
        predictions_fake = discriminator(generation, training=False)
        prediction_fake, prediction_fake_sentiment = tf.transpose(predictions_fake)
        discriminator_input = vae.encode(inputs)
        predictions_real = discriminator(discriminator_input, training=False)
        prediction_real, prediction_real_sentiment = tf.transpose(predictions_real)
        loss_generator = tf.math.negative(tf.reduce_mean(prediction_fake))
        loss_discriminator = tf.reduce_mean(prediction_fake - prediction_real)
        # Adding the Losses for the Sentiment
        loss_generator = tf.add(loss_generator, loss_function_sentiment(prediction_fake_sentiment, tf.cast(sentiment, tf.float32)))
        loss_discriminator = tf.add(loss_discriminator, loss_function_sentiment(prediction_real_sentiment, tf.cast(sentiment, tf.float32)))
        # calculating the gradients
        losses_generator.append(loss_generator)
        losses_discriminator.append(loss_discriminator)
    loss_generator_average = tf.reduce_mean(losses_generator)
    loss_discriminator_average = tf.reduce_mean(losses_discriminator)

    return loss_generator_average, loss_discriminator_average
