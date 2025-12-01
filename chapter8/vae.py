import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Task: VAE Encoder Network
original_dim = 784
intermediate_dim = 64
latent_dim = 2

inputs = tf.keras.Input(shape=(original_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)

# Task: VAE Sampling Layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# Task: VAE Decoder Network
decoder_h = layers.Dense(intermediate_dim, activation='relu')
decoder_mean = layers.Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# Task: VAE Custom Loss Layer
xent_loss = original_dim * tf.keras.metrics.binary_crossentropy(inputs, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae = tf.keras.Model(inputs, x_decoded_mean)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')