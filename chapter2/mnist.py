import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Task: Loading the MNIST Dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Task: Building the Network Architecture
model = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Task: Compiling the Network
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Task: Preparing Image Data
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# Task: Preparing Labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Task: Training the Network
model.fit(train_images, train_labels, epochs=5, batch_size=128)