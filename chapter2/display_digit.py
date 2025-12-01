import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Task: Displaying a Digit (4th digit, index 3)
digit = train_images[3]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()