import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Task: Visualizing Intermediate Activations
model = VGG16(weights='imagenet', include_top=False)

img_path = '/path/to/image.jpg'  # Update path
img = image.load_img(img_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()

# Task: Visualizing Convnet Filters
model = VGG16(weights='imagenet', include_top=False)

def compute_loss(input_image, filter_index, feature_map):
    activation = feature_map[:, :, :, filter_index]
    return tf.reduce_mean(activation)

@tf.function
def gradient_ascent_step(img, filter_index, feature_map, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index, feature_map)
    grads = tape.gradient(loss, img)
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

def visualize_filter(filter_index):
    iterations = 30
    learning_rate = 10.
    img = tf.random.uniform((1, 224, 224, 3))
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, model.get_layer('block3_conv1').output, learning_rate)
    return img

# Task: Class Activation Heatmap (Grad-CAM)
model = VGG16(weights='imagenet')

img_path = '/path/to/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.vgg16.preprocess_input(x)

preds = model.predict(x)
predicted_class = np.argmax(preds[0])

last_conv_layer = model.get_layer('block5_conv3')
last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in ['block5_pool', 'flatten', 'fc1', 'fc2', 'predictions']:
    x = model.get_layer(layer_name)(x)
classifier_model = tf.keras.Model(classifier_input, x)

with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(x)
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]

grads = tape.gradient(top_class_channel, last_conv_layer_output)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

last_conv_layer_output = last_conv_layer_output.numpy()[0]
pooled_grads = pooled_grads.numpy()
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]

heatmap = np.mean(last_conv_layer_output, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()