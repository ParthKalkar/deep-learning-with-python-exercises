# Deep Learning with Python Exercises - Tasks and Results

This file contains the task descriptions and outputs from running the exercise scripts.

## Tasks Overview

### Chapter 2: Getting Started with Neural Networks

#### Section 2.1: First Look at a Neural Network
- **Loading the MNIST Dataset**: Load the MNIST dataset (train_images, train_labels, test_images, test_labels) using the keras.datasets module.
- **Building the Network Architecture**: Initialize a Sequential model and add two Dense layers. The first layer should have 512 units with 'relu' activation, and the second should be a 10-way softmax layer.
- **Compiling the Network**: Compile the model using the 'rmsprop' optimizer, 'categorical_crossentropy' loss function, and 'accuracy' metrics.
- **Preparing Image Data**: Reshape the train and test images into float32 vectors of shape (60000, 28 * 28) and (10000, 28 * 28) respectively, and scale the pixel values to be between 0 and 1.
- **Preparing Labels**: Use 'to_categorical' from keras.utils to categorically encode the training and test labels.
- **Training the Network**: Fit the model to the training data using 5 epochs and a batch size of 128.

#### Section 2.2: Data Representations for Neural Networks
- **Displaying a Digit**: Write a script using Matplotlib to display the 4th digit from the training images.

#### Section 2.3: The Gears of Neural Networks
- **Naive Vector Operations**: Implement a naive Python function 'naive_relu' (element-wise relu) and 'naive_add' (element-wise addition) using for-loops, without using Numpy's built-in vectorization.

### Chapter 3: Introduction to Keras and TensorFlow

#### Section 3.4: Classifying Movie Reviews
- **Loading IMDB Dataset**: Load the IMDB dataset keeping only the top 10,000 most frequently occurring words.
- **Vectorizing Sequences**: Create a function 'vectorize_sequences' to one-hot encode the integer sequence lists into binary matrices of shape (samples, 10000).
- **Building the IMDB Network**: Build a Sequential model with two hidden Dense layers (16 units, relu activation) and a final Dense layer (1 unit, sigmoid activation).
- **Validating IMDB Approach**: Set aside the first 10,000 samples for validation. Train the model for 20 epochs with a batch size of 512, passing the validation data.
- **Plotting Training History**: Use Matplotlib to plot the training and validation loss side by side, and the training and validation accuracy side by side.

#### Section 3.5: Classifying Newswires
- **Loading Reuters Dataset**: Load the Reuters newswire dataset restricting it to the top 10,000 most frequent words.
- **Encoding Labels (One-Hot)**: Write a function 'to_one_hot' (or use built-in Keras utilities) to encode the labels into vectors of dimension 46.
- **Building Reuters Network**: Create a model with two hidden layers of 64 units (relu) and a final softmax layer with 46 units. Compile using categorical_crossentropy.

#### Section 3.6: Predicting House Prices
- **Loading Boston Housing Data**: Load the Boston Housing Price dataset.
- **Normalizing Data**: Perform feature-wise normalization: subtract the mean of the feature and divide by the standard deviation. (Compute mean/std on training data only).
- **K-Fold Validation**: Implement a K-fold cross-validation loop (k=4). Instantiate, train, and evaluate the model for each fold, then calculate the average score.

### Chapter 5: Deep Learning for Computer Vision

#### Section 5.1: Introduction to ConvNets
- **Instantiating a Small Convnet**: Build a Sequential model for MNIST using Conv2D (32 filters, 3x3) and MaxPooling2D (2x2) layers, followed by a flattening step and Dense layers.

#### Section 5.2: Training a ConvNet from Scratch on a Small Dataset
- **Copying Images to Directories**: Write a script to create a directory structure for the Dogs vs. Cats dataset (train, validation, test) and copy specific ranges of images (e.g., first 1000 cats for training) into them.
- **Building Dogs vs. Cats Model**: Build a deeper convnet with 4 convolution blocks (Conv2D + MaxPooling2D), flattening, and Dense layers ending in a single sigmoid unit.
- **Using ImageDataGenerator**: Configure ImageDataGenerator to rescale images by 1./255. Create generators using `flow_from_directory` for both train and validation directories.
- **Data Augmentation Configuration**: Configure an ImageDataGenerator with rotation, width shift, height shift, shear, zoom, and horizontal flip.
- **Convnet with Dropout**: Define a new convnet that includes a Dropout layer (rate 0.5) before the fully connected classifier.

#### Section 5.3: Using a Pretrained ConvNet
- **Feature Extraction (No Augmentation)**: Instantiate the VGG16 base. Run the training and validation data through the VGG16 base to record the output (features). Train a standalone Dense classifier on these recorded features.
- **Feature Extraction (With Augmentation)**: Add the VGG16 base (frozen) to a Sequential model, add a Dense classifier on top, and train end-to-end using data augmentation generators.
- **Fine-Tuning**: Unfreeze the top layers of the VGG16 base (e.g., block5_conv1 onwards) and train the model with a very low learning rate.

#### Section 5.4: Visualizing What ConvNets Learn
- **Visualizing Intermediate Activations**: Create a model that outputs layer activations. Feed an image, retrieve activations, and use Matplotlib to display specific channels of the feature maps.
- **Visualizing Convnet Filters**: Define a loss function that maximizes the activation of a specific filter. Use stochastic gradient descent to adjust an input image (starting from noise) to maximize this activation.
- **Class Activation Heatmap (Grad-CAM)**: Implement Grad-CAM: Compute the gradient of the top predicted class with respect to the last conv layer, weight the feature map channels by these gradients, and generate a heatmap.

### Chapter 6: Deep Learning for Text and Sequences

#### Section 6.1: Working with Text Data
- **One-Hot Encoding with Hashing**: Implement a one-hot encoding scheme using the hashing trick (without building an explicit index).
- **Training with Embedding Layer**: Prepare IMDB data (pad sequences). Build a model with an Embedding layer, Flatten, and Dense classifier. Train on the data.
- **Using Pretrained Embeddings (GloVe)**: Download and parse GloVe embeddings. Create an embedding matrix. Load this matrix into an Embedding layer, freeze it, and train the model.

#### Section 6.2: Understanding Recurrent Neural Networks
- **Numpy SimpleRNN**: Implement the forward pass of a simple RNN using Numpy (a loop over timesteps where state_t = activation(dot(W, input_t) + dot(U, state_t) + b)).
- **Training LSTM on IMDB**: Build a model using an Embedding layer and an LSTM layer. Train on the IMDB dataset.

#### Section 6.3: Advanced Usage of Recurrent Neural Networks
- **Jena Weather Data Generator**: Create a generator function that yields batches of timeseries data (samples) and targets (temperature 24 hours later) based on `lookback`, `delay`, and `step` parameters.
- **Recurrent Baseline (GRU)**: Train a model using a GRU layer on the temperature forecasting problem.
- **Stacked GRU with Dropout**: Train a model with stacked GRU layers, utilizing `dropout` and `recurrent_dropout` arguments.
- **Bidirectional LSTM**: Train a Bidirectional LSTM on the IMDB dataset (or the reversed-sequence temperature problem).

#### Section 6.4: Sequence Processing with ConvNets
- **1D Convnet for IMDB**: Build and train a model using Conv1D and MaxPooling1D layers for sentiment analysis.
- **CNN + RNN Combination**: Build a model that uses 1D Conv layers to downsample the input sequence, followed by a GRU layer, for the temperature forecasting problem.

### Chapter 7: Advanced Deep-Learning Best Practices

#### Section 7.1: Going Beyond the Sequential Model: The Keras Functional API
- **Functional API: Multi-Input Model**: Implement a question-answering model with two input branches (text and question) that merge via concatenation.
- **Functional API: Multi-Output Model**: Implement a model that predicts three different attributes (age, income, gender) from a single input source.
- **Inception Module**: Implement a basic Inception module using the functional API (parallel branches with 1x1, 3x3 convolutions and pooling, concatenated at the end).
- **Residual Connection**: Implement a residual connection where the input tensor is added to the output tensor of a convolution block.
- **Layer Sharing**: Implement a Siamese LSTM model where the same LSTM layer instance is applied to two different inputs.

#### Section 7.2: Inspecting and Monitoring Deep-Learning Models Using Keras Callbacks and TensorBoard
- **Using Callbacks**: Train a model using `EarlyStopping` and `ModelCheckpoint` callbacks.
- **Custom Callback**: Write a custom callback class that saves the activations of a specific layer to disk at the end of every epoch.
- **TensorBoard Logging**: Train a model using the TensorBoard callback, specifying a log directory, and launch the TensorBoard server.

### Chapter 8: Generative Deep Learning

#### Section 8.1: Text Generation with LSTM
- **Text Generation Sampling**: Implement a 'sample' function that reweights a probability distribution based on a 'temperature' parameter and draws a sample.
- **Character-level LSTM Training**: Train an LSTM on a text corpus (e.g., Nietzsche) to predict the next character. Use a loop to generate text at different temperatures after each epoch.

#### Section 8.2: DeepDream
- **DeepDream Loss**: Define a loss function that is the weighted sum of the L2 norm of activations of specific layers in Inception V3.
- **DeepDream Gradient Ascent**: Implement a gradient ascent loop that modifies the input image to maximize the defined loss, over multiple scales (octaves).

#### Section 8.3: Neural Style Transfer
- **Style Transfer Loss**: Define the content loss (L2 norm between feature maps) and style loss (Gram matrix correlations) using VGG19.
- **Style Transfer Optimization**: Use SciPy's L-BFGS optimizer to minimize the total loss (content + style + variational) by iteratively adjusting the generated image.

#### Section 8.4: Generating Images with Variational Autoencoders
- **VAE Encoder Network**: Build an encoder that maps inputs to 'z_mean' and 'z_log_var'.
- **VAE Sampling Layer**: Implement a custom Lambda layer (or function) that samples 'z' from the latent space using the reparameterization trick.
- **VAE Decoder Network**: Build a decoder that maps the latent vector 'z' back to an image.
- **VAE Custom Loss Layer**: Implement a custom layer that calculates the VAE loss (reconstruction loss + KL divergence) and adds it using `self.add_loss`.

#### Section 8.5: Introduction to Generative Adversarial Networks
- **GAN Generator**: Build a generator network using Dense, Reshape, and Conv2DTranspose layers.
- **GAN Discriminator**: Build a discriminator network using Conv2D and LeakyReLU layers.
- **GAN Training Loop**: Implement the training loop: 1. Train discriminator on real vs fake images. 2. Train the generator (via the combined GAN model) to fool the discriminator.

## Results

This section contains the outputs from running the exercise scripts.

## Chapter 2

### Naive Vector Operations

```
naive_relu(x): [[1 0]
 [3 4]]
naive_add(x, y): [[ 6  4]
 [10 12]]
```

### MNIST Training

```
Epoch 1/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9247 - loss: 0.2619    
Epoch 2/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9686 - loss: 0.1059 
Epoch 3/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9794 - loss: 0.0706 
Epoch 4/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9846 - loss: 0.0522 
Epoch 5/5
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9880 - loss: 0.0392
```

## Chapter 3

### IMDB Sentiment Analysis

```
Epoch 1/2
30/30 ━━━━━━━━━━━━━━━━━━━━ 1s 25ms/step - acc: 0.7741 - loss: 0.5334 - val_acc: 0.8413 - val_loss: 0.4269
Epoch 2/2
30/30 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - acc: 0.8937 - loss: 0.3382 - val_acc: 0.8698 - val_loss: 0.3401
```

### Boston Housing K-Fold Validation

```
Processing fold #0
Processing fold #1
Processing fold #2
Processing fold #3
All scores: [2.160963535308838, 2.6503875255584717, 2.61151123046875, 2.566434860229492]
Mean MAE: 2.497324287891388
```

### Reuters Newswire Classification

```
Epoch 1/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 9ms/step - accuracy: 0.5046 - loss: 2.6237  
Epoch 2/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6806 - loss: 1.5271 
Epoch 3/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.7474 - loss: 1.1692 
Epoch 4/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.7914 - loss: 0.9520 
Epoch 5/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.8264 - loss: 0.7898 
Epoch 6/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.8560 - loss: 0.6585 
Epoch 7/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.8825 - loss: 0.5522 
Epoch 8/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9040 - loss: 0.4636 
Epoch 9/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9189 - loss: 0.3937 
Epoch 10/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9293 - loss: 0.3346 
Epoch 11/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9330 - loss: 0.2945 
Epoch 12/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9410 - loss: 0.2528 
Epoch 13/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9443 - loss: 0.2320 
Epoch 14/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9485 - loss: 0.2014 
Epoch 15/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9507 - loss: 0.1891 
Epoch 16/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9526 - loss: 0.1719 
Epoch 17/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9535 - loss: 0.1602 
Epoch 18/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9549 - loss: 0.1503 
Epoch 19/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9547 - loss: 0.1481 
Epoch 20/20
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9534 - loss: 0.1397
```

## Chapter 5

### Instantiating a Small Convnet

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 26, 26, 32)          │             320 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 13, 13, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 11, 11, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 5, 5, 64)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 3, 3, 64)            │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 576)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 64)                  │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 10)                  │             650 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 93,322 (364.54 KB)
 Trainable params: 93,322 (364.54 KB)
 Non-trainable params: 0 (0.00 B)
```

## Chapter 6

### Numpy SimpleRNN

(Output is the final output sequence shape and values - implementation runs without print)

### Training LSTM on IMDB

```
Epoch 1/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 21s 129ms/step - acc: 0.6719 - loss: 0.5876 - val_acc: 0.8130 - val_loss: 0.4386
Epoch 2/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 20s 130ms/step - acc: 0.8358 - loss: 0.3857 - val_acc: 0.8322 - val_loss: 0.3989
Epoch 3/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 22s 139ms/step - acc: 0.8627 - loss: 0.3321 - val_acc: 0.8418 - val_loss: 0.3668
Epoch 4/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 21s 133ms/step - acc: 0.8812 - loss: 0.2949 - val_acc: 0.8718 - val_loss: 0.3129
Epoch 5/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 21s 132ms/step - acc: 0.8929 - loss: 0.2673 - val_acc: 0.8030 - val_loss: 0.4740
Epoch 6/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 21s 131ms/step - acc: 0.9017 - loss: 0.2503 - val_acc: 0.8416 - val_loss: 0.3651
Epoch 7/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 21s 136ms/step - acc: 0.9093 - loss: 0.2380 - val_acc: 0.8556 - val_loss: 0.3985
Epoch 8/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 22s 137ms/step - acc: 0.9142 - loss: 0.2232 - val_acc: 0.8402 - val_loss: 0.3709
Epoch 9/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 22s 142ms/step - acc: 0.9238 - loss: 0.2001 - val_acc: 0.8858 - val_loss: 0.3040
Epoch 10/10
157/157 ━━━━━━━━━━━━━━━━━━━━ 22s 141ms/step - acc: 0.9334 - loss: 0.1865 - val_acc: 0.8288 - val_loss: 0.4278
```

## Chapter 8

### Text Generation Sampling

```
Sample at temp 1.0: 1
Sample at temp 0.5: 1
```