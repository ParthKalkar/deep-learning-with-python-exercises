import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Task: One-Hot Encoding with Hashing
def hashing_trick(text, n, hash_function=hash):
    vector = [0] * n
    for word in text.split():
        index = abs(hash_function(word)) % n
        vector[index] = 1
    return vector

# Task: Training with Embedding Layer
from tensorflow.keras.datasets import imdb
max_features = 10000
maxlen = 20
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

model = tf.keras.Sequential([
    layers.Embedding(max_features, 8, input_length=maxlen),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Task: Using Pretrained Embeddings (GloVe)
embeddings_index = {}
with open('/path/to/glove.6B.100d.txt', encoding='utf-8') as f:  # Update path
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_dim = 100
embedding_matrix = np.zeros((max_features, embedding_dim))
word_index = imdb.get_word_index()
for word, i in word_index.items():
    if i < max_features:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

model = tf.keras.Sequential([
    layers.Embedding(max_features, embedding_dim, input_length=maxlen),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)