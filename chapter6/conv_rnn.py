import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Task: 1D Convnet for IMDB
from tensorflow.keras.datasets import imdb
max_features = 10000
max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = tf.keras.Sequential([
    layers.Embedding(max_features, 128, input_length=max_len),
    layers.Conv1D(32, 7, activation='relu'),
    layers.MaxPooling1D(5),
    layers.Conv1D(32, 7, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

# Task: CNN + RNN Combination
model = tf.keras.Sequential([
    layers.Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])),
    layers.MaxPooling1D(3),
    layers.Conv1D(32, 5, activation='relu'),
    layers.GRU(32, dropout=0.1, recurrent_dropout=0.5),
    layers.Dense(1)
])