import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Task: Text Generation Sampling
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Task: Character-level LSTM Training
# Assuming text is loaded
# text = open('nietzsche.txt').read().lower()
# chars = sorted(list(set(text)))
# char_indices = dict((c, i) for i, c in enumerate(chars))
# indices_char = dict((i, c) for i, c in enumerate(chars))

# maxlen = 60
# step = 3
# sentences = []
# next_chars = []
# for i in range(0, len(text) - maxlen, step):
#     sentences.append(text[i: i + maxlen])
#     next_chars.append(text[i + maxlen])

# x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
# y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
# for i, sentence in enumerate(sentences):
#     for t, char in enumerate(sentence):
#         x[i, t, char_indices[char]] = 1
#     y[i, char_indices[next_chars[i]]] = 1

# model = tf.keras.Sequential([
#     layers.LSTM(128, input_shape=(maxlen, len(chars))),
#     layers.Dense(len(chars), activation='softmax')
# ])

# model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.01))

# for epoch in range(1, 60):
#     model.fit(x, y, batch_size=128, epochs=1)
#     start_index = random.randint(0, len(text) - maxlen - 1)
#     generated_text = text[start_index: start_index + maxlen]
#     for temperature in [0.2, 0.5, 1.0]:
#         print('------ temperature:', temperature)
#         sys.stdout.write(generated_text)
#         for i in range(400):
#             sampled = np.zeros((1, maxlen, len(chars)))
#             for t, char in enumerate(generated_text):
#                 sampled[0, t, char_indices[char]] = 1.
#             preds = model.predict(sampled, verbose=0)[0]
#             next_index = sample(preds, temperature)
#             next_char = indices_char[next_index]
#             generated_text += next_char
#             generated_text = generated_text[1:]
#             sys.stdout.write(next_char)