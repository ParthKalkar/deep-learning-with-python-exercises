import tensorflow as tf
from tensorflow.keras import layers, Input, Model

# Task: Functional API: Multi-Input Model
text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(64, 32)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(32, 16)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
answer = layers.Dense(1, activation='sigmoid')(concatenated)

model = Model([text_input, question_input], answer)

# Task: Functional API: Multi-Output Model
posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, 16)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(1, name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

# Task: Inception Module
branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(input_tensor)
branch_b = layers.Conv2D(128, 1, activation='relu')(input_tensor)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
branch_c = layers.AveragePooling2D(3, strides=2)(input_tensor)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)
branch_d = layers.Conv2D(128, 1, activation='relu')(input_tensor)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)
output = layers.Concatenate()([branch_a, branch_b, branch_c, branch_d])

# Task: Residual Connection
x = layers.Conv2D(128, 3, activation='relu', padding='same')(input_tensor)
x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = layers.Add()([x, input_tensor])

# Task: Layer Sharing
lstm = layers.LSTM(32)
left_input = Input(shape=(None, 128))
left_output = lstm(left_input)
right_input = Input(shape=(None, 128))
right_output = lstm(right_input)
merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)
model = Model([left_input, right_input], predictions)