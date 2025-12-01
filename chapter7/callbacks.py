import tensorflow as tf
from tensorflow.keras import layers
import os

# Task: Using Callbacks
model = tf.keras.Sequential([layers.Dense(1)])
model.compile(optimizer='adam', loss='mse')

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5', monitor='val_loss', save_best_only=True)
]

# model.fit(x, y, epochs=10, callbacks=callbacks, validation_data=(x_val, y_val))

# Task: Custom Callback
class ActivationLogger(tf.keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activation_model = tf.keras.models.Model(model.input, layer_outputs)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')
        validation_sample = self.validation_data[0][0:1]
        activations = self.activation_model.predict(validation_sample)
        # Save activations to disk

# Task: TensorBoard Logging
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir='/my_log_dir',
        histogram_freq=1,
        embeddings_freq=1,
    )
]

# model.fit(x, y, epochs=10, callbacks=callbacks)