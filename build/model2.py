import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import os, sys
sys.path.append('\VN-music-classification')  # Add parent directory to import varibles from config.py
from config import *



def get_model2(input_shape = INPUT_SHAPE, n_class = N_CLASS):
    model = tf.keras.Sequential(layers=[
            tf.keras.layers.InputLayer(input_shape= (input_shape[0], input_shape[1], 3)),
            # first convolution
            tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            # second convolution
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            # third convolution
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            # FC 
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(n_class, activation="softmax")
        ])

    ckptdir = os.path.join(CHECKPOINT_FILEPATH, 'model2')
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)

    # Define checkpoint
    checkpoint= tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(ckptdir, 'model2_{epoch:02d}_{val_accuracy:.4f}.keras'),  # Keras 3 only supports .keras models, versions < 3 still support .h5  # https://stackoverflow.com/questions/78692707/valueerror-the-filepath-provided-must-end-in-keras-keras-model-format-rec
    monitor = CHECKPOINT_MONITOR,
    save_best_only=True,
    # save_weights_only=True,
    verbose=1
    )

    # Define callbacks
    early = EarlyStopping(monitor = EARLY_MONITOR,
        patience = PATIENCE,
        verbose = VERBOSE,
        mode='auto',
        baseline= None,
        restore_best_weights= True)
    return model, checkpoint, early
