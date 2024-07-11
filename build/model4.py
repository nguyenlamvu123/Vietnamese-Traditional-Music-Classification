import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import os, sys
sys.path.append('\VN-music-classification')  # Add parent directory to import varibles from config.py
from config import *



def get_model4(input_shape = INPUT_SHAPE, n_class = N_CLASS):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(INPUT_SHAPE,)))
    # model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    ckptdir = os.path.join(CHECKPOINT_FILEPATH, 'model3')
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)

    # Define checkpoint
    checkpoint= tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(ckptdir, 'model3_{epoch:02d}_{val_accuracy:.4f}.keras'),  # Keras 3 only supports .keras models, versions < 3 still support .h5  # https://stackoverflow.com/questions/78692707/valueerror-the-filepath-provided-must-end-in-keras-keras-model-format-rec
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
