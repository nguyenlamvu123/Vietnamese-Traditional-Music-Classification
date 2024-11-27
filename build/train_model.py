import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt

from config import INPUT_SHAPE, N_CLASS, LOSS, OPTIMIZER, METRICS, BATCH_SIZE, EPOCHS, VALIDATION_BATCH_SIZE, SAVED_MODEL_PATH
from .model1 import get_model1
from .model2 import get_model2
from .model3 import get_model3


def fit_model(model, checkpoint, early, train_set, val_set, model_index):
    model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics= METRICS)
    model_history = model.fit(
        train_set,
        batch_size= BATCH_SIZE,
        epochs = EPOCHS,
        callbacks=[
            # early,
            checkpoint
        ],
        validation_data = val_set,
        validation_batch_size = VALIDATION_BATCH_SIZE
    )
    model.save(f'{SAVED_MODEL_PATH}{os.sep}best_model_{model_index}.keras')
    return model_history


def Validation_plot(model_index, history):
    print("Validation Accuracy", max(history.history["val_accuracy"]))
    hist_df = pd.DataFrame(history.history)
    hist_df.plot(figsize=(12, 6))
    with open(os.path.join(SAVED_MODEL_PATH, f'traininghistory_{model_index}.json'), mode='w') as f:
        hist_df.to_json(f)
    plt.savefig(os.path.join(SAVED_MODEL_PATH, f'Validation_plot_{model_index}.jpg'))


def load_best_model(model_index, val_set):
    """
    Load best model after training
    model_index = {1, 2, 3} - Load best model with highest val_acc of model1, 2, 3
    """

    if model_index == 1:
        # best_model, _ , _ = get_model1(input_shape = INPUT_SHAPE, n_class = N_CLASS)
        # checkpoint_model_path = CHECKPOINT_FILEPATH + "/model1"
        # best_model_path = checkpoint_model_path + "/" + str(os.listdir(checkpoint_model_path)[-1])
        metrics = [METRICS[0], tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    elif model_index == 2:
        # best_model, _ , _ = get_model2(input_shape = INPUT_SHAPE, n_class = N_CLASS)
        # checkpoint_model_path = CHECKPOINT_FILEPATH + "/model2"
        # best_model_path = checkpoint_model_path + "/" + str(os.listdir(checkpoint_model_path)[-1])
        metrics = [METRICS[0], tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    elif model_index == 3:
        # best_model, _ , _ = get_model3(input_shape = INPUT_SHAPE, n_class = N_CLASS)
        # checkpoint_model_path = CHECKPOINT_FILEPATH + "/model3"
        # best_model_path = checkpoint_model_path + "/" + str(os.listdir(checkpoint_model_path)[-1])
        metrics = [METRICS[0]]
    elif model_index == 0:
        print("all 3 models are trained!")
        return
    else:
        raise Exception("Sorry, there are just 3 models")

    # Re-evaluate val_set to check whether this is the best model
    # best_model.load_weights(best_model_path)
    best_model = tf.keras.models.load_model(f'{SAVED_MODEL_PATH}{os.sep}best_model_{model_index}.keras')

    best_model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics=metrics)
    print("The best model's metrics: ")
    best_model.evaluate(val_set)
    return best_model

def train_model(model_index, train_set, val_set, test_set):
    """
    Train model and save the best model 
    Input:
    - model_index : 1 or 2 or 3
    Output:
    - While training, checkpoints are saved at CHECKPOINT_FILEPATH
    - Save the best model at SAVE_MODEL_PATH
    """
    model_history = None
    if model_index in (0, 1, ):
        model1, checkpoint1, early1 = get_model1(INPUT_SHAPE, N_CLASS)
        model_history = fit_model(model1, checkpoint1, early1, train_set=train_set, val_set=val_set, model_index=1)
        Validation_plot(1, model_history)
        model1.save(f'{SAVED_MODEL_PATH}{os.sep}best_model_1.keras')
    if model_index in (0, 2, ):
        model2, checkpoint2, early2 = get_model2(INPUT_SHAPE, N_CLASS)
        model_history = fit_model(model2, checkpoint2, early2, train_set=train_set, val_set=val_set, model_index=2)
        Validation_plot(2, model_history)
        model2.save(f'{SAVED_MODEL_PATH}{os.sep}best_model_2.keras')
    if model_index in (0, 3, ):
        model3, checkpoint3, early3 = get_model3(INPUT_SHAPE, N_CLASS)
        model_history = fit_model(model3, checkpoint3, early3, train_set=train_set, val_set=val_set, model_index=3)
        Validation_plot(3, model_history)
        model3.save(f'{SAVED_MODEL_PATH}{os.sep}best_model_3.keras')
    if model_history is None:
        raise Exception("Sorry, there are just 3 models")
    return load_best_model(model_index = model_index, val_set=val_set)
