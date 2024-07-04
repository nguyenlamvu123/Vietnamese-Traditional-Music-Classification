from build import generate_data, train_model, load_best_model
from config import TRAIN_ROOT, VAL_ROOT, TEST_ROOT


if __name__ == "__main__":
    train_generator, val_generator, test_generator = generate_data(TRAIN_ROOT, VAL_ROOT, TEST_ROOT)
    # # Can change index from 1 to 3, or 0
    # best_model = train_model(0, train_set=train_generator, val_set=val_generator, test_set=test_generator)
    for model_index in range(3):
        best_model = load_best_model(model_index=model_index + 1, val_set=val_generator)
