import os

# Define directories
path = os.path.dirname(__file__)
RAW_ROOT = os.path.join(path, 'rawdata')
FOLDER_ROOT = os.path.join(path, "mel-images")
DATASET_ROOT = os.path.join(path, "dataset")
TRAIN_ROOT = os.path.join(path, "dataset", "train")
VAL_ROOT = os.path.join(path, "dataset", "val")
TEST_ROOT = os.path.join(path, "dataset", "test")
CHECKPOINT_FILEPATH = os.path.join(path, 'checkpoint')
SAVED_MODEL_PATH = os.path.join(path, 'model')
TEST_AUDIO_PATH = os.path.join(path, "test_audio")
TEST_IMAGES_ROOT = os.path.join(path, "test_images")  # Store mel-spec img of new audio to predict
AUDIO_FROM_USER = os.path.join(path, "audio_from_user")  # Store audio uploaded from user in app

demo = False 

# For testing create these below folders
# RAW_ROOT = 'rawdata'
# FOLDER_ROOT = "check_mel-images"
# DATASET_ROOT = "check_dataset"
# TRAIN_ROOT = "check_dataset\\train"
# VAL_ROOT = "check_dataset\\val"
# TEST_ROOT = "check_dataset\\test"
# CHECKPOINT_FILEPATH = 'check_checkpoint'
# SAVED_MODEL_PATH = 'check_model'
# TEST_AUDIO_PATH = "test_audio"
# TEST_IMAGES_ROOT = "test_images"    # Store mel-spec img of new audio to predict
# AUDIO_FROM_USER = "audio_from_user" # Store audio uploaded from user in app


# Define global variable
danhmuc: tuple = (
    ["cailuong", "CaiLuong", "Cải lương", ],
    ["catru", "Catru", "Ca trù", ],
    ["chauvan", "Chauvan", "Chầu văn", ],
    ["cheo", "Cheo", "Chèo", ],
    ["hatxam", "Xam", "Hát xẩm", ],
    ["cachmang", "cachmang", "Cách mạng", ],
    ["nhactre", "nhactre", "Nhạc trẻ", ],
    ["thieunhi", "thieunhi", "Thiếu nhi", ],
    ["trutinh", "trutinh", "Trữ tình", ],
)
type_list: dict = {stt: [theloai[0], theloai[1]] for stt, theloai in enumerate(danhmuc)}
class_list: dict = {stt: theloai[2] for stt, theloai in enumerate(danhmuc)}

# Define processing parameters
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512

# Train/Val/Test split
TRAIN_RATE = 0.75
VAL_RATE = 0.15
TEST_RATE = 0.1

# Input/ Output
N_CLASS = 9
INPUT_SHAPE = (128, 1292)

# Num of samples of each class
amount_of_each_genre: dict = {
    "NUM_OF_CAILUONG": len(os.listdir(os.path.join(RAW_ROOT, "cailuong"))),
    "NUM_OF_CATRU": len(os.listdir(os.path.join(RAW_ROOT, "catru"))),
    "NUM_OF_CHAUVAN": len(os.listdir(os.path.join(RAW_ROOT, "chauvan"))),
    "NUM_OF_CHEO": len(os.listdir(os.path.join(RAW_ROOT, "cheo"))),
    "NUM_OF_HATXAM": len(os.listdir(os.path.join(RAW_ROOT, "hatxam"))),
    "NUM_OF_CACHMANG": len(os.listdir(os.path.join(RAW_ROOT, "cachmang"))),
    "NUM_OF_NHACTRE": len(os.listdir(os.path.join(RAW_ROOT, "nhactre"))),
    "NUM_OF_THIEUNHI": len(os.listdir(os.path.join(RAW_ROOT, "thieunhi"))),
    "NUM_OF_TRUTINH": len(os.listdir(os.path.join(RAW_ROOT, "trutinh"))),
}


# Data augmentation configs
RESCALE = 1./255
WIDTH_SHIFT_RANGE = 0.05
HEIGHT_SHIFT_RANGE = 0.05
ZOOM_RANGE = 0.025

# Checkpoint configs
CHECKPOINT_MONITOR = "val_accuracy"  # val_loss

# Early stopping configs
PATIENCE = 5
VERBOSE = 1
EARLY_MONITOR = "loss"

# Model config
OPTIMIZER = "adam"  # rmsrop, sgd
METRICS = ["accuracy"]   # tf.Metrics.Precision(), #tf.Metrics.Recall
LOSS ='categorical_crossentropy' # ...
BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 32
EPOCHS = 3 if demo else 300


# EXCEL URL

EXCEL_URL = "app/user_input.xlsx"