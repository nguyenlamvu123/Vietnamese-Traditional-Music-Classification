# Define directories 
RAW_ROOT = 'rawdata'
FOLDER_ROOT = "mel-images"
DATASET_ROOT = "dataset"
TRAIN_ROOT = "dataset\\train"
VAL_ROOT = "dataset\\val"
TEST_ROOT = "dataset\\test"
CHECKPOINT_FILEPATH = 'checkpoint'
SAVED_MODEL_PATH = 'model'
TEST_AUDIO_PATH = "test_audio"

# Define global variable
type_list = {0: ["cailuong", "CaiLuong"], 1: ["catru", "Catru"], 2:["chauvan", "Chauvan"], 3: ["cheo", "Cheo"], 4: ["hatxam", "Xam"]}

class_list = {0: "cailuong", 1: "catru", 2:"chauvan", 3: "cheo", 4: "hatxam"}

# Define processing parameters
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512

# Train/Val/Test split
TRAIN_RATE = 0.75
VAL_RATE = 0.15
TEST_RATE = 0.1

# Input/ Output
N_CLASS = 5
INPUT_SHAPE = (128, 1292)

# Data augmentation config
RESCALE = 1./255
WIDTH_SHIFT_RANGE = 0.05
HEIGHT_SHIFT_RANGE = 0.05
ZOOM_RANGE = 0.025