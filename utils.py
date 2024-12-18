import os
import librosa as lb
import random
import numpy as np
from config import *
from sklearn.model_selection import train_test_split
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from pydub import AudioSegment


def plot_waveform(samples, type_index, sr = SR):
    """
    Waveform plot of samples
    """

    color = random.choice(["blue", "red", "yellow", "brown", "purple"])
    for index, sample in samples.items():
      plt.figure(figsize = (16, 5))
      lb.display.waveshow(y = sample["sampling"], sr = sr, color = color)
      plt.title("Sound Waves of sample {} of class {}".format(index, type_list[type_index][0]), fontsize = 23)


def mp3_2_wav(dir, dst, sample_rate = SR):
    """
    Convert mp3 to wav and save wav file to dst
    Input: dir (mp3)
    """
    # convert wav to mp3.
    sound = AudioSegment.from_mp3(dir)
    sound.set_frame_rate(sample_rate)
    sound.export(dst, format="wav")


def break_down_downloaded_data(
        originaldata=os.path.join('/', 'home', 'zaibachkhoa', 'Downloads', 'GTZAN Dataset - Music Genre Classification', 'genres_original_vnmesemusi_cachmang-nhactre-thieunhi-trutinh', 'genres_original'),
        lim: int = 500,
        onlygenre: str or None = None,
):
    """
    breaks down data into series of 30 seconds songs
    TODO parameter
    """
    for theloai_ in os.listdir(originaldata):
        if onlygenre is not None:
            if not theloai_ == onlygenre:
                continue
        theloai = theloai_.replace('_', '')
        assert os.path.isdir(os.path.join(RAW_ROOT, theloai))
        i = 0
        for baihat in os.listdir(os.path.join(originaldata, theloai_)):
            dir = os.path.join(originaldata, theloai_, baihat)
            dst = os.path.join(originaldata, theloai_, baihat[:-len('.mp3')] + '.wav')
            mp3: bool = False
            if not any([
                baihat.endswith('.mp3'),
                baihat.endswith('.wav')
            ]):
                print('@@@@@@@@@@@@@@@@@@', baihat)
                continue
            if baihat.endswith('.mp3'):
                mp3 = True
                mp3_2_wav(dir, dst)
            newAudio = AudioSegment.from_wav(dst)
            i_ = 0
            t1 = 0
            # print(newAudio.duration_seconds)  # print(type(newAudio.duration_seconds))  # float
            while t1 < newAudio.duration_seconds:
                t1 = i_ * 30 * 1000  # Works in milliseconds
                t2 = (i_ + 1) * 30
                if t2 > int(newAudio.duration_seconds):
                    break
                t2 *= 1000
                newAudio_ = newAudio[t1:t2]
                try:
                    newAudio_.export(os.path.join(RAW_ROOT, theloai, f'{theloai}.{str(i).zfill(3)}.wav'), format="wav")
                except Exception as e:
                    print()
                    continue
                del newAudio_
                i += 1
                if i > lim-1:
                    break
                i_ += 1
                t1 /= 1000
            if mp3:
                os.remove(dst)
            print(f'{dir}_______{i}')
            if i > lim-1:
                break



def plot_fft(samples, type_index):
    """
    Get frequency domain representation
    """

    for index, item in samples.items():
        plt.figure(figsize = (16, 6))
        plt.plot(item["stft"])
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title("STFT of sample {} of class {}".format(index, type_list[type_index][0]))



def plot_spectrogram(samples, type_index, hop_length):
    """
    Plot spectrogram
    """

    for index, item in samples.items():
        DB = lb.amplitude_to_db(item["stft"], ref = np.max)
        plt.figure(figsize = (25, 10))
        lb.display.specshow(DB, hop_length= hop_length, x_axis = "time", y_axis = "log")
        plt.title("Spectrogram of sample {} of class {}".format(index, type_list[type_index][0]), fontsize = 20)
        plt.colorbar()



def train_val_test_split(folder_root, dataset_root, type_index):
    """
    Split and save train/val/test set
    Input:
    - folder_root: folder_root containing mel-spec images
    - dataset_root: Directory to save dataset
    - type_root : train_root, val_root or test_root
    - type_index: class index in type_list
    """

    def save_set(subset, dataset_root, typeset, type_index):
        """
        Save X_train, X_val, X_test to their respective dir
        Input:
          - subset - X_train, X_val, X_test
          - dataset_root: Directory to save dataset
          - typeset - train, val, test
          - type index - Class index
        """
        # Copy file from subset to train/val/test folder
        for file in subset:
            srcpath = os.path.join(src_dir, file)
            dst_dir = os.path.join(dataset_root, typeset, type_list[type_index][0])
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            shutil.copy(srcpath, dst_dir)


    src_dir = os.path.join(folder_root, type_list[type_index][0])
    X = os.listdir(src_dir)
    Y = ["{}".format(type_list[type_index][0]) for i in range(0, len(X))]

    # Train 75%, test 25%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 1 - TRAIN_RATE, random_state=42, shuffle = True)
    # Val 15 %, test 10%
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = TEST_RATE / (TEST_RATE + VAL_RATE), random_state=42, shuffle = True)

    # Create dataset_root to save dataset
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)

    # Save train/val/test of each class
    save_set(X_train, dataset_root, "train", type_index)
    save_set(X_val, dataset_root, "val", type_index)
    save_set(X_test, dataset_root, "test", type_index)



def plot_result(history):
    """
    Plot loss and acc of train/val
    Input: history - model_history
    """

    fig, ag = plt.subplots(1,2,figsize = (15,6))
    ag[0].plot(history.history['loss'],label = 'train')
    ag[0].plot(history.history['val_loss'],label = 'val')
    ag[0].legend()
    ag[0].set_title('Loss versus epochs')

    ag[1].plot(history.history['accuracy'],label='train')
    ag[1].plot(history.history['val_accuracy'],label='val')
    ag[1].legend()
    ag[1].set_title('Accuracy versus epochs')
    plt.show()


def predict(file_names, labels, class_list, typeset, model):
    """
    Predict samples in a set (val/test set) (just .wav)
    Input: 
    - file_name: files directory (got from val(test)_generator.filenames)
    - labels: labels respective to file_name (got from val(test)_generator.labels)
    - class_list: Global variables 'class_list'
    - typeset: "val" or "test"
    - model: model used to predict

    Output:
    - y_pred_index: List of index class prediction of all samples in file_name
    - y_class_pred: List of class respective to y_pred_index
    """

    y_pred_index = []
    y_class_pred = []
    for file in file_names:
        file_root = os.path.join(DATASET_ROOT, typeset, str(file))
        image = load_img(file_root, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3))
        image_array = img_to_array(image)
        image_array = image_array * 1./255
        input_data = tf.expand_dims(image_array, 0)
        pred = model.predict(input_data, verbose = 0)
        pred_index = np.argmax(np.squeeze(pred))
        y_pred_index.append(pred_index)
        y_class_pred.append(class_list[pred_index])
    print("---Predicted----")
    print("Accuracy on {} set : {}".format(typeset, (labels == y_pred_index).sum()/ len(labels)))
    return y_pred_index, y_class_pred



def get_cfm(y_pred, labels, class_list, typeset):
    """
    Get confusion matrix 
    """
    ax= plt.subplot()
    cfm = confusion_matrix(y_pred, labels)
    sns.heatmap(cfm, annot=True, fmt='g', ax=ax)  # annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix on {} set'.format(typeset), fontsize = 20)
    ax.xaxis.set_ticklabels(list(class_list.values()))
    ax.yaxis.set_ticklabels(list(class_list.values()))
    plt.show()



def predict_new30s(audio_dir, model, save_dir = TEST_IMAGES_ROOT):
    """
    Predict new 30s-length audio (arround 30 is accepted)
    Input:
    - audio_dir : List of audios directory (.wav)
    - model: model to predict
    - save_dir: TEST_IMAGES_ROOT - directory save log-mel-spec image of new audio

    Output:
    - y_pred: List of index class prediction of all samples in file_name
    - y_class: List of class respective to y_pred_index
    """

    y_pred = []
    y_class = []

    for dir in audio_dir:
        load_dir, sr = lb.load(dir)
        S = lb.feature.melspectrogram(y = load_dir, sr=sr)
        S_db = lb.amplitude_to_db(S, ref=np.max)

        # Create TEST_IMAGE_ROOT if it does not exist yey
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        audio_file_name = dir.split("\\")[-1][:-4]

        saved_img_root = save_dir + "\\{}".format(audio_file_name) + ".png"
        plt.imsave(saved_img_root, S_db)

        image = load_img(saved_img_root, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3))
        image_array = img_to_array(image)
        input_data = tf.expand_dims(image_array, 0)
        pred = model.predict(input_data, verbose = 0)
        pred_index = np.argmax(np.squeeze(pred))
        y_pred.append(pred_index)
        y_class.append(class_list[pred_index])

        return y_pred, y_class
    


def predict_new(audio_dir, src_folder, model, save_dir, unit_length = 661500):
    """
    Predict audio of any length using one model
    Split each audio into several equal samples which length = unit_length (661500 = 30s), then feed to NN
    Get predict class by votting each sample's prediction

    Input:
    - audio_dir: List of audio directory to predict
    - src_folder: Dir of folder containning audio_dir
    - model: Model to predict
    - save_dir: Directory to save log-mel-spec image of samples splitted from each audio in audio_dir
    Output:
    - y_pred_index: List of index predicted of each audio in audio_dir
    - y_pred_class: Respective class predicted of y_pred_index
    """

    def process(samples_split, save_dir, file_name, is_saved):
        """
        End to end processing steps of each audio

        Input:
        - samples_split: List of samples splitted from each audio in audio_dir
        - save_dir: Directory to save log-mel-spec image of samples splitted from each audio in audio_dir
        - is_save: If False, do not save log-mel-spec image of samples, just make prediction

        Output:
        - np.array(samples_db_list): A batch of samples of each audio file (nums_of_sample, input_shape[0], input_shape[1], 3) to feed to NN
        """
        
        samples_db_list = []
        for i, sample in enumerate(samples_split):
            S = lb.feature.melspectrogram(y = sample, sr=sr)
            S_db = lb.amplitude_to_db(S, ref=np.max)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            sample_root = os.path.join(save_dir, f'{file_name}_sample{i}.png')
            plt.imsave(sample_root, S_db)
            image = load_img(sample_root, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3))
            img_array = img_to_array(image)
            img_array = img_array * 1./ 255
            samples_db_list.append(img_array)
            if not is_saved: # Not save mode
                for file in os.listdir(save_dir):
                    if file.endswith('.png'):
                        os.remove(os.path.join(save_dir, file))
        return np.array(samples_db_list)

    # Define result
    y_pred_index = []
    y_pred_class = []

    # List of samples of each audio
    samples_split = []
    y_pred_split = []

    for dir in audio_dir:
        if dir.endswith(".mp3"):
            print("Convert {} to .wav".format(dir))
            wav_dir = os.path.join(src_folder, dir.split(os.sep)[-1][:-4] + ".wav")  # src_folder = TEST_AUDIO_PATH (trainning), AUDIO_FROM_USER (web)
            mp3_2_wav(dir, wav_dir)
            dir = wav_dir  # Take wav dir for sampling
        audio, sr = lb.load(dir)
        if (len(audio) >= unit_length):
            # Number of sample of each audio
            nums_of_samples = len(audio) // unit_length
        else:
            err = "Audio length must be greater than 30s"
            print(err)
            return err
        for i in range(0, nums_of_samples):
            curr_sample = audio[i * unit_length : i * unit_length + unit_length]
            if (len(curr_sample) != unit_length): # Cannot sampling this curr_sample
                break
            samples_split.append(audio[i * unit_length : i * unit_length + unit_length])

        file_name = dir.split(os.sep)[-1][:-4]

        input_data = process(samples_split, save_dir, file_name, False)

        pred_candidates = model.predict(input_data, verbose = 0)

        pred_index_candidates = [np.argmax(sample) for sample in pred_candidates]

        pred_index = max(pred_index_candidates, key = pred_index_candidates.count)
        pred_class = class_list[pred_index]

        y_pred_index.append(pred_index)
        y_pred_class.append(pred_class)

        # Reset samples_split after passing one dir of audio_dir
        samples_split = []

    return y_pred_index, y_pred_class



def PROD_predict(audio_dir, src_folder, save_dir, model1, model2, model3, unit_length = 661500, gra: bool = False):
    """
    Predict audio of any length using PROD fusion of three models predicted probability vectors
    Split each audio into several equal sample which length = unit_length, then feed to NN
    Get predict class by most votting of each sample's prediction

    Input:
    - audio_dir: List of audio directory to predict
    - src_folder: Dir of folder containning audio_dir
    - model: Model to predict
    - save_dir: Directory to save log-mel-spec image of samples splitted from each audio in audio_dir
    Output:
    - y_pred_index: List of index predicted of each audio in audio_dir
    - y_pred_class: Respective class predicted of y_pred_index
    """

    def process(samples_split, save_dir, file_name, is_saved):
        """
        End to end processing steps of each audio

        Input:
        - samples_split: List of samples splitted from each audio in audio_dir
        - save_dir: Directory to save log-mel-spec image of samples splitted from each audio in audio_dir
        - is_save: If False, do not save log-mel-spec image of samples, just make prediction

        Output:
        - np.array(samples_db_list): A batch of samples of each audio file (nums_of_sample, input_shape[0], input_shape[1], 3) to feed to NN
        """
        samples_db_list = []
        for i, sample in enumerate(samples_split):
            S = lb.feature.melspectrogram(y = sample, sr=sr)
            S_db = lb.amplitude_to_db(S, ref=np.max)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            sample_root = os.path.join(save_dir, f"{file_name}_sample{i}.png")
            plt.imsave(sample_root, S_db)
            image = load_img(sample_root, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3))
            img_array = img_to_array(image)
            img_array = img_array / 255
            samples_db_list.append(img_array)
            if not is_saved:  # Not save mode
                for file in os.listdir(save_dir):
                    if file.endswith('.png'):
                        os.remove(os.path.join(save_dir, file))
        return np.array(samples_db_list)

    # Define result
    y_pred_index = []
    y_pred_class = []

    # List of samples of each audio
    samples_split = []
    y_pred_split = []
    resdic = {cl: list() for cl in class_list.values()}

    for dir in audio_dir:
        if dir.endswith(".mp3"):
            # Get file name
            wav_dir = os.path.join(src_folder, dir.split(os.sep)[-1][:-4] + ".wav")
            mp3_2_wav(dir, wav_dir)
            dir = wav_dir       # Take wav dir for sampling
        print(dir)
        audio, sr = lb.load(dir, mono=True)
        if (len(audio) >= unit_length):
            # Number of sample of each audio
            nums_of_samples = len(audio) // unit_length
        else:
            err = "Audio length must be greater than 30s"
            print(err)
            return err
        for i in range(nums_of_samples):
            curr_sample = audio[i * unit_length : i * unit_length + unit_length]
            if (len(curr_sample) != unit_length): # Cannot sampling this curr_sample
                break
            samples_split.append(audio[i * unit_length : i * unit_length + unit_length])

        file_name = dir.split(os.sep)[-1][:-4]

        input_data = process(samples_split, save_dir, file_name, False)

        pred_candidates1 = model1.predict(input_data, verbose = 0)

        pred_candidates2 = model2.predict(input_data, verbose = 0)

        pred_candidates3 = model3.predict(input_data, verbose = 0)

        PROD_probs = []

        # PROD fusion
        for i in range(pred_candidates1.shape[0]):
            PROD_probs.append(1/3 * pred_candidates1[i] * pred_candidates2[i] * pred_candidates3[i])

        pred_index_candidates = [np.argmax(sample) for sample in PROD_probs]

        pred_index = max(pred_index_candidates, key = pred_index_candidates.count)
        pred_class = class_list[pred_index]

        resdic[pred_class].append(file_name)
        y_pred_index.append(pred_index)
        y_pred_class.append(pred_class)

        # Reset samples_split after passing one dir of audio_dir
        samples_split = []

    return (y_pred_index, y_pred_class) if not gra else {k: v for k, v in resdic.items() if v}


if __name__ == '__main__':
    break_down_downloaded_data(
        # originaldata=None,
        lim=100,
        onlygenre='cach_mang',
    )