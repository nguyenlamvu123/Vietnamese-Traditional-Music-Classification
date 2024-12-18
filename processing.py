import os
import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np
from config import *
from sklearn.model_selection import train_test_split
import shutil
from utils import * 


class Preprocessing():
    def __init__(self, root, save_root, dataset_root, train_root, val_root, test_root, samples, type_index, num_of_samples):
        self.root = root                            # Directory of folder
        self.save_root = save_root                  # Directory of folder saving mel-spec images
        self.dataset_root = dataset_root            # Directory to save dataset
        self.train_root = train_root                # Directory of trainset
        self.val_root = val_root                    # Directory of valset
        self.test_root = test_root                  # Directory of test_root
        self.samples = samples                      # Dictionary store information of each class folder
        self.type_index = type_index                # Class index
        self.num_of_samples = num_of_samples        # Num of samples in class to process

    def _load_dir_samples(self, mode):
        """
        Load directory into 'samples' dictionary
        - Random: Load random dir
        - All: Load all dir

        Return:
        - Samples: Dictionary {index: {"dir": "/...."}}
        """
        def padding(index) -> str:
            return str(index).zfill(3)
        for i in range(self.num_of_samples):
            if mode == "random":  # Mode load random samples
                random_index = np.random.randint(0, 500)
                index = random_index
                self.samples[index] = {}  # For futher append values
                random_index = padding(random_index)
                self.samples[index]["dir"] = self.root + os.sep + type_list[self.type_index][0] + os.sep + type_list[self.type_index][1] + "." + random_index + ".wav"
                # self.samples_list[index]["dir"] = (os.path.join(self.root, type_list[self.type_index][0], type_list[self.type_index][1] + "." + random_index + ".wav"))
            
            elif mode == "all":  # Mode load all samples
                index = i
                self.samples[index] = {}
                i = padding(i)
                self.samples[index]["dir"] = self.root + os.sep + type_list[self.type_index][0] + os.sep + type_list[self.type_index][1] + "." + i + ".wav"
                # self.samples[index]["dir"] = os.path.join(self.root, type_list[self.type_index][0], type_list[self.type_index][1] + "." + i + ".wav")

        return self.samples


    def _load_samples(self):
        """
        Load and sampling
        Input: samples_listdir - Dictionary {index: {"dir": "/...."}}
        Output: samples_listdir - Dictionary {index: {"dir": "/....", "sampling": array}}
        """
        for index, sample in self.samples.items():
            fil = sample["dir"]
            if not os.path.isfile(fil):  # No such file or directory: 'rawdata/catru/Catru.089.wav'
                print(f'@@@@@@@@@@@@@@@@@ No such file or directory: {fil}')
                continue
            file, sr = lb.load(fil)  # TODO
            if len(self.samples[index]) == 1:  # Avoid adding multiple times
                self.samples[index]["sampling"] = file
        return self.samples
    

    def _get_fft(self, n_fft, hop_length):
        """
        Input: samples: {index: {"dir": "/..."}}
        Output: samples: {index: {"dir": "/...", "stft:" array}}
        """
        for index, item in self.samples.items():
            if not "sampling" in item:  # KeyError: 'sampling' (<- 'rawdata/catru/Catru.089.wav')
                print(f'@@@@@@@@@@@@@@@@ KeyError: "sampling" in {item["dir"]}')
                continue
            # Get STFT
            D = np.abs(lb.stft(item["sampling"], n_fft = n_fft, hop_length = hop_length))
            self.samples[index]["stft"] = D
        return self.samples


    def _get_mel_spectrogram(self, sr):
        """
        Get log-mel-spectrogram (db)
        Input: {index: {"dir": "/...", "sampling": array, "stft": array, }}
        Output: {index: {"dir": "/...", "sampling": array, "stft": array, "mel-spec-db": array}}
        """
        for index, item in self.samples.items():
            if not "sampling" in item:  # KeyError: 'sampling' (<- 'rawdata/catru/Catru.089.wav')
                print(f'@@@@@@@@@@@@@@@ KeyError: "sampling" in {item["dir"]}')
                continue
            S = lb.feature.melspectrogram(y = item["sampling"], sr = sr)
            S_db = lb.amplitude_to_db(S, ref=np.max)
            self.samples[index]["mel-spec-db"] = S_db
        return self.samples

    def _save_mel_spec(self):
        """
        Save log-mel-spec
        After running, images of a class will be saved in : root/class/file_name.png
        """

        for _, item in self.samples.items():
            if not "mel-spec-db" in item:  # KeyError: 'sampling' (<- 'rawdata/catru/Catru.089.wav')
                print(f'@@@@@@@@@@@@@@@ KeyError: "mel-spec-db" in {item["dir"]}')
                continue
            S_db = item["mel-spec-db"]
            images_root = self.save_root + os.sep + type_list[self.type_index][0]
            if not os.path.exists(images_root):
                os.makedirs(images_root)
                print("Create new root: {}".format(images_root))
            # Get file name from fir
            file_name = item["dir"].split(os.sep)[-1][:-4]
            plt.imsave(f'{images_root}{os.sep}{file_name}.png', S_db)
            print("Saved {}".format(f'{images_root}{os.sep}{file_name}.png'))



# --------------OUTSIDE CLASS-------------------------------

def end_to_end_process(raw_root, save_root, dataset_root, train_root, val_root, test_root, type_index, num_of_samples):
    """
    End to end process from raw audio to train/val/test split
    Input:
    - raw_root: Directory of raw data
    - save_root: Directory to save mel-images
    - dataset_root: Directory to save dataset
    - train_root: Directory to save train set
    - val_root: Directory to save val set
    - test_root: Directory to save test set
    - type_index: Class index
    - num_of_samples: Num of samples of each class to train (Train all the raw audio -> 500)
    """
    class_samples = Preprocessing(root=raw_root, save_root=save_root, dataset_root=dataset_root, 
                                  train_root=train_root, val_root=val_root, test_root=test_root, samples={}, 
                                  type_index=type_index, num_of_samples=num_of_samples)
    class_samples._load_dir_samples(mode = "all")
    class_samples._load_samples()
    class_samples._get_fft(n_fft=N_FFT, hop_length=HOP_LENGTH)
    class_samples._get_mel_spectrogram(sr=SR)
    class_samples._save_mel_spec()
    train_val_test_split(folder_root=save_root, dataset_root= dataset_root, type_index= type_index)
    return 


if __name__ == "__main__":

# -----------END-TO-END PROCESS EACH CLASS---------------------------------
    for thutu, danhmuc in enumerate(amount_of_each_genre):
        end_to_end_process(
            raw_root=RAW_ROOT,
            save_root=FOLDER_ROOT,
            dataset_root=DATASET_ROOT,
            train_root=TRAIN_ROOT,
            val_root=VAL_ROOT,
            test_root=TEST_ROOT,
            type_index=thutu,
            num_of_samples=amount_of_each_genre[danhmuc]
        )
