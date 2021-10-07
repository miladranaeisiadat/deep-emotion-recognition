import os
import re
import json
from typing import Tuple
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

def read_emotions_json(name: str):
    # Opening JSON file
    f = open('emotions.json',)
    
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    return data[name]

def extract_feature_from_mfcc(wav_file_path: str, 
                              len_mfcc: int = 50, 
                              sample_rate: int= 22050, 
                              hop_length: int= 552) -> np.ndarray:
    """
    This function extract feature vector from Mel-frequency cepstrum(MFCC) for the given sound file 
    and obtain the mean of each dimension.

    Args:
        file_path (str): path to the .wav file for the current database.
        mfcc_len (int): Number of cepestral co efficients to be consider.
    Returns:
        numpy.ndarray: feature vector of the wav file which extracted from mfcc.
    """
    signal, sample_rate = librosa.load(wav_file_path) #  dtype="float32"
    result = np.array([])
    mfccs = np.mean(librosa.feature.mfcc(y=signal, 
                                        sr=sample_rate,                     # sample rate which defult is 22050
                                        n_mfcc=len_mfcc,                  # number of mfcc
                                        # n_fft= n_fft,                       # number of fast fourier transform
                                        hop_length= hop_length ).T, axis=0) # The number of samples between successive frame
    result = np.hstack((result, mfccs))
    return result


def load_train_data(path_csv_data: str, name_of_dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    """
        Loads training data from the csv files.
    """
    X_train = []
    Y_train = []
    file_properties = pd.read_csv(path_csv_data)
    label_emotions = pd.DataFrame({'emotion' : read_emotions_json(name_of_dataset)})
    trains = file_properties[file_properties['set'] == 'train'].copy()
    trains.drop('Unnamed: 0', inplace = True, axis = 1)
    for train in tqdm(trains.iterrows(), desc='Extrating MFCCs from train data. It maybe take a short time. Please wait... '):
        wav_sound = str(train[1]['filename'])
        mfcc = extract_feature_from_mfcc(wav_sound)
        X_train.append(mfcc)
        Y_train.append(label_emotions[label_emotions['emotion'] == train[1]['emotion']].index[0])
    return np.array(X_train), np.array(Y_train)


def load_test_data(path_csv_data: str, name_of_dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    """
        Loads test data from the csv files
        """
    X_test = []
    Y_test = []
    file_properties = pd.read_csv(path_csv_data)
    label_emotions = pd.DataFrame({'emotion' : read_emotions_json(name_of_dataset)})
    tests = file_properties[file_properties['set'] == 'test'].copy()
    tests.drop('Unnamed: 0', inplace = True, axis = 1)
    for test in tqdm(tests.iterrows(), desc='Extrating MFCCs from test data. It maybe take a short time. Please wait... '):
        wav_sound = str(test[1]['filename'])
        mfcc = extract_feature_from_mfcc(wav_sound)
        X_test.append(mfcc)
        Y_test.append(label_emotions[label_emotions['emotion'] == test[1]['emotion']].index[0])
    return np.array(X_test), np.array(Y_test)

    

def load_datas(path_csv_data : str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads features of wav audio files and return X_train, Y_train, X_test, Y_test
    Params:
        path_csv_data (str): path of description files (csv files) to read from
    """

    name_of_dataset = str(os.path.splitext(os.path.basename(path_csv_data))[0])
    # Loads training data
    X_train, Y_train = load_train_data(path_csv_data, name_of_dataset)
    # Loads test data
    X_test, Y_test = load_test_data(path_csv_data, name_of_dataset)
    return X_train, Y_train, X_test, Y_test
