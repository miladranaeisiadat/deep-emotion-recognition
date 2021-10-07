## Python
import sys

## Package
import glob
import librosa
import pandas as pd
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from optparse import OptionParser

from sklearn.model_selection import train_test_split

class EmotionDatasets():
    """
    Class for reads specific dataset from directory and write it to a metadata CSV file
    with the 6 targets such as: dataset, filename, actor, emotion, length, gender.
    """
    def __init__(self, dataset_path: str, dataset_type: str, verbose = 1):

        self.dataset_type = dataset_type.upper()
        self.verbose = verbose
        if self.dataset_type == "EMOVO":
            self.emovo_dataset_to_csv(dataset_path)
        elif self.dataset_type == "SAVEE":
            self.savee_dataset_to_csv(dataset_path)
        elif self.dataset_type == "TESS":
            self.tess_dataset_to_csv(dataset_path)
        elif self.dataset_type == "RAVDESS":
            self.ravdess_dataset_to_csv(dataset_path)
        elif self.dataset_type == "EMODB":
            self.emodb_dataset_to_csv(dataset_path)
        elif self.dataset_type == "SHEMO":
            self.get_shemo_dataset(dataset_path)
        elif self.dataset_type == "CREMA":
            self.crema_dataset_to_csv(dataset_path)
        else:
            print("[Error] - please choose a correct name for dataset")

    @staticmethod
    def save_csv(list_dataframe, name_of_dataset):
        """
        Get a dataframe list and name of dataset and seve dataframe to the path: './datasets/csv/'.
        params:
            list_dataframe : data for each dataset for saving.
            name_of_dataset : name of dataset which wanna save.
        """
        path = './datasets/csv/'
        list_dataframe.to_csv(f'{path}{name_of_dataset}.csv')
        print(f'[{name_of_dataset.capitalize()}] dataset converted to CSV successfully.')


    @staticmethod
    def split_data(all_emotion_list):
        """
        This function spliting data to train and test set to 80/20.
        For tune hyper parameters just need to uncomment some lines to also get a validation set.

        """
        train, test = train_test_split(all_emotion_list, test_size = 0.20, random_state = 42)
        #validation, test = train_test_split(_test, test_size = 0.5, random_state = 42)
        set_list = []
        for file in all_emotion_list['filename']:
            if file in list(train['filename']):
                set_list.append('train')
    #         if file in list(validation['filename']):
    #             set_list.append('validation')
            if file in list(test['filename']):
                set_list.append('test')

        all_emotion_list['set'] = set_list
        return all_emotion_list

    @staticmethod
    def details_datasets(dataset):
        df_emotion = dataset["emotion"].value_counts().rename_axis('list_of_emotion').reset_index(name='counts').set_index('list_of_emotion')
        df_to_dict = df_emotion.to_dict()['counts']
        train = dataset[dataset['set'] == 'train']['emotion'].value_counts().rename_axis('list_of_emotion').reset_index(name='counts').set_index('list_of_emotion')
        train_dict = train.to_dict()['counts']
        test = dataset[dataset['set'] == 'test']['emotion'].value_counts().rename_axis('list_of_emotion').reset_index(name='counts').set_index('list_of_emotion')
        test_dict = test.to_dict()['counts']

        table = Table(title="EMOTIONS")
        table.add_column("Emotions", justify="right", style="cyan", no_wrap=True)
        table.add_column("Total number of emotions",style="magenta")
        table.add_column("Train(80%) samples", justify="right", style="green")
        table.add_column("Test(20%) samples", justify="right", style="red")

        for key, value in df_to_dict.items():
          for train_key, train_value in train_dict.items():
            for test_key, test_value in test_dict.items():
              if key == train_key and key == test_key:
                table.add_row(key, str(value), str(train_value), str(test_value))
        table.add_row("", "Shape : " + str(dataset.shape),"","")

        console = Console()
        console.print(table)

    def savee_dataset_to_csv(self, path: str):
        """
        Reads speech SAVEE datasets from directory and write it to a CSV file.
        params:
            path : path of dataset's directory
        """
        savee_listdir = glob.glob(path + '/*/*.wav')
        if not savee_listdir:
            print("Directory not exist.")
        emotion_savee_key = {
                            'n': 'neutral', 
                            'h': 'happy', 
                            'sa': 'sad', 
                            'a': 'angry', 
                            'f': 'fearful', 
                            'd': 'disgusted', 
                            'su': 'surprised'
                            }
        targets = {'dataset': [], 'filename': [],
                   'actor': [], 'emotion': [],
                   'length': [], 'gender': []}
        for file in tqdm(savee_listdir, desc='[SAVEE] to CSV and split into train and test(80/20)'):
            targets['dataset'].append('SAVEE')
            targets['filename'].append(file)
            props = file.split('/')
            targets['actor'].append(props[-2])
            targets['emotion'].append(emotion_savee_key[props[-1][:-6]])
            targets['gender'].append('male')
            y, sr = librosa.load(file)
            targets['length'].append(y.shape[0]/sr)

        file_props = pd.DataFrame(targets)
        common_cols = ['dataset', 'filename', 'actor', 'emotion', 'length', 'gender']
        complete_datasets = pd.concat([file_props[common_cols]], axis = 0)
        split_dataset = self.split_data(complete_datasets)
        if self.verbose:
            self.details_datasets(split_dataset)
            self.save_csv(split_dataset, 'savee')

    def crema_dataset_to_csv(self, path: str):
        """
        Reads speech CREMA datasets from directory and write it to a CSV file.
        params:
            path : path of dataset's directory
        """
        crema_listdir = glob.glob(path + '/*.wav')
        if not crema_listdir:
            print("Directory not exist.")
        emotion_crema_key = {
                            'NEU': 'neutral', 
                            'HAP': 'happy', 
                            'SAD': 'sad',
                            'SA' : 'sad',
                            'ANG': 'angry', 
                            'FEA': 'fearful', 
                            'DIS': 'disgusted'
                            }
        targets = {'dataset': [], 'filename': [],
                   'actor': [], 'emotion': [],
                   'length': [], 'gender': []}
        for file in tqdm(crema_listdir, desc='[CREMA] to CSV and split into train and test(80/20)'):
            targets['dataset'].append('CREMA')
            targets['filename'].append(file)
            props = file.split('/')
            targets['actor'].append(props[3][0:4])
            targets['emotion'].append(emotion_crema_key[props[3][9:-7]])
            targets['gender'].append('male')

            y, sr = librosa.load(file)
            targets['length'].append(y.shape[0]/sr)

        file_props = pd.DataFrame(targets)
        common_cols = ['dataset', 'filename', 'actor', 'emotion', 'length', 'gender']
        complete_datasets = pd.concat([file_props[common_cols]], axis = 0)
        split_dataset = self.split_data(complete_datasets)
        if self.verbose:
            self.details_datasets(split_dataset)
            self.save_csv(split_dataset, 'crema')

    def emovo_dataset_to_csv(self, path: str):
        """
        Reads speech EMOVO datasets from directory and write it to a CSV file.
        params:
            path : path of dataset's directory
        """
        emovo_listdir = glob.glob(path + '/*/*.wav')
        if not emovo_listdir:
            print("Directory not exist.")
        emotion_emovo_key = {'neu': 'neutral',
                             'gio': 'happy',
                             'tri': 'sad',
                             'rab': 'angry',
                             'pau': 'fearful',
                             'dis': 'disgusted',
                             'sor': 'surprised'}
        targets = {'dataset': [], 'filename': [],
                   'actor': [], 'emotion': [],
                   'length': [], 'gender': []}
        males = ['m1', 'm2', 'm3']
        females = ['f1', 'f2', 'f3']
        for file in tqdm(emovo_listdir, desc='[EMOVO] to CSV and split into train and test(80/20)'):
            targets['dataset'].append('EMOVO')
            targets['filename'].append(file)
            props = file.split('/')
            targets['actor'].append(props[-1][-9:-7])
            targets['emotion'].append(emotion_emovo_key[props[-1][:-10]])
            # targets['repitition'].append(props[-1][-6:-4])
            if int(props[-1][-9:-7] in males):
              targets['gender'].append('male')
            elif int(props[-1][-9:-7] in females):
              targets['gender'].append('female')

            y, sr = librosa.load(file)
            targets['length'].append(y.shape[0]/sr)

        file_props = pd.DataFrame(targets)
        common_cols = ['dataset', 'filename', 'actor', 'emotion', 'length', 'gender']
        complete_datasets = pd.concat([file_props[common_cols]], axis = 0)
        split_dataset = self.split_data(complete_datasets)
        if self.verbose:
            self.details_datasets(split_dataset)
            self.save_csv(split_dataset, 'emovo')

    def emodb_dataset_to_csv(self, path: str) -> None:
        """
        Reads speech EMODB datasets from directory and write it to a CSV file.
        params:
            path : path of dataset's directory
        """
        emodb_listdir = glob.glob(path + '*.wav')
        if not emodb_listdir:
            print("Directory not exist.")
        emotion_emodb_key = {
                            "W": "angry",
                            "L": "calm",
                            "E": "disgusted",
                            "A": "fearful",
                            "F": "happy",
                            "T": "sad",
                            "N": "neutral"
                            }
        targets = {'dataset': [], 'filename': [],
                   'actor': [], 'emotion': [],
                   'length': [], 'gender': []}
        males = ['03', '10', '11', '12', '15']
        females = ['08', '09', '13', '14', '16']
        for file in tqdm(emodb_listdir, desc='[EMO_DB] to CSV and split into train and test(80/20)'):
            targets['dataset'].append('emodb')
            targets['filename'].append(file)
            props = file.split('/')[3]
            # print(props[2:5])
            targets['actor'].append(props[0:2])
            targets['emotion'].append(emotion_emodb_key[props[5][:6]])
            if int(props[0:2] in males):
              targets['gender'].append('male')
            elif int(props[0:2] in females):
              targets['gender'].append('female')


            y, sr = librosa.load(file)
            targets['length'].append(y.shape[0]/sr)

        file_props = pd.DataFrame(targets)
        common_cols = ['dataset', 'filename', 'actor', 'emotion', 'length', 'gender']
        complete_datasets = pd.concat([file_props[common_cols]], axis = 0)
        split_dataset = self.split_data(complete_datasets)
        if self.verbose:
            self.details_datasets(split_dataset)
            self.save_csv(split_dataset, 'emodb')

    def tess_dataset_to_csv(self, path: str):
        """
        Reads speech TESS datasets from directory and write it to a CSV file.
        params:
            path : path of dataset's directory
        """
        tess_listdir = glob.glob(path + '/*/*.wav')
        if not tess_listdir:
            print("Directory not exist.")
        emotion_tess_key = {'neutral': 'neutral',
                             'happy': 'happy',
                             'sad': 'sad',
                             'angry': 'angry',
                             'fear': 'fearful',
                             'disgust': 'disgusted',
                             'ps': 'surprised'}

        targets = {'dataset': [], 'filename': [],
                   'actor': [], 'emotion': [],
                   'length': [], 'gender': []}

        for file in tqdm(tess_listdir, desc='[TESS] to CSV and split into train and test(80/20)'):
            targets['dataset'].append('TESS')
            targets['filename'].append(file)
            props = file.split('/')[4].split('_')
            targets['actor'].append(props[0])
            targets['emotion'].append(emotion_tess_key[props[2][:-4]])
            targets['gender'].append('female')
            y, sr = librosa.load(file)
            targets['length'].append(y.shape[0]/sr)

        file_props = pd.DataFrame(targets)
        common_cols = ['dataset', 'filename', 'actor', 'emotion', 'length', 'gender']
        complete_datasets = pd.concat([file_props[common_cols]], axis = 0)
        split_dataset = self.split_data(complete_datasets)
        if self.verbose:
            self.details_datasets(split_dataset)
            self.save_csv(split_dataset, 'tess')

    def ravdess_dataset_to_csv(self, path: str):
        """
        Reads speech RAVDESS datasets from directory and write it to a CSV file.
        params:
            path : path of dataset's directory
        """
        ravdess_listdir = glob.glob(path + '/*/*.wav')
        emotion_ravdess_key = {'01': 'neutral',
                               '02': 'calm',
                               '03': 'happy',
                               '04': 'sad',
                               '05': 'angry',
                               '06': 'fearful',
                               '07': 'disgusted',
                               '08': 'surprised'}
        targets = {'dataset': [], 'filename': [], 'actor': [], 'emotion': [], 'intensity': [], 'statement': [],
                   'length': [], 'gender': []}
        intensity_key = {'01': 'normal', '02': 'strong'}

        for file in tqdm(ravdess_listdir, desc='[RAVDESS] to CSV and split into train and test(80/20)'):
            targets['dataset'].append('RAVDESS')
            targets['filename'].append(file)
            props = file.split('/')[4].split('.')[0].split('-')
            targets['actor'].append(props[6])
            targets['emotion'].append(emotion_ravdess_key[props[2]])
            targets['intensity'].append(intensity_key[props[3]])
            targets['statement'].append(props[4])

            if int(props[6]) % 2 == 0:
                targets['gender'].append('female')
            else:
                targets['gender'].append('male')

            y, sr = librosa.load(file)
            targets['length'].append(y.shape[0]/sr)
        file_props = pd.DataFrame(targets)
        common_cols = ['dataset', 'filename', 'actor', 'emotion', 'length', 'gender']
        complete_datasets = pd.concat([file_props[common_cols]], axis = 0)
        #complete_datasets.reset_index(drop = True, inplace = True)
        split_dataset = self.split_data(complete_datasets)
        if self.verbose:
            self.details_datasets(split_dataset)
            self.save_csv(split_dataset, 'ravdess')




if __name__ == '__main__':
    '''
    Example : python3 dataset.py -d ./datasets/EMODB/ -n emodb
    '''
    parser = OptionParser()
    parser.add_option('-d', '--dataset_path', dest='path', default='')
    parser.add_option('-n', '--dataset_name', dest='name', default='')

    (options, args) = parser.parse_args(sys.argv)

    dataset_path = options.path
    dataset_name = options.name

    EmotionDatasets(dataset_path, dataset_name)


