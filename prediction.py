import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from optparse import OptionParser

from keras.models import Model, load_model
from numpy.core.records import array

from utilities import extract_feature_from_mfcc, read_emotions_json

from rich.console import Console
from rich.table import Table

def predict_new_data(raw_audio: str, model: Model, emotions: array) -> str:
    """
    Predict labels for given wav sound.
    Args:
        raw_audio : Audio wav sound which blong to one of the dataset.
        model: Appropriate mode which traind and saved.
        emotions: Array of available emotions of each dataset. 
    Returns:
        str: returns the name of emotion which predict from one sample data.
    """
    feature = extract_feature_from_mfcc(raw_audio).reshape(1,50,1)
    predict_emotion = emotions[np.argmax(model.predict(feature))]
    return predict_emotion

def predict_new_data_value(raw_audio: str, model: Model, emotions: array) -> array:
    """
    Predict label or category with probability for given wav sound. 
    Args:
        raw_audio : Audio wav sound which blong to one of the dataset.
        model: Appropriate mode which traind and saved.
        emotions: Array of available emotions of each dataset. 
    Returns:
        int: returns the label for the sample.
    """
    feature = extract_feature_from_mfcc(raw_audio).reshape(1,50,1)
    predict_value = model.predict(feature)[0]
    result = {}
    for prob, emotion in zip(predict_value, emotions):
        result[emotion] = prob
    return result


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--predicted_wav_path')
    parser.add_option('-m', '--model_path' )

    (options, args) = parser.parse_args()

    wav_path = options.predicted_wav_path
    model_path = options.model_path

    name_dataset = str(os.path.splitext(os.path.basename(model_path))[0]).split('_')[0]
    emotions = read_emotions_json(name_dataset)

    # load model
    model = load_model(model_path)
    res_predict = predict_new_data(wav_path, model, emotions)
    res_predict_value = predict_new_data_value(wav_path, model, emotions)


    # Create table for demonstrate prediction of wav sound 
    print('\n')
    table = Table(title="Prediction wav sound")

    table.add_column()
    table.add_column("Prediction", justify="center", style="#FFCB0B bold")

    table.add_row("prediction is: ", res_predict)

    console = Console()
    console.print(table, '\n')

    print("------------------------------------------------------- \n")
    # Create table for demonstrate prediction value of wav sound 
    table = Table(title="Score value of prediction of wav sound")

    table.add_column()
    table.add_column("Prediction", justify="center", style="#4BFCEA bold")

    for key in res_predict_value:
        table.add_row(key, str(res_predict_value[key]))


    console = Console()
    console.print(table, '\n')


