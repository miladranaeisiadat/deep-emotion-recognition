from optparse import OptionParser
from re import T
import sys
import numpy as np
from numpy.core.records import array

import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import load_model

from sklearn.metrics import confusion_matrix

from utilities import load_datas, read_emotions_json

import matplotlib.pyplot as plt

import seaborn as sns



class DeepEmotionRecognition():
    """
    The Deep Learning class for detect emotion from sound.
    This class uses 1D convolutional neural network (CNN).
    """
    def __init__(self, input_shape, name_dataset : str):
        """
        Constructor to initialize the deep neural network model. Takes the input
        shape and number of classes.
        Args:
            input_shape (tuple): shape of the input
            num_classes (int): number of different classes ( labels ) in the data.
        """


        self.trained = False
        self.history = ''
        self.name = name_dataset.lower()

        # List of available emotion for each dataset
        self.emotions = read_emotions_json(self.name)

        self.input_shape = input_shape
        self.num_classes = len(self.emotions)
        self.model = Sequential()

        # call create_model for create our neural network
        self.create_model()


    def create_model(self):
        self.model.add(Conv1D(8, 5, padding='same',input_shape=(self.input_shape[1], 1)))  # X_train.shape[1] = No. of Columns
        self.model.add(Activation('relu'))
        self.model.add(Conv1D(16, 5, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(32, 5, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.num_classes)) # Target class number
        self.model.add(Activation('softmax'))
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        self.model.compile(loss='binary_crossentropy', optimizer=opt,
                           metrics=['accuracy'])
        print(self.model.summary(), file=sys.stderr)
        self.save_path_model = 'model/' + self.name + '_best_model.h5'


    def load_model(self, w_path: str):
        """
        Load the model weights from the given path.
        Args:
            to_load (str): path to the saved model file in h5 format.
        """
        try:
            self.model.load_weights(w_path)
        except:
            sys.stderr.write("Invalid saved file provided")
            sys.exit(-1)

    
    def save_model(self):
        """
        Save the model weights to `save_path` provided while creating the model.
        """
        self.model.save(self.save_path_model)

    def train (self, X_train: np.ndarray, 
                     Y_train: np.ndarray, 
                     X_test: np.ndarray, 
                     Y_test: np.ndarray, 
                     n_epochs=500,
                     shuffle = False):
        """
        Train the current model on the given training data which extraxted using MFCCs.
        Args:
            X_train (numpy.ndarray): samples of training data.
            Y_train (numpy.ndarray): labels for training data.
            X_test (numpy.ndarray): Optional, samples in the validation data.
            Y_test (numpy.ndarray): Optional, labels of the validation data.
            n_epochs (int): Number of epochs to be trained.
        """
        #reshape data for feeding to neural network
        self.X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
        self.X_test =  X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

        # make categorical labels
        self.Y_train = to_categorical(Y_train, self.num_classes)
        self.Y_test = to_categorical(Y_test, self.num_classes)

        # initite history variable to plot train and test accuracy
        self.history = self.model.fit(self.X_train, self.Y_train, epochs = n_epochs, shuffle= shuffle, validation_data=(self.X_test, self.Y_test))
        self.model.evaluate(self.X_test, self.Y_test)
        self.trained = True

    
    def show_history(self):
        """
        plot loss for train and test data 
        """
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'ro', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='test loss')
        plt.title('Training and test loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


    def plot_confusion(self, normalize = False):
        """
        print the relevant metrics like confusion matrix.
        """
        Y_test = np.argmax(self.Y_test, axis=-1)
        Y_pred = np.argmax(self.model.predict(self.X_test), axis=-1)
        matrix = confusion_matrix(Y_test, Y_pred)
        if normalize == True:
            matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10,8))
        sns.heatmap(matrix , annot = True,
                    xticklabels = self.emotions,
                    yticklabels = self.emotions)
        plt.xlabel('Prediction')
        plt.ylabel('Truth')
        plt.show()



if __name__ == '__main__':
    '''
    Example : python3 deep_emotion_recognition -d emodb -p datasets/csv/emodb.csv -l -a -s -c
    '''
    parser = OptionParser()
    parser.add_option('-d', '--dataset', help='Name of dataset which wanna train')
    parser.add_option('-p', '--csv_path', help='Path of csv file where stored on disk')
    parser.add_option('-l', '--load_data', action='store_true', help='Load data and feature extracting and split to train and test')
    parser.add_option('-a', '--show_accuracy', action='store_true', help='Show accuracy plot for train and test data')
    parser.add_option('-s', '--save_model', action='store_true', help='Save model into the model folder')
    parser.add_option('-c', '--plot_confusion', action='store_true', help='Plot confusion matrix for determine how model predict well')


    (options, args) = parser.parse_args()

    dataset = options.dataset
    path = options.csv_path
    load_data = options.load_data
    show_accuracy = options.show_accuracy
    save_model = options.save_model
    plot_confusion = options.plot_confusion

    if load_data:
        X_train, Y_train, X_test, Y_test = load_datas(path)
        model = DeepEmotionRecognition(input_shape = X_train.shape, name_dataset=dataset)
        model.train(X_train, Y_train, X_test, Y_test)
        if save_model:
            model.save_model()
        if show_accuracy:
            model.show_history()
        if plot_confusion:
            model.plot_confusion()

    else:
        print("[Error] - please load data first")
    
