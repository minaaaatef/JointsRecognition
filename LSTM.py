from keras.models import Sequential,model_from_json
from keras.layers import LSTM, Dense, BatchNormalization,Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import random
from keras.utils import Sequence
import multiprocessing
import matplotlib.pyplot as plt
import pickle

import numpy as np
from numpy.testing import assert_allclose


class data_generator(Sequence):

    def __init__(self, mode="dev"):
        if mode == "dev":
            pickle_in = open("data/dev.pickle", "rb")
            self.all_data = pickle.load(pickle_in)
        else:
            pickle_in = open("data/dataset.pickle", "rb")
            self.all_data = pickle.load(pickle_in)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
            return self.all_data[idx-1][0],self.all_data[idx-1][1]


def Init_model ():
    data_dim = 34
    num_classes = 16
    
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,input_shape=(None, data_dim)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model


def load_last_model(path,model,model_name):
    models = []
    for file in os.listdir(path):
        if file.startswith(model_name):
            models.append(file)
    models.sort()
    inital_epoch = 0
    if models.__len__() > 2:
        model.load_weights(path + model[-1])
        inital_epoch = int(models[-1][:-6][:3])
    return model,inital_epoch

def train (model_name,path = "",epochs=100):
    model = Init_model()
    save_model = ModelCheckpoint(model_name+'weights{epoch:08d}.h5',
                                         save_weights_only=False, period=10)
    model,inital_epoch_num = load_last_model(path,model,model_name)
    model.fit_generator(data_generator('dataset'), epochs, callbacks=[save_model],initial_epoch=inital_epoch)
    

def ecaluate(model_name,mode = 'dev',path = ""):
    valid = ['dev','dataset']
    if mode not in valid:
        raise ValueError("results: status must be one of %r." % valid)

    model = Init_model()
    model = load_last_model(path,model,model_name)
    print(model.evaluate_generator(data_generator(mode)))



def draw_plots(path,model_name):
    model = Init_model()
    
    models = []
    for file in os.listdir(path):
        if file.startswith(model_name):
            models.append(file)
    models.sort()

    accuracylist = []
    for x in models:
        model.load_weights(path+x)
        loss,accuracy = model.evaluate_generator(data_generator('dataset'))
        accuracylist.append(accuracy)

    plt.plot(accuracylist)
    plt.title('Test Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epochs#')
    plt.savefig('dataset.png')

    accuracylist = []
    for x in models:
        model.load_weights(path+x)
        loss,accuracy = model.evaluate_generator(data_generator('dev'))
        accuracylist.append(accuracy)

    plt.plot(accuracylist)
    plt.title('Test Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epochs#')
    plt.savefig('dev.png')




