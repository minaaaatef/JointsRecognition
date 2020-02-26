from keras.models import Sequential,model_from_json
from keras.layers import LSTM, Dense, BatchNormalization,Dropout,Flatten
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import random
from keras.utils import Sequence
import multiprocessing
import matplotlib.pyplot as plt
import pickle

from numpy.testing import assert_allclose

from keras.layers import Input
from keras import regularizers

from keras.utils.test_utils import layer_test
from keras.layers import normalization
from keras.models import Sequential, Model
from keras import backend as K


class data_generator(Sequence):

    def __init__(self, mode="dev"):
        if mode == "dev":
            pickle_in = open("../data/dev.pickle", "rb")
            self.all_data = pickle.load(pickle_in)
        else:
            pickle_in = open("../data/dataset.pickle", "rb")
            self.all_data = pickle.load(pickle_in)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
            return self.all_data[idx-1][0],self.all_data[idx-1][1]


def DenseModel ():
    model_name='DENSE-Model'
    data_dim = 34
    num_classes = 16


    # model
    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=(data_dim,)))
    model.add(Dense(8, activation='relu')) 
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    # callback to save model every 10 epochs
    save_model = ModelCheckpoint(model_name+'weights{epoch:08d}.h5',
                                         save_weights_only=False, period=10)


    model.fit_generator(generator=data_generator("dataset"),steps_per_epoch=2129,epochs=500,callbacks=[save_model]
                        ,use_multiprocessing=True, workers=2,shuffle=True)



DenseModel()




