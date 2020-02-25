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

from keras.layers import Input
from keras import regularizers
from keras.utils.test_utils import layer_test
from keras.layers import normalization
from keras.models import Sequential, Model
from keras import backend as K


class data_generator(Sequence):

    def __init__(self, mode="dev"):
        if mode == "dev":
            pickle_in = open("../../data/dev.pickle", "rb")
            self.all_data = pickle.load(pickle_in)
        else:
            pickle_in = open("../../data/dataset.pickle", "rb")
            self.all_data = pickle.load(pickle_in)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
            return self.all_data[idx-1][0],self.all_data[idx-1][1]


def biggerModel ():
    data_dim = 34
    num_classes = 16


    # model
    model = Sequential()
#    model.add(Dropout(0.2, input_shape=(None,data_dim)))
    model.add(Dense(units=64,input_shape=(None,data_dim)))
    model.add(BatchNormalization(trainable=False))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
#    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))


    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    # callback to save model every 10 epochs
    #save_model = ModelCheckpoint('Sec_weights/weights{epoch:08d}.h5',
    #                                     save_weights_only=False, period=10)


  #  model.fit_generator(generator=data_generator("dataset"),steps_per_epoch=2129,epochs=130,callbacks=[save_model]
  #                      ,use_multiprocessing=True, workers=2,shuffle=True)
    model.load_weights("weights00000020.h5")
    test=model.evaluate_generator(generator=data_generator("dataset"), steps=1000
                        , max_queue_size=10, workers=2, use_multiprocessing=True, verbose=0)
    print(test)
biggerModel()


