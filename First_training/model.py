from keras.models import Sequential,model_from_json
from keras.layers import LSTM, Dense, BatchNormalization,Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
from InputProcessingv2 import InputProccessing
import os
import random
from keras.utils import Sequence
import multiprocessing
import matplotlib.pyplot as plt
import pickle
from keras.backend import cntk_backend


'''
Class to feed the data to the model while training
Thread save so we could use use_multiprocessing = True
'''
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


#a fn to save the date to the to a pickle file
def save_list (path):
    print("training The data")
    y_train = []
    all_data = []

    labels = {'clap': 0,
              'climb': 1,
              'climb_stairs': 2,
              'hit': 3,
              'jump': 4,
              'kick': 5,
              'pick': 6,
              'punch': 7,
              'push': 8,
              'run': 9,
              'sit': 10,
              'situp': 11,
              'stand': 12,
              'turn': 13,
              'walk': 14,
              'wave': 15,
              }

    epochs = 1

    count = 1
    print("is training {}".format(path))
    for subdir, dirs, files in os.walk(path, topdown=True):
        for file in files:
            try:
                count += 1
                if (count % 100 == 0):
                    print("preparing data # {}".format(count))

                x_train = InputProccessing(0.9, os.path.join(subdir, file), 20)
                y_train = labels[os.path.basename(subdir)]
                y_train = to_categorical(y_train, num_classes=16)
                data = (np.array(x_train), np.array(y_train).reshape((1, 16)))
                all_data.append(data)
            except:
                print("try failed")


    pickle_out = open("dev.pickle", "wb")
    pickle.dump(all_data, pickle_out)
    pickle_out.close()

'''
A function could replace the class data_generator
But it couldn't be used with use_multiprocessing = True
'''
def prepare_input_Matrix(mode="dev"):
    if mode == "dev":
        pickle_in = open("data/dev.pickle", "rb")
        all_data = pickle.load(pickle_in)
    else:
        pickle_in = open("data/dataset.pickle", "rb")
        all_data = pickle.load(pickle_in)
    while True:
        random.shuffle(all_data)
        for data in all_data:
            data = list(data)
            yield data[0], data[1]

# the model fn
def base ():
    data_dim = 34
    num_classes = 16


    # model
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,input_shape=(None, data_dim)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    # callback to save model every 10 epochs
    save_model = ModelCheckpoint('weights{epoch:08d}.h5',
                                         save_weights_only=False, period=10)


    model.fit_generator(generator=data_generator("dataset"),steps_per_epoch=2129,epochs=500,callbacks=[save_model]
                        ,use_multiprocessing=True, workers=2 )





base()


