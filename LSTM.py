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
#
# class data_generator(Sequence):
#
#     def __init__(self, path):
#         self.batch_size = 1
#         print("training The data")
#         self.y_train = []
#         self.all_data = []
#
#         labels = {'clap': 0,
#                   'climb': 1,
#                   'climb_stairs': 2,
#                   'hit': 3,
#                   'jump': 4,
#                   'kick': 5,
#                   'pick': 6,
#                   'punch': 7,
#                   'push': 8,
#                   'run': 9,
#                   'sit': 10,
#                   'situp': 11,
#                   'stand': 12,
#                   'turn': 13,
#                   'walk': 14,
#                   'wave': 15,
#                   }
#
#         self.epochs = 1
#         self.count = 1
#         print(path)
#         for subdir, dirs, files in os.walk(path, topdown=True):
#             for file in files:
#                 try:
#                     self.count += 1
#                     if (self.count % 100 == 0):
#                         print("preparing data # {}".format(self.count))
#
#                     x_train = InputProccessing(0.9, os.path.join(subdir, file), 20)
#                     y_train = labels[os.path.basename(subdir)]
#                     y_train = to_categorical(y_train, num_classes=16)
#                     data = (np.array(x_train), np.array(y_train).reshape((1, 16)))
#                     self.all_data.append(data)
#                 except:
#                     print("try failed")
#         print(self.all_data.__len__())
#         random.shuffle(self.all_data)
#
#     def __len__(self):
#         return 2129
#
#     def __getitem__(self, idx):
#             return self.all_data[idx-1][0],self.all_data[idx-1][1]
#

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


def patchnormmodel ():

    data_dim = 34
    num_classes = 16

    model = Sequential()
    model.add(LSTM(32, return_sequences=True,input_shape=(None, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(BatchNormalization())
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(BatchNormalization())
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    save_model = ModelCheckpoint('patchnormmodel/weights{epoch:08d}.h5',
                                         save_weights_only=False, period=10)
    model.summary()



    model.fit_generator(generator=prepare_input_Matrix("dataset"),steps_per_epoch=2129,epochs=500,callbacks=[save_model])
    # print ('training finshed')






def biggermodel ():


    data_dim = 34
    num_classes = 16

    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(None, data_dim)))  # returns a sequence of vectors of dimension 32
    # model.add(BatchNormalization())
    model.add(LSTM(256, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(256, return_sequences=True))  # returns a sequence of vectors of dimension 32
    # model.add(BatchNormalization())
    model.add(LSTM(256))  # return a single vector of dimension 32
    # model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    save_model = ModelCheckpoint('biggermodel/weights{epoch:08d}.h5',
                                         save_weights_only=False, period=10)
    model.summary()



    model.fit_generator(generator=prepare_input_Matrix("dataset"),steps_per_epoch=2129,epochs=500,callbacks=[save_model])
    print ('training finshed')



def dropout ():


    data_dim = 34
    num_classes = 16

    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(None, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(256, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(Dropout(0.5))
    model.add(LSTM(256))  # return a single vector of dimension 32
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    save_model = ModelCheckpoint('dropout/weights{epoch:08d}.h5',
                                         save_weights_only=False, period=10)
    model.summary()



    model.fit_generator(generator=prepare_input_Matrix("dataset"),steps_per_epoch=2129,epochs=500,callbacks=[save_model])
    print ('training finshed')


patchnormmodel()
dropout ()
biggermodel()