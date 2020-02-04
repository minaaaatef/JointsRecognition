from keras.models import Sequential,model_from_json
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
import numpy as np
from Inputprocessing.InputProcessingv2 import InputProccessing
import os





def prepare_input_Matrix():
    print("training The data")
    path = r'dataset'
    y_train = []
    count = 1

    labels = {'clap':0,
              'climb':1,
              'climb_stairs':2,
              'hit':3,
              'jump':4,
              'kick':5,
              'pick':6,
              'punch':7,
              'push':8,
              'run':9,
              'sit':10,
              'situp':11,
              'stand':12,
              'turn':13,
              'walk':14,
              'wave':15,
              }
    epochs = 1

    while True:
        print("epochs {}".format(epochs))
        epochs += 1
        for subdir, dirs, files in os.walk(path,topdown=True):
            for file in files:

                try:
                    x_train = InputProccessing(0.9,os.path.join(subdir,file),20)
                    y_train = labels[os.path.basename(subdir)]

                    y_train = to_categorical(y_train,num_classes=16)
                    yield np.array(x_train),np.array(y_train).reshape((1,16))
                except:
                    pass



def model ():


    data_dim = 34
    num_classes = 16


    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                input_shape=(None, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(16, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.summary()

    model.fit_generator(generator=prepare_input_Matrix(),steps_per_epoch=2593,epochs=1,use_multiprocessing=True,workers=16)
    print ('training finshed')

    model.evaluate_generator(generator=prepare_input_Matrix(),steps=2593)
    print ('evaluate finshed')

    model.save('model.h5')


model ()

