from keras.models import Sequential,model_from_json
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
import numpy as np
from Inputprocessing.InputProcessingv2 import InputProccessing
import os
import random




def prepare_input_Matrix(path):
    print("training The data")
    y_train = []
    all_data = []
    
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

    print(path)
    for subdir, dirs, files in os.walk(path,topdown=True):
        print("outerloop")
        for file in files:
            try:
                print("innerloop")
                x_train = InputProccessing(0.9,os.path.join(subdir,file),20)
                y_train = labels[os.path.basename(subdir)]
                y_train = to_categorical(y_train,num_classes=16)
                data = (np.array(x_train),np.array(y_train).reshape((1,16)))
                all_data.append(data)
            except:
                print("try failed")

    while True:
        random.shuffle(all_data)
        print("epochs {}".format(epochs))
        epochs += 1
        for data in all_data:
            data = list(data)
            yield data[0],data[1]

                



def model ():


    data_dim = 34
    num_classes = 16

    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                input_shape=(None, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.summary()
    
    model.fit_generator(generator=prepare_input_Matrix(r'dataset'),steps_per_epoch=2130,epochs=500)
    print ('training finshed')

    model.evaluate_generator(generator=prepare_input_Matrix(r'validtion set'),steps=514)
    print ('evaluate finshed')

    model.save('model.h5')


model ()

